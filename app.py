from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import serial
from spatialmath import SE3

from config import AppConfig
from enum import Enum
from session import KeyOffsetStore, PerformanceTracker, TuneSessionManager
from tracgenaruco import PianoArmController
from pianokeymodel import PianoKeyModel
from worldbasetrackerhomography import WorldBaseTrackerHomography
from greencircle import GreenBlobDetector


@dataclass
class SafetyState:
    stop_requested: bool = False
    serial_ok: bool = True
    marker_ok: bool = False
    blob_ok: bool = False
    last_marker_seen_t: float = 0.0
    last_blob_seen_t: float = 0.0
    last_stop_reason: str = ""

class OperatorMode(str, Enum):
    PLAY = "PLAY"
    CALIBRATE = "CALIBRATE"
    EVALUATE = "EVALUATE"

class VisionPID2D:
    """
    Visual alignment PID controller used by align_to_key_center_pid().

    This implements the closed-loop visual feedback portion of the alignment
    pipeline.  It measures the 2D world-frame error between the detected
    end-effector position and the desired strike target, then computes a
    proportional-dominant correction command.

    In the current tuning, ki and kd are both zero, so the controller
    operates as a bounded P-only controller.  The PID structure is retained
    in the software so that I and D terms can be enabled during future
    experiments without changing the calling code.

    Key safety features:
      - Integral clamping to prevent wind-up
      - Sign-change reset of the integral term
      - Derivative filtering via exponential smoothing
    """
    def __init__(
        self,
        dt: float,
        kp: tuple[float, float] = (0.55, 0.55),
        ki: tuple[float, float] = (0.03, 0.03),
        kd: tuple[float, float] = (0.00, 0.00),
        i_limit: tuple[float, float] = (0.006, 0.006),
        d_alpha: float = 0.7,
    ) -> None:
        self.dt_default = dt
        self.kp = np.array(kp, dtype=float)
        self.ki = np.array(ki, dtype=float)
        self.kd = np.array(kd, dtype=float)
        self.i_limit = np.array(i_limit, dtype=float)
        self.d_alpha = float(d_alpha)
        self.reset()

    def reset(self) -> None:
        self.e_int = np.zeros(2, dtype=float)
        self.e_prev = None
        self.t_prev = None
        self.d_filt = np.zeros(2, dtype=float)

    def update(self, error_xy: np.ndarray, t_now: float) -> np.ndarray:
        e = np.asarray(error_xy, dtype=float)
        if self.t_prev is None:
            dt = self.dt_default
            d = np.zeros(2, dtype=float)
        else:
            dt = max(1e-3, t_now - self.t_prev)
            d_raw = (e - self.e_prev) / dt
            self.d_filt = self.d_alpha * self.d_filt + (1.0 - self.d_alpha) * d_raw
            d = self.d_filt

        if self.e_prev is not None:
            sign_flip = np.sign(e) != np.sign(self.e_prev)
            self.e_int[sign_flip] *= 0.25

        self.e_int += e * dt
        self.e_int = np.clip(self.e_int, -self.i_limit, self.i_limit)
        output = self.kp * e + self.ki * self.e_int + self.kd * d

        self.e_prev = e.copy()
        self.t_prev = t_now
        return output


class PianoBotApp:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.cfg = config or AppConfig()

        self.arm: PianoArmController | None = None
        self.cap: cv2.VideoCapture | None = None
        self.key_model: PianoKeyModel | None = None
        self.tracker: WorldBaseTrackerHomography | None = None
        self.blob: GreenBlobDetector | None = None

        self.mode = OperatorMode.PLAY
        self.song_names: list[str] = []
        self.current_song_idx: int = 0

        self.last_ui_message: str = ""
        self.last_ui_message_t: float = 0.0
        self.ui_message_timeout_s: float = 2.0

        self.safety = SafetyState(
            last_marker_seen_t=time.time(),
            last_blob_seen_t=time.time(),
        )

        self.offsets = KeyOffsetStore(self.cfg.n_keys, self.cfg)
        self.stats = PerformanceTracker(self.cfg.n_keys, self.cfg)
        self.tune = TuneSessionManager(self.cfg)

        self.last_tracker_ok = False
        self.last_play_xy: np.ndarray | None = None
        self.last_play_idx: int | None = None
        self.last_align_ok = False
        self.last_best_err_m: float | None = None
        self.last_finetune_ref_xy: np.ndarray | None = None
        self.align_good_streak = 0
        self.last_motion_class = "none"
        self.last_first_over_mA_ms: float | None = None
        self.auto_press_profile_enabled = self.cfg.use_auto_press_profile
        self.press_profile_name = self.cfg.default_press_profile
        self.last_press_profile_used = self.cfg.default_press_profile
        self.auto_press_base_key_idx = self.cfg.auto_press_base_key_fallback_idx

        self.ik_x_scale = self.cfg.ik_x_scale_default
        self.ik_x_bias_m = self.cfg.ik_x_bias_m_default
        self.tune_dx_m = 0.0
        self.last_launch_to_contact_ms: float | None = None

    # ----------------------------
    # Lifecycle
    # ----------------------------
    def connect(self) -> None:
        self.arm = PianoArmController(port=self.cfg.com_port, dt=self.cfg.dt)
        self.arm.connect()
        print("✅ Robot connected")
        time.sleep(2.0)
        self._clear_serial_input()

        if self.cfg.use_ina219_logger:
            try:
                time.sleep(0.2)
                self._clear_serial_input()
                print("✅ INA219 logger enabled")
            except Exception as exc:
                print(f"⚠️ Could not initialize shared INA219 logger: {exc}")

        self.arm.set_servo5_smooth(self.cfg.servo5_neutral_deg, steps=12, dt=0.02)
        print(f"✅ Servo 5 neutral set to {self.cfg.servo5_neutral_deg:.1f} deg")

        self.key_model = PianoKeyModel(
            self.cfg.n_keys,
            self.cfg.x_id10_to_f1,
            self.cfg.key_pitch,
            self.cfg.y_key,
        )
        self.tracker = WorldBaseTrackerHomography(
            id10=self.cfg.id10,
            id11=self.cfg.id11,
            id_base=self.cfg.id_base,
            marker_size_m=self.cfg.marker_size_m,
            baseline_m=self.cfg.baseline_m,
            mr_to_base_xyz_m=self.cfg.mr_to_base_xyz,
            mr_to_base_yaw_deg=self.cfg.mr_to_base_yaw_deg,
        )
        self.blob = GreenBlobDetector(
            lower_hsv=(35, 70, 70),
            upper_hsv=(85, 255, 255),
            min_area=6,
        )

        self.cap = cv2.VideoCapture(self.cfg.cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError("❌ Could not open camera")

        self.offsets.load()
        self.refresh_song_library()
        current_song = self.get_current_song_name()
        if current_song is not None:
            print(f"✅ Current song: {current_song}")
        print(f"PID_MAX_TOTAL_CORR_M = {self.cfg.pid_max_total_corr_m * 1000:.1f} mm")

    def disconnect(self) -> None:
        try:
            if self.cfg.enable_end_reset_pose and self.arm is not None:
                self.goto_system_neutral_pose(reason="program exit")
        except Exception:
            pass

        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

        if self.arm is not None:
            self.arm.disconnect()

    def goto_system_neutral_pose(self, reason: str = "") -> bool:
        try:
            if reason:
                print(f"↩ Moving to neutral pose: {reason}")
            else:
                print("↩ Moving to neutral pose")
            assert self.arm is not None
            self.arm.goto_neutral_pose(
                q_neutral_deg=(0.0, 0.0, 180.0),
                servo5_deg=self.cfg.neutral_servo5_deg,
                steps=self.cfg.neutral_move_steps,
                dt=self.cfg.neutral_move_dt,
            )
            return True
        except Exception as exc:
            print(f"⚠️ Could not move to neutral pose: {exc}")
            return False

    def _clear_serial_input(self) -> None:
        assert self.arm is not None
        if hasattr(self.arm, "clear_serial_input"):
            self.arm.clear_serial_input()
        elif hasattr(self.arm, "_ser") and self.arm._ser is not None:
            self.arm._ser.reset_input_buffer()

    def cycle_mode(self) -> None:
        order = [
            OperatorMode.PLAY,
            OperatorMode.CALIBRATE,
            OperatorMode.EVALUATE,
        ]
        idx = order.index(self.mode)
        self.mode = order[(idx + 1) % len(order)]
        print(f"\n=== Mode: {self.mode.value} ===")
        self.print_controls()


    def set_mode(self, mode: OperatorMode) -> None:
        self.mode = mode
        print(f"\n=== Mode: {self.mode.value} ===")
        self.print_controls()

    def handle_play_mode_keypress(self, key_code: int) -> bool:
        if key_code == 13 and self.tune.typed:
            idx = int(self.tune.typed)
            self.tune.typed = ""
            if 0 <= idx < self.cfg.n_keys:
                self.execute_key(idx)
            else:
                print("Out of range 0..21")
            return True

        elif key_code == ord('m'):
            self.play_song()
            return True

        elif key_code == ord('['):
            self.cycle_song(-1)
            return True

        elif key_code == ord(']'):
            self.cycle_song(1)
            return True

        elif key_code == ord('p'):
            self.print_song_library()
            return True

        elif key_code == ord('-'):
            self.cycle_press_profile(-1)
            return True

        elif key_code == ord('='):
            self.cycle_press_profile(1)
            return True

        elif key_code == ord(';'):
            self.print_current_press_profile()
            return True

        elif key_code == ord('k'):
            self.auto_press_profile_enabled = not self.auto_press_profile_enabled
            print(f"auto press profile = {'ON' if self.auto_press_profile_enabled else 'OFF'}")
            return True

        return False
    
    def handle_calibrate_mode_keypress(self, key_code: int) -> bool:
        if key_code == 13 and self.tune.typed:
            idx = int(self.tune.typed)
            self.tune.typed = ""
            if 0 <= idx < self.cfg.n_keys:
                self.execute_key(idx)
            else:
                print("Out of range 0..21")

        elif key_code in {ord('a'), ord('d'), ord('w'), ord('x')}:
            self._handle_offset_nudge(key_code)

        elif key_code == ord('z'):
            idx_sel = self.get_selected_idx()
            if idx_sel is None:
                print("Type a key index first, then press 'z'")
            else:
                self.offsets.reset_one(idx_sel)
                x_tgt, y_tgt = self.strike_target_world_xy(idx_sel)
                print(f"key {idx_sel} target offset reset to zero   strike_tgt=({x_tgt:.4f}, {y_tgt:.4f})")

        elif key_code == ord('e'):
            idx_sel = self.get_selected_idx()
            if idx_sel is None:
                print("Type a key index first, then press 'e'")
            else:
                self.offsets.print_one(idx_sel)

        elif key_code == ord('v'):
            self.offsets.save()

        elif key_code == ord('b'):
            self.offsets.load()

        elif key_code == ord('n'):
            self.offsets.reset_all()

        elif key_code == ord('t'):
            idx_sel = self.get_selected_idx()
            if idx_sel is None:
                print("Type a key index first, then press 't'")
            else:
                self.offsets.print_suggestion(idx_sel)

        elif key_code == ord('y'):
            idx_sel = self.get_selected_idx()
            if idx_sel is None:
                print("Type a key index first, then press 'y'")
            else:
                self.offsets.apply_suggestion(idx_sel)

        elif key_code == ord('u'):
            idx_sel = self.get_selected_idx()
            if idx_sel is None:
                print("Type a key index first, then press 'u'")
            else:
                self.offsets.clear_suggestion(idx_sel)

        elif key_code == ord('g'):
            self.run_white_key_sweep()

        elif key_code == ord('h'):
            self.stats.print_key_perf_stats()

        elif key_code == ord('i'):
            self.stats.reset_key_perf_stats()

        elif key_code == ord('l'):
            self.stats.print_ranked_white_key_report()

        elif key_code == ord('/'):
            self.run_servo5_motion_test()

        elif key_code == ord('j'):
            self.select_worst_tested_white_key()

        elif key_code == ord('.'):
            self.run_selected_key_repeated_test()

        elif key_code == ord('['):
            self.select_prev_weak_key()

        elif key_code == ord(']'):
            self.select_next_weak_key()

        elif key_code == ord('s'):
            self.tune.save(self.stats.build_white_key_report_rows())

        elif key_code == ord('p'):
            self.tune.load()
            if self.tune.weak_key_queue and 0 <= self.tune.weak_key_queue_pos < len(self.tune.weak_key_queue):
                self.print_focus_key_summary(self.tune.weak_key_queue[self.tune.weak_key_queue_pos])

        elif key_code == ord('f'):
            self.print_tuning_todo_list()
            self.print_remaining_weak_queue()

        elif key_code == ord('c'):
            self.print_automatic_tuning_checklist()

        elif key_code == ord('J'):
            self.assist_selected_key()

        return True

    def handle_evaluate_mode_keypress(self, key_code: int) -> bool:
        if key_code == ord('r'):
            self.run_selected_key_profile_validation()

        elif key_code == ord('o'):
            self.stats.summarize_validation_run()

        elif key_code == ord('A'):
            self.run_repeat_timing_test()

        elif key_code == ord('B'):
            self.stats.print_timing_error_stats()

        elif key_code == ord('C'):
            self.stats.print_timing_session_summary()

        elif key_code == ord('D'):
            self.stats.print_session_evaluation_summary()

        elif key_code == ord('E'):
            self.stats.save_session_evaluation_summary()

        return True
    
    def flash_ui_message(self, text: str) -> None:
        self.last_ui_message = str(text)
        self.last_ui_message_t = time.time()
        print(text)


    def refresh_song_library(self) -> None:
        songs = sorted(p.stem for p in self.cfg.paths.songs_dir.glob("*.json"))
        self.song_names = songs

        if not self.song_names:
            self.current_song_idx = 0
            return

        if self.cfg.default_song_name in self.song_names:
            if not (0 <= self.current_song_idx < len(self.song_names)):
                self.current_song_idx = self.song_names.index(self.cfg.default_song_name)
        else:
            self.current_song_idx = min(self.current_song_idx, len(self.song_names) - 1)


    def get_current_song_name(self) -> str | None:
        if not self.song_names:
            return None
        if 0 <= self.current_song_idx < len(self.song_names):
            return self.song_names[self.current_song_idx]
        return self.song_names[0]


    def cycle_song(self, delta: int) -> None:
        self.refresh_song_library()
        if not self.song_names:
            self.flash_ui_message("No song JSON files found in songs/")
            return

        self.current_song_idx = (self.current_song_idx + int(delta)) % len(self.song_names)
        self.flash_ui_message(f"Current song: {self.get_current_song_name()}")


    def print_song_library(self) -> None:
        self.refresh_song_library()
        print("\n=== Song library ===")
        if not self.song_names:
            print("No songs found in songs/\n")
            return

        current = self.get_current_song_name()
        for i, name in enumerate(self.song_names):
            marker = "->" if name == current else "  "
            print(f"{marker} [{i}] {name}")
        print()

    def draw_text_panel(
        self,
        frame: np.ndarray,
        lines: list[str],
        x: int,
        y: int,
        line_h: int = 22,
        pad: int = 8,
    ) -> None:
        if not lines:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1

        widths = []
        for line in lines:
            (w, _), _ = cv2.getTextSize(line, font, scale, thickness)
            widths.append(w)

        box_w = max(widths) + 2 * pad
        box_h = len(lines) * line_h + 2 * pad

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        for i, line in enumerate(lines):
            yy = y + pad + (i + 1) * line_h - 6
            cv2.putText(frame, line, (x + pad, yy), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    def get_mode_help_lines(self) -> list[str]:
        lines = [f"MODE: {self.mode.value}", "TAB cycle mode"]

        if self.mode == OperatorMode.PLAY:
            lines += [
                "ENTER play selected key",
                "m play current song",
                "* start trigger (while ready)",
                "[ / ] prev/next song",
                "p print song list",
                "- / = press profile",
                "; profile info",
            ]
        elif self.mode == OperatorMode.CALIBRATE:
            lines += [
                "ENTER play selected key",
                "a/d/w/x nudge offset",
                "z reset key offset",
                "t/y/u suggestion",
                ". repeated test",
                "j worst key",
                "J assist selected key",
            ]
        else:
            lines += [
                "r validation",
                "o validation summary",
                "A repeat timing",
                "B/C/D/E reports",
            ]

        return lines

    def get_selected_key_panel_lines(self) -> list[str]:
        lines: list[str] = []

        idx_sel = self.get_selected_idx()
        lines.append(f"Selected key: {'-' if idx_sel is None else idx_sel}")

        if self.mode == OperatorMode.PLAY:
            lines.append(f"Song: {self.get_current_song_name() or '-'}")

        prof_name = self.press_profile_name if not self.auto_press_profile_enabled else "AUTO"
        lines.append(f"Press profile: {prof_name}")

        if idx_sel is not None:
            dx_i, dy_i = self.offsets.get(idx_sel)
            lines.append(f"Offset dx={dx_i*1000:.1f} mm")
            lines.append(f"Offset dy={dy_i*1000:.1f} mm")

            suggestion, n_sugg = self.offsets.get_suggestion(idx_sel)
            if suggestion is None:
                lines.append("Suggestion: none")
            else:
                lines.append(f"Suggest n={n_sugg}")
                lines.append(f"sx={suggestion[0]*1000:.1f} mm")
                lines.append(f"sy={suggestion[1]*1000:.1f} mm")

            if self.mode == OperatorMode.CALIBRATE:
                action, note = self.recommend_key_action(idx_sel)
                lines.append(f"Action: {action}")
                lines.append(note)

        return lines
    
    def draw_ui_message(self, frame: np.ndarray) -> None:
        if not self.last_ui_message:
            return
        if (time.time() - self.last_ui_message_t) > self.ui_message_timeout_s:
            return

        self.draw_text_panel(frame, [self.last_ui_message], 20, 90)

    # ----------------------------
    # Safety
    # ----------------------------
    def request_safe_stop(self, reason: str) -> None:
        self.safety.stop_requested = True
        self.safety.last_stop_reason = str(reason)
        print(f"🛑 SAFE STOP requested: {reason}")

    def clear_safe_stop(self) -> None:
        self.safety.stop_requested = False
        self.safety.last_stop_reason = ""
        self.safety.serial_ok = True
        print("✅ Safe stop cleared")

    def handle_serial_failure(self, reason: str, try_reconnect: bool = True) -> bool:
        self.safety.serial_ok = False
        self.request_safe_stop(reason)
        print(f"⚠️ Serial failure: {reason}")

        if not try_reconnect:
            return False

        try:
            assert self.arm is not None
            print("Attempting serial reconnect...")
            self.arm.disconnect()
            time.sleep(0.5)
            self.arm.connect()
            time.sleep(0.2)
            self._clear_serial_input()
            self.safety.serial_ok = True
            print("✅ Serial reconnect successful")
            return True
        except Exception as exc:
            print(f"❌ Serial reconnect failed: {exc}")
            return False

    def can_play_note_now(self) -> bool:
        return (not self.safety.stop_requested) and self.safety.serial_ok

    def update_safety_flags_from_vision(self, tracker_ok: bool, blob_seen: bool) -> None:
        now = time.time()
        if tracker_ok:
            self.safety.marker_ok = True
            self.safety.last_marker_seen_t = now
        else:
            self.safety.marker_ok = (now - self.safety.last_marker_seen_t) <= self.cfg.marker_loss_grace_s

        if blob_seen:
            self.safety.blob_ok = True
            self.safety.last_blob_seen_t = now
        else:
            self.safety.blob_ok = (now - self.safety.last_blob_seen_t) <= self.cfg.blob_loss_grace_s

    def get_command_mode_hint(self, key_code: int) -> str | None:
        play_keys = {13, ord('m'), ord('['), ord(']'), ord('p'), ord('-'), ord('='), ord(';'), ord('k')}
        calibrate_keys = {
            ord('a'), ord('d'), ord('w'), ord('x'), ord('z'), ord('e'),
            ord('v'), ord('b'), ord('n'), ord('t'), ord('y'), ord('u'),
            ord('g'), ord('h'), ord('i'), ord('l'), ord('/'),
            ord('j'), ord('.'), ord('J'), ord('['), ord(']'),
            ord('s'), ord('p'), ord('f'), ord('c'),
        }
        evaluate_keys = {ord('r'), ord('o'), ord('A'), ord('B'), ord('C'), ord('D'), ord('E')}

        if key_code in play_keys and self.mode != OperatorMode.PLAY:
            return "Switch to PLAY mode"
        if key_code in calibrate_keys and self.mode != OperatorMode.CALIBRATE:
            return "Switch to CALIBRATE mode"
        if key_code in evaluate_keys and self.mode != OperatorMode.EVALUATE:
            return "Switch to EVALUATE mode"
        return None

    def assist_selected_key(self) -> None:
        idx_sel = self.get_selected_idx()

        if idx_sel is None:
            idx_sel = self.select_worst_tested_white_key()
            if idx_sel is None:
                self.flash_ui_message("No key selected and no weak key available")
                return

        action, note = self.recommend_key_action(idx_sel)
        print(f"\n[ASSIST] key {idx_sel}: {action} -- {note}")

        if action == "APPLY_LEARNED_SUGGESTION":
            self.offsets.apply_suggestion(idx_sel)
            self.flash_ui_message(f"Applied learned suggestion to key {idx_sel}")
            return

        if action in {"COLLECT_MORE_TESTS", "TEST_KEY", "RETEST_LATER"}:
            self.tune.typed = str(idx_sel)
            self.run_selected_key_repeated_test()
            self.flash_ui_message(f"Ran repeated test for key {idx_sel}")
            return

        if action == "DONE":
            self.flash_ui_message(f"Key {idx_sel} already looks good")
            return

        if action == "OPTIONAL_APPLY_SUGGESTION":
            self.flash_ui_message(f"Suggestion available for key {idx_sel}; press y to apply")
            return

        if action == "MANUAL_NUDGE_SMALL":
            suggestion, n_sugg = self.offsets.get_suggestion(idx_sel)
            if suggestion is not None and n_sugg > 0:
                dx = self.cfg.key_offset_step_m if suggestion[0] > 0 else -self.cfg.key_offset_step_m
                dy = self.cfg.key_offset_step_m if suggestion[1] > 0 else -self.cfg.key_offset_step_m
                if abs(suggestion[0]) >= abs(suggestion[1]):
                    self.offsets.nudge(idx_sel, ddx=dx)
                else:
                    self.offsets.nudge(idx_sel, ddy=dy)
                self.flash_ui_message(f"Nudged key {idx_sel}; rerun repeated test")
            else:
                self.flash_ui_message(f"No strong suggestion for key {idx_sel}")
            return

        self.flash_ui_message(f"{action}: {note}")

    # ----------------------------
    # Low-level geometry / selection
    # ----------------------------
    def get_selected_idx(self) -> int | None:
        if self.tune.typed.isdigit():
            idx = int(self.tune.typed)
            if 0 <= idx < self.cfg.n_keys:
                return idx
        return None

    def strike_target_world_xy(self, idx: int) -> tuple[float, float]:
        assert self.key_model is not None
        x_key, y_key = self.key_model.key_world_xy(idx)
        dx_i, dy_i = self.offsets.get(idx)
        return (
            x_key + self.cfg.strike_dx_w + dx_i,
            y_key + self.cfg.strike_dy_w + dy_i,
        )

    def compute_base_origin_world_xy(self) -> tuple[float, float] | None:
        if not self.last_tracker_ok:
            return None
        assert self.tracker is not None
        try:
            w_t_b = np.linalg.inv(self.tracker.B_T_W)
            p0_w = w_t_b[:3, 3]
            return float(p0_w[0]), float(p0_w[1])
        except Exception as exc:
            print(f"⚠️ Could not compute base origin in world frame: {exc}")
            return None

    # ----------------------------
    # Press profile selection
    # ----------------------------
    def update_auto_press_base_key_idx(self, verbose: bool = False) -> int:
        base_xy = self.compute_base_origin_world_xy()
        if base_xy is None:
            return self.auto_press_base_key_idx

        x_base_w, _ = base_xy
        assert self.key_model is not None
        best_idx = min(
            range(self.cfg.n_keys),
            key=lambda i: abs(float(self.key_model.key_world_xy(i)[0]) - x_base_w),
        )
        changed = best_idx != self.auto_press_base_key_idx
        self.auto_press_base_key_idx = int(best_idx)

        if changed and verbose:
            print(f"auto base-reference key -> {self.auto_press_base_key_idx} (x_base={x_base_w:.4f})")
        return self.auto_press_base_key_idx

    def choose_auto_press_profile_from_base_x(self, idx: int) -> str:
        ref_idx = self.update_auto_press_base_key_idx(verbose=False)
        assert self.key_model is not None
        x_ref, _ = self.key_model.key_world_xy(ref_idx)
        x_key, _ = self.key_model.key_world_xy(idx)
        dx = abs(float(x_key) - float(x_ref))
        if dx <= self.cfg.auto_press_soft_x_dist_m:
            return "soft"
        if dx <= self.cfg.auto_press_medium_x_dist_m:
            return "medium"
        return "hard"

    def get_effective_press_profile_name_for_key(self, idx: int) -> str:
        if not self.auto_press_profile_enabled:
            return self.press_profile_name
        return self.choose_auto_press_profile_from_base_x(idx)

    def get_current_press_profile(self) -> dict[str, float | int]:
        return self.cfg.hybrid_press_profiles[self.press_profile_name]

    def print_current_press_profile(self) -> None:
        prof = self.get_current_press_profile()
        print(f"\n=== Press profile: {self.press_profile_name} ===")
        print(f"preload_dz         = {prof['preload_dz'] * 1000:.1f} mm")
        print(f"preload_steps      = {prof['preload_steps']}")
        print(f"preload_hold_s     = {prof['preload_hold_s']:.3f}")
        print(f"servo5_delta_deg   = {prof['servo5_delta_deg']:.1f} deg")
        print(f"servo5_steps       = {prof['servo5_steps']}")
        print(f"servo5_hold_s      = {prof['servo5_hold_s']:.3f}")
        print(f"final_extra_dz     = {prof['final_extra_dz'] * 1000:.1f} mm")
        print(f"final_extra_steps  = {prof['final_extra_steps']}")
        print(f"up_steps           = {prof['up_steps']}")
        print()

    def cycle_press_profile(self, delta: int) -> None:
        idx = self.cfg.press_profile_order.index(self.press_profile_name)
        idx = max(0, min(idx + int(delta), len(self.cfg.press_profile_order) - 1))
        self.press_profile_name = self.cfg.press_profile_order[idx]
        print(f"✅ Press profile set to: {self.press_profile_name}")
        self.print_current_press_profile()

    # ----------------------------
    # Motion helpers
    # ----------------------------
    def build_note_context(self, idx, next_idx=None, beat_sec=0.5, phrase_start=False, phrase_end=False, prev_idx=None) -> dict[str, Any]:
        if prev_idx is None:
            prev_idx = self.last_play_idx

        key_jump = None if prev_idx is None else abs(int(idx) - int(prev_idx))
        same_key = (prev_idx == idx) if prev_idx is not None else False

        x_tgt, y_tgt = self.strike_target_world_xy(idx)

        if prev_idx is None:
            dxy_prev = None
        else:
            x_prev, y_prev = self.strike_target_world_xy(prev_idx)
            dxy_prev = float(np.hypot(x_tgt - x_prev, y_tgt - y_prev))

        if next_idx is None:
            dxy_next = None
        else:
            x_next, y_next = self.strike_target_world_xy(next_idx)
            dxy_next = float(np.hypot(x_next - x_tgt, y_next - y_tgt))

        return {
            "idx": idx,
            "prev_idx": prev_idx,
            "next_idx": next_idx,
            "same_key": same_key,
            "key_jump": key_jump,
            "dxy_prev": dxy_prev,
            "dxy_next": dxy_next,
            "beat_sec": float(beat_sec),
            "phrase_start": bool(phrase_start),
            "phrase_end": bool(phrase_end),
        }

    @staticmethod
    def classify_motion_context(ctx: dict[str, Any]) -> str:
        if ctx["phrase_start"] or ctx["prev_idx"] is None:
            return "phrase_start"
        if ctx["same_key"]:
            return "repeat"
        if ctx["key_jump"] is not None and ctx["key_jump"] <= 1:
            return "step"
        if ctx["key_jump"] is not None and ctx["key_jump"] <= 3:
            return "near"
        return "far"

    def get_align_z_for_motion_class(self, motion_class: str) -> float:
        return {
            "repeat": self.cfg.repeat_align_z,
            "step": self.cfg.step_align_z,
            "near": self.cfg.near_align_z,
            "phrase_start": self.cfg.phrase_start_align_z,
            "far": self.cfg.far_align_z,
        }.get(motion_class, self.cfg.far_align_z)

    def should_use_light_align(self, idx: int) -> bool:
        return (
            self.last_align_ok
            and self.last_play_idx is not None
            and self.last_best_err_m is not None
            and abs(idx - self.last_play_idx) <= self.cfg.light_align_max_key_jump
            and self.align_good_streak >= self.cfg.light_align_min_streak
            and self.last_best_err_m <= self.cfg.light_align_good_err_m
        )

    def get_travel_profile(self, dxy: float) -> tuple[int, float]:
        alpha = 0.0
        if self.cfg.travel_far_dist_m > self.cfg.travel_near_dist_m:
            alpha = (dxy - self.cfg.travel_near_dist_m) / (self.cfg.travel_far_dist_m - self.cfg.travel_near_dist_m)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        steps = int(round(self.cfg.travel_steps_min + alpha * (self.cfg.travel_steps_max - self.cfg.travel_steps_min)))
        dt = float(self.cfg.travel_dt_min + alpha * (self.cfg.travel_dt_max - self.cfg.travel_dt_min))
        return steps, dt

    def get_pid_profile(self, err_m: float) -> tuple[float, float]:
        if err_m > self.cfg.pid_err_large_m:
            return self.cfg.pid_step_large_m, self.cfg.pid_sleep_large_s
        if err_m > self.cfg.pid_err_medium_m:
            return self.cfg.pid_step_medium_m, self.cfg.pid_sleep_medium_s
        return self.cfg.pid_step_small_m, self.cfg.pid_sleep_small_s

    def move_to_align_pose_safely(
        self,
        idx: int,
        x_cmd: float,
        y_w: float,
        motion_class: str = "far",
        align_z: float | None = None,
    ) -> np.ndarray:
        align_z = self.cfg.align_z_above_key if align_z is None else align_z
        assert self.arm is not None
        p_align_b = self.arm.piano_xy_to_base_xyz(x_cmd, y_w, z_p=align_z)

        if self.last_play_xy is None or self.last_play_idx is None:
            steps0, dt0 = self.get_travel_profile(0.060)
            self.arm.goto_cartesian(p_align_b, steps=min(10, steps0), dt=dt0)
            self.last_play_xy = np.array([x_cmd, y_w], dtype=float)
            return p_align_b

        dxy = np.linalg.norm(np.array([x_cmd, y_w], dtype=float) - self.last_play_xy)
        steps_prof, dt_prof = self.get_travel_profile(float(dxy))

        if motion_class == "repeat":
            self.arm.goto_cartesian(p_align_b, steps=4, dt=0.016)
        elif motion_class == "step":
            self.arm.goto_cartesian(p_align_b, steps=10, dt=0.018)
        elif motion_class == "near":
            self.arm.goto_cartesian(p_align_b, steps=min(14, steps_prof), dt=min(0.022, dt_prof))
        elif dxy > self.cfg.far_key_lift_threshold_m:
            x_prev, y_prev = self.last_play_xy
            p_lift_b = self.arm.piano_xy_to_base_xyz(float(x_prev), float(y_prev), z_p=self.cfg.safe_travel_z_above_key)
            p_high_target_b = self.arm.piano_xy_to_base_xyz(x_cmd, y_w, z_p=self.cfg.safe_travel_z_above_key)
            self.arm.goto_cartesian(p_lift_b, steps=self.cfg.far_lift_steps, dt=self.cfg.far_lift_dt)
            self.arm.goto_cartesian(p_high_target_b, steps=self.cfg.far_high_travel_steps, dt=self.cfg.far_high_travel_dt)
            self.arm.goto_cartesian(p_align_b, steps=self.cfg.far_descend_steps, dt=self.cfg.far_descend_dt)
        else:
            self.arm.goto_cartesian(p_align_b, steps=steps_prof, dt=dt_prof)

        self.last_play_xy = np.array([x_cmd, y_w], dtype=float)
        return p_align_b

    def goto_hover_pose_world(self, xw: float, yw: float, steps: int = 8, dt: float | None = None, hover_z: float | None = None) -> np.ndarray:
        assert self.arm is not None
        hover_z = self.cfg.align_z_above_key if hover_z is None else hover_z
        p_hover_b = self.arm.piano_xy_to_base_xyz(xw, yw, z_p=hover_z)
        self.arm.goto_cartesian(p_hover_b, steps=steps, dt=dt)
        return p_hover_b

    # ----------------------------
    # Vision / INA helpers
    # ----------------------------
    def detect_world_xy_avg(self, n: int = 6, flush: int = 2, wait_s: float = 0.01) -> np.ndarray | None:
        assert self.cap is not None and self.blob is not None and self.tracker is not None
        points: list[np.ndarray] = []

        for _ in range(flush):
            self.cap.grab()

        for _ in range(n):
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(wait_s)
                continue

            uv, _ = self.blob.detect(frame)
            if uv is None:
                time.sleep(wait_s)
                continue

            xw, yw = self.tracker.pixel_to_world(float(uv[0]), float(uv[1]))
            points.append(np.array([xw, yw], dtype=float))
            time.sleep(wait_s)

        min_points = max(1, min(3, n))
        if len(points) < min_points:
            return None

        return np.median(np.stack(points, axis=0), axis=0)

    def parse_current_mA_from_ina_line(self, line: str) -> float | None:
        s = line.strip()
        m_ma = re.search(r"(?:current|i)\s*[:=]?\s*([-+]?\d*\.?\d+)\s*mA\b", s, re.IGNORECASE)
        if m_ma:
            return float(m_ma.group(1))
        m_a = re.search(r"(?:current|i)\s*[:=]?\s*([-+]?\d*\.?\d+)\s*A\b", s, re.IGNORECASE)
        if m_a:
            return 1000.0 * float(m_a.group(1))
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)]
        return float(nums[-1]) if nums else None

    def summarize_ina219_press(self, lines: list[str]) -> dict[str, Any]:
        for line in lines:
            s = line.strip()
            if s.startswith("INA219_SUMMARY"):
                def _extract(pattern: str, cast=float) -> Any:
                    match = re.search(pattern, s)
                    if not match:
                        return None
                    return cast(match.group(1))

                return {
                    "n_samples": _extract(r"samples=(\d+)", int) or 0,
                    "peak_current_mA": _extract(r"peak_mA=([-+]?\d*\.?\d+)"),
                    "mean_current_mA": _extract(r"avg_mA=([-+]?\d*\.?\d+)"),
                    "min_current_mA": None,
                    "min_bus_V": _extract(r"min_bus_V=([-+]?\d*\.?\d+)"),
                    "first_over_mA_ms": _extract(r"first_over_mA_ms=(\d+)", int),
                    "peak_ms": _extract(r"peak_ms=(\d+)", int),
                }

        currents = [self.parse_current_mA_from_ina_line(line) for line in lines]
        currents = [c for c in currents if c is not None]
        if not currents:
            return {
                "n_samples": 0,
                "peak_current_mA": None,
                "mean_current_mA": None,
                "min_current_mA": None,
                "min_bus_V": None,
                "first_over_mA_ms": None,
                "peak_ms": None,
            }

        arr = np.array(currents, dtype=float)
        return {
            "n_samples": int(arr.size),
            "peak_current_mA": float(np.max(arr)),
            "mean_current_mA": float(np.mean(arr)),
            "min_current_mA": float(np.min(arr)),
            "min_bus_V": None,
            "first_over_mA_ms": None,
            "peak_ms": None,
        }

    def ina_start_log(self) -> None:
        if not self.cfg.use_ina219_logger:
            return
        try:
            assert self.arm is not None
            self.arm.ina_start_log()
            print("[INA219] start command sent")
        except Exception as exc:
            print(f"⚠️ INA219 start log failed: {exc}")

    def ina_end_log(self) -> list[str]:
        if not self.cfg.use_ina219_logger:
            return []
        try:
            assert self.arm is not None
            lines = self.arm.ina_end_log(timeout_s=2.0)
            print(f"[INA219] captured {len(lines)} raw lines")
            for i, line in enumerate(lines[:8]):
                print(f"[INA219 raw {i}] {line}")
            if not any(line.startswith("INA219_SUMMARY") for line in lines):
                print("⚠️ INA219 summary line not seen before timeout")
            return lines
        except Exception as exc:
            print(f"⚠️ INA219 end log failed: {exc}")
            return []

    def estimate_release_tail_s(
        self,
        profile_name: str,
        motion_class: str = "far",
    ) -> float:
        """Estimate the time from end-of-contact-hold to arm arriving at hover.

        This is used by compute_playback_hold_s to decide how early the
        contact hold must end so that the arm is free in time for the next
        note launch.

        For the repeat path (execute_key reuse branch), the pre-press overhead
        includes a goto_hover_pose_world call (2 steps for playback repeat).
        This is NOT part of the release tail, but the NEXT NOTE's pre-press
        overhead IS included in predict_note_latency_s, so we don't double count.

        For the post-press path: servo_release + pause + arm_up + hover_settle.
        The hover_settle steps must match what execute_key actually uses after
        pressing, which varies by motion_class.
        """
        assert self.arm is not None, "estimate_release_tail_s called before arm is connected"
        prof = self.cfg.hybrid_press_profiles[profile_name]

        servo_release_s = float(prof["servo5_release_steps"]) * self.arm.dt
        arm_up_s        = float(prof["up_steps"]) * self.arm.dt
        pause_s         = float(prof["release_pause_s"])

        # Match the hover_steps used in execute_key post-press for each motion class.
        # For repeat path: post-press hover is 2 steps, then the next repeat
        # does a 2-step pre-press hover before INA_START.  Both are included here
        # so the total overhead from end-of-hold to next INA_START is captured.
        hover_steps = {
            "repeat":       2,
            "step":         3,
            "near":         3,
            "phrase_start": 4,
            "far":          4,
        }.get(motion_class, 4)

        # For repeat: add the pre-press goto_hover of the NEXT repeat_reuse note (2 steps).
        # This closes the 75ms accounting gap between computed hold and actual timing.
        next_prepress_steps = 2 if motion_class == "repeat" else 0
        hover_settle_s = float(hover_steps + next_prepress_steps) * self.arm.dt

        return servo_release_s + pause_s + arm_up_s + hover_settle_s

    def estimate_next_note_lead_s(
        self,
        current_idx: int,
        next_idx: int | None,
        next_next_idx: int | None = None,
    ) -> float:
        if next_idx is None:
            return 0.0

        next_ctx = self.build_note_context(
            idx=next_idx,
            prev_idx=current_idx,
            next_idx=next_next_idx,
            beat_sec=0.5,
            phrase_start=False,
            phrase_end=False,
        )
        next_motion_class = self.classify_motion_context(next_ctx)

        return (
            self.stats.predict_note_latency_s(next_motion_class)
            + self.cfg.playback_latency_margin_s
        )

    def compute_sequential_hold_s(
        self,
        idx: int,
        beat_duration_s: float,
        next_idx: int | None,
        next_next_idx: int | None,
        forced_profile_name: str | None = None,
    ) -> float:
        if next_idx is None:
            return beat_duration_s

        if forced_profile_name is not None:
            prof_name = forced_profile_name
        else:
            prof_name = self.get_effective_press_profile_name_for_key(idx)

        release_tail_s = self.estimate_release_tail_s(prof_name)
        next_lead_s = self.estimate_next_note_lead_s(idx, next_idx, next_next_idx)

        transition_budget_s = release_tail_s + next_lead_s
        hold_s = beat_duration_s - transition_budget_s

        hold_s = max(self.cfg.song_min_press_hold_s, hold_s)

        print(
            f"sequential timing idx={idx}: "
            f"beat_dur={beat_duration_s:.3f}s  "
            f"release_tail={release_tail_s:.3f}s  "
            f"next_lead={next_lead_s:.3f}s  "
            f"hold={hold_s:.3f}s"
        )

        return hold_s

    def play_score_events_sequential(self, score_events: list[dict[str, Any]], bpm: float) -> None:
        beat_sec = 60.0 / bpm

        self.stats.latency_stats["phrase_start"].clear()

        self.last_play_idx = None
        self.last_align_ok = False
        self.last_best_err_m = None
        self.last_finetune_ref_xy = None
        self.align_good_streak = 0

        time.sleep(self.cfg.song_start_lead_in_s)

        last_press_wall_t = None

        if score_events:
            first_event = score_events[0]
            first_idx = first_event["idx"]
            first_next_idx = score_events[1]["idx"] if len(score_events) > 1 else None
            first_dur_beats = float(first_event.get("dur_beats", 1.0))

            print(f"preparing first note idx={first_idx} before starting playback timing")

            self.execute_key(
                first_idx,
                hold_s=0.0,
                force_full_align=not self.cfg.song_force_light_align,
                next_idx=first_next_idx,
                beat_sec=first_dur_beats * beat_sec,
                phrase_start=first_event.get("phrase_start", True),
                phrase_end=first_event.get("phrase_end", False),
                playback_mode=True,
                prepare_only=True,
                forced_profile_name="song_fast",
            )

        for n, event in enumerate(score_events):
            idx = event["idx"]
            dur_beats = float(event.get("dur_beats", 1.0))
            phrase_start = event.get("phrase_start", False)
            phrase_end = event.get("phrase_end", False)

            prev_idx = score_events[n - 1]["idx"] if n > 0 else None
            next_idx = score_events[n + 1]["idx"] if (n + 1) < len(score_events) else None

            beat_duration_s = dur_beats * beat_sec
            next_next_idx = score_events[n + 2]["idx"] if (n + 2) < len(score_events) else None

            if self.cfg.song_use_full_duration_hold:
                hold_s = self.compute_sequential_hold_s(
                    idx=idx,
                    beat_duration_s=beat_duration_s,
                    next_idx=next_idx,
                    next_next_idx=next_next_idx,
                    forced_profile_name=event.get("press_profile", None),
                )
            else:
                hold_ratio = float(event.get("hold_ratio", self.cfg.default_song_hold_ratio))
                hold_s = hold_ratio * beat_duration_s
                
            ctx = self.build_note_context(
                idx=idx,
                prev_idx=prev_idx,
                next_idx=next_idx,
                beat_sec=beat_duration_s,
                phrase_start=phrase_start,
                phrase_end=phrase_end,
            )
            motion_class = self.classify_motion_context(ctx)

            print(
                f"sequential play idx={idx}  motion_class={motion_class}  "
                f"hold_s={hold_s:.3f}"
            )

            note_exec_t0 = time.perf_counter()

            self.last_first_over_mA_ms = None
            self.execute_key(
                idx,
                hold_s=hold_s,
                next_idx=next_idx,
                beat_sec=beat_duration_s,
                phrase_start=phrase_start,
                phrase_end=phrase_end,
                playback_mode=True,
                forced_profile_name="song_fast",
            )

            note_exec_s = time.perf_counter() - note_exec_t0
            print(f"note service time idx={idx}: {note_exec_s*1000:.1f} ms")

            actual_ms = self.last_first_over_mA_ms
            if actual_ms is not None:
                actual_press_wall_t = note_exec_t0 + actual_ms / 1000.0

                if last_press_wall_t is not None:
                    delta_s = actual_press_wall_t - last_press_wall_t
                    print(f"actual press-to-press interval: {delta_s:.3f}s")

                last_press_wall_t = actual_press_wall_t

            # optional rest if the score has a gap larger than the note duration
            if (n + 1) < len(score_events):
                next_onset = float(score_events[n + 1].get("onset_beats", 0.0))
                this_onset = float(event.get("onset_beats", 0.0))
                rest_beats = max(0.0, next_onset - this_onset - dur_beats)
                if rest_beats > 0.0:
                    rest_s = rest_beats * beat_sec
                    print(f"rest after idx={idx}: {rest_s:.3f}s")
                    time.sleep(rest_s)

        if self.cfg.enable_end_reset_pose and not self.safety.stop_requested:
            self.goto_system_neutral_pose(reason="end of score")
            
    def do_current_press(
        self,
        idx,
        hold_s,
        forced_profile_name=None,
        min_press_hold_s=None,
        playback_mode: bool = False,
    ) -> tuple[str, str]:
        
        assert self.arm is not None

        if forced_profile_name is not None:
            prof_name = str(forced_profile_name)
            profile_source = "forced"
            ref_idx = self.update_auto_press_base_key_idx(verbose=False)
        else:
            ref_idx = self.update_auto_press_base_key_idx(verbose=False)
            prof_name = self.get_effective_press_profile_name_for_key(idx)
            profile_source = "auto" if self.auto_press_profile_enabled else "manual"

        prof = self.cfg.hybrid_press_profiles[prof_name]
        self.last_press_profile_used = prof_name

        if self.cfg.auto_press_verbose:
            assert self.key_model is not None
            x_ref, _ = self.key_model.key_world_xy(ref_idx)
            x_key, _ = self.key_model.key_world_xy(idx)
            dx_mm = 1000.0 * abs(float(x_key) - float(x_ref))
            print(
                f"press profile for key {idx}: {prof_name} "
                f"(source={profile_source}, auto base-ref key={ref_idx}, dx_from_ref={dx_mm:.1f} mm)"
            )

        if min_press_hold_s is None:
            min_press_hold_s = self.cfg.min_press_hold_s

        press_hold = max(hold_s, min_press_hold_s)

        if playback_mode:
            press_hold = min(press_hold, self.cfg.song_playback_contact_cap_s)

        pre_contact_hold = float(prof["servo5_hold_s"])
        contact_hold = press_hold

        print(
            f"press timing: pre_contact_hold={pre_contact_hold:.3f}s  "
            f"contact_hold={contact_hold:.3f}s"
        )

        if self.cfg.use_hybrid_press:
            self.arm.hybrid_tap_current(
                preload_dz=prof["preload_dz"],
                preload_steps=prof["preload_steps"],
                preload_hold_s=prof["preload_hold_s"],

                servo5_delta_deg=prof["servo5_delta_deg"],
                servo5_steps=prof["servo5_steps"],
                servo5_hold_s=pre_contact_hold,

                final_extra_dz=prof["final_extra_dz"],
                final_extra_steps=prof["final_extra_steps"],
                final_extra_hold_s=contact_hold,

                up_steps=prof["up_steps"],
                servo5_release_steps=prof["servo5_release_steps"],
                release_pause_s=prof["release_pause_s"],
                dt=self.arm.dt,
            )
        else:
            self.arm.simple_tap_current(
                dz=self.cfg.press_dz,
                down_steps=14,
                up_steps=8,
                hold_s=press_hold,
                dt=self.arm.dt,
            )

        return prof_name, profile_source

    # ----------------------------
    # Vision alignment
    # ----------------------------
    def align_to_key_center_pid(
        self,
        idx: int,
        x_ref0: float,
        y_ref0: float,
        timeout_s: float | None = None,
        tol_m: float | None = None,
        settle_frames: int | None = None,
        align_z: float | None = None,
        max_iters: int | None = None,
        detect_n: int = 4,
        detect_flush: int = 1,
        detect_wait_s: float = 0.006,
    ) -> tuple[bool, float | None, float | None, float, float]:
        """
        Close-loop visual alignment routine.

        Starting from the IK-based coarse position (x_ref0, y_ref0), this
        method iteratively measures the green end-effector marker position
        and applies proportional corrections until the alignment error falls
        within tol_m for settle_frames consecutive frames.

        Returns (ok, best_err_m, last_err_m, x_ref_final, y_ref_final).
          ok            : True if alignment converged within tol_m
          best_err_m    : smallest error seen during alignment (metres)
          last_err_m    : error in the final iteration
          x_ref_final   : final aligned X position in world frame
          y_ref_final   : final aligned Y position in world frame

        The controller uses a P-dominant VisionPID2D instance (ki=kd=0 in
        current tuning), with deadbands and per-step limits to keep corrections
        local and stable.  If the blob is lost during alignment, the routine
        exits with ok=False.

        This method is called both for full alignment (with the default timeout)
        and for quick/light alignment during song playback (with reduced
        timeout and iteration limits).
        """
        timeout_s = self.cfg.align_timeout_s if timeout_s is None else timeout_s
        tol_m = self.cfg.align_tol_m if tol_m is None else tol_m
        settle_frames = self.cfg.align_settle_frames if settle_frames is None else settle_frames
        align_z = self.cfg.align_z_above_key if align_z is None else align_z

        pid = VisionPID2D(
            dt=self.cfg.dt,
            kp=(self.cfg.pid_kp_x, self.cfg.pid_kp_y),
            ki=(self.cfg.pid_ki_x, self.cfg.pid_ki_y),
            kd=(self.cfg.pid_kd_x, self.cfg.pid_kd_y),
            i_limit=(self.cfg.pid_int_lim_x, self.cfg.pid_int_lim_y),
            d_alpha=self.cfg.pid_d_alpha,
        )

        xw_tgt, yw_tgt = self.strike_target_world_xy(idx)
        x_ref = float(x_ref0)
        y_ref = float(y_ref0)
        stable = 0
        t0 = time.perf_counter()
        best_err_m = None
        last_err_m = None
        last_log = 0.0
        n_iters = 0

        w_filt: np.ndarray | None = None
        prev_step = np.zeros(2, dtype=float)
        miss_count = 0

        deadband_x_m = 0.0007
        deadband_y_m = 0.0009
        max_step_x_m = 0.0012
        max_step_y_m = 0.0018
        step_blend = 0.60

        time.sleep(self.cfg.align_start_settle_s)

        while time.perf_counter() - t0 < timeout_s:
            if max_iters is not None and n_iters >= max_iters:
                break

            w = self.detect_world_xy_avg(
                n=detect_n,
                flush=detect_flush,
                wait_s=detect_wait_s,
            )

            if w is None:
                miss_count += 1
                if miss_count <= 1:
                    time.sleep(detect_wait_s)
                    continue

                stable = 0
                pid.reset()
                w_filt = None
                print("PID align: blob lost")
                break

            miss_count = 0
            w = np.asarray(w, dtype=float)

            if w_filt is None:
                w_filt = w.copy()
            else:
                w_filt = 0.65 * w_filt + 0.35 * w

            ex = xw_tgt - float(w_filt[0])
            ey = yw_tgt - float(w_filt[1])

            if abs(ex) < deadband_x_m:
                ex = 0.0
            if abs(ey) < deadband_y_m:
                ey = 0.0

            error = np.array([ex, ey], dtype=float)
            err_m = float(np.linalg.norm(error))
            n_iters += 1

            last_err_m = err_m
            best_err_m = err_m if best_err_m is None else min(best_err_m, err_m)

            now = time.perf_counter()
            if now - last_log > 0.08:
                print(f"PID align: ex={ex*1000:.2f}mm ey={ey*1000:.2f}mm err={err_m*1000:.2f}mm")
                last_log = now

            if err_m < tol_m:
                stable += 1
                if stable >= settle_frames:
                    return True, best_err_m, last_err_m, x_ref, y_ref
            else:
                stable = 0

            raw_step = pid.update(error, now)
            _, pid_sleep_s = self.get_pid_profile(err_m)

            step = step_blend * prev_step + (1.0 - step_blend) * raw_step
            step[0] = float(np.clip(step[0], -max_step_x_m, max_step_x_m))
            step[1] = float(np.clip(step[1], -max_step_y_m, max_step_y_m))
            prev_step = step.copy()

            x_ref += float(step[0])
            y_ref += float(step[1])

            total_corr = np.array([x_ref - x_ref0, y_ref - y_ref0], dtype=float)
            total_mag = np.linalg.norm(total_corr)
            if total_mag > self.cfg.pid_max_total_corr_m and total_mag > 1e-9:
                total_corr *= self.cfg.pid_max_total_corr_m / total_mag
                x_ref = x_ref0 + float(total_corr[0])
                y_ref = y_ref0 + float(total_corr[1])
                print(f"CLAMP total correction to {self.cfg.pid_max_total_corr_m * 1000:.1f} mm")

            print(
                f"cmd step: dx={step[0]*1000:.2f}mm dy={step[1]*1000:.2f}mm   "
                f"total=({(x_ref - x_ref0)*1000:.2f}, {(y_ref - y_ref0)*1000:.2f}) mm"
            )

            assert self.arm is not None
            p_align_b = self.arm.piano_xy_to_base_xyz(x_ref, y_ref, z_p=align_z)
            self.arm.goto_cartesian(p_align_b, steps=self.cfg.pid_move_steps, dt=self.arm.dt)
            time.sleep(pid_sleep_s)

        return False, best_err_m, last_err_m, x_ref, y_ref

    def compute_playback_hold_s(
        self,
        score_events: list[dict[str, Any]],
        n: int,
        beat_sec: float,
        forced_profile_name: str | None = None,
    ) -> float:
        """Compute how long to hold the key in contact during playback.

        The hold must end early enough that:
          release_tail  (servo release + arm up + hover settle)
          + next_launch_lead  (alignment + press latency for next note)
          + playback_release_guard  (single safety margin, in config)
        all fit within the gap to the next note onset.

        Key fix: release_tail now uses the motion_class of the CURRENT note
        (which determines how many hover_steps execute_key actually uses after
        pressing), rather than a fixed worst-case step count.
        """
        event = score_events[n]
        idx = int(event["idx"])
        onset_beats = float(event["onset_beats"])
        dur_beats = float(event.get("dur_beats", 1.0))
        nominal_note_s = dur_beats * beat_sec

        if forced_profile_name is not None:
            prof_name = str(forced_profile_name)
        else:
            prof_name = self.get_effective_press_profile_name_for_key(idx)

        # Determine current note's motion class so release_tail uses the right
        # hover_steps (matches what execute_key will actually do post-press)
        prev_idx = score_events[n - 1]["idx"] if n > 0 else None
        next_idx_for_ctx = (
            int(score_events[n + 1]["idx"]) if (n + 1) < len(score_events) else None
        )
        current_ctx = self.build_note_context(
            idx=idx,
            prev_idx=prev_idx,
            next_idx=next_idx_for_ctx,
            beat_sec=nominal_note_s,
            phrase_start=bool(event.get("phrase_start", False)),
            phrase_end=bool(event.get("phrase_end", False)),
        )
        current_motion_class = self.classify_motion_context(current_ctx)

        # Time from end-of-hold to arm reaching hover (uses correct hover_steps)
        release_tail_s = self.estimate_release_tail_s(
            prof_name, motion_class=current_motion_class
        )
        # Single safety guard (config value, no second add-on elsewhere)
        release_tail_s += self.cfg.playback_release_guard_s

        # Last note: hold as long as the note duration allows
        if (n + 1) >= len(score_events):
            return max(self.cfg.song_min_press_hold_s, nominal_note_s - release_tail_s)

        next_event = score_events[n + 1]
        next_idx = int(next_event["idx"])
        next_next_idx = int(score_events[n + 2]["idx"]) if (n + 2) < len(score_events) else None
        next_onset_beats = float(next_event["onset_beats"])

        # Time from this note onset to next note onset
        gap_to_next_s = max(0.0, (next_onset_beats - onset_beats) * beat_sec)

        next_ctx = self.build_note_context(
            idx=next_idx,
            prev_idx=idx,
            next_idx=next_next_idx,
            beat_sec=gap_to_next_s,
            phrase_start=bool(next_event.get("phrase_start", False)),
            phrase_end=bool(next_event.get("phrase_end", False)),
        )
        next_motion_class = self.classify_motion_context(next_ctx)

        # How early the next note must be launched before its onset
        next_launch_lead_s = (
            self.stats.predict_note_latency_s(next_motion_class)
            + self.cfg.playback_latency_margin_s
        )

        # Constraint 1: hold must allow release + next launch to fit in gap
        hold_timing = gap_to_next_s - release_tail_s - next_launch_lead_s

        # Constraint 2: hold must not make the current note's service exceed the gap.
        # Without this cap, a step note followed by a fast repeat_reuse gets assigned
        # a very large hold (because next_lead is only 200ms), but the step's own
        # alignment+press overhead (~1100ms) means service = overhead + hold >> gap.
        # The overhead estimate is conservative (typical measured values).
        current_note_overhead_s = {
            "phrase_start": 1.10,
            "step":         1.10,
            "near":         1.10,
            "far":          1.20,
            "repeat":       0.67,
            "repeat_reuse": 0.64,
        }.get(current_motion_class, 1.10)
        hold_overhead_cap = gap_to_next_s - current_note_overhead_s

        hold_s = min(hold_timing, hold_overhead_cap)
        return max(self.cfg.song_min_press_hold_s, hold_s)

    def _wait_for_start_trigger(self, song_title: str) -> bool:
        """
        Pause after arm preparation and wait for the operator to press
        cfg.song_trigger_key (default: '*') to start the song clock.

        While waiting the camera loop keeps running so the live view stays
        updated and the operator can see the arm hovering over the first key.

        Returns True if the trigger was received, False if the operator
        pressed q / ESC to abort (safe stop is also respected).
        """
        assert self.cap is not None and self.tracker is not None
        assert self.blob is not None and self.key_model is not None

        trigger_char = chr(self.cfg.song_trigger_key) if 32 <= self.cfg.song_trigger_key < 127 else f"code={self.cfg.song_trigger_key}"
        print(
            f"\n🎹 Arm ready at first key.  "
            f"Press '{trigger_char}' in the camera window to start '{song_title}'.  "
            f"Press q/ESC to abort."
        )
        self.flash_ui_message(f"READY — press '{trigger_char}' to start")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                return False

            disp = frame.copy()
            ok = self.tracker.update(frame)

            if ok:
                self.last_tracker_ok = True
                xw_ref, yw_ref = self.key_model.key_world_xy(10)
                self.tracker.z_lock_keyplane(xw_ref, yw_ref, z_des_base=self.cfg.z_des_base)
                self.tracker.draw_markers(disp)

                uv_show, _ = self.blob.detect(frame)
                if uv_show is not None:
                    cv2.circle(disp, uv_show, 8, (0, 255, 0), -1)
                    cv2.putText(disp, "READY", (uv_show[0] + 10, uv_show[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                for i, (xw, yw) in enumerate(self.key_model.all_keys_world_xy()):
                    u, v = self.tracker.world_to_pixel(xw, yw)
                    cv2.circle(disp, (u, v), 4, (0, 0, 255), -1)
                    if i % 2 == 0:
                        cv2.putText(disp, str(i), (u + 6, v - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                cv2.putText(disp, "Need markers visible", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Overlay the waiting prompt prominently
            prompt = f"Press '{trigger_char}' to START  |  q to abort"
            cv2.rectangle(disp, (0, self.cfg.frame_h - 55),
                          (self.cfg.frame_w, self.cfg.frame_h), (20, 20, 20), -1)
            cv2.putText(disp, prompt,
                        (20, self.cfg.frame_h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Piano-Playing Robot", disp)
            key_code = cv2.waitKey(15) & 0xFF

            if key_code in (ord('q'), 27):          # q or ESC — abort
                print("▶ Song start aborted by operator.")
                return False

            if key_code == ord('F'):                # honour safe stop
                self.request_safe_stop("operator stop during song trigger wait")
                return False

            if key_code == self.cfg.song_trigger_key:
                print(f"▶ Trigger received — starting '{song_title}'")
                return True

            if self.safety.stop_requested:
                return False

    # ----------------------------
    # Play / execute
    # ----------------------------
    def execute_key(
        self,
        idx: int,
        hold_s: float = 0.12,
        force_full_align: bool = False,
        forced_profile_name: str | None = None,
        validation_run_id: str = "",
        validation_trial: str = "",
        prev_idx: int | None = None,
        next_idx: int | None = None,
        beat_sec: float = 0.5,
        phrase_start: bool = False,
        phrase_end: bool = False,
        playback_mode: bool = False,
        prepare_only: bool = False,
    ) -> None:
        if not self.last_tracker_ok:
            print("❌ Need markers visible at least once before playing")
            return
        if not self.can_play_note_now():
            print("⚠️ Cannot play note now: unsafe state")
            return
        
        self.last_first_over_mA_ms = None
        self.last_launch_to_contact_ms = None
        exec_t0 = time.perf_counter()
        
        assert self.arm is not None and self.tracker is not None and self.key_model is not None
        try:
            if not prepare_only:
                self.stats.record_attempt(idx)

            xw_ref, yw_ref = self.key_model.key_world_xy(10)
            self.tracker.z_lock_keyplane(xw_ref, yw_ref, z_des_base=self.cfg.z_des_base)
            self.arm.T_B_P = SE3(self.tracker.B_T_W)

            x_nom, y_w = self.key_model.key_world_xy(idx)


            ctx = self.build_note_context(
                idx=idx,
                prev_idx=prev_idx,
                next_idx=next_idx,
                beat_sec=beat_sec,
                phrase_start=phrase_start,
                phrase_end=phrase_end,
            )
            motion_class = self.classify_motion_context(ctx)
            align_z = self.get_align_z_for_motion_class(motion_class)
            self.last_motion_class = motion_class

            x_cmd = self.ik_x_scale * x_nom + self.ik_x_bias_m + self.tune_dx_m
            x_tgt_w, y_tgt_w = self.strike_target_world_xy(idx)

            x_seed = float(x_cmd + (x_tgt_w - x_nom))
            y_seed = float(y_tgt_w)

            if (
                self.last_align_ok
                and self.last_play_idx is not None
                and self.last_finetune_ref_xy is not None
                and abs(idx - self.last_play_idx) == 1
            ):
                x_prev_tgt, y_prev_tgt = self.strike_target_world_xy(self.last_play_idx)
                dx_tgt = x_tgt_w - x_prev_tgt

                reuse_seed = np.array(
                    [
                        self.last_finetune_ref_xy[0] + dx_tgt,
                        self.last_finetune_ref_xy[1],   # keep Y unchanged for neighbour key
                    ],
                    dtype=float,
                )

                coarse_seed = np.array([x_cmd, y_w], dtype=float)
                delta_seed = reuse_seed - coarse_seed
                delta_mag = np.linalg.norm(delta_seed)

                if delta_mag > self.cfg.reuse_max_seed_offset_m and delta_mag > 1e-9:
                    delta_seed *= self.cfg.reuse_max_seed_offset_m / delta_mag
                    reuse_seed = coarse_seed + delta_seed

                x_seed = float(reuse_seed[0])
                y_seed = float(reuse_seed[1])
                print(
                    f"reuse fine-tune seed: prev_key={self.last_play_idx} -> idx={idx}   "
                    f"x_seed={x_seed:.4f}, y_seed={y_seed:.4f}"
                )

            if (
                not force_full_align
                and idx == self.last_play_idx
                and self.last_align_ok
                and self.last_finetune_ref_xy is not None
            ):
                print(f"➡️ repeat same key idx={idx}: reuse current XY, tap in Z only")
                repeat_hover_steps = 2 if playback_mode else 6
                self.goto_hover_pose_world(
                    float(self.last_finetune_ref_xy[0]),
                    float(self.last_finetune_ref_xy[1]),
                    steps=repeat_hover_steps,
                    dt=self.arm.dt,
                )

                ina_t0 = time.perf_counter()
                self.ina_start_log()
                press_min_hold = self.cfg.song_min_press_hold_s if playback_mode else self.cfg.min_press_hold_s
                used_profile_name, used_profile_source = self.do_current_press(
                    idx,
                    hold_s,
                    forced_profile_name=forced_profile_name,
                    min_press_hold_s=press_min_hold,
                    playback_mode=playback_mode,
                )
                ina_lines = self.ina_end_log()
                repeat_hover_steps = 2 if playback_mode else 6
                self.goto_hover_pose_world(
                    float(self.last_finetune_ref_xy[0]),
                    float(self.last_finetune_ref_xy[1]),
                    steps=repeat_hover_steps,
                    dt=self.arm.dt,
                )
                ina_summary = self.summarize_ina219_press(ina_lines)
                press_only_ms = ina_summary.get("first_over_mA_ms")
                launch_to_contact_ms = None

                # Compute launch_to_contact regardless of first_over value.
                # first_over=34ms is valid: it means contact happened 34ms after
                # INA_START.  launch_to_contact = alignment_phase + 34ms = ~613ms,
                # which IS a plausible step-press latency.
                if press_only_ms is not None:
                    launch_to_contact_ms = (ina_t0 - exec_t0) * 1000.0 + float(press_only_ms)

                self.last_first_over_mA_ms = press_only_ms
                self.last_launch_to_contact_ms = launch_to_contact_ms

                # Detect bad contact: first_over > 400ms on a repeat_reuse tap
                # means the arm descended but didn't reach the key until very late.
                # This usually means the hover Z was wrong (e.g. after an overrun).
                # Force fresh alignment on the next repeat rather than reusing this pose.
                if press_only_ms is not None and press_only_ms > self.cfg.late_contact_invalidate_ms:
                    self.last_align_ok = False
                    self.align_good_streak = 0
                    print(
                        f"⚠️ repeat_reuse late contact: first_over={press_only_ms}ms "
                        f"— invalidating aligned pose, next repeat will realign"
                    )

                if launch_to_contact_ms is not None:
                    allow_timing_learn = True

                    # Reject if launch_to_contact is physically implausible.
                    # Minimum: alignment alone (~150ms) + fastest press (~150ms) = 300ms.
                    # Maximum: if the note was very late to launch, the measurement is
                    # not representative of normal operation.
                    if launch_to_contact_ms < self.cfg.min_launch_to_contact_ms:
                        allow_timing_learn = False
                        print(
                            f"skip timing-model update: repeat_reuse "
                            f"launch_to_contact={launch_to_contact_ms:.0f}ms implausibly short"
                        )

                    if allow_timing_learn and self.last_best_err_m is not None and self.last_best_err_m > 0.008:
                        allow_timing_learn = False
                        print(
                            f"skip timing-model update: motion_class=repeat "
                            f"best_err={self.last_best_err_m*1000:.2f} mm"
                        )

                    if allow_timing_learn:
                        timing_summary = dict(ina_summary)
                        timing_summary["first_over_mA_ms"] = launch_to_contact_ms
                        self.stats.update_latency_stats("repeat_reuse", timing_summary)

                self.stats.record_success(idx, best_err_m=self.last_best_err_m, last_err_m=self.last_best_err_m)
                self.stats.record_session_result(
                    idx=idx,
                    best_err_mm=None if self.last_best_err_m is None else 1000.0 * self.last_best_err_m,
                    ina_summary=ina_summary,
                )
                self.stats.append_press_validation_row(
                    {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "key_idx": int(idx),
                        "press_profile": str(used_profile_name),
                        "profile_source": str(used_profile_source),
                        "forced_profile_name": "" if forced_profile_name is None else str(forced_profile_name),
                        "validation_run_id": str(validation_run_id),
                        "validation_trial": str(validation_trial),
                        "auto_base_ref_key": int(self.auto_press_base_key_idx),
                        "best_err_mm": None if self.last_best_err_m is None else 1000.0 * self.last_best_err_m,
                        "last_err_mm": None if self.last_best_err_m is None else 1000.0 * self.last_best_err_m,
                        "n_ina_samples": ina_summary["n_samples"],
                        "peak_current_mA": ina_summary["peak_current_mA"],
                        "mean_current_mA": ina_summary["mean_current_mA"],
                        "min_current_mA": ina_summary["min_current_mA"],
                        "min_bus_V": ina_summary["min_bus_V"],
                        "first_over_mA_ms": ina_summary["first_over_mA_ms"],
                        "peak_ms": ina_summary["peak_ms"],
                        "motion_class": "repeat",
                    }
                )
                return

            self.move_to_align_pose_safely(idx, x_seed, y_seed, motion_class=motion_class, align_z=align_z)
            print(
                f"➡️ play idx={idx}  "
                f"x_nom={x_nom:.4f}  x_cmd={x_cmd:.4f}  "
                f"(a={self.ik_x_scale:.6f}, b={self.ik_x_bias_m:.5f}, tune_dx={self.tune_dx_m*1000:.1f}mm)"
            )
            print(f"target key centre: x={x_tgt_w:.4f}, y={y_tgt_w:.4f}")

            use_light_align = self.should_use_light_align(idx) and (not force_full_align)

            if playback_mode and self.cfg.song_force_light_align:
                use_light_align = True

            best_err_m = None
            last_err_m = None
            x_ref_final = float(x_seed)
            y_ref_final = float(y_seed)

            if use_light_align:
                print("using LIGHT align")

                if playback_mode and self.cfg.song_force_light_align:
                    quick_timeout_s = self.cfg.song_quick_align_timeout_s
                    quick_tol_m = self.cfg.song_quick_align_tol_m
                    quick_iters = self.cfg.song_quick_align_max_iters
                    quick_detect_n = self.cfg.song_quick_detect_n
                    quick_detect_flush = self.cfg.song_quick_detect_flush
                    quick_detect_wait_s = self.cfg.song_quick_detect_wait_s

                    if prepare_only or motion_class == "phrase_start":
                        quick_timeout_s = max(quick_timeout_s, 0.24)
                        quick_iters = max(quick_iters, 3)
                        quick_detect_n = max(quick_detect_n, 3)
                        quick_detect_flush = max(quick_detect_flush, 1)

                    ok_align, best_err_m, last_err_m, x_ref_final, y_ref_final = self.align_to_key_center_pid(
                        idx=idx,
                        x_ref0=x_seed,
                        y_ref0=y_seed,
                        timeout_s=quick_timeout_s,
                        tol_m=quick_tol_m,
                        settle_frames=self.cfg.song_quick_align_settle_frames,
                        align_z=align_z,
                        max_iters=quick_iters,
                        detect_n=quick_detect_n,
                        detect_flush=quick_detect_flush,
                        detect_wait_s=quick_detect_wait_s,
                    )

                    accept_err_by_motion = {
                        "repeat": self.cfg.song_quick_accept_repeat_m,
                        "step": self.cfg.song_quick_accept_step_m,
                        "near": self.cfg.song_quick_accept_near_m,
                        "far": self.cfg.song_quick_accept_far_m,
                        "phrase_start": self.cfg.song_quick_accept_far_m,
                    }
                    playback_accept_err_m = accept_err_by_motion.get(
                        motion_class,
                        self.cfg.song_quick_accept_far_m,
                    )

                    if (not ok_align) and best_err_m is not None:
                        if (
                            motion_class in ("step", "near")
                            and best_err_m <= self.cfg.song_micro_retry_trigger_m
                        ):
                            print("quick LIGHT align incomplete -> micro retry")

                            ok2, best2, last2, x_ref_final, y_ref_final = self.align_to_key_center_pid(
                                idx=idx,
                                x_ref0=x_ref_final,
                                y_ref0=y_ref_final,
                                timeout_s=self.cfg.song_micro_retry_timeout_s,
                                tol_m=quick_tol_m,
                                settle_frames=1,
                                align_z=align_z,
                                max_iters=self.cfg.song_micro_retry_max_iters,
                                detect_n=2,
                                detect_flush=0,
                                detect_wait_s=0.002,
                            )

                            if best2 is not None:
                                best_err_m = min(best_err_m, best2)
                                last_err_m = last2
                            ok_align = ok2

                            micro_accept_by_motion = {
                                "step": self.cfg.song_micro_retry_accept_step_m,
                                "near": self.cfg.song_micro_retry_accept_near_m,
                            }
                            playback_accept_err_m = micro_accept_by_motion.get(
                                motion_class,
                                playback_accept_err_m,
                            )

                    if not ok_align:
                        if playback_mode:
                            playback_press_limit_by_motion = {
                                "repeat": self.cfg.song_quick_accept_repeat_m,
                                "step": self.cfg.song_quick_accept_step_m,
                                "near": self.cfg.song_quick_accept_near_m,
                                "far": self.cfg.song_quick_accept_far_m,
                                "phrase_start": self.cfg.song_quick_accept_near_m,
                            }
                            playback_press_limit_m = playback_press_limit_by_motion.get(
                                motion_class,
                                self.cfg.song_quick_accept_far_m,
                            )

                            hard_skip_limit_by_motion = {
                                "repeat": self.cfg.song_hard_skip_repeat_m,
                                "step": self.cfg.song_hard_skip_step_m,
                                "near": self.cfg.song_hard_skip_near_m,
                                "far": self.cfg.song_hard_skip_far_m,
                                "phrase_start": self.cfg.song_hard_skip_near_m,
                            }
                            hard_skip_limit_m = hard_skip_limit_by_motion.get(
                                motion_class,
                                self.cfg.song_hard_skip_far_m,
                            )

                            if best_err_m is None:
                                if self.last_align_ok and self.last_finetune_ref_xy is not None:
                                    print("⚠️ No blob feedback; reusing previous aligned pose for playback")
                                    x_ref_final = float(self.last_finetune_ref_xy[0])
                                    y_ref_final = float(self.last_finetune_ref_xy[1])
                                    best_err_m = self.last_best_err_m if self.last_best_err_m is not None else playback_press_limit_m
                                    last_err_m = best_err_m
                                    ok_align = True
                                else:
                                    self.last_align_ok = False
                                    self.align_good_streak = 0
                                    print("⚠️ No blob feedback and no reusable pose, skipping press")
                                    return
                            elif best_err_m <= playback_press_limit_m:
                                print(
                                    f"⚠️ Playback press allowed despite incomplete convergence "
                                    f"(best={best_err_m*1000:.2f} mm, soft_limit={playback_press_limit_m*1000:.2f} mm)"
                                )
                                ok_align = True
                            elif best_err_m <= hard_skip_limit_m:
                                print(
                                    f"⚠️ Playback press forced despite large error "
                                    f"(best={best_err_m*1000:.2f} mm, hard_skip={hard_skip_limit_m*1000:.2f} mm)"
                                )
                                ok_align = True
                            else:
                                self.last_align_ok = False
                                self.align_good_streak = 0
                                print(
                                    f"⚠️ Error too large, skipping press "
                                    f"(best={best_err_m*1000:.2f} mm, hard_skip={hard_skip_limit_m*1000:.2f} mm)"
                                )
                                return
                        else:
                            manual_limit_m = 0.0035
                            if best_err_m is None or best_err_m > manual_limit_m:
                                self.last_align_ok = False
                                self.align_good_streak = 0
                                self.last_finetune_ref_xy = None
                                self.last_play_idx = None
                                print("⚠️ PID alignment did not converge fully, skipping press")
                                return

                else:
                    # LIGHT align requested outside playback, but no special playback quick profile
                    ok_align, best_err_m, last_err_m, x_ref_final, y_ref_final = self.align_to_key_center_pid(
                        idx=idx,
                        x_ref0=x_seed,
                        y_ref0=y_seed,
                        align_z=align_z,
                    )

                    manual_limit_m = 0.0035
                    if best_err_m is None or best_err_m > manual_limit_m:
                        self.last_align_ok = False
                        self.align_good_streak = 0
                        self.last_finetune_ref_xy = None
                        self.last_play_idx = None
                        print("⚠️ PID alignment did not converge fully, skipping press")
                        return

            else:
                print("using FULL align")
                ok_align, best_err_m, last_err_m, x_ref_final, y_ref_final = self.align_to_key_center_pid(
                    idx=idx,
                    x_ref0=x_seed,
                    y_ref0=y_seed,
                    align_z=align_z,
                )

                manual_limit_m = 0.0035
                if playback_mode:
                    if best_err_m is None:
                        self.last_align_ok = False
                        self.align_good_streak = 0
                        print("⚠️ FULL align got no blob feedback, skipping press")
                        return
                else:
                    if best_err_m is None or best_err_m > manual_limit_m:
                        self.last_align_ok = False
                        self.align_good_streak = 0
                        self.last_finetune_ref_xy = None
                        self.last_play_idx = None
                        print("⚠️ FULL PID alignment did not converge, skipping press")
                        return

            print("✅ aligned, move to exact hover before press")
            pre_press_hover_steps = 4
            if playback_mode:
                if motion_class == "repeat":
                    pre_press_hover_steps = 1
                elif motion_class in ("step", "near"):
                    pre_press_hover_steps = 2
                else:
                    pre_press_hover_steps = 3

            self.goto_hover_pose_world(
                x_ref_final,
                y_ref_final,
                steps=pre_press_hover_steps,
                dt=self.arm.dt,
                hover_z=align_z,
            )

            if prepare_only:
                self.last_play_idx = idx
                self.last_align_ok = True
                self.last_best_err_m = best_err_m
                self.align_good_streak = (
                    self.align_good_streak + 1
                    if (best_err_m is not None and best_err_m <= self.cfg.light_align_good_err_m)
                    else 0
                )
                self.last_finetune_ref_xy = np.array([x_ref_final, y_ref_final], dtype=float)

                print("✅ prepared first note at hover; timing will start on press")
                return

            ina_t0 = time.perf_counter()
            self.ina_start_log()
            print("✅ pressing")
            press_min_hold = self.cfg.song_min_press_hold_s if playback_mode else self.cfg.min_press_hold_s
            used_profile_name, used_profile_source = self.do_current_press(
                idx,
                hold_s,
                forced_profile_name=forced_profile_name,
                min_press_hold_s=press_min_hold,
                playback_mode=playback_mode,
            )
            ina_lines = self.ina_end_log()
            print("✅ return to hover after press")
            hover_steps = 6
            if playback_mode:
                if motion_class == "repeat":
                    hover_steps = 2
                elif motion_class in ("step", "near"):
                    hover_steps = 3
                else:
                    hover_steps = 4

            self.goto_hover_pose_world(
                x_ref_final,
                y_ref_final,
                steps=hover_steps,
                dt=self.arm.dt,
                hover_z=align_z,
            )

            self.stats.append_ina_raw_log(ina_lines, key_idx=idx, trial_label=validation_trial, run_id=validation_run_id)
            ina_summary = self.summarize_ina219_press(ina_lines)

            press_only_ms = ina_summary.get("first_over_mA_ms")
            launch_to_contact_ms = None
            # Compute launch_to_contact from any non-None first_over, including
            # small values like 34ms.  The plausibility check below will decide
            # whether it is usable for the timing model.
            if press_only_ms is not None:
                launch_to_contact_ms = (ina_t0 - exec_t0) * 1000.0 + float(press_only_ms)

            self.last_first_over_mA_ms = press_only_ms
            self.last_launch_to_contact_ms = launch_to_contact_ms

            # Detect service overrun: if this note took longer than its onset gap,
            # the arm is not in a reliable hover position for the next note.
            # Invalidate the aligned seed so the next note performs fresh alignment
            # rather than using a stale position from a compromised press.
            if next_idx is not None:
                note_service_s = time.perf_counter() - exec_t0
                if note_service_s > beat_sec * 1.10:   # >10% over beat duration
                    self.last_align_ok = False
                    self.align_good_streak = 0
                    print(
                        f"⚠️ service overrun detected ({note_service_s*1000:.0f}ms > "
                        f"{beat_sec*1000:.0f}ms): invalidating seed for next note"
                    )

            if launch_to_contact_ms is not None:
                allow_timing_learn = True

                # Reject if launch_to_contact is physically implausible.
                # Too short (< 200ms): would imply contact before press mechanics finish.
                # Too long (> 2000ms): note was launched very late, not representative.
                if launch_to_contact_ms < self.cfg.min_launch_to_contact_ms:
                    allow_timing_learn = False
                    print(
                        f"skip timing-model update: motion_class={motion_class} "
                        f"launch_to_contact={launch_to_contact_ms:.0f}ms implausibly short"
                    )

                # reject spatially poor samples
                if allow_timing_learn and best_err_m is not None and best_err_m > 0.012:
                    allow_timing_learn = False

                # step/near: tighter spatial requirement during playback
                if (allow_timing_learn and playback_mode
                        and motion_class in ("step", "near")
                        and best_err_m is not None and best_err_m > 0.010):
                    allow_timing_learn = False

                # reject latency outliers — window widened to cover slow BPM
                if allow_timing_learn:
                    if motion_class == "repeat":
                        if not (0.15 <= launch_to_contact_ms / 1000.0 <= 1.00):
                            allow_timing_learn = False
                    elif motion_class in ("step", "near"):
                        if not (0.40 <= launch_to_contact_ms / 1000.0 <= 2.00):
                            allow_timing_learn = False

                if allow_timing_learn:
                    timing_summary = dict(ina_summary)
                    timing_summary["first_over_mA_ms"] = launch_to_contact_ms
                    self.stats.update_latency_stats(motion_class, timing_summary)
                else:
                    print(
                        f"skip timing-model update: motion_class={motion_class} "
                        f"best_err={best_err_m*1000:.2f} mm  "
                        f"launch={launch_to_contact_ms:.1f} ms"
                        if best_err_m is not None
                        else f"skip timing-model update: motion_class={motion_class} launch={launch_to_contact_ms:.1f} ms"
                    )


            best_err_mm = None if best_err_m is None else 1000.0 * float(best_err_m)
            last_err_mm = None if last_err_m is None else 1000.0 * float(last_err_m)

            self.stats.record_session_result(idx=idx, best_err_mm=best_err_mm, ina_summary=ina_summary)
            self.stats.append_press_validation_row(
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "key_idx": int(idx),
                    "press_profile": str(used_profile_name),
                    "profile_source": str(used_profile_source),
                    "forced_profile_name": "" if forced_profile_name is None else str(forced_profile_name),
                    "validation_run_id": str(validation_run_id),
                    "validation_trial": str(validation_trial),
                    "auto_base_ref_key": int(self.auto_press_base_key_idx),
                    "best_err_mm": best_err_mm,
                    "last_err_mm": last_err_mm,
                    "n_ina_samples": ina_summary["n_samples"],
                    "peak_current_mA": ina_summary["peak_current_mA"],
                    "mean_current_mA": ina_summary["mean_current_mA"],
                    "min_current_mA": ina_summary["min_current_mA"],
                    "min_bus_V": ina_summary["min_bus_V"],
                    "first_over_mA_ms": ina_summary["first_over_mA_ms"],
                    "peak_ms": ina_summary["peak_ms"],
                    "motion_class": str(self.last_motion_class),
                }
            )

            print(
                f"[INA219 summary] "
                f"n={ina_summary['n_samples']}  "
                f"peak={ina_summary['peak_current_mA']} mA  "
                f"mean={ina_summary['mean_current_mA']} mA  "
                f"min_bus_V={ina_summary['min_bus_V']} V"
            )

            self.stats.record_success(idx, best_err_m=best_err_m, last_err_m=last_err_m)

            reuse_seed_limit_by_motion = {
                "repeat": self.cfg.song_reuse_seed_repeat_m,
                "step": self.cfg.song_reuse_seed_step_m,
                "near": self.cfg.song_reuse_seed_near_m,
                "far": self.cfg.song_reuse_seed_far_m,
                "phrase_start": self.cfg.song_reuse_seed_far_m,
            }

            reuse_seed_limit_m = reuse_seed_limit_by_motion.get(motion_class, self.cfg.song_reuse_seed_far_m)

            self.last_play_idx = idx
            self.last_best_err_m = best_err_m

            if best_err_m is not None and best_err_m <= reuse_seed_limit_m:
                self.last_align_ok = True
                self.last_finetune_ref_xy = np.array([x_ref_final, y_ref_final], dtype=float)
                self.align_good_streak = (
                    self.align_good_streak + 1
                    if best_err_m <= self.cfg.light_align_good_err_m
                    else 0
                )
            else:
                # Press may be allowed, but do not reuse this pose as a seed
                self.last_align_ok = False
                self.align_good_streak = 0
                print(
                    f"not promoting align to reusable seed "
                    f"(best={None if best_err_m is None else best_err_m*1000:.2f} mm)"
                )

            if not playback_mode:
                self.offsets.learn_from_success(
                    idx=idx,
                    x_seed=x_seed,
                    y_seed=y_seed,
                    x_ref_final=x_ref_final,
                    y_ref_final=y_ref_final,
                    best_err_m=best_err_m,
                )

        except (serial.SerialTimeoutException, serial.SerialException) as exc:
            print(f"❌ Serial error during execute_key: {exc}")
            self.handle_serial_failure(f"execute_key serial error: {exc}")
            self.last_align_ok = False
            self.last_finetune_ref_xy = None

    def get_song_path(self, song_name_or_path: str) -> str:
        if song_name_or_path.endswith(".json"):
            return str(self.cfg.paths.songs_dir / song_name_or_path)
        return str(self.cfg.paths.songs_dir / f"{song_name_or_path}.json")
        
    def load_song_json(self, song_name_or_path: str) -> dict[str, Any]:
        path = self.get_song_path(song_name_or_path)
        with open(path, "r", encoding="utf-8") as f:
            song = json.load(f)

        if "events" not in song or not isinstance(song["events"], list):
            raise ValueError(f"Song file missing 'events' list: {path}")

        if "bpm" not in song:
            raise ValueError(f"Song file missing 'bpm': {path}")

        return song
        
    def build_score_events_from_song(self, song: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []

        for i, ev in enumerate(song["events"]):
            if "key" not in ev:
                raise ValueError(f"Song event #{i} missing 'key'")

            idx = int(ev["key"])
            if not (0 <= idx < self.cfg.n_keys):
                raise ValueError(f"Song event #{i} key out of range: {idx}")

            onset_beats = float(ev.get("beat", ev.get("onset_beats", 0.0)))
            dur_beats = float(ev.get("dur", ev.get("dur_beats", 1.0)))
            hold_ratio = float(ev.get("hold_ratio", self.cfg.default_song_hold_ratio))

            out.append({
                "idx": idx,
                "onset_beats": onset_beats,
                "dur_beats": dur_beats,
                "hold_ratio": hold_ratio,
                "phrase_start": bool(ev.get("phrase_start", False)),
                "phrase_end": bool(ev.get("phrase_end", False)),
            })

        out.sort(key=lambda e: e["onset_beats"])
        return out
        
    def play_song(self, song_name_or_path: str | None = None) -> None:
        self.refresh_song_library()

        if song_name_or_path is None:
            song_name_or_path = self.get_current_song_name()

        if song_name_or_path is None:
            self.flash_ui_message("No song selected")
            return

        song = self.load_song_json(song_name_or_path)
        score_events = self.build_score_events_from_song(song)
        bpm = float(song["bpm"])
        title = song.get("name", song_name_or_path)

        print(f"\n▶ Playing song: {title}")
        print(f"bpm={bpm}, events={len(score_events)}")

        self._current_song_title = title
        self.play_score_events(score_events, bpm=bpm)

    def play_score_events(self, score_events: list[dict[str, Any]], bpm: float) -> None:
        beat_sec = 60.0 / bpm

        self.last_play_idx = None
        self.last_align_ok = False
        self.last_best_err_m = None
        self.last_finetune_ref_xy = None
        self.align_good_streak = 0

        # Prepare the first note before the song clock starts
        if score_events:
            first_event = score_events[0]
            first_idx = first_event["idx"]
            first_next_idx = score_events[1]["idx"] if len(score_events) > 1 else None
            first_dur_beats = float(first_event.get("dur_beats", 1.0))

            print(f"preparing first note idx={first_idx} before starting playback timing")

            self.execute_key(
                first_idx,
                hold_s=0.0,
                force_full_align=not self.cfg.song_force_light_align,
                forced_profile_name=self.cfg.song_force_profile_name,
                prev_idx=None,
                next_idx=first_next_idx,
                beat_sec=first_dur_beats * beat_sec,
                phrase_start=first_event.get("phrase_start", True),
                phrase_end=first_event.get("phrase_end", False),
                playback_mode=True,
                prepare_only=True,
            )

        # ── Arm preparation complete ──────────────────────────────────────
        # The arm is now hovering over the first key, aligned and ready.
        # If song_wait_for_trigger is enabled, pause here and wait for the
        # operator to press the trigger key before the song clock starts.
        # This mirrors a pianist positioning their hand before playing.
        if self.cfg.song_wait_for_trigger:
            song_title = getattr(self, '_current_song_title', 'song')
            triggered = self._wait_for_start_trigger(song_title)
            if not triggered:
                # Operator aborted — return arm to neutral and exit
                if self.cfg.enable_end_reset_pose:
                    self.goto_system_neutral_pose(reason="song start aborted")
                return

        start_t = time.perf_counter() + self.cfg.song_start_lead_in_s

        for n, event in enumerate(score_events):
            idx = event["idx"]
            onset_beats = float(event["onset_beats"])
            dur_beats = float(event.get("dur_beats", 1.0))
            phrase_start = bool(event.get("phrase_start", False))
            phrase_end = bool(event.get("phrase_end", False))

            prev_idx = score_events[n - 1]["idx"] if n > 0 else None
            next_idx = score_events[n + 1]["idx"] if (n + 1) < len(score_events) else None
            next_next_idx = score_events[n + 2]["idx"] if (n + 2) < len(score_events) else None

            beat_duration_s = dur_beats * beat_sec

            ctx = self.build_note_context(
                idx=idx,
                prev_idx=prev_idx,
                next_idx=next_idx,
                beat_sec=beat_duration_s,
                phrase_start=phrase_start,
                phrase_end=phrase_end,
            )

            motion_class = self.classify_motion_context(ctx)
            effective_motion_class = motion_class

            if (
                n == 0
                and self.last_align_ok
                and self.last_play_idx == idx
                and self.last_finetune_ref_xy is not None
            ):
                motion_class = "repeat"
                effective_motion_class = "repeat_reuse"
            elif (
                motion_class == "repeat"
                and self.last_align_ok
                and self.last_play_idx == idx
                and self.last_finetune_ref_xy is not None
            ):
                effective_motion_class = "repeat_reuse"


            predicted_latency_s = (
                self.stats.predict_note_latency_s(effective_motion_class)
                + self.cfg.playback_latency_margin_s
            )

            hold_s = self.compute_playback_hold_s(
                score_events=score_events,
                n=n,
                beat_sec=beat_sec,
                forced_profile_name=self.cfg.song_force_profile_name,
            )
            # No second clamp here: compute_playback_hold_s already accounts for
            # release_tail + playback_release_guard + next_launch_lead correctly.
            # The old reserve_s = 0.35 clamp was double-counting those margins.

            onset_time = start_t + onset_beats * beat_sec
            launch_time = onset_time - predicted_latency_s

            now = time.perf_counter()
            lateness_s = now - launch_time

            if launch_time > now:
                time.sleep(launch_time - now)
            else:
                print(f"⚠️ late launch for key {idx}: {lateness_s*1000:.1f} ms")

            print(
                f"scheduled idx={idx}  motion_class={motion_class}  "
                f"launch_class={effective_motion_class}  "
                f"pred_latency={predicted_latency_s*1000:.1f} ms  hold_s={hold_s:.3f}"
            )
            self.last_first_over_mA_ms = None
            note_exec_t0 = time.perf_counter()

            self.execute_key(
                idx,
                hold_s=hold_s,
                forced_profile_name=self.cfg.song_force_profile_name,
                prev_idx=prev_idx,
                next_idx=next_idx,
                beat_sec=beat_duration_s,
                phrase_start=phrase_start,
                phrase_end=phrase_end,
                playback_mode=True,
            )

            note_exec_s = time.perf_counter() - note_exec_t0
            print(f"note service time idx={idx}: {note_exec_s*1000:.1f} ms")

            if next_idx is not None:
                next_onset_beats = float(score_events[n + 1]["onset_beats"])
                gap_to_next_s = max(0.0, (next_onset_beats - onset_beats) * beat_sec)
                overrun_s = note_exec_s - gap_to_next_s
                if overrun_s > 0:
                    print(
                        f"⚠️ service overrun idx={idx}: "
                        f"{overrun_s*1000:.1f} ms beyond next onset gap"
                    )

            launch_to_contact_ms = self.last_launch_to_contact_ms
            if launch_to_contact_ms is not None:
                pred_ms = predicted_latency_s * 1000.0
                timing_err_ms = float(launch_to_contact_ms) - pred_ms
                self.stats.update_timing_error_stats(motion_class, predicted_latency_s, launch_to_contact_ms)

                print(
                    f"timing check idx={idx}  motion_class={motion_class}  "
                    f"pred_launch={pred_ms:.1f} ms  "
                    f"actual_launch={launch_to_contact_ms:.1f} ms  "
                    f"err={timing_err_ms:+.1f} ms"
                )
            else:
                print(f"timing check idx={idx}  motion_class={motion_class}  actual_launch=missing")
                
        if self.cfg.enable_end_reset_pose and not self.safety.stop_requested:
            self.goto_system_neutral_pose(reason="end of score")

    # ----------------------------
    # Operator workflows
    # ----------------------------
    def run_selected_key_profile_validation(self) -> None:
        idx_sel = self.get_selected_idx()
        if idx_sel is None:
            print("Type/select a key first, then run profile validation.")
            return

        bpm = self.cfg.validation_bpm
        beat_sec = 60.0 / bpm
        hold_s = self.cfg.validation_hold_ratio * beat_sec
        run_id = time.strftime("%Y%m%d_%H%M%S") + f"_key{idx_sel}"

        print(f"\n▶ Press-profile validation on key {idx_sel}")
        print(f"run_id = {run_id}")
        print(f"profiles = {self.cfg.press_profile_order}, reps/profile = {self.cfg.validation_reps_per_profile}")

        t_deadline = time.perf_counter()
        for prof_name in self.cfg.press_profile_order:
            print(f"\n=== Validation profile: {prof_name} ===")
            for rep in range(self.cfg.validation_reps_per_profile):
                now = time.perf_counter()
                if t_deadline > now:
                    time.sleep(t_deadline - now)

                trial_label = f"{prof_name}_{rep+1}"
                print(f"\n--- validation {trial_label} ---")
                self.execute_key(
                    idx_sel,
                    hold_s=hold_s,
                    force_full_align=self.cfg.validation_force_full_align,
                    forced_profile_name=prof_name,
                    validation_run_id=run_id,
                    validation_trial=trial_label,
                )
                t_deadline += beat_sec

        print(f"\n✅ Press-profile validation complete for key {idx_sel}")
        print(f"CSV log: {self.cfg.paths.press_log_csv}\n")

    def run_repeat_timing_test(self) -> None:
        idx_sel = self.get_selected_idx()
        if idx_sel is None:
            print("Type/select a key first, then run repeat timing test.")
            return

        score_events = [
            {"idx": idx_sel, "onset_beats": 0.0, "dur_beats": 0.5, "hold_ratio": 0.45, "phrase_start": True},
            {"idx": idx_sel, "onset_beats": 0.5, "dur_beats": 0.5, "hold_ratio": 0.45},
            {"idx": idx_sel, "onset_beats": 1.0, "dur_beats": 0.5, "hold_ratio": 0.45},
            {"idx": idx_sel, "onset_beats": 1.5, "dur_beats": 0.5, "hold_ratio": 0.45},
            {"idx": idx_sel, "onset_beats": 2.0, "dur_beats": 0.5, "hold_ratio": 0.45, "phrase_end": True},
        ]
        print(f"\n▶ Repeat timing test on key {idx_sel}")
        self.play_score_events(score_events, bpm=100)

    def run_white_key_sweep(self) -> None:
        bpm = self.cfg.white_key_test_bpm
        beat_sec = 60.0 / bpm
        hold_s = self.cfg.white_key_test_hold_ratio * beat_sec
        t_deadline = time.perf_counter()

        print("\n▶ Running white-key sweep test...")
        for key_idx in self.cfg.white_key_test_sequence:
            now = time.perf_counter()
            if t_deadline > now:
                time.sleep(t_deadline - now)
            print(f"\n--- sweep key {key_idx} ---")
            self.execute_key(key_idx, hold_s=hold_s)
            t_deadline += beat_sec

        print("\n✅ White-key sweep finished")
        self.stats.print_ranked_white_key_report()
        self.print_automatic_tuning_checklist()
        self._save_tune_session_if_enabled()

    def run_selected_key_repeated_test(self) -> None:
        idx_sel = self.get_selected_idx()
        if idx_sel is None:
            print("Type/select a key first, or press 'j' to select the worst tested key.")
            return

        bpm = self.cfg.focus_key_test_bpm
        beat_sec = 60.0 / bpm
        hold_s = self.cfg.focus_key_test_hold_ratio * beat_sec

        print(f"\n▶ Repeated test on key {idx_sel} for {self.cfg.focus_key_test_reps} reps...")
        t_deadline = time.perf_counter()

        for rep in range(self.cfg.focus_key_test_reps):
            now = time.perf_counter()
            if t_deadline > now:
                time.sleep(t_deadline - now)
            print(f"\n--- key {idx_sel} rep {rep+1}/{self.cfg.focus_key_test_reps} ---")
            self.execute_key(idx_sel, hold_s=hold_s, force_full_align=True)
            t_deadline += beat_sec

        self.print_focus_key_summary(idx_sel)
        self.print_automatic_tuning_checklist()

        if self.cfg.focus_auto_advance_on_pass:
            status = self.key_status_from_report(idx_sel)
            print(f"focus key {idx_sel} status after repeated test: {status}")
            if status != "RETUNE":
                print("✅ Focus key no longer flagged for retune. Advancing to next weak key...")
                self.rebuild_weak_key_queue()
                if idx_sel in self.tune.weak_key_queue:
                    pos_now = self.tune.weak_key_queue.index(idx_sel)
                    if pos_now + 1 < len(self.tune.weak_key_queue):
                        self.select_weak_key_queue_pos(pos_now + 1)
                    else:
                        print("🎉 No further weak keys in queue.")
                elif self.tune.weak_key_queue:
                    self.select_weak_key_queue_pos(0)
                else:
                    print("🎉 Weak key queue is now empty.")

        self._save_tune_session_if_enabled()

    def run_servo5_motion_test(self) -> None:
        idx_sel = self.get_selected_idx()
        if idx_sel is None:
            print("Type a key index first, then press '/'")
            return
        if not self.last_tracker_ok:
            print("❌ Need markers visible at least once before servo 5 test")
            return
        if not self.can_play_note_now():
            print("⚠️ Cannot play note now: unsafe state")
            return

        assert self.arm is not None and self.key_model is not None and self.tracker is not None
        print(f"\n▶ Servo 5 motion test on key {idx_sel}")

        xw_ref, yw_ref = self.key_model.key_world_xy(10)
        self.tracker.z_lock_keyplane(xw_ref, yw_ref, z_des_base=self.cfg.z_des_base)
        self.arm.T_B_P = SE3(self.tracker.B_T_W)

        x_nom, y_w = self.key_model.key_world_xy(idx_sel)
        x_cmd = self.ik_x_scale * x_nom + self.ik_x_bias_m + self.tune_dx_m
        x_seed, y_seed = float(x_cmd), float(y_w)

        if (
            self.last_align_ok
            and self.last_play_idx is not None
            and self.last_finetune_ref_xy is not None
            and self.last_best_err_m is not None
            and self.last_best_err_m <= 0.0030   # only reuse if previous key was aligned within 3 mm
            and 0 < abs(idx_sel - self.last_play_idx) <= 1
        ):
            
            x_prev_tgt, y_prev_tgt = self.strike_target_world_xy(self.last_play_idx)
            x_new_tgt, y_new_tgt = self.strike_target_world_xy(idx_sel)
            dx_tgt, dy_tgt = x_new_tgt - x_prev_tgt, y_new_tgt - y_prev_tgt
            reuse_seed = np.array(
                [self.last_finetune_ref_xy[0] + dx_tgt, y_new_tgt],
                dtype=float,
            )
            coarse_seed = np.array([x_cmd, y_w], dtype=float)
            delta_seed = reuse_seed - coarse_seed
            delta_mag = np.linalg.norm(delta_seed)
            if delta_mag > self.cfg.reuse_max_seed_offset_m and delta_mag > 1e-9:
                delta_seed *= self.cfg.reuse_max_seed_offset_m / delta_mag
                reuse_seed = coarse_seed + delta_seed
            x_seed, y_seed = float(reuse_seed[0]), float(reuse_seed[1])

        self.move_to_align_pose_safely(idx_sel, x_seed, y_seed)
        if self.cfg.servo5_test_use_pid_align:
            ok_align, best_err_m, last_err_m, _, _ = self.align_to_key_center_pid(idx=idx_sel, x_ref0=x_seed, y_ref0=y_seed)
            print(f"servo5 test align result: ok={ok_align}, best={best_err_m}, last={last_err_m}")
            if best_err_m is None:
                print("❌ No blob feedback, aborting servo 5 test")
                return

        self.arm.set_servo5_smooth(self.cfg.servo5_test_base_deg, steps=self.cfg.servo5_test_move_steps, dt=self.cfg.servo5_test_move_dt)
        time.sleep(self.cfg.servo5_test_settle_s)

        w0 = self.detect_world_xy_avg(n=8, flush=2, wait_s=0.01)
        if w0 is None:
            print("❌ Could not detect blob at baseline")
            return

        base_deg = float(self.arm.servo5_current_deg)
        print(f"baseline blob world: x={w0[0]:.4f}, y={w0[1]:.4f}")
        print("\nservo5_deg | dx_mm | dy_mm | drift_mm")
        print("-" * 40)

        try:
            for ddeg in self.cfg.servo5_test_offsets_deg:
                test_deg = base_deg + float(ddeg)
                self.arm.set_servo5_smooth(test_deg, steps=self.cfg.servo5_test_move_steps, dt=self.cfg.servo5_test_move_dt)
                time.sleep(self.cfg.servo5_test_settle_s)
                w = self.detect_world_xy_avg(n=8, flush=2, wait_s=0.01)
                if w is None:
                    print(f"{test_deg:>10.1f} |  lost |  lost |   lost")
                    continue
                dx_mm = 1000.0 * (w[0] - w0[0])
                dy_mm = 1000.0 * (w[1] - w0[1])
                drift_mm = float(np.hypot(dx_mm, dy_mm))
                print(f"{test_deg:>10.1f} | {dx_mm:>5.2f} | {dy_mm:>5.2f} | {drift_mm:>7.2f}")
            print("\nReturning servo 5 to baseline...")
        finally:
            self.arm.set_servo5_smooth(base_deg, steps=self.cfg.servo5_test_move_steps, dt=self.cfg.servo5_test_move_dt)
            time.sleep(self.cfg.servo5_test_settle_s)
            print("✅ Servo 5 test finished\n")

    # ----------------------------
    # Reports / tune queue
    # ----------------------------
    def get_report_row_for_key(self, idx: int) -> dict[str, Any] | None:
        for row in self.stats.build_white_key_report_rows():
            if row["key"] == idx:
                return row
        return None

    def key_status_from_report(self, idx: int) -> str:
        row = self.get_report_row_for_key(idx)
        return "NO_DATA" if row is None else str(row["status"])

    def suggestion_to_nudge_text(self, suggestion_xy: np.ndarray, nudge_mm: float | None = None) -> str:
        nudge_mm = self.cfg.checklist_nudge_mm if nudge_mm is None else nudge_mm
        dx_m, dy_m = float(suggestion_xy[0]), float(suggestion_xy[1])
        if abs(dx_m) >= abs(dy_m):
            return f"NUDGE {'+X' if dx_m >= 0.0 else '-X'} {nudge_mm:.0f} mm"
        return f"NUDGE {'+Y' if dy_m >= 0.0 else '-Y'} {nudge_mm:.0f} mm"

    def recommend_key_action(self, idx: int) -> tuple[str, str]:
        row = self.get_report_row_for_key(idx)
        if row is None:
            return "NO_DATA", "Run test first"

        status = row["status"]
        attempts = row["attempts"]
        success_pct = row["success_pct"]
        mean_best_mm = row["mean_best_mm"]

        dx_i, dy_i = self.offsets.get(idx)
        offset_mag_mm = 1000.0 * float(np.hypot(dx_i, dy_i))
        suggestion, n_sugg = self.offsets.get_suggestion(idx)
        sugg_mag_mm = 0.0 if suggestion is None else 1000.0 * float(np.hypot(suggestion[0], suggestion[1]))

        if attempts == 0:
            return "TEST_KEY", "Run focused test first"
        if status == "GOOD":
            return "DONE", "No action needed"
        if suggestion is not None and n_sugg >= self.cfg.auto_cal_min_samples and sugg_mag_mm >= self.cfg.checklist_min_suggestion_mm:
            if status == "RETUNE":
                return "APPLY_LEARNED_SUGGESTION", "Press 'y' to apply learned suggestion"
            return "OPTIONAL_APPLY_SUGGESTION", "Suggestion looks usable; apply if desired"
        if attempts < 3:
            return "COLLECT_MORE_TESTS", "Run repeated focused test"
        if status == "RETUNE" and suggestion is not None and n_sugg > 0:
            return self.suggestion_to_nudge_text(suggestion), "Then rerun focused test"
        if status == "RETUNE":
            if offset_mag_mm > self.cfg.checklist_max_manual_offset_mm:
                return "RETEST_AFTER_MANUAL_TUNE", "Offset already large; inspect strike point/mechanics"
            return "MANUAL_NUDGE_SMALL", "Try ±1 mm on dominant observed error direction"
        if status == "OK":
            if success_pct < 90.0 or (mean_best_mm is not None and mean_best_mm > 3.5):
                return "RETEST_LATER", "Keep in queue and gather more samples"
            return "LEAVE_AS_IS", "Acceptable for now"
        return "REVIEW", "Inspect manually"

    def print_automatic_tuning_checklist(self) -> None:
        rows = self.stats.build_white_key_report_rows()
        if self.cfg.checklist_print_all_keys:
            checklist_rows = [r for r in rows if r["attempts"] > 0]
        else:
            checklist_rows = [r for r in rows if r["status"] in ("RETUNE", "OK") and r["attempts"] > 0]

        checklist_rows = sorted(
            checklist_rows,
            key=lambda r: (r["status"] != "RETUNE", r["success_pct"], -(r["mean_best_mm"] if r["mean_best_mm"] is not None else 1e9)),
        )

        print("\n=== Automatic tuning checklist ===")
        if not checklist_rows:
            print("🎉 No keys currently need tuning action.\n")
            return

        print("key | status | offset(dx,dy) mm | suggestion | action | note")
        for row in checklist_rows:
            idx = row["key"]
            dx_i, dy_i = self.offsets.get(idx)
            suggestion, n_sugg = self.offsets.get_suggestion(idx)
            if suggestion is None:
                sugg_str = "none"
            else:
                sugg_str = f"n={n_sugg} ({suggestion[0]*1000:.1f},{suggestion[1]*1000:.1f})"
            action, note = self.recommend_key_action(idx)
            print(
                f"{idx:>3} | {row['status']:<6} | ({dx_i*1000:>5.1f},{dy_i*1000:>5.1f}) | "
                f"{sugg_str:<18} | {action:<26} | {note}"
            )
        print()

    def print_tuning_todo_list(self, top_n: int | None = None) -> None:
        top_n = top_n or self.cfg.todo_list_top_n
        rows = self.stats.build_white_key_report_rows()
        todo_rows = [r for r in rows if r["status"] == "RETUNE"]
        todo_rows = sorted(
            todo_rows,
            key=lambda r: (r["success_pct"], -(r["mean_best_mm"] if r["mean_best_mm"] is not None else 1e9)),
        )

        print("\n=== Tuning to-do list ===")
        if not todo_rows:
            print("🎉 No keys currently flagged for retune.\n")
            return

        print("key | succ% | mean_best_mm | min_best_mm | attempts | successes")
        for row in todo_rows[:top_n]:
            mean_best = "-" if row["mean_best_mm"] is None else f"{row['mean_best_mm']:.2f}"
            min_best = "-" if row["min_best_mm"] is None else f"{row['min_best_mm']:.2f}"
            print(
                f"{row['key']:>3} | {row['success_pct']:>5.1f}% | {mean_best:>12} | "
                f"{min_best:>11} | {row['attempts']:>8} | {row['successes']:>9}"
            )
        print()

    def rebuild_weak_key_queue(self) -> None:
        rows = self.stats.build_white_key_report_rows()
        tested = [r for r in rows if r["attempts"] > 0]
        if self.cfg.focus_include_ok_keys_in_queue:
            queue_rows = [r for r in tested if r["status"] in ("RETUNE", "OK")]
        else:
            queue_rows = [r for r in tested if r["status"] == "RETUNE"]
        queue_rows = sorted(
            queue_rows,
            key=lambda r: (r["success_pct"], -(r["mean_best_mm"] if r["mean_best_mm"] is not None else 1e9)),
        )
        self.tune.weak_key_queue = [r["key"] for r in queue_rows]
        self.tune.weak_key_queue_pos = 0 if self.tune.weak_key_queue else -1
        print(f"weak key queue rebuilt: {self.tune.weak_key_queue}")

    def select_weak_key_queue_pos(self, pos: int) -> int | None:
        if not self.tune.weak_key_queue:
            print("ℹ️ Weak key queue is empty. Run a sweep/report first.")
            return None
        pos = max(0, min(int(pos), len(self.tune.weak_key_queue) - 1))
        self.tune.weak_key_queue_pos = pos
        key_idx = self.tune.weak_key_queue[self.tune.weak_key_queue_pos]
        self.tune.typed = str(key_idx)

        print(f"🎯 Weak key queue {self.tune.weak_key_queue_pos+1}/{len(self.tune.weak_key_queue)} -> key {key_idx}")
        self.print_focus_key_summary(key_idx)
        self._save_tune_session_if_enabled()
        return key_idx

    def select_worst_tested_white_key(self) -> int | None:
        self.rebuild_weak_key_queue()
        if not self.tune.weak_key_queue:
            print("ℹ️ No weak tested white keys found.")
            return None
        return self.select_weak_key_queue_pos(0)

    def select_next_weak_key(self) -> int | None:
        if not self.tune.weak_key_queue:
            self.rebuild_weak_key_queue()
        if not self.tune.weak_key_queue:
            print("ℹ️ No weak keys available.")
            return None
        next_pos = min(self.tune.weak_key_queue_pos + 1, len(self.tune.weak_key_queue) - 1)
        if next_pos == self.tune.weak_key_queue_pos:
            print("ℹ️ Already at the last weak key.")
        return self.select_weak_key_queue_pos(next_pos)

    def select_prev_weak_key(self) -> int | None:
        if not self.tune.weak_key_queue:
            self.rebuild_weak_key_queue()
        if not self.tune.weak_key_queue:
            print("ℹ️ No weak keys available.")
            return None
        prev_pos = max(self.tune.weak_key_queue_pos - 1, 0)
        if prev_pos == self.tune.weak_key_queue_pos:
            print("ℹ️ Already at the first weak key.")
        return self.select_weak_key_queue_pos(prev_pos)

    def print_focus_key_summary(self, idx: int) -> None:
        st = self.stats.key_perf_stats[idx]
        success_pct = (100.0 * st.successes / st.attempts) if st.attempts > 0 else 0.0
        mean_best_mm = (1000.0 * st.best_err_sum_m / st.successes) if st.successes > 0 else None
        min_best_mm = (1000.0 * st.best_err_min_m) if st.best_err_min_m is not None else None
        dx_i, dy_i = self.offsets.get(idx)
        suggestion, n_sugg = self.offsets.get_suggestion(idx)

        print(f"\n=== Focus key {idx} ===")
        print(f"attempts={st.attempts}, successes={st.successes}, success%={success_pct:.1f}")
        print("mean_best_mm=-" if mean_best_mm is None else f"mean_best_mm={mean_best_mm:.2f}")
        print("min_best_mm=-" if min_best_mm is None else f"min_best_mm={min_best_mm:.2f}")
        print(f"current per-key offset: dx={dx_i*1000:.2f} mm, dy={dy_i*1000:.2f} mm")
        if suggestion is None:
            print("learned suggestion: none")
        else:
            print(f"learned suggestion: n={n_sugg}, dx={suggestion[0]*1000:.2f} mm, dy={suggestion[1]*1000:.2f} mm")
        print()

    def print_remaining_weak_queue(self) -> None:
        print("\n=== Remaining weak-key queue ===")
        if not self.tune.weak_key_queue:
            print("Queue is empty.\n")
            return
        for pos, key_idx in enumerate(self.tune.weak_key_queue):
            marker = "->" if pos == self.tune.weak_key_queue_pos else "  "
            status = self.key_status_from_report(key_idx)
            print(f"{marker} [{pos+1}/{len(self.tune.weak_key_queue)}] key {key_idx}  status={status}")
        print()

    def _save_tune_session_if_enabled(self) -> None:
        if self.cfg.auto_save_tune_session:
            self.tune.save(self.stats.build_white_key_report_rows())

    # ----------------------------
    # UI / keyboard loop
    # ----------------------------
    def print_controls(self) -> None:
        print("\nControls:")
        print("  TAB -> cycle mode")
        print("  Type digits (0..21) to select key")
        print("  BACKSPACE -> edit selected key")
        print("  F / G -> request / clear safe stop")
        print("  q / ESC -> quit")
        print(f"  Current mode: {self.mode.value}")

        if self.mode == OperatorMode.PLAY:
            print("\n[PLAY]")
            print("  ENTER -> play selected key")
            print(f"  m -> play default song ({self.cfg.default_song_name})")
            print(f"  * -> start trigger (press after arm reaches hover position)")
            print("  - / = -> softer / harder press profile")
            print("  ; -> print current press profile")
            print("  k -> toggle auto press profile on/off")
            print("  [ / ] -> previous / next song")
            print("  p -> print song library")

        elif self.mode == OperatorMode.CALIBRATE:
            print("\n[CALIBRATE]")
            print("  ENTER -> play selected key")
            print("  a / d -> selected key target X -/+ 1 mm")
            print("  w / x -> selected key target Y + / - 1 mm")
            print("  z -> reset selected key target offset")
            print("  e -> print selected key target offset")
            print("  v -> save all per-key target offsets to JSON")
            print("  b -> load per-key target offsets from JSON")
            print("  n -> reset ALL per-key target offsets")
            print("  t -> print selected key learned suggestion")
            print("  y -> apply selected key learned suggestion")
            print("  u -> clear selected key learned suggestion")
            print("  g -> run white-key sweep test")
            print("  h -> print per-key performance stats")
            print("  i -> reset per-key performance stats")
            print("  l -> print ranked white-key report")
            print("  / -> run servo 5 motion test on selected key")
            print("  j -> select worst tested white key")
            print("  . -> run repeated test on selected key")
            print("  [ / ] -> previous / next weak key")
            print("  s / p -> save / load tune session state")
            print("  f -> print tuning to-do list")
            print("  c -> print automatic tuning checklist")

        elif self.mode == OperatorMode.EVALUATE:
            print("\n[EVALUATE]")
            print("  r -> run soft/medium/hard validation on selected key")
            print("  o -> summarize latest validation run")
            print("  A -> run repeat timing test on selected key")
            print("  B -> print timing error stats")
            print("  C -> print timing session summary")
            print("  D -> print session evaluation summary")
            print("  E -> save session evaluation summary")

        print()

    def _handle_offset_nudge(self, key_code: int) -> None:
        idx_sel = self.get_selected_idx()
        if idx_sel is None:
            print("Type a key index first.")
            return

        delta = self.cfg.key_offset_step_m
        if key_code == ord('a'):
            dx, dy = self.offsets.nudge(idx_sel, ddx=-delta)
        elif key_code == ord('d'):
            dx, dy = self.offsets.nudge(idx_sel, ddx=delta)
        elif key_code == ord('w'):
            dx, dy = self.offsets.nudge(idx_sel, ddy=delta)
        else:
            dx, dy = self.offsets.nudge(idx_sel, ddy=-delta)

        x_tgt, y_tgt = self.strike_target_world_xy(idx_sel)
        print(
            f"key {idx_sel} target offset updated: "
            f"dx={dx*1000:.1f} mm, dy={dy*1000:.1f} mm   "
            f"strike_tgt=({x_tgt:.4f}, {y_tgt:.4f})"
        )

    def handle_keypress(self, key_code: int) -> bool:
        if key_code in (ord('q'), 27):
            return False

        if key_code == 9:
            self.cycle_mode()
            return True

        if key_code == ord('F'):
            self.request_safe_stop("manual operator stop")
            return True

        if key_code == ord('G'):
            self.clear_safe_stop()
            return True

        if ord('0') <= key_code <= ord('9'):
            self.tune.typed += chr(key_code)
            return True

        if key_code == 8:
            self.tune.typed = self.tune.typed[:-1]
            return True

        handled = False
        if self.mode == OperatorMode.PLAY:
            handled = self.handle_play_mode_keypress(key_code)
        elif self.mode == OperatorMode.CALIBRATE:
            handled = self.handle_calibrate_mode_keypress(key_code)
        elif self.mode == OperatorMode.EVALUATE:
            handled = self.handle_evaluate_mode_keypress(key_code)

        if not handled:
            hint = self.get_command_mode_hint(key_code)
            if hint is not None:
                self.flash_ui_message(hint)

        return True

    def run(self) -> None:
        self.connect()
        self.print_controls()

        assert self.cap is not None and self.tracker is not None and self.blob is not None and self.key_model is not None
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                disp = frame.copy()
                ok = self.tracker.update(frame)

                if ok:
                    self.last_tracker_ok = True
                    xw_ref, yw_ref = self.key_model.key_world_xy(10)
                    self.tracker.z_lock_keyplane(xw_ref, yw_ref, z_des_base=self.cfg.z_des_base)
                    self.tracker.draw_markers(disp)

                    ref_idx = self.update_auto_press_base_key_idx(verbose=False)
                    x_r, y_r = self.key_model.key_world_xy(ref_idx)
                    u_r, v_r = self.tracker.world_to_pixel(x_r, y_r)
                    cv2.circle(disp, (u_r, v_r), 8, (0, 255, 255), 2)
                    cv2.putText(disp, f"BASE-REF {ref_idx}", (u_r + 8, v_r + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    blob_seen = False
                    uv_show, _ = self.blob.detect(frame)
                    if uv_show is not None:
                        blob_seen = True
                        cv2.circle(disp, uv_show, 6, (0, 255, 0), -1)
                        cv2.putText(disp, "EE", (uv_show[0] + 8, uv_show[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    self.update_safety_flags_from_vision(tracker_ok=True, blob_seen=blob_seen)

                    for i, (xw, yw) in enumerate(self.key_model.all_keys_world_xy()):
                        u, v = self.tracker.world_to_pixel(xw, yw)
                        cv2.circle(disp, (u, v), 4, (0, 0, 255), -1)
                        if i % 2 == 0:
                            cv2.putText(disp, str(i), (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    idx_show = self.get_selected_idx()
                    if idx_show is not None:
                        x_k, y_k = self.strike_target_world_xy(idx_show)
                        u_k, v_k = self.tracker.world_to_pixel(x_k, y_k)
                        cv2.circle(disp, (u_k, v_k), 8, (255, 0, 0), 2)
                        cv2.putText(disp, "KEY", (u_k + 8, v_k - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    self.update_safety_flags_from_vision(tracker_ok=False, blob_seen=False)
                    cv2.putText(disp, "Need ID10, ID11, ID_BASE visible", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                status = (
                    f"mode={self.mode.value}  "
                    f"typed={self.tune.typed}  "
                    f"marker={'OK' if self.safety.marker_ok else 'LOST'}  "
                    f"blob={'OK' if self.safety.blob_ok else 'LOST'}  "
                    f"serial={'OK' if self.safety.serial_ok else 'FAIL'}  "
                    f"stop={'YES' if self.safety.stop_requested else 'NO'}"
                )

                help_lines = self.get_mode_help_lines()
                self.draw_text_panel(disp, help_lines, 20, 20)

                panel_lines = self.get_selected_key_panel_lines()
                panel_x = max(20, self.cfg.frame_w - 260)
                self.draw_text_panel(disp, panel_lines, panel_x, 20)

                self.draw_ui_message(disp)

                if self.safety.last_stop_reason:
                    cv2.putText(disp, f"reason={self.safety.last_stop_reason}", (20, self.cfg.frame_h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                cv2.putText(disp, status, (20, self.cfg.frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if self.safety.stop_requested:
                    cv2.putText(disp, f"SAFE STOP: {self.safety.last_stop_reason}", (20, self.cfg.frame_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Piano-Playing Robot", disp)
                key_code = cv2.waitKey(1) & 0xFF
                if not self.handle_keypress(key_code):
                    break

        finally:
            self.disconnect()