"""
config.py
=========
All configuration parameters for the piano-playing robot system.

Edit AppConfig to match your hardware setup before running:
  - com_port: serial port for the Arduino (e.g. "COM8" on Windows,
              "/dev/ttyUSB0" on Linux)
  - cam_index: camera index (1 for DroidCam over USB, 0 for built-in webcam)
  - mr_to_base_xyz: measured offset from base ArUco marker to actual robot base

The PathsConfig dataclass manages all file paths automatically and
should not need editing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

@dataclass
class PathsConfig:
    """Manages all file paths relative to the project directory."""

    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)

    # Auto-generated paths (do not edit directly)
    key_offset_json: Path = field(init=False)
    press_log_csv: Path = field(init=False)
    ina219_raw_log_txt: Path = field(init=False)
    session_summary_csv: Path = field(init=False)
    key_session_csv: Path = field(init=False)
    tune_session_json: Path = field(init=False)
    songs_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.key_offset_json     = self.base_dir / "key_target_offsets.json"
        self.press_log_csv       = self.base_dir / "press_validation_log.csv"
        self.ina219_raw_log_txt  = self.base_dir / "ina219_raw_log.txt"
        self.session_summary_csv = self.base_dir / "session_summary_log.csv"
        self.key_session_csv     = self.base_dir / "key_session_log.csv"
        self.tune_session_json   = self.base_dir / "tune_session_state.json"
        self.songs_dir           = self.base_dir / "songs"
        self.songs_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Main application configuration
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    """
    Central configuration for the entire piano robot system.

    Hardware settings
    -----------------
    com_port      : serial port string for the Arduino (platform-specific)
    cam_index     : OpenCV camera index (1 = DroidCam, 0 = built-in)
    frame_w/h     : camera resolution in pixels
    use_ina219_logger : enable electrical press monitoring via INA219

    Marker and workspace geometry
    -----------------------------
    id10, id11, id_base : ArUco marker IDs for left keyboard, right keyboard, base
    marker_size_m       : printed physical side length of each marker (metres)
    baseline_m          : measured centre-to-centre distance ID10 → ID11 (metres)
    mr_to_base_xyz      : (dx, dy, dz) offset from base marker centre to robot base origin
    mr_to_base_yaw_deg  : yaw angle between marker frame and robot base frame

    Key model
    ---------
    x_id10_to_f1  : world-frame X distance from ID10 to the first key (metres)
    key_pitch     : key-to-key spacing (metres)
    n_keys        : number of playable keys modelled
    y_key         : world-frame Y position of the key row
    strike_dy_w   : Y offset applied to all nominal key centres for striking

    Alignment (unified controller)
    --------------------------------
    align_tol_m         : spatial convergence tolerance (metres)
    align_settle_frames : frames inside tolerance before convergence declared
    align_timeout_s     : hard safety timeout (fault fallback only)
    pid_kp_x/y          : proportional gain for the visual alignment controller
    """

    # ---- Hardware ----
    com_port: str = "COM8"       # Windows: "COM8" | Linux: "/dev/ttyUSB0"
    cam_index: int = 1           # 1 = DroidCam (external), 0 = built-in webcam
    frame_w: int = 1280
    frame_h: int = 720
    use_ina219_logger: bool = True

    # ---- ArUco marker IDs ----
    id10: int = 10               # left keyboard marker (world origin)
    id11: int = 11               # right keyboard marker (defines +X axis)
    id_base: int = 20            # base marker (near robot)

    marker_size_m: float = 0.040  # 40 mm physical side length
    baseline_m: float = 0.415     # measured ID10-to-ID11 distance in metres

    # Measured offset from base marker centre to robot base origin (metres)
    mr_to_base_xyz: tuple[float, float, float] = (-0.057, 0.070, 0.023)
    mr_to_base_yaw_deg: float = 0.0

    # ---- Key geometry ----
    x_id10_to_f1: float = 0.036   # X distance from ID10 to first key (metres)
    key_pitch: float = 0.0162      # key spacing (metres)
    n_keys: int = 22
    y_key: float = -0.01           # Y row position of keys
    strike_dx_w: float = 0.0       # global X strike offset (metres)
    strike_dy_w: float = -0.008    # global Y strike offset — press toward near edge

    # ---- Z heights and basic motion ----
    z_des_base: float = -0.02      # desired Z of key plane in base frame
    dt: float = 0.02               # default servo step interval (s)
    align_z_above_key: float = 0.0060    # hover height during alignment (metres)
    press_dz: float = 0.017              # downward press distance for simple_tap
    min_press_hold_s: float = 0.5        # minimum hold duration (manual mode)
    song_min_press_hold_s: float = 0.12  # minimum hold duration (playback mode)
    song_force_light_align: bool = True  # use quick alignment during songs
    song_hold_ratio_default: float = 0.25

    # ---- Safe travel parameters for large key jumps ----
    safe_travel_z_above_key: float = 0.030   # lift height for far key moves
    far_key_lift_threshold_m: float = 0.035  # distance above which lift-and-travel is used

    far_high_travel_steps: int = 12
    far_high_travel_dt: float = 0.02
    far_lift_steps: int = 8
    far_lift_dt: float = 0.02
    far_descend_steps: int = 8
    far_descend_dt: float = 0.02

    # ---- Alignment / convergence (unified controller) ----
    align_timeout_s: float = 4.0          # safety timeout — fault fallback only
    align_tol_m: float = 0.0035           # spatial convergence tolerance
    align_settle_frames: int = 2          # frames inside tol before "done"
    align_start_settle_s: float = 0.08    # brief settle before first alignment iteration

    # ---- Neighbour-key seed reuse ----
    neighbor_key_jump: int = 2
    reuse_fine_tune_key_jump: int = 1
    reuse_max_seed_offset_m: float = 0.004
    key_offset_step_m: float = 0.001     # nudge step size for manual calibration

    # ---- Learned offset / calibration thresholds ----
    auto_cal_min_good_err_m: float = 0.0025
    auto_cal_max_shift_per_apply_m: float = 0.006
    auto_cal_min_samples: int = 3

    # ---- White-key sweep test ----
    white_key_test_sequence: list[int] = field(default_factory=lambda: list(range(4, 18)))
    white_key_test_hold_ratio: float = 0.55
    white_key_test_bpm: int = 90
    ranked_report_top_n: int = 5

    good_success_pct: float = 95.0
    warn_success_pct: float = 80.0
    good_mean_best_mm: float = 2.5
    warn_mean_best_mm: float = 4.5

    # ---- Servo 5 test ----
    servo5_test_base_deg: float = 65.0
    servo5_test_offsets_deg: list[int] = field(default_factory=lambda: [-8, -6, -4, -2, 0, 2, 4, 6, 8])
    servo5_test_move_steps: int = 10
    servo5_test_move_dt: float = 0.015
    servo5_test_settle_s: float = 0.10
    servo5_test_use_pid_align: bool = True

    # ---- Hybrid press profiles ----
    # soft / medium / hard are used for calibration and validation.
    # song_fast is a playback-optimised variant of the hybrid press that
    # reduces hold and release overhead to fit within short note gaps.
    use_hybrid_press: bool = True
    servo5_neutral_deg: float = 60.0
    press_profile_order: list[str] = field(default_factory=lambda: ["soft", "medium", "hard"])
    default_press_profile: str = "medium"
    hybrid_press_profiles: dict[str, dict[str, float | int]] = field(
        default_factory=lambda: {
            "soft": {
                "preload_dz": 0.0070,
                "preload_steps": 4,
                "preload_hold_s": 0.01,
                "servo5_delta_deg": 4.0,
                "servo5_steps": 4,
                "servo5_hold_s": 0.06,
                "final_extra_dz": 0.022,
                "final_extra_steps": 4,
                "final_extra_hold_s": 0.05,
                "up_steps": 12,
                "servo5_release_steps": 10,
                "release_pause_s": 0.05,
            },
            "medium": {
                "preload_dz": 0.0070,
                "preload_steps": 4,
                "preload_hold_s": 0.01,
                "servo5_delta_deg": 7.5,
                "servo5_steps": 4,
                "servo5_hold_s": 0.03,
                "final_extra_dz": 0.024,
                "final_extra_steps": 4,
                "final_extra_hold_s": 0.05,
                "up_steps": 12,
                "servo5_release_steps": 10,
                "release_pause_s": 0.05,
            },
            "hard": {
                "preload_dz": 0.0070,
                "preload_steps": 4,
                "preload_hold_s": 0.01,
                "servo5_delta_deg": 9.0,
                "servo5_steps": 4,
                "servo5_hold_s": 0.03,
                "final_extra_dz": 0.029,
                "final_extra_steps": 4,
                "final_extra_hold_s": 0.06,
                "up_steps": 12,
                "servo5_release_steps": 10,
                "release_pause_s": 0.05,
            },
            "song_fast": {
                # Playback-optimised hybrid press.  Reduced servo hold and
                # release times allow the arm to recover faster between notes.
                "preload_dz": 0.0070,
                "preload_steps": 4,
                "preload_hold_s": 0.01,
                "servo5_delta_deg": 8.0,
                "servo5_steps": 4,
                "servo5_hold_s": 0.015,
                "final_extra_dz": 0.025,   # 25 mm gives ~4 mm margin over 21 mm baseline
                "final_extra_steps": 3,
                "final_extra_hold_s": 0.03,
                "up_steps": 6,
                "servo5_release_steps": 4,
                "release_pause_s": 0.015,
            },
        }
    )

    # ---- Automatic press profile selection ----
    use_auto_press_profile: bool = True
    auto_press_verbose: bool = True
    auto_press_base_key_fallback_idx: int = 10
    auto_press_soft_x_dist_m: float = 0.015    # keys within 15 mm of base -> soft
    auto_press_medium_x_dist_m: float = 0.055  # keys within 55 mm -> medium; beyond -> hard

    # ---- Validation run ----
    validation_reps_per_profile: int = 5
    validation_bpm: int = 70
    validation_hold_ratio: float = 0.55
    validation_force_full_align: bool = True

    # ---- Motion context / align Z per motion class ----
    phrase_start_align_z: float = 0.010
    step_align_z: float = 0.006
    near_align_z: float = 0.007
    far_align_z: float = 0.009
    repeat_align_z: float = 0.005
    light_align_timeout_s: float = 0.45
    light_align_settle_frames: int = 1
    light_align_good_err_m: float = 0.0020
    light_align_max_key_jump: int = 1
    light_align_min_streak: int = 3

    # ---- Safety ----
    marker_loss_grace_s: float = 0.40    # tolerate brief marker dropout
    blob_loss_grace_s: float = 0.25      # tolerate brief blob dropout
    auto_stop_on_serial_error: bool = True
    enable_end_reset_pose: bool = True
    neutral_servo5_deg: float = 60.0
    neutral_move_steps: int = 14
    neutral_move_dt: float = 0.02

    # ---- Travel speed profile ----
    travel_near_dist_m: float = 0.020
    travel_far_dist_m: float = 0.080
    travel_dt_min: float = 0.018
    travel_dt_max: float = 0.040
    travel_steps_min: int = 10
    travel_steps_max: int = 30

    # ---- PID step size profile (used by align_to_key_center_pid) ----
    pid_err_large_m: float = 0.015
    pid_err_medium_m: float = 0.008
    pid_step_large_m: float = 0.0032
    pid_step_medium_m: float = 0.0020
    pid_step_small_m: float = 0.0009
    pid_sleep_large_s: float = 0.008
    pid_sleep_medium_s: float = 0.012
    pid_sleep_small_s: float = 0.018

    # ---- Affine X correction (compensates IK X-axis bias) ----
    ik_x_scale_default: float = 1.18993352
    ik_x_bias_m_default: float = -0.036068

    # ---- Visual alignment proportional gains ----
    pid_kp_x: float = 0.28
    pid_ki_x: float = 0.00
    pid_kd_x: float = 0.00
    pid_kp_y: float = 0.24
    pid_ki_y: float = 0.00
    pid_kd_y: float = 0.00
    pid_int_lim_x: float = 0.010
    pid_int_lim_y: float = 0.010
    pid_d_alpha: float = 0.7
    pid_max_total_corr_m: float = 0.050   # global correction bound
    pid_max_step_m: float = 0.0022        # per-step correction limit

    pid_move_steps: int = 3

    # ---- Per-key calibration / tune session ----
    focus_key_test_reps: int = 3
    focus_key_test_bpm: int = 85
    focus_key_test_hold_ratio: float = 0.55
    focus_auto_advance_on_pass: bool = True
    focus_include_ok_keys_in_queue: bool = False
    auto_save_tune_session: bool = True
    todo_list_top_n: int = 10
    checklist_nudge_mm: float = 1.0
    checklist_min_suggestion_mm: float = 0.8
    checklist_max_manual_offset_mm: float = 8.0
    checklist_print_all_keys: bool = False

    # ---- Song playback ----
    song_use_full_duration_hold: bool = False
    default_song_name: str = "timing_test"
    song_start_lead_in_s: float = 0.20

    # When True, the arm prepares (aligns to first key) then pauses and waits
    # for the operator to press song_trigger_key before the song clock starts.
    # This mirrors a pianist positioning their hand before playing.
    song_wait_for_trigger: bool = True
    song_trigger_key: int = ord('*')     # press '*' in the camera window to start

    playback_latency_margin_s: float = 0.000
    default_song_hold_ratio: float = 0.60

    song_force_profile_name: str | None = "song_fast"
    song_quick_align_timeout_s: float = 0.18
    song_quick_align_tol_m: float = 0.006
    song_quick_align_settle_frames: int = 1
    song_quick_align_max_iters: int = 3
    song_quick_detect_n: int = 3
    song_quick_detect_flush: int = 1
    song_quick_detect_wait_s: float = 0.003

    # Per-motion-class acceptance thresholds for playback (soft limit)
    song_quick_accept_repeat_m: float = 0.010
    song_quick_accept_step_m: float = 0.012
    song_quick_accept_near_m: float = 0.020
    song_quick_accept_far_m: float = 0.024

    # Micro-retry parameters for step/near transitions
    song_micro_retry_timeout_s: float = 0.10
    song_micro_retry_max_iters: int = 2
    song_micro_retry_trigger_m: float = 0.018
    song_micro_retry_accept_step_m: float = 0.0085
    song_micro_retry_accept_near_m: float = 0.0075

    # Seed reuse limits per motion class
    song_reuse_seed_repeat_m: float = 0.0070
    song_reuse_seed_step_m: float = 0.0055
    song_reuse_seed_near_m: float = 0.0050
    song_reuse_seed_far_m: float = 0.0035

    # Contact hold cap — raised from 0.5 s to accommodate 2-beat notes at slow tempos
    song_playback_contact_cap_s: float = 2.0
    song_release_guard_s: float = 0.03
    song_service_guard_s: float = 0.20

    # Safety guard added to the release tail in compute_playback_hold_s.
    # Reduced from 0.20 → 0.06 s: the old value caused 200 ms of idle time
    # after each release.  Increase this if you see late launches.
    playback_release_guard_s: float = 0.06

    # Minimum plausible launch-to-contact time for timing model updates.
    # Values below this are considered noise and excluded.
    min_launch_to_contact_ms: float = 200.0

    # If repeat_reuse contact takes longer than this, the aligned pose is
    # invalidated so the next repeat performs fresh alignment.
    late_contact_invalidate_ms: float = 400.0

    # Hard skip limits per motion class — if best_err exceeds this, the note
    # is skipped rather than forced (prevents clearly wrong key presses)
    song_hard_skip_repeat_m: float = 0.014
    song_hard_skip_step_m: float = 0.025
    song_hard_skip_near_m: float = 0.028
    song_hard_skip_far_m: float = 0.032

    # Playback release lead times (used by estimate_next_note_lead_s)
    playback_release_lead_repeat_s: float = 0.45
    playback_release_lead_step_s: float = 0.65
    playback_release_lead_near_s: float = 0.75
    playback_release_lead_far_s: float = 0.85

    paths: PathsConfig = field(default_factory=PathsConfig)
