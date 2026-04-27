"""
tracgenaruco.py
===============
Robot arm motion controller for the piano-playing system.

Provides PianoArmController — a class that wraps the 3-DOF Adeept robotic arm,
its serial link to the Arduino, inverse kinematics, trajectory generation, and
the hybrid key-press actuation routine.

Key methods called by the higher-level application (app.py):
  arm.piano_xy_to_base_xyz(x_p, y_p, z_p)   -- coordinate frame conversion
  arm.goto_cartesian(p_B, steps, dt)          -- smooth Cartesian motion
  arm.current_cartesian_position()            -- FK-based EE position readout
  arm.hybrid_tap_current(...)                 -- staged key press + release
  arm.goto_neutral_pose(...)                  -- move to safe resting position

Hardware context
----------------
The arm is modelled as a 3-DOF serial manipulator using an Elementary
Transform Sequence (ETS) representation.  IK is solved numerically via
Levenberg-Marquardt with random restarts for robustness.

This module is import-safe: importing it does not move the robot.
Connection only occurs when arm.connect() is called explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Iterable, List

import numpy as np
import time
import serial
import roboticstoolbox as rtb
from spatialmath import SE3


@dataclass
class ServoMap:
    """Maps joint radians -> servo degrees."""
    offset_deg: np.ndarray  # shape (3,)
    sign: np.ndarray        # shape (3,)
    min_deg: np.ndarray     # shape (3,)
    max_deg: np.ndarray     # shape (3,)

    def qrad_to_servo_deg(self, q_rad: np.ndarray) -> np.ndarray:
        q = np.asarray(q_rad, dtype=float).reshape(3,)
        deg = self.offset_deg + self.sign * np.degrees(q)
        return np.clip(deg, self.min_deg, self.max_deg)


class PianoArmController:
    """A small controller wrapper around your 3-DOF ETS arm + Arduino streaming."""

    def __init__(
        self,
        port: Optional[str] = None,
        baud: int = 115200,
        dt: float = 0.03,
        # Piano frame -> robot base frame translation (your previous SE3)
        T_B_P: SE3 = SE3(-0.178, 0.10,-0.015),
        # Key geometry (used if you want index-based targets)
        n_keys: int = 22,
        key_spacing_m: float = 0.0162,
        # Arm link lengths (your tuned values)
        L1: float = 0.0425,
        L2: float = 0.065,
        L3: float = 0.147,
        L4: float = 0.022,
        L5: float = 0.001,

        # Pressing motion
        hover_height_m: float = 0.02,
        press_depth_m: float = -0.02,
        # Servo mapping
        servo_map: Optional[ServoMap] = None,
        # Safety / behaviour
        ik_restarts: int = 10,
    ):
        self.port = port
        self.baud = int(baud)
        self.dt = float(dt)

        self.T_B_P = T_B_P

        self.n_keys = int(n_keys)
        self.key_spacing_m = float(key_spacing_m)

        self.hover_height_m = float(hover_height_m)
        self.press_depth_m = float(press_depth_m)

        self.ik_restarts = int(ik_restarts)
        
        # Build key model in piano frame: x along key index
        self.keys_P = np.array(
            [[i * self.key_spacing_m, 0.0, 0.0] for i in range(self.n_keys)], dtype=float
        )
        self.keys_B = np.array([self.P_to_B(p) for p in self.keys_P], dtype=float)

        # Build ETS model
        ets = (
            rtb.ET.Rz() *
            rtb.ET.tz(L1) *
            rtb.ET.Rx() *
            rtb.ET.tz(L2) *
            rtb.ET.Rx() *
            rtb.ET.tz(L3) *
            rtb.ET.tx(-L4) *
            rtb.ET.ty(L5)
        )
        self.robot = rtb.ERobot(ets, name="PianoRoboticArm")
        self.robot.qlim = np.array(
            [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]], dtype=float
        ).T  # shape (2,3)

        # Default servo mapping = your existing constants
        if servo_map is None:
            servo_map = ServoMap(
                offset_deg=np.array([84.0, 38.0, 180.0], dtype=float),
                sign=np.array([1.0, -1.0, 1.0], dtype=float),
                min_deg=np.array([0.0, 0.0, 0.0], dtype=float),
                max_deg=np.array([180.0, 180.0, 180.0], dtype=float),
            )
        self.servo_map = servo_map

        # Runtime state
        self._ser: Optional[serial.Serial] = None
        self.q_current = np.radians([0.0, 0.0, 180.0]).astype(float)  # your previous start
        self.servo5_current_deg = 60.0   # choose a safe neutral default
        self.serial_line_delay_s = 0.005

    # ---------------------------
    # Serial lifecycle
    # ---------------------------
    def connect(self):
        if self.port is None:
            raise ValueError("No serial port set. Pass port='COM8' (or similar) to PianoArmController.")
        if self._ser is not None and self._ser.is_open:
            return

        self._ser = serial.Serial(
            self.port,
            self.baud,
            timeout=1.0,
            write_timeout=2.0,
        )

        # Arduino Uno resets when serial opens
        t0 = time.time()
        ready = False

        while time.time() - t0 < 6.0:
            try:
                if self._ser.in_waiting:
                    line = self._ser.readline().decode("utf-8", errors="ignore").strip()
                    if line:
                        print(f"[ARM FW] {line}")
                        if line == "ARDUINO_READY":
                            ready = True
                            break
                else:
                    time.sleep(0.01)
            except Exception:
                time.sleep(0.01)

        # Safety wait in case READY was missed
        if not ready:
            time.sleep(1.5)

        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

    def disconnect(self):
        if self._ser is not None:
            try:
                self._ser.close()
            finally:
                self._ser = None

    def clear_serial_input(self):
        if getattr(self, "_ser", None) is not None:
            self._ser.reset_input_buffer()


    def send_text_command(self, cmd: str, clear_input: bool = False):
        """
        Send a newline-terminated text command over the same serial port
        used for servo control.
        """
        if getattr(self, "_ser", None) is None:
            raise RuntimeError("Serial port is not connected")

        if clear_input:
            self._ser.reset_input_buffer()

        msg = (cmd.strip() + "\n").encode("ascii", errors="ignore")
        self._ser.write(msg)
        self._ser.flush()


    def read_serial_lines_until(self, stop_prefix: str = None, timeout_s: float = 2.0, max_lines: int = 100):
        """
        Read lines from serial until:
        - timeout, or
        - a line starting with stop_prefix is seen and input buffer drains
        """
        if getattr(self, "_ser", None) is None:
            raise RuntimeError("Serial port is not connected")

        lines = []
        saw_stop = False
        t0 = time.time()

        while time.time() - t0 < timeout_s:
            while self._ser.in_waiting:
                raw = self._ser.readline()
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                lines.append(line)

                if stop_prefix is not None and line.startswith(stop_prefix):
                    saw_stop = True

                if len(lines) >= max_lines:
                    return lines

            if saw_stop and self._ser.in_waiting == 0:
                break

            time.sleep(0.01)

        return lines


    def ina_start_log(self):
        """
        Start INA219 logging on the same Arduino/serial link.
        """
        self.send_text_command("INA_START", clear_input=True)
        time.sleep(0.02)


    def ina_end_log(self, timeout_s: float = 2.0):
        """
        Stop INA219 logging and collect tagged output lines.
        """
        self.send_text_command("INA_STOP", clear_input=False)
        return self.read_serial_lines_until(
            stop_prefix="INA219_SUMMARY",
            timeout_s=timeout_s,
            max_lines=32,
        )

    # ---------------------------
    # Coordinate transforms
    # ---------------------------
    def P_to_B(self, p_P: np.ndarray) -> np.ndarray:
        """Piano frame point -> robot base frame point (3D)."""
        p_P = np.asarray(p_P, dtype=float).reshape(3,)
        p_P_h = np.r_[p_P, 1.0]
        p_B_h = self.T_B_P.A @ p_P_h
        return p_B_h[:3]

    def piano_xy_to_base_xyz(self, x_p: float, y_p: float, z_p: float = 0.0) -> np.ndarray:
        """Convenience wrapper: piano plane (x,y) in meters -> base xyz."""
        return self.P_to_B(np.array([x_p, y_p, z_p], dtype=float))

    # ---------------------------
    # IK + trajectory
    # ---------------------------
    def solve_ik_pos(self, p_B: np.ndarray, q_seed: Optional[np.ndarray] = None) -> np.ndarray:
        """Solve IK for a position-only target in base frame."""
        p_B = np.asarray(p_B, dtype=float).reshape(3,)
        if q_seed is None:
            q_seed = self.q_current
        q_seed = np.asarray(q_seed, dtype=float).reshape(3,)

        T = SE3(p_B[0], p_B[1], p_B[2])

        qmin = self.robot.qlim[0, :]
        qmax = self.robot.qlim[1, :]

        seeds: List[np.ndarray] = [q_seed]
        for _ in range(self.ik_restarts):
            seeds.append(qmin + (qmax - qmin) * np.random.rand(3))

        best_q = None
        best_err = np.inf

        for qs in seeds:
            sol = self.robot.ikine_LM(
                T,
                q0=qs,
                mask=[1, 1, 1, 0, 0, 0],
                joint_limits=True,
            )
            if sol.success:
                return sol.q

            if hasattr(sol, "residual") and sol.residual < best_err:
                best_err = sol.residual
                best_q = sol.q

        raise RuntimeError(f"IK failed for target {p_B}. best_residual={best_err}, best_q={best_q}")

    def joint_trajectory(self, q_start: np.ndarray, q_goal: np.ndarray, steps: int = 60) -> np.ndarray:
        q_start = np.asarray(q_start, dtype=float).reshape(3,)
        q_goal = np.asarray(q_goal, dtype=float).reshape(3,)
        traj = rtb.jtraj(q_start, q_goal, int(steps))
        return traj.q

    # ---------------------------
    # Servo streaming
    # ---------------------------
    def _write_servo_line(self, servo_deg: np.ndarray):
        if self._ser is None or not self._ser.is_open:
            raise RuntimeError("Serial not connected. Call arm.connect() first.")

        a = np.asarray(servo_deg, dtype=float).ravel()
        line = ",".join(f"{v:.1f}" for v in a) + "\n"
        payload = line.encode("ascii")

        for attempt in range(3):
            try:
                self._ser.write(payload)
                return
            except serial.SerialTimeoutException:
                if attempt < 2:
                    time.sleep(0.05)
                    try:
                        self._ser.reset_output_buffer()
                    except Exception:
                        pass
                else:
                    raise
        
    def send_q_path(self, q_path: np.ndarray, dt: Optional[float] = None):
        if dt is None:
            dt = self.dt

        q_path = np.asarray(q_path, dtype=float)
        if q_path.ndim != 2 or q_path.shape[1] != 3:
            raise ValueError("q_path must have shape (N, 3)")

        # serial_line_delay_s (5 ms) is a hardware minimum; never sleep less
        # than this even if dt is smaller, to avoid overflowing the serial buffer.
        sleep_s = max(float(dt), self.serial_line_delay_s)

        for q in q_path:
            servo = self.servo_map.qrad_to_servo_deg(q)
            self._write_servo_line(servo)
            time.sleep(sleep_s)

        self.q_current = np.asarray(q_path[-1], dtype=float).reshape(3,)
        
    # ---------------------------
    # High-level motion helpers
    # ---------------------------

    def goto_neutral_pose(
        self,
        q_neutral_deg: Iterable[float] = (0.0, 0.0, 180.0),
        servo5_deg: float = 45.0,
        steps: int = 14,
        dt: Optional[float] = None,
    ):
        """
        Move arm joints to a known neutral pose, then set servo 5 to a safe neutral angle.
        q_neutral_deg is in arm joint degrees before servo mapping.
        """
        if dt is None:
            dt = self.dt

        q_neutral = np.radians(np.asarray(list(q_neutral_deg), dtype=float).reshape(3,))
        q_path = self.linear_joint_path(self.q_current, q_neutral, steps=steps)
        self.send_q_path(q_path, dt=dt)
        self.set_servo5_smooth(float(servo5_deg), steps=max(8, steps), dt=dt)
        
    def key_targets(self, p_key_B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return hover and press targets in base frame."""
        p_key_B = np.asarray(p_key_B, dtype=float).reshape(3,)
        hover = p_key_B + np.array([0.0, 0.0, self.hover_height_m])
        press = p_key_B + np.array([0.0, 0.0, self.press_depth_m])
        return hover, press

    def goto_cartesian(self, p_B: np.ndarray, steps: int = 60, dt: Optional[float] = None):
        """Move end-effector to a Cartesian *position* (base frame) with IK + jtraj."""
        q_goal = self.solve_ik_pos(p_B, q_seed=self.q_current)
        q_path = self.joint_trajectory(self.q_current, q_goal, steps=steps)
        self.send_q_path(q_path, dt=dt)

    # Legacy helper — used for testing.  Prefer goto_cartesian() in the main application.
    def goto_key_index(self, key_idx: int, steps: int = 60, dt: Optional[float] = None):
        p_key = self.keys_B[int(key_idx)]
        hover, _ = self.key_targets(p_key)
        # old: self.goto_cartesian(hover, steps=steps, dt=dt)
        self.goto_cartesian_via_safe_z(
            hover,
            travel_clearance_m=0.03,
            lift_steps=40,
            move_steps=steps,
            lower_steps=40,
            dt=dt
        )

    def goto_cartesian_via_safe_z(
        self,
        p_B: np.ndarray,
        travel_clearance_m: float = 0.03,
        lift_steps: int = 30,
        move_steps: int = 60,
        lower_steps: int = 30,
        dt: Optional[float] = None,
        min_clearance_m: float = 0.005,   # never try less than 5mm
        clearance_step_m: float = 0.005,  # reduce by 5mm if IK fails
    ):
        """
        Robust safe-Z move:
        - tries to lift to z_safe = max(z_now, z_target) + clearance
        - if IK fails, reduces clearance step-by-step
        - if still fails, falls back to direct goto_cartesian(p_B)
        """
        if dt is None:
            dt = self.dt

        p_B = np.asarray(p_B, dtype=float).reshape(3,)

        # current Cartesian from FK
        T_fk = self.robot.fkine(self.q_current)
        p_now = np.array(T_fk.t).reshape(3,)

        z_base = float(max(p_now[2], p_B[2]))
        clearance = float(travel_clearance_m)

        # helper to test IK feasibility without moving
        def can_solve(p_test):
            try:
                _ = self.solve_ik_pos(p_test, q_seed=self.q_current)
                return True
            except Exception:
                return False

        # Try decreasing clearance until all waypoints are solvable
        while clearance >= min_clearance_m - 1e-9:
            z_safe = z_base + clearance

            p_up  = np.array([p_now[0], p_now[1], z_safe], dtype=float)
            p_mid = np.array([p_B[0],   p_B[1],   z_safe], dtype=float)

            if can_solve(p_up) and can_solve(p_mid) and can_solve(p_B):
                # Execute the 3-leg path
                self.goto_cartesian(p_up,  steps=lift_steps,  dt=dt)
                self.goto_cartesian(p_mid, steps=move_steps,  dt=dt)
                self.goto_cartesian(p_B,   steps=lower_steps, dt=dt)
                return

            clearance -= clearance_step_m

        # If we get here, safe-Z path is not feasible; fallback
        # (Still might fail if p_B itself is unreachable, but that indicates a different issue.)
        self.goto_cartesian(p_B, steps=move_steps, dt=dt)

    def press_at(
        self,
        p_key_B: np.ndarray,
        hover_steps: int = 60,
        press_steps: int = 30,
        release_steps: int = 30,
        hold_s: float = 1.0, 
        dt: Optional[float] = None,
    ):
        """Hover -> press -> release at a base-frame key point."""
        if dt is None:
            dt = self.dt

        hover, press = self.key_targets(p_key_B)

        q_hover = self.solve_ik_pos(hover, q_seed=self.q_current)
        q_press = self.solve_ik_pos(press, q_seed=q_hover)

        traj1 = self.joint_trajectory(self.q_current, q_hover, steps=hover_steps)
        traj2 = self.joint_trajectory(q_hover, q_press, steps=press_steps)
        traj3 = self.joint_trajectory(q_press, q_hover, steps=release_steps)

        # Move to hover then press
        self.send_q_path(np.vstack([traj1, traj2]), dt=dt)

        # Hold at press pose (repeat q_press so servo maintains contact)
        n_hold = max(1, int(hold_s / dt))
        hold_path = np.repeat(q_press.reshape(1, 3), n_hold, axis=0)
        self.send_q_path(hold_path, dt=dt)

        # Release back to hover
        self.send_q_path(traj3, dt=dt)

    def current_cartesian_position(self) -> np.ndarray:
        """Return the current end-effector position in base frame via FK.

        Used by UnifiedVisualController._visual_step() to compute incremental
        world-frame XY corrections without re-solving full IK from scratch.
        """
        T_fk = self.robot.fkine(self.q_current)
        return np.array(T_fk.t, dtype=float).reshape(3,)

    def nudge_xy(self, dx: float, dy: float, steps: int = 25, dt: Optional[float] = None):
        """Small XY correction in base frame at current Z (Cartesian position-only IK).

        NOTE: With position-only IK, this assumes your tool stays approximately above the same plane.
        """
        # Estimate current EE position with FK
        T_fk = self.robot.fkine(self.q_current)
        p = np.array(T_fk.t).reshape(3,)
        p2 = p + np.array([dx, dy, 0.0])
        self.goto_cartesian(p2, steps=steps, dt=dt)

    def linear_joint_path(self, q_start: np.ndarray, q_goal: np.ndarray, steps: int = 10) -> np.ndarray:
        """Very simple linear interpolation in joint space (no jtraj)."""
        q_start = np.asarray(q_start, dtype=float).reshape(3,)
        q_goal  = np.asarray(q_goal, dtype=float).reshape(3,)
        return np.linspace(q_start, q_goal, int(steps))

    def set_servo5(self, deg: float):
        servo123 = self.servo_map.qrad_to_servo_deg(self.q_current)
        deg = float(deg)
        self._write_servo_line(np.r_[servo123, deg])
        self.servo5_current_deg = deg
        time.sleep(self.dt)

    def set_servo5_smooth(self, deg: float, steps: int = 10, dt: Optional[float] = None):
        if dt is None:
            dt = self.dt

        deg0 = float(self.servo5_current_deg)
        deg1 = float(deg)

        vals = np.linspace(deg0, deg1, int(steps))
        for v in vals:
            servo123 = self.servo_map.qrad_to_servo_deg(self.q_current)
            self._write_servo_line(np.r_[servo123, float(v)])
            self.servo5_current_deg = float(v)
            time.sleep(max(dt, self.serial_line_delay_s))

    def hybrid_tap_current(
        self,
        preload_dz: float = 0.006,
        preload_steps: int = 10,
        preload_hold_s: float = 0.06,

        servo5_delta_deg: float = 6.0,
        servo5_steps: int = 8,
        servo5_hold_s: float = 0.02,   # short pre-contact settle only

        final_extra_dz: float = 0.0,
        final_extra_steps: int = 4,
        final_extra_hold_s: float = 0.30,   # actual musical key-down hold

        up_steps: int = 10,
        servo5_release_steps: int = 8,
        release_pause_s: float = 0.02,

        dt: Optional[float] = None,
    ):
        """
        Hybrid press:
        1) arm gently preloads downward
        2) servo 5 adds local press
        3) optional tiny extra arm depth to reach full contact
        4) hold while actually pressing the key
        5) undo extra depth
        6) release servo 5
        7) arm returns up
        """
        if dt is None:
            dt = self.dt

        preload_dz = abs(float(preload_dz))
        final_extra_dz = abs(float(final_extra_dz))

        q0 = self.q_current.copy()
        servo5_0 = float(self.servo5_current_deg)
        p0 = np.array(self.robot.fkine(q0).t).reshape(3,)

        # Stage 1: preload with arm
        p_pre = p0 + np.array([0.0, 0.0, -preload_dz], dtype=float)
        q_pre = self.solve_ik_pos(p_pre, q_seed=q0)
        path_pre = self.linear_joint_path(q0, q_pre, steps=preload_steps)
        self.send_q_path(path_pre, dt=dt)
        time.sleep(preload_hold_s)

        # Stage 2: servo 5 assist
        self.set_servo5_smooth(
            servo5_0 + float(servo5_delta_deg),
            steps=servo5_steps,
            dt=dt
        )

        # Short settle only, not the full note hold
        time.sleep(servo5_hold_s)

        # Stage 3: optional tiny extra arm depth to reach true key-down contact
        q_contact = q_pre
        if final_extra_dz > 1e-6:
            p_final = p_pre + np.array([0.0, 0.0, -final_extra_dz], dtype=float)
            q_final = self.solve_ik_pos(p_final, q_seed=q_pre)
            path_extra_down = self.linear_joint_path(q_pre, q_final, steps=final_extra_steps)
            self.send_q_path(path_extra_down, dt=dt)
            q_contact = q_final

        # Stage 4: HOLD while actually pressing the key
        time.sleep(final_extra_hold_s)

        # Stage 5: undo extra arm depth first
        if final_extra_dz > 1e-6:
            path_extra_up = self.linear_joint_path(q_contact, q_pre, steps=final_extra_steps)
            self.send_q_path(path_extra_up, dt=dt)

        # Stage 6: release servo 5
        self.set_servo5_smooth(
            servo5_0,
            steps=servo5_release_steps,
            dt=dt
        )
        time.sleep(release_pause_s)

        # Stage 7: return arm up
        path_up = self.linear_joint_path(q_pre, q0, steps=up_steps)
        self.send_q_path(path_up, dt=dt)

    def simple_tap_current(
        self,
        dz: float = 0.010,          # 10 mm downward
        down_steps: int = 8,
        up_steps: int = 8,
        hold_s: float = 0.08,
        dt: Optional[float] = None,
    ):
        """
        SIMPLE press/tap at the *current pose*:
        go down in Z by dz, hold, then return.
        Assumes you're already above the key (aligned XY).
        """
        if dt is None:
            dt = self.dt

        dz = abs(float(dz))

        # Save where we are now (joint + cartesian)
        q0 = self.q_current.copy()
        p0 = np.array(self.robot.fkine(q0).t).reshape(3,)

        # Go down in Z only
        p_down = p0 + np.array([0.0, 0.0, -dz], dtype=float)

        # IK for down pose (seed with current)
        q_down = self.solve_ik_pos(p_down, q_seed=q0)

        # Down (simple linear joints)
        path_down = self.linear_joint_path(q0, q_down, steps=down_steps)
        self.send_q_path(path_down, dt=dt)

        time.sleep(hold_s)

        # Up (back to q0)
        path_up = self.linear_joint_path(q_down, q0, steps=up_steps)
        self.send_q_path(path_up, dt=dt)

if __name__ == "__main__":
    # Optional: simple smoke test that does NOT run unless you execute tracgen.py directly.
    arm = PianoArmController(port="COM8")
    print("Key 0 base xyz:", arm.keys_B[0])
    print("Key 10 base xyz:", arm.keys_B[10])
    print("Key 14 base xyz:", arm.keys_B[14])
    print("(Run your main closed-loop script for real operation.)")
