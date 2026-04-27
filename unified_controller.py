"""
unified_controller.py
=====================
Unified Sensor-Gated Visual Controller for Piano Key Alignment.

Background
----------
The green end-effector marker is NOT visible when the arm sits at its neutral
(rest) pose.  It only enters the camera's usable field of view once the arm
has moved partway toward the keyboard.  This means every note execution has a
guaranteed initial blind phase — not merely an occasional one.

Why unify the two old stages?
------------------------------
An earlier version of the system used two separate stages:

  Stage 1 — Open-loop IK trajectory
      Termination: fixed step count → TIME-BASED
      Problem: the arm has no spatial awareness of when it has arrived.
               It stops because the timer expired, not because it reached
               the target.  Overshoot or undershoot goes undetected.

  Stage 2 — Closed-loop PID refinement
      Termination: |e| < tolerance → SPACE-BASED
      Problem: Stage 2 inherits whatever error Stage 1 left behind.
               If Stage 1 lands poorly, Stage 2 has to do much more work,
               and the handoff is entirely time-based with no principled
               criterion for "Stage 1 is done".

The supervisor correctly identified this as incoherent: the two stages answer
"am I done?" differently, and the handoff between them has no principled basis.

This module replaces both stages with a single loop that has ONE termination
criterion throughout: spatial convergence.

How it works
------------
The unified loop has two internal phases, but these are NOT separate
controllers.  They are two branches of the same iteration, selected
automatically by whether the blob is currently visible:

  Phase A — Blind approach
      Condition   : blob not yet detected in the camera frame
      Action      : one IK-based Cartesian step toward the hover target
      Termination : Phase A ends the moment the blob FIRST appears.
                    This is a perceptual criterion, not a time criterion.
                    There is no step count, no timer.

  Phase B — Visual feedback
      Condition   : blob is visible
      Action      : gain-scheduled proportional correction
      Termination : |e| < tolerance for N_settle consecutive frames.
                    This is a spatial criterion.

Both phases terminate on a sensor or spatial observation — never on
elapsed time alone.  A global safety timeout exists only as a hardware
fault fallback, not as a normal termination path.

Gain scheduling in Phase B
--------------------------
Rather than a hard mode switch, the gain varies continuously with error:

    If |e| > thresh_far  :  coarse gain + large max_step
    If |e| < thresh_fine :  fine gain   + small max_step
    In between           :  linear interpolation

This makes the arm naturally decelerate as it approaches the key without
any explicit state machine.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Callable

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class UnifiedControllerConfig:
    """All tuning parameters for the unified controller in one place."""

    # ---- Gain schedule thresholds (metres) ----
    thresh_far: float = 0.020    # above this -> use coarse gain
    thresh_fine: float = 0.006   # below this -> use fine gain

    # ---- Proportional gains ----
    kp_far: float  = 0.60        # aggressive during coarse approach
    kp_fine: float = 0.28        # conservative near the key surface

    # ---- Per-step correction limits (metres) ----
    max_step_far: float  = 0.006   # maximum single correction step (coarse)
    max_step_fine: float = 0.0022  # maximum single correction step (fine)

    # ---- Total accumulated correction bound ----
    max_total_corr: float = 0.050  # keeps alignment local, not a global reposition

    # ---- Convergence criteria ----
    tol: float = 0.0035          # spatial tolerance (m) to declare convergence
    settle_frames: int = 2       # consecutive frames inside tol before "done"

    # ---- Timing ----
    dt: float = 0.020            # control loop interval (s)
    timeout_s: float = 4.0       # hard safety timeout — fault fallback only

    # ---- Blind-phase IK parameters ----
    blind_steps: int  = 8        # IK interpolation steps per blind move
    blind_step_dt: float = 0.018 # servo step interval during each blind move (s)

    # ---- Derivative filter (d-gain is 0 in current tuning, but plumbing exists) ----
    d_alpha: float = 0.7


# ---------------------------------------------------------------------------
# Result returned to the caller after run() completes
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    """Summary of one alignment run, returned by UnifiedVisualController.run()."""
    success: bool
    final_error_m: float
    best_error_m: float
    n_iterations: int
    elapsed_s: float
    n_blind_iters: int        # how many Phase A iterations occurred
    n_visual_iters: int       # how many Phase B iterations occurred
    termination_reason: str   # "converged" | "timeout" | "fault"

    @property
    def final_error_mm(self) -> float:
        return self.final_error_m * 1000.0

    @property
    def best_error_mm(self) -> float:
        return self.best_error_m * 1000.0


# ---------------------------------------------------------------------------
# Internal derivative/integral filter state
# ---------------------------------------------------------------------------

class _FilterState:
    """Tracks derivative and integral state across iterations."""

    def __init__(self, alpha: float):
        self._alpha = alpha
        self.d_filt = np.zeros(2, dtype=float)
        self.e_int  = np.zeros(2, dtype=float)
        self.e_prev: Optional[np.ndarray] = None

    def reset(self):
        """Clear all filter state — call this when the blob is lost."""
        self.d_filt[:] = 0.0
        self.e_int[:] = 0.0
        self.e_prev = None

    def update(self, e: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (filtered_derivative, integral) for the current error sample."""
        if self.e_prev is not None:
            d_raw = (e - self.e_prev) / dt
            self.d_filt = self._alpha * self.d_filt + (1 - self._alpha) * d_raw
        else:
            self.d_filt[:] = 0.0

        # Clamp integral to ±10 mm to prevent wind-up
        self.e_int = np.clip(self.e_int + e * dt, -0.010, 0.010)
        self.e_prev = e.copy()
        return self.d_filt.copy(), self.e_int.copy()


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------

class UnifiedVisualController:
    """
    Unified sensor-gated visual controller.

    Every note execution starts in Phase A (blind approach, blob not visible)
    and transitions automatically to Phase B (visual feedback) the moment the
    blob is detected.  There is no explicit handoff — the transition is simply
    the first iteration where blob_px is not None.

    Typical usage
    -------------
        ctrl = UnifiedVisualController(cfg, arm, tracker, blob_detector)
        result = ctrl.run(target_xy_world=(x, y), frame_source=lambda: cap.read()[1])
        if result.success:
            arm.hybrid_tap_current(...)
    """

    def __init__(self, cfg: UnifiedControllerConfig, arm, tracker, blob_detector):
        self._cfg  = cfg
        self._arm  = arm
        self._trk  = tracker
        self._blob = blob_detector
        self._filt = _FilterState(cfg.d_alpha)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        target_xy_world: Tuple[float, float],
        frame_source: Callable[[], np.ndarray],
    ) -> AlignmentResult:
        """
        Drive the end-effector to target_xy_world and return when converged.

        Parameters
        ----------
        target_xy_world : (x, y) target in world frame (metres)
        frame_source    : callable that returns the latest BGR camera frame
        """
        cfg = self._cfg
        self._filt.reset()

        target     = np.array(target_xy_world, dtype=float)
        total_corr = np.zeros(2, dtype=float)   # tracks accumulated correction
        best_err   = np.inf
        settle_n   = 0
        n_iter     = 0
        n_blind    = 0
        n_visual   = 0
        t_start    = time.perf_counter()

        while True:
            elapsed = time.perf_counter() - t_start

            # ---- Safety timeout: fault fallback, NOT the normal exit path ----
            if elapsed > cfg.timeout_s:
                log.warning(
                    "UnifiedController: safety timeout %.2f s  best_err=%.2f mm",
                    elapsed, best_err * 1e3,
                )
                return self._make_result(
                    False, best_err, best_err,
                    n_iter, elapsed, n_blind, n_visual, "timeout"
                )

            # ---- Read frame and update tracker / blob detector ----
            frame      = frame_source()
            tracker_ok = self._trk.update(frame)
            blob_px, _ = self._blob.detect(frame)

            # ==============================================================
            # PHASE A — Blind approach
            #
            # The blob is not yet visible (guaranteed at the start of every
            # note because the arm begins from a neutral pose outside the
            # camera's usable field of view).
            #
            # Action    : issue one IK-based Cartesian step toward the target.
            # Termination: Phase A ends when blob_px is not None — a
            #              perceptual criterion, not a step count or timer.
            # ==============================================================
            if blob_px is None or not tracker_ok:
                n_blind += 1
                self._filt.reset()          # discard stale derivative state
                self._blind_step(target_xy_world)
                n_iter += 1
                time.sleep(cfg.dt)
                continue

            # ==============================================================
            # PHASE B — Visual feedback
            #
            # The blob is visible.  Every iteration from here uses the
            # measured error to drive the correction command.
            #
            # Termination: |e| < tol for settle_frames consecutive frames.
            #              This is the ONLY normal termination criterion.
            # ==============================================================
            n_visual += 1

            # Convert blob pixel location to world-frame position
            ee_xy = np.array(
                self._trk.pixel_to_world(blob_px[0], blob_px[1]),
                dtype=float,
            )
            e       = target - ee_xy
            err_mag = float(np.linalg.norm(e))

            if err_mag < best_err:
                best_err = err_mag

            # ---- Check for spatial convergence ----
            if err_mag < cfg.tol:
                settle_n += 1
                if settle_n >= cfg.settle_frames:
                    elapsed = time.perf_counter() - t_start
                    log.info(
                        "UnifiedController converged: iter=%d t=%.1f ms "
                        "blind=%d visual=%d err=%.2f mm",
                        n_iter, elapsed * 1e3, n_blind, n_visual, err_mag * 1e3,
                    )
                    return self._make_result(
                        True, err_mag, best_err,
                        n_iter, elapsed, n_blind, n_visual, "converged"
                    )
            else:
                settle_n = 0

            # ---- Compute gain-scheduled proportional correction ----
            gain, max_step = self._gain_schedule(err_mag)

            # Update filter state (currently P-only; I and D gains are zero)
            self._filt.update(e, cfg.dt)
            u = gain * e
            u = _clip_to_magnitude(u, max_step)

            # Enforce the global correction bound to keep this a local refinement
            proposed = total_corr + u
            if np.linalg.norm(proposed) > cfg.max_total_corr:
                proposed = _clip_to_magnitude(proposed, cfg.max_total_corr)
                u = proposed - total_corr
            total_corr = proposed

            self._visual_step(u)
            n_iter += 1
            time.sleep(cfg.dt)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _gain_schedule(self, err_mag: float) -> Tuple[float, float]:
        """Return (gain, max_step) as a continuous function of error magnitude."""
        cfg = self._cfg
        if err_mag >= cfg.thresh_far:
            return cfg.kp_far, cfg.max_step_far
        elif err_mag <= cfg.thresh_fine:
            return cfg.kp_fine, cfg.max_step_fine
        # Linear interpolation between fine and far
        t = (err_mag - cfg.thresh_fine) / (cfg.thresh_far - cfg.thresh_fine)
        gain     = float(cfg.kp_fine  + t * (cfg.kp_far      - cfg.kp_fine))
        max_step = float(cfg.max_step_fine + t * (cfg.max_step_far - cfg.max_step_fine))
        return gain, max_step

    def _blind_step(self, target_xy: Tuple[float, float]):
        """Phase A: move one IK Cartesian step toward the hover pose above target."""
        try:
            x, y    = target_xy
            hover_z = getattr(self._arm, 'hover_height_m', 0.010)
            p_B     = self._arm.piano_xy_to_base_xyz(x, y, z_p=0.0)
            p_B[2] += hover_z
            self._arm.goto_cartesian(
                p_B,
                steps=self._cfg.blind_steps,
                dt=self._cfg.blind_step_dt,
            )
        except Exception as exc:
            log.warning("UnifiedController blind step failed: %s", exc)

    def _visual_step(self, delta_xy_world: np.ndarray):
        """Phase B: apply a small world-frame XY correction to the current pose."""
        try:
            current_B = self._arm.current_cartesian_position()
            # Rotate the world-frame correction into the base frame
            R       = self._trk.B_T_W[:3, :3]
            delta_B = R @ np.array([delta_xy_world[0], delta_xy_world[1], 0.0])
            self._arm.goto_cartesian(current_B + delta_B, steps=3, dt=self._cfg.dt)
        except Exception as exc:
            log.warning("UnifiedController visual step failed: %s", exc)

    @staticmethod
    def _make_result(
        success, final_err, best_err, n_iter, elapsed,
        n_blind, n_visual, reason,
    ) -> AlignmentResult:
        return AlignmentResult(
            success=success,
            final_error_m=final_err,
            best_error_m=best_err,
            n_iterations=n_iter,
            elapsed_s=elapsed,
            n_blind_iters=n_blind,
            n_visual_iters=n_visual,
            termination_reason=reason,
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _clip_to_magnitude(v: np.ndarray, vmax: float) -> np.ndarray:
    """Scale vector v so its magnitude does not exceed vmax."""
    mag = np.linalg.norm(v)
    if mag > vmax and mag > 1e-12:
        return v * (vmax / mag)
    return v.copy()
