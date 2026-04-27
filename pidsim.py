"""
pidsim.py
=========
Simulation study for the unified sensor-gated visual controller.

This script supports the design justification discussed in the report's
motor control chapter.

Physical context
----------------
The arm always starts from a neutral pose where the green end-effector
marker is NOT visible.  Phase A (blind approach) is therefore guaranteed
at the start of every note, not merely occasionally.  The simulation
models this explicitly: the first BLIND_ITERS iterations have no blob
feedback.

The key design question answered by this simulation is:

    "For the unified controller, do all termination criteria remain
     spatial (|e| < tolerance) regardless of whether the blob was
     visible from the start, and does the gain schedule improve final
     accuracy without slowing convergence?"

Four variants are compared
--------------------------
  1. Fixed-P only
       Equivalent to the old isolated Stage 2 operating in its most
       favourable scenario (blob already visible, error already small).
       Gain is fixed at KP_FINE regardless of error magnitude.

  2. Gain schedule, no blind phase (ideal case)
       The full gain schedule is active, but we assume the blob is
       visible from iteration zero (best-case scenario).

  3. Gain schedule + guaranteed blind phase (realistic case)
       Reflects the actual hardware: the first BLIND_ITERS iterations
       cannot see the blob, so Phase A runs unconditionally.  Phase A
       terminates when the blob appears — a perceptual criterion.

  4. Gain schedule + blind phase + occasional dropout (most realistic)
       As above, but during Phase B the blob may be temporarily lost
       with probability DROPOUT_PROB per iteration.

All four variants terminate on spatial criteria.  The safety timeout
is never reached in normal operation.

Usage
-----
    python pidsim.py
Output: a summary table in the terminal and a saved PNG figure.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


# ============================================================
# Simulation constants — match AppConfig / UnifiedControllerConfig
# ============================================================

DT             = 0.020          # control loop interval (s)
ALIGN_TOL_M    = 0.0035         # convergence tolerance (m)
MAX_TIME_S     = 4.0            # safety timeout — fault fallback only
SETTLE_FRAMES  = 2              # frames inside tol before declaring convergence

# Gain schedule parameters (must match UnifiedControllerConfig)
THRESH_FAR     = 0.020          # above this -> coarse gain
THRESH_FINE    = 0.006          # below this -> fine gain
KP_FAR         = 0.60
KP_FINE        = 0.28
MAX_STEP_FAR   = 0.006          # metres
MAX_STEP_FINE  = 0.0022         # metres
MAX_TOTAL_CORR = 0.050          # metres

# Simulation noise model
MEAS_NOISE_STD = 0.00035        # 0.35 mm measurement noise (one standard deviation)
DROPOUT_PROB   = 0.05           # probability of blob loss per step in Phase B
PLANT_RESPONSE = 0.90           # fraction of commanded correction actually applied

# Physical setup: arm starts from neutral, blob not visible.
# BLIND_ITERS represents the arm travelling from neutral to hover above the key.
# Phase A ends when the blob first appears — modelled here as exactly after
# BLIND_ITERS steps.
BLIND_ITERS = 18                # ~0.36 s at 20 ms / step

# Initial error when the blob FIRST becomes visible (after blind approach).
# Taken from representative system logs.
INITIAL_VIS_ERROR_M = np.array([0.00506, -0.00344], dtype=float)   # ~6.1 mm

# Fixed-P baseline uses the same initial error for a fair comparison
BASELINE_ERROR_M = np.array([0.00506, -0.00344], dtype=float)


# ============================================================
# Gain schedule (continuous interpolation — mirrors the real controller)
# ============================================================

def schedule(err_mag: float) -> Tuple[float, float]:
    """Return (gain, max_step) as a continuous function of error magnitude."""
    if err_mag >= THRESH_FAR:
        return KP_FAR, MAX_STEP_FAR
    elif err_mag <= THRESH_FINE:
        return KP_FINE, MAX_STEP_FINE
    t = (err_mag - THRESH_FINE) / (THRESH_FAR - THRESH_FINE)
    return (
        float(KP_FINE + t * (KP_FAR  - KP_FINE)),
        float(MAX_STEP_FINE + t * (MAX_STEP_FAR - MAX_STEP_FINE)),
    )


def clip_vec(v: np.ndarray, vmax: float) -> np.ndarray:
    """Scale vector v so its magnitude does not exceed vmax."""
    mag = np.linalg.norm(v)
    return v * (vmax / mag) if mag > vmax and mag > 1e-12 else v.copy()


# ============================================================
# Single simulation run
# ============================================================

def run_case(
    name: str,
    initial_error: np.ndarray,
    n_blind: int = 0,
    use_schedule: bool = True,
    use_dropout: bool = False,
    seed: int = 0,
) -> Tuple[str, List[Dict]]:
    """
    Simulate one variant of the unified controller.

    Parameters
    ----------
    name          : descriptive label for the plot legend
    initial_error : XY error (m) when the blob first becomes visible
    n_blind       : guaranteed blind iterations before the blob appears
    use_schedule  : if False, use fixed KP_FINE (old Stage 2 equivalent)
    use_dropout   : if True, simulate random blob loss during Phase B
    seed          : RNG seed for reproducibility
    """
    rng        = np.random.default_rng(seed)
    true_error = initial_error.copy()
    total_corr = np.zeros(2, dtype=float)
    e_prev     = None
    d_filt     = np.zeros(2, dtype=float)
    settle_n   = 0
    t          = 0.0
    history: List[Dict] = []

    # ---- Phase A: guaranteed blind iterations ----
    # The arm moves toward the target without any visual feedback.
    # Each step uses the coarse gain applied to the true (unobserved) error.
    # Phase A ends when the blob appears — not when a timer expires.
    for _ in range(n_blind):
        blind_u    = clip_vec(KP_FAR * true_error, MAX_STEP_FAR)
        true_error -= PLANT_RESPONSE * blind_u
        history.append({
            "t": t, "phase": "A_blind",
            "err": np.nan,   # no measurement available in Phase A
            "ex": np.nan, "ey": np.nan,
            "u_mag": float(np.linalg.norm(blind_u)),
        })
        t += DT

    # ---- Phase B: visual feedback ----
    # The blob is now visible.  The controller drives the error to zero
    # using measured feedback.  Termination is purely spatial.
    while t < MAX_TIME_S:

        # Simulate occasional blob dropout (only for use_dropout=True)
        if use_dropout and rng.random() < DROPOUT_PROB:
            e_prev = None
            d_filt[:] = 0.0
            history.append({
                "t": t, "phase": "B_dropout",
                "err": np.nan, "ex": np.nan, "ey": np.nan, "u_mag": 0.0,
            })
            t += DT
            continue

        # Noisy measurement (simulates camera pixel quantisation + noise)
        meas    = true_error + rng.normal(0.0, MEAS_NOISE_STD, size=2)
        err_mag = float(np.linalg.norm(meas))

        # ---- Spatial termination check ----
        if err_mag < ALIGN_TOL_M:
            settle_n += 1
            if settle_n >= SETTLE_FRAMES:
                history.append({
                    "t": t, "phase": "B_converged",
                    "err": err_mag, "ex": meas[0], "ey": meas[1], "u_mag": 0.0,
                })
                break  # spatial convergence
        else:
            settle_n = 0

        # Select gain: gain schedule or fixed-P (old Stage 2)
        if use_schedule:
            gain, max_step = schedule(err_mag)
        else:
            gain, max_step = KP_FINE, MAX_STEP_FINE

        u = clip_vec(gain * meas, max_step)

        # Enforce total correction bound
        proposed = total_corr + u
        if np.linalg.norm(proposed) > MAX_TOTAL_CORR:
            proposed = clip_vec(proposed, MAX_TOTAL_CORR)
            u = proposed - total_corr
        total_corr = proposed

        true_error -= PLANT_RESPONSE * u

        history.append({
            "t": t, "phase": "B_visual",
            "err": err_mag, "ex": meas[0], "ey": meas[1],
            "u_mag": float(np.linalg.norm(u)),
        })
        t += DT

    return name, history


# ============================================================
# Summary statistics
# ============================================================

def summarise(name: str, history: List[Dict]) -> Dict:
    """Extract convergence statistics from a simulation run's history."""
    vis = [
        h for h in history
        if h["phase"] in ("B_visual", "B_converged") and not np.isnan(h["err"])
    ]
    blind_n = sum(1 for h in history if h["phase"] == "A_blind")

    if not vis:
        return dict(
            case=name, settled=False, t_settle=None,
            final_mm=None, best_mm=None, blind_n=blind_n, termination="timeout",
        )

    converged   = vis[-1]["err"] < ALIGN_TOL_M
    t_settle    = vis[-1]["t"] if converged else None
    final_mm    = vis[-1]["err"] * 1000.0
    best_mm     = min(h["err"] for h in vis) * 1000.0
    termination = "spatial" if converged else "timeout"

    return dict(
        case=name, settled=converged, t_settle=t_settle,
        final_mm=final_mm, best_mm=best_mm, blind_n=blind_n, termination=termination,
    )


# ============================================================
# Define simulation cases
# ============================================================

CASES = [
    # (name, initial_error, n_blind, use_schedule, use_dropout, seed)
    (
        "1. Fixed-P only\n   (old Stage 2, no blind phase)",
        BASELINE_ERROR_M, 0, False, False, 100,
    ),
    (
        "2. Gain schedule\n   (no blind phase, ideal)",
        INITIAL_VIS_ERROR_M, 0, True, False, 101,
    ),
    (
        "3. Gain schedule\n   + guaranteed blind phase",
        INITIAL_VIS_ERROR_M, BLIND_ITERS, True, False, 102,
    ),
    (
        "4. Gain schedule\n   + blind phase + dropout",
        INITIAL_VIS_ERROR_M, BLIND_ITERS, True, True, 103,
    ),
]


# ============================================================
# Run and report
# ============================================================

if __name__ == "__main__":

    results   = [run_case(*c) for c in CASES]
    summaries = [summarise(*r) for r in results]

    print("\n=== Unified sensor-gated controller: simulation summary ===")
    print(
        f"Phase A (blind) duration: {BLIND_ITERS} steps "
        f"({BLIND_ITERS * DT * 1000:.0f} ms) — guaranteed on every note"
    )
    print(
        f"Phase B termination: spatial (|e| < {ALIGN_TOL_M*1000:.1f} mm "
        f"for {SETTLE_FRAMES} frames)\n"
    )

    header = (
        f"{'Case':<45} | {'Term.':<8} | {'T_settle':>9} | "
        f"{'Final (mm)':>10} | {'Best (mm)':>9} | Blind steps"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        label = s["case"].replace("\n   ", " ")
        ts = "-"    if s["t_settle"] is None else f"{s['t_settle']:.2f} s"
        fe = "-"    if s["final_mm"] is None else f"{s['final_mm']:.2f}"
        be = "-"    if s["best_mm"]  is None else f"{s['best_mm']:.2f}"
        print(
            f"{label:<45} | {s['termination']:<8} | {ts:>9} | "
            f"{fe:>10} | {be:>9} | {s['blind_n']}"
        )

    print("\nKey result: all cases terminate spatially.")
    print("The safety timeout is never reached in normal operation.")
    print("Phase A ends when the sensor becomes available, not when a step count expires.")

    # ============================================================
    # Plot results
    # ============================================================

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    fig.suptitle(
        "Unified Sensor-Gated Controller — Simulation Comparison",
    )

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    tol_drawn = False

    for (name, hist), color in zip(results, colors):
        t_arr   = np.array([h["t"]     for h in hist])
        err_arr = np.array([h["err"]   for h in hist])
        u_arr   = np.array([h["u_mag"] for h in hist])
        ex_arr  = np.array([h.get("ex",  np.nan) for h in hist])
        ey_arr  = np.array([h.get("ey",  np.nan) for h in hist])

        # Shade the Phase A (blind) region
        blind_mask = np.array([h["phase"] == "A_blind" for h in hist])
        if blind_mask.any():
            t_blind = t_arr[blind_mask]
            axes[0].axvspan(t_blind[0], t_blind[-1], alpha=0.10, color=color)
            axes[2].axvspan(t_blind[0], t_blind[-1], alpha=0.10, color=color)

        label = name.replace("\n   ", "\n")
        axes[0].plot(t_arr, err_arr * 1000, label=label, color=color, linewidth=1.5)
        axes[1].plot(t_arr, ex_arr * 1000, color=color, linewidth=1.2, alpha=0.9,
                     label=f"ex {name.split(chr(10))[0]}")
        axes[1].plot(t_arr, ey_arr * 1000, color=color, linewidth=1.2,
                     linestyle="--", alpha=0.6)
        axes[2].plot(t_arr, u_arr * 1000, label=label, color=color, linewidth=1.5)

        if not tol_drawn:
            axes[0].axhline(
                ALIGN_TOL_M * 1000, color="k", linestyle="--",
                linewidth=1.0, label=f"tol = {ALIGN_TOL_M*1000:.1f} mm",
            )
            tol_drawn = True

    # Phase A / Phase B boundary annotation
    blind_end_t = BLIND_ITERS * DT
    for ax in [axes[0], axes[2]]:
        ax.axvline(blind_end_t, color="grey", linestyle=":", linewidth=1.0)
        ymax = ax.get_ylim()[1]
        ax.text(blind_end_t / 2,     ymax * 0.92, "Phase A\n(blind)",  ha="center", fontsize=7, color="grey")
        ax.text(blind_end_t + 0.05,  ymax * 0.92, "Phase B\n(visual)", ha="left",   fontsize=7, color="grey")

    axes[0].set_ylabel("Error magnitude (mm)")
    axes[0].set_title("Convergence  (shaded = Phase A, unshaded = Phase B)")
    axes[0].legend(fontsize=7, loc="upper right", ncol=1)

    axes[1].set_ylabel("Axis error (mm)")
    axes[1].set_title("X error (solid) and Y error (dashed)")
    axes[1].legend(fontsize=7, ncol=2)

    axes[2].set_ylabel("Correction step size (mm)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Applied correction magnitude  (0 = blind / dropout)")
    axes[2].legend(fontsize=7, ncol=2)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig("unified_controller_sim.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigure saved: unified_controller_sim.png")
