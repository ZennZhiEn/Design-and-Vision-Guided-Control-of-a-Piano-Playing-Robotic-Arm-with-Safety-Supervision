"""
session.py
==========
Session management, performance tracking, and per-key calibration storage.

This module provides three main classes used by PianoBotApp:

  KeyOffsetStore
      Stores a per-key (dx, dy) strike-target correction for each of the
      22 playable keys.  Corrections are loaded from / saved to a JSON file
      and can be nudged manually or updated from accumulated alignment data.

  PerformanceTracker
      Records spatial accuracy (best alignment error) and electrical press
      data (INA219 peak current, first-contact time) per key and per session.
      Also maintains an adaptive latency model: for each motion class
      (repeat, step, near, far, phrase_start) it keeps a rolling median of
      measured launch-to-contact times, which the playback engine uses to
      decide when to launch each note.

  TuneSessionManager
      Persists the weak-key tuning queue across sessions so calibration work
      can be resumed after restarting the program.
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from config import AppConfig


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class KeyPerformance:
    """Cumulative accuracy stats for a single key across multiple presses."""
    attempts: int = 0
    successes: int = 0
    best_err_sum_m: float = 0.0
    best_err_min_m: float | None = None
    last_err_sum_m: float = 0.0


@dataclass
class KeySession:
    """Per-key data collected during the current evaluation session."""
    attempts: int = 0
    successes: int = 0
    best_err_mm: list[float] = field(default_factory=list)
    first_over_mA_ms: list[float] = field(default_factory=list)   # INA219 contact time
    peak_current_mA: list[float] = field(default_factory=list)
    mean_current_mA: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-key strike target offset store
# ---------------------------------------------------------------------------

class KeyOffsetStore:
    """
    Stores individual (dx, dy) corrections applied on top of the nominal
    keyboard geometry model for each key.

    These corrections compensate for local camera projection error, mechanical
    tolerances, and any geometric mismatch between the model and the physical
    keyboard.  They are accumulated through repeated test presses and can be
    saved/loaded across sessions.
    """

    def __init__(self, n_keys: int, config: AppConfig) -> None:
        self.n_keys = n_keys
        self.config = config

        # Current active offsets: {key_index: (dx_metres, dy_metres)}
        self.target_offsets: dict[int, tuple[float, float]] = {
            i: (0.0, 0.0) for i in range(n_keys)
        }

        # Accumulated learned corrections from successful presses
        self.suggestion_sum: dict[int, np.ndarray] = {
            i: np.zeros(2, dtype=float) for i in range(n_keys)
        }
        self.suggestion_count: dict[int, int] = {i: 0 for i in range(n_keys)}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load per-key offsets from JSON file (silently skipped if missing)."""
        path = self.config.paths.key_offset_json
        if not path.exists():
            print(f"ℹ️ No offset file found: {path}")
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        loaded = 0
        for key, item in data.items():
            try:
                idx = int(key)
            except ValueError:
                continue
            if 0 <= idx < self.n_keys:
                self.target_offsets[idx] = (
                    float(item.get("dx_m", 0.0)),
                    float(item.get("dy_m", 0.0)),
                )
                loaded += 1
        print(f"✅ Loaded per-key target offsets from {path} ({loaded} keys)")

    def save(self) -> None:
        """Save all per-key offsets to JSON file."""
        path = self.config.paths.key_offset_json
        data = {
            str(i): {
                "dx_m": float(self.target_offsets.get(i, (0.0, 0.0))[0]),
                "dy_m": float(self.target_offsets.get(i, (0.0, 0.0))[1]),
            }
            for i in range(self.n_keys)
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"✅ Saved per-key target offsets to {path}")

    def reset_all(self) -> None:
        """Reset all keys to zero offset."""
        for i in range(self.n_keys):
            self.target_offsets[i] = (0.0, 0.0)
        print("✅ Reset ALL per-key target offsets to zero")

    # ------------------------------------------------------------------
    # Single-key operations
    # ------------------------------------------------------------------

    def nudge(self, idx: int, ddx: float = 0.0, ddy: float = 0.0) -> tuple[float, float]:
        """Apply a small incremental correction to key idx and return new offset."""
        dx, dy = self.target_offsets.get(idx, (0.0, 0.0))
        self.target_offsets[idx] = (dx + float(ddx), dy + float(ddy))
        return self.target_offsets[idx]

    def reset_one(self, idx: int) -> tuple[float, float]:
        """Reset a single key's offset to zero."""
        self.target_offsets[idx] = (0.0, 0.0)
        return self.target_offsets[idx]

    def get(self, idx: int) -> tuple[float, float]:
        """Return (dx, dy) offset for key idx (both zero if not set)."""
        return self.target_offsets.get(idx, (0.0, 0.0))

    def print_one(self, idx: int) -> None:
        dx, dy = self.get(idx)
        print(f"key {idx} per-key target offset: dx={dx*1000:.1f} mm, dy={dy*1000:.1f} mm")

    # ------------------------------------------------------------------
    # Learned correction suggestions
    # ------------------------------------------------------------------

    def learn_from_success(
        self,
        idx: int,
        x_seed: float,
        y_seed: float,
        x_ref_final: float,
        y_ref_final: float,
        best_err_m: float | None,
    ) -> None:
        """
        Record the difference between the IK seed and the visually aligned
        final position as a candidate correction for this key.

        Only samples with best_err_m < auto_cal_min_good_err_m are accepted,
        ensuring the suggestion is built from genuinely accurate presses.
        """
        if best_err_m is None or best_err_m > self.config.auto_cal_min_good_err_m:
            return

        corr = np.array([x_ref_final - x_seed, y_ref_final - y_seed], dtype=float)
        self.suggestion_sum[idx] += corr
        self.suggestion_count[idx] += 1

        n         = self.suggestion_count[idx]
        mean_corr = self.suggestion_sum[idx] / max(1, n)
        print(
            f"learn key {idx}: sample #{n}   "
            f"corr=({corr[0]*1000:.2f}, {corr[1]*1000:.2f}) mm   "
            f"mean=({mean_corr[0]*1000:.2f}, {mean_corr[1]*1000:.2f}) mm"
        )

    def get_suggestion(self, idx: int) -> tuple[np.ndarray | None, int]:
        """Return (mean_correction_vector, n_samples) for key idx."""
        n = self.suggestion_count[idx]
        if n <= 0:
            return None, 0
        return self.suggestion_sum[idx] / n, n

    def print_suggestion(self, idx: int) -> None:
        suggestion, n = self.get_suggestion(idx)
        if suggestion is None:
            print(f"key {idx} suggestion: no samples yet")
            return
        print(
            f"key {idx} suggestion: n={n}   "
            f"dx={suggestion[0]*1000:.2f} mm, dy={suggestion[1]*1000:.2f} mm"
        )

    def apply_suggestion(self, idx: int) -> bool:
        """
        Apply the accumulated learned suggestion to this key's stored offset.

        The shift magnitude is capped at auto_cal_max_shift_per_apply_m to
        prevent a single noisy suggestion from over-correcting.
        Returns True on success, False if not enough samples exist.
        """
        suggestion, n = self.get_suggestion(idx)
        if suggestion is None or n < self.config.auto_cal_min_samples:
            print(
                f"key {idx} suggestion not applied: "
                f"need at least {self.config.auto_cal_min_samples} good samples"
            )
            return False

        shift = suggestion.copy()
        mag   = np.linalg.norm(shift)
        if mag > self.config.auto_cal_max_shift_per_apply_m and mag > 1e-9:
            shift *= self.config.auto_cal_max_shift_per_apply_m / mag

        dx, dy = self.target_offsets.get(idx, (0.0, 0.0))
        self.target_offsets[idx] = (dx + float(shift[0]), dy + float(shift[1]))
        self.clear_suggestion(idx)
        print(
            f"✅ Applied suggestion to key {idx}: "
            f"new dx={self.target_offsets[idx][0]*1000:.2f} mm, "
            f"dy={self.target_offsets[idx][1]*1000:.2f} mm"
        )
        return True

    def clear_suggestion(self, idx: int) -> None:
        """Discard accumulated suggestion data for key idx."""
        self.suggestion_sum[idx]   = np.zeros(2, dtype=float)
        self.suggestion_count[idx] = 0
        print(f"Cleared learned suggestion for key {idx}")


# ---------------------------------------------------------------------------
# Performance tracker
# ---------------------------------------------------------------------------

class PerformanceTracker:
    """
    Tracks spatial accuracy, press behaviour, and timing statistics
    across the current session.

    Adaptive timing model
    ---------------------
    For each motion class (repeat, step, near, far, phrase_start) a rolling
    median of measured launch-to-contact times is maintained.  Once at least
    three valid samples have been collected for a motion class, the playback
    engine uses the learned median rather than the built-in default.  This
    allows the system to adapt to the specific mechanical characteristics of
    the current hardware setup.
    """

    def __init__(self, n_keys: int, config: AppConfig) -> None:
        self.n_keys   = n_keys
        self.config   = config
        self.session_id         = time.strftime("%Y%m%d_%H%M%S")
        self.session_start_time = time.time()

        # Per-key cumulative stats (across all attempts this session)
        self.key_perf_stats: dict[int, KeyPerformance] = {
            i: KeyPerformance() for i in range(n_keys)
        }
        # Per-key stats for the session evaluation summary report
        self.key_session_stats: dict[int, KeySession] = {
            i: KeySession() for i in range(n_keys)
        }

        # Rolling latency measurements (up to 50 per class)
        self.latency_stats: dict[str, list[float]] = {
            "repeat_reuse": [], "repeat": [], "step": [],
            "near": [], "far": [], "phrase_start": [],
        }
        # Timing prediction errors for reporting
        self.timing_error_stats: dict[str, list[float]] = {
            "repeat": [], "step": [], "near": [], "far": [], "phrase_start": [],
        }

    # ------------------------------------------------------------------
    # Recording results
    # ------------------------------------------------------------------

    def record_attempt(self, idx: int) -> None:
        """Increment attempt count for key idx."""
        self.key_perf_stats[idx].attempts += 1

    def record_success(
        self,
        idx: int,
        best_err_m: float | None,
        last_err_m: float | None,
    ) -> None:
        """Record a successful press with its best and final alignment errors."""
        st = self.key_perf_stats[idx]
        st.successes += 1
        if best_err_m is not None:
            st.best_err_sum_m += float(best_err_m)
            st.best_err_min_m = (
                float(best_err_m) if st.best_err_min_m is None
                else min(st.best_err_min_m, float(best_err_m))
            )
        if last_err_m is not None:
            st.last_err_sum_m += float(last_err_m)

    def record_session_result(
        self,
        idx: int,
        best_err_mm: float | None,
        ina_summary: dict[str, Any],
    ) -> None:
        """Record per-key session data for the evaluation summary."""
        st = self.key_session_stats[idx]
        st.attempts += 1
        if best_err_mm is not None:
            st.successes += 1
            st.best_err_mm.append(float(best_err_mm))
        if ina_summary.get("first_over_mA_ms") is not None:
            st.first_over_mA_ms.append(float(ina_summary["first_over_mA_ms"]))
        if ina_summary.get("peak_current_mA") is not None:
            st.peak_current_mA.append(float(ina_summary["peak_current_mA"]))
        if ina_summary.get("mean_current_mA") is not None:
            st.mean_current_mA.append(float(ina_summary["mean_current_mA"]))

    # ------------------------------------------------------------------
    # White-key sweep report
    # ------------------------------------------------------------------

    def build_white_key_report_rows(self) -> list[dict[str, Any]]:
        """Build the ranked white-key report data for all tested keys."""
        rows = []
        for i in self.config.white_key_test_sequence:
            st = self.key_perf_stats[i]
            success_pct  = (100.0 * st.successes / st.attempts) if st.attempts > 0 else 0.0
            mean_best_mm = (1000.0 * st.best_err_sum_m / st.successes) if st.successes > 0 else None
            min_best_mm  = (1000.0 * st.best_err_min_m) if st.best_err_min_m is not None else None

            if st.attempts == 0:
                status = "NO_DATA"
            elif (
                success_pct >= self.config.good_success_pct
                and mean_best_mm is not None
                and mean_best_mm <= self.config.good_mean_best_mm
            ):
                status = "GOOD"
            elif (
                success_pct < self.config.warn_success_pct
                or (mean_best_mm is not None and mean_best_mm > self.config.warn_mean_best_mm)
            ):
                status = "RETUNE"
            else:
                status = "OK"

            rows.append({
                "key": i,
                "attempts": st.attempts,
                "successes": st.successes,
                "success_pct": success_pct,
                "mean_best_mm": mean_best_mm,
                "min_best_mm": min_best_mm,
                "status": status,
            })
        return rows

    def print_ranked_white_key_report(self, top_n: int | None = None) -> None:
        top_n  = top_n or self.config.ranked_report_top_n
        rows   = self.build_white_key_report_rows()
        tested = [r for r in rows if r["attempts"] > 0]
        if not tested:
            print("\nℹ️ No white-key test data yet.\n")
            return

        worst = sorted(
            tested,
            key=lambda r: (r["success_pct"], -(r["mean_best_mm"] or 1e9)),
        )
        best = sorted(
            tested,
            key=lambda r: (-r["success_pct"], (r["mean_best_mm"] or 1e9)),
        )

        print("\n=== Ranked white-key report ===")
        print("Scope:", self.config.white_key_test_sequence)

        for title, data in (
            (f"Worst {min(top_n, len(worst))} keys", worst[:top_n]),
            (f"Best {min(top_n, len(best))} keys",  best[:top_n]),
        ):
            print(f"\n{title}:")
            print("key | attempts | success | succ% | mean_best_mm | min_best_mm | status")
            for row in data:
                mean_best = "-" if row["mean_best_mm"] is None else f"{row['mean_best_mm']:.2f}"
                min_best  = "-" if row["min_best_mm"]  is None else f"{row['min_best_mm']:.2f}"
                print(
                    f"{row['key']:>3} | {row['attempts']:>8} | {row['successes']:>7} | "
                    f"{row['success_pct']:>5.1f}% | {mean_best:>12} | {min_best:>11} | {row['status']}"
                )

        retune_keys = [row["key"] for row in tested if row["status"] == "RETUNE"]
        if retune_keys:
            print(f"\nKeys likely needing retune: {retune_keys}")
        else:
            print("\nNo keys currently flagged for retune.")
        print()

    def print_key_perf_stats(self) -> None:
        print("\n=== Per-key performance stats ===")
        print("key | attempts | success | succ% | mean_best_mm | min_best_mm")
        for i in range(self.n_keys):
            st           = self.key_perf_stats[i]
            succ_pct     = (100.0 * st.successes / st.attempts) if st.attempts > 0 else 0.0
            mean_best_mm = (1000.0 * st.best_err_sum_m / st.successes) if st.successes > 0 else None
            min_best_mm  = (1000.0 * st.best_err_min_m) if st.best_err_min_m is not None else None
            print(
                f"{i:>3} | {st.attempts:>8} | {st.successes:>7} | {succ_pct:>5.1f}% | "
                f"{'-' if mean_best_mm is None else f'{mean_best_mm:6.2f}':>12} | "
                f"{'-' if min_best_mm  is None else f'{min_best_mm:6.2f}':>11}"
            )
        print()

    def reset_key_perf_stats(self) -> None:
        self.key_perf_stats = {i: KeyPerformance() for i in range(self.n_keys)}
        print("✅ Reset per-key performance stats")

    # ------------------------------------------------------------------
    # Adaptive timing model
    # ------------------------------------------------------------------

    def update_latency_stats(
        self,
        motion_class: str,
        ina_summary: dict[str, Any],
    ) -> None:
        """
        Add a new launch-to-contact measurement for the given motion class.

        Uses the first_over_mA_ms field from the INA219 summary, which
        represents the time from INA_START to first electrical contact.
        Keeps only the 50 most recent samples per class.
        """
        values = self.latency_stats.get(motion_class)
        ms     = ina_summary.get("first_over_mA_ms")
        if values is not None and ms is not None and ms > 0:
            values.append(float(ms))
            if len(values) > 50:
                self.latency_stats[motion_class] = values[-50:]

    def predict_note_latency_s(self, motion_class: str) -> float:
        """
        Return the predicted launch-to-contact latency for motion_class (seconds).

        Uses the rolling median once 3+ samples are available; otherwise
        falls back to calibrated defaults from representative system logs.
        """
        data = self.latency_stats.get(motion_class, [])
        if len(data) >= 3:
            return statistics.median(data) / 1000.0

        # Startup defaults (measured from representative logs)
        defaults = {
            "repeat_reuse": 0.20,
            "repeat":       0.45,
            "step":         0.75,
            "near":         0.75,
            "far":          1.05,
            "phrase_start": 0.80,
        }
        return defaults.get(motion_class, 0.75)

    def update_timing_error_stats(
        self,
        motion_class: str,
        predicted_latency_s: float,
        actual_first_over_ms: float | None,
    ) -> None:
        """Record the difference between predicted and measured launch timing."""
        if actual_first_over_ms is None or motion_class not in self.timing_error_stats:
            return
        err_ms = float(actual_first_over_ms) - 1000.0 * float(predicted_latency_s)
        self.timing_error_stats[motion_class].append(err_ms)
        # Keep only the last 100 samples
        if len(self.timing_error_stats[motion_class]) > 100:
            self.timing_error_stats[motion_class] = self.timing_error_stats[motion_class][-100:]

    def print_latency_stats(self) -> None:
        print("\n=== Latency stats ===")
        for key in ["repeat", "step", "near", "far", "phrase_start"]:
            vals = self.latency_stats.get(key, [])
            if vals:
                print(f"{key:<12} n={len(vals):>2}  mean={statistics.mean(vals):.1f} ms")
            else:
                print(f"{key:<12} n= 0  mean=-")
        print()

    def print_timing_error_stats(self) -> None:
        print("\n=== Timing error stats ===")
        print("motion_class | n | mean_err_ms | median_err_ms | std_err_ms")
        for key in ["repeat", "step", "near", "far", "phrase_start"]:
            vals = self.timing_error_stats.get(key, [])
            if vals:
                mean_v = statistics.mean(vals)
                med_v  = statistics.median(vals)
                std_v  = statistics.stdev(vals) if len(vals) >= 2 else 0.0
                print(
                    f"{key:<12} | {len(vals):>2} | {mean_v:>11.1f} | "
                    f"{med_v:>13.1f} | {std_v:>10.1f}"
                )
            else:
                print(f"{key:<12} |  0 |           - |             - |          -")
        print()

    def print_timing_session_summary(self) -> None:
        print("\n=== Timing session summary ===")
        self.print_latency_stats()
        self.print_timing_error_stats()

    # ------------------------------------------------------------------
    # CSV / logging helpers
    # ------------------------------------------------------------------

    def append_press_validation_row(self, row_dict: dict[str, Any]) -> None:
        """Append one row to the press validation CSV log."""
        path = self.config.paths.press_log_csv
        fieldnames = [
            "timestamp", "key_idx", "press_profile", "profile_source",
            "forced_profile_name", "validation_run_id", "validation_trial",
            "auto_base_ref_key", "best_err_mm", "last_err_mm",
            "n_ina_samples", "peak_current_mA", "mean_current_mA",
            "min_current_mA", "min_bus_V", "first_over_mA_ms", "peak_ms",
            "motion_class",
        ]
        file_exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_dict)

    def append_ina_raw_log(
        self,
        lines: list[str],
        key_idx: int,
        trial_label: str = "",
        run_id: str = "",
    ) -> None:
        """Append raw INA219 lines to the running text log."""
        path = self.config.paths.ina219_raw_log_txt
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n" + "=" * 70 + "\n")
            handle.write(f"time={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            handle.write(f"key={key_idx}, run_id={run_id}, trial={trial_label}\n")
            for line in lines:
                handle.write(line + "\n")

    def summarize_validation_run(self, run_id: str | None = None) -> None:
        """Print a summary of soft/medium/hard press statistics for a validation run."""
        path = self.config.paths.press_log_csv
        if not path.exists():
            print(f"ℹ️ No validation CSV found: {path}")
            return

        with path.open("r", encoding="utf-8") as _f:
            rows = list(csv.DictReader(_f))
        if not rows:
            print("ℹ️ Validation CSV is empty")
            return

        # Default to the most recent run_id if none specified
        if not run_id:
            run_ids = [r.get("validation_run_id", "") for r in rows if r.get("validation_run_id")]
            if not run_ids:
                print("ℹ️ No validation_run_id found in CSV")
                return
            run_id = run_ids[-1]

        run_rows = [r for r in rows if r.get("validation_run_id") == run_id]
        if not run_rows:
            print(f"ℹ️ No rows found for validation_run_id={run_id}")
            return

        # Group rows by press profile
        grouped: dict[str, dict[str, list[float]]] = {}
        for row in run_rows:
            prof = row.get("press_profile", "")
            if prof not in grouped:
                grouped[prof] = {
                    "peak_current_mA": [], "mean_current_mA": [],
                    "best_err_mm": [], "min_bus_V": [],
                }
            for col in ("peak_current_mA", "mean_current_mA", "best_err_mm", "min_bus_V"):
                value = row.get(col, "")
                if value not in ("", None):
                    try:
                        grouped[prof][col].append(float(value))
                    except ValueError:
                        pass

        def _mean(values: list[float]) -> float | None:
            return statistics.mean(values) if values else None

        def _std(values: list[float]) -> float | None:
            if not values:
                return None
            return statistics.stdev(values) if len(values) >= 2 else 0.0

        print(f"\n=== Validation summary: {run_id} ===")
        print("profile | n | peak_mean | peak_std | meanI_mean | meanI_std | err_mean | busV_min")
        for prof in self.config.press_profile_order:
            if prof not in grouped:
                continue
            g = grouped[prof]
            peak_mean  = _mean(g["peak_current_mA"])
            peak_std   = _std(g["peak_current_mA"])
            meanI_mean = _mean(g["mean_current_mA"])
            meanI_std  = _std(g["mean_current_mA"])
            err_mean   = _mean(g["best_err_mm"])
            busV_min   = min(g["min_bus_V"]) if g["min_bus_V"] else None
            n = len(g["peak_current_mA"])
            print(
                f"{prof:<7} | {n:>1} | "
                f"{peak_mean:>9.2f} | {peak_std:>8.2f} | "
                f"{meanI_mean:>10.2f} | {meanI_std:>9.2f} | "
                f"{err_mean:>8.2f} | {busV_min:>7.3f}"
            )
        print()

    # ------------------------------------------------------------------
    # Session evaluation summary
    # ------------------------------------------------------------------

    def print_session_evaluation_summary(self) -> None:
        print("\n=== Session evaluation summary ===")
        print("key | attempts | succ% | mean_best_mm | mean_latency_ms | mean_peak_mA")

        worst_key   = None
        worst_score = None

        total_attempts  = 0
        total_successes = 0
        all_best: list[float] = []
        tested_keys = 0

        for idx in range(self.n_keys):
            st = self.key_session_stats[idx]
            if st.attempts == 0:
                continue
            tested_keys     += 1
            total_attempts  += st.attempts
            total_successes += st.successes

            success_pct  = 100.0 * st.successes / st.attempts
            mean_best    = statistics.mean(st.best_err_mm) if st.best_err_mm else None
            mean_first   = statistics.mean(st.first_over_mA_ms) if st.first_over_mA_ms else None
            mean_peak    = statistics.mean(st.peak_current_mA)  if st.peak_current_mA  else None
            if st.best_err_mm:
                all_best.extend(st.best_err_mm)

            print(
                f"{idx:>3} | {st.attempts:>8} | {success_pct:>5.1f}% | "
                f"{('-' if mean_best  is None else f'{mean_best:.2f}'):>12} | "
                f"{('-' if mean_first is None else f'{mean_first:.1f}'):>15} | "
                f"{('-' if mean_peak  is None else f'{mean_peak:.1f}'):>12}"
            )

            score = (success_pct, -(mean_best if mean_best is not None else 1e9))
            if worst_score is None or score < worst_score:
                worst_score = score
                worst_key   = idx

        overall_success_pct = (100.0 * total_successes / total_attempts) if total_attempts > 0 else 0.0
        mean_best_all       = statistics.mean(all_best) if all_best else None

        print()
        print(f"tested_keys={tested_keys}")
        print(f"total_attempts={total_attempts}")
        print(f"overall_success_pct={overall_success_pct:.1f}%")
        print(f"mean_best_err_mm={'-' if mean_best_all is None else f'{mean_best_all:.2f}'}")
        print(f"worst_key={worst_key}")
        print()

    def save_session_evaluation_summary(self) -> None:
        """Persist the session summary to CSV files for later analysis."""
        total_attempts  = 0
        total_successes = 0
        all_best: list[float] = []
        tested_keys = 0
        worst_key = None
        worst_key_success_pct = None
        worst_key_mean_best_mm = None
        worst_score = None

        for idx in range(self.n_keys):
            st = self.key_session_stats[idx]
            if st.attempts == 0:
                continue
            tested_keys     += 1
            total_attempts  += st.attempts
            total_successes += st.successes
            success_pct      = 100.0 * st.successes / st.attempts
            mean_best        = statistics.mean(st.best_err_mm) if st.best_err_mm else None
            if st.best_err_mm:
                all_best.extend(st.best_err_mm)
            score = (success_pct, -(mean_best if mean_best is not None else 1e9))
            if worst_score is None or score < worst_score:
                worst_score            = score
                worst_key              = idx
                worst_key_success_pct  = success_pct
                worst_key_mean_best_mm = mean_best

        duration_s           = time.time() - self.session_start_time
        overall_success_pct  = (100.0 * total_successes / total_attempts) if total_attempts > 0 else 0.0
        mean_best_all        = statistics.mean(all_best) if all_best else None

        self._append_session_summary_row({
            "session_id":              self.session_id,
            "timestamp":               time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_s":              round(duration_s, 1),
            "tested_keys":             tested_keys,
            "total_attempts":          total_attempts,
            "total_successes":         total_successes,
            "overall_success_pct":     overall_success_pct,
            "mean_best_err_mm":        mean_best_all,
            "worst_key":               worst_key,
            "worst_key_success_pct":   worst_key_success_pct,
            "worst_key_mean_best_mm":  worst_key_mean_best_mm,
        })
        self._append_key_session_rows()
        print(
            f"✅ Saved session evaluation to "
            f"{self.config.paths.session_summary_csv} and {self.config.paths.key_session_csv}"
        )

    def _append_session_summary_row(self, row_dict: dict[str, Any]) -> None:
        path = self.config.paths.session_summary_csv
        fieldnames = [
            "session_id", "timestamp", "duration_s", "tested_keys",
            "total_attempts", "total_successes", "overall_success_pct",
            "mean_best_err_mm", "worst_key", "worst_key_success_pct",
            "worst_key_mean_best_mm",
        ]
        file_exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_dict)

    def _append_key_session_rows(self) -> None:
        path = self.config.paths.key_session_csv
        fieldnames = [
            "session_id", "timestamp", "key_idx", "attempts", "successes",
            "success_pct", "mean_best_err_mm", "median_best_err_mm",
            "mean_first_over_mA_ms", "std_first_over_mA_ms",
            "mean_peak_current_mA", "std_peak_current_mA", "mean_mean_current_mA",
        ]
        file_exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for idx in range(self.n_keys):
                st = self.key_session_stats[idx]
                if st.attempts == 0:
                    continue
                success_pct = (100.0 * st.successes / st.attempts) if st.attempts > 0 else 0.0
                writer.writerow({
                    "session_id":            self.session_id,
                    "timestamp":             time.strftime("%Y-%m-%d %H:%M:%S"),
                    "key_idx":               idx,
                    "attempts":              st.attempts,
                    "successes":             st.successes,
                    "success_pct":           success_pct,
                    "mean_best_err_mm":      statistics.mean(st.best_err_mm) if st.best_err_mm else None,
                    "median_best_err_mm":    statistics.median(st.best_err_mm) if st.best_err_mm else None,
                    "mean_first_over_mA_ms": statistics.mean(st.first_over_mA_ms) if st.first_over_mA_ms else None,
                    "std_first_over_mA_ms":  statistics.stdev(st.first_over_mA_ms) if len(st.first_over_mA_ms) >= 2 else None,
                    "mean_peak_current_mA":  statistics.mean(st.peak_current_mA) if st.peak_current_mA else None,
                    "std_peak_current_mA":   statistics.stdev(st.peak_current_mA) if len(st.peak_current_mA) >= 2 else None,
                    "mean_mean_current_mA":  statistics.mean(st.mean_current_mA) if st.mean_current_mA else None,
                })


# ---------------------------------------------------------------------------
# Tune session persistence
# ---------------------------------------------------------------------------

class TuneSessionManager:
    """
    Persists the weak-key calibration queue and session state across
    program restarts, so tuning can be resumed where it left off.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config             = config
        self.weak_key_queue: list[int] = []
        self.weak_key_queue_pos: int   = -1
        self.typed: str = ""

    def save(self, report_rows: list[dict[str, Any]]) -> None:
        """Write current queue state and report snapshot to JSON."""
        path = self.config.paths.tune_session_json
        data = {
            "weak_key_queue":     list(self.weak_key_queue),
            "weak_key_queue_pos": int(self.weak_key_queue_pos),
            "typed":              str(self.typed),
            "report_snapshot": [
                {
                    "key":          int(r["key"]),
                    "attempts":     int(r["attempts"]),
                    "successes":    int(r["successes"]),
                    "success_pct":  float(r["success_pct"]),
                    "mean_best_mm": None if r["mean_best_mm"] is None else float(r["mean_best_mm"]),
                    "min_best_mm":  None if r["min_best_mm"]  is None else float(r["min_best_mm"]),
                    "status":       str(r["status"]),
                }
                for r in report_rows
            ],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"✅ Saved tune session state to {path}")

    def load(self) -> None:
        """Restore queue state from JSON (silently skipped if missing)."""
        path = self.config.paths.tune_session_json
        if not path.exists():
            print(f"ℹ️ No tune session file found: {path}")
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self.weak_key_queue     = [int(x) for x in data.get("weak_key_queue", [])]
        self.weak_key_queue_pos = int(data.get("weak_key_queue_pos", -1))
        self.typed              = str(data.get("typed", ""))
        print(f"✅ Loaded tune session state from {path}")
        print(f"weak_key_queue = {self.weak_key_queue}")
        print(f"weak_key_queue_pos = {self.weak_key_queue_pos}")
        print(f"typed = '{self.typed}'")
