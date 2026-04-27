import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Simulated visual alignment for the robotic-arm key-refinement stage
# ============================================================

DT = 0.03                    # matches cfg.dt
ALIGN_TOL_M = 0.0035         # matches cfg.align_tol_m
MAX_STEP_M = 0.0030          # matches cfg.pid_max_step_m
MAX_TOTAL_CORR_M = 0.050     # matches cfg.pid_max_total_corr_m
MAX_TIME_S = 4.0             # matches cfg.align_timeout_s
D_ALPHA = 0.7                # matches cfg.pid_d_alpha

# Example initial alignment error in metres.
# You can replace this with a measured case from your logs.
# For example: ex=5.06 mm, ey=-3.44 mm  -> 0.00506, -0.00344
INITIAL_ERROR_M = np.array([0.00506, -0.00344], dtype=float)

# Simulated measurement conditions
MEASUREMENT_NOISE_STD_M = 0.00035   # 0.35 mm measurement noise
BLOB_DROPOUT_PROB = 0.03            # chance of losing the blob on any step

# Plant response factor:
# 1.0 means the robot exactly applies the commanded correction next step.
# <1.0 means lag or incomplete correction.
PLANT_RESPONSE = 0.9


class SimVisionPID2D:
    def __init__(self, dt, kp, ki, kd, i_limit=(0.010, 0.010), d_alpha=0.7):
        self.dt_default = float(dt)
        self.kp = np.array(kp, dtype=float)
        self.ki = np.array(ki, dtype=float)
        self.kd = np.array(kd, dtype=float)
        self.i_limit = np.array(i_limit, dtype=float)
        self.d_alpha = float(d_alpha)
        self.reset()

    def reset(self):
        self.e_int = np.zeros(2, dtype=float)
        self.e_prev = None
        self.t_prev = None
        self.d_filt = np.zeros(2, dtype=float)

    def update(self, error_xy, t_now):
        e = np.asarray(error_xy, dtype=float)

        if self.t_prev is None:
            dt = self.dt_default
            d = np.zeros(2, dtype=float)
        else:
            dt = max(1e-3, t_now - self.t_prev)
            d_raw = (e - self.e_prev) / dt
            self.d_filt = self.d_alpha * self.d_filt + (1.0 - self.d_alpha) * d_raw
            d = self.d_filt

        self.e_int += e * dt
        self.e_int = np.clip(self.e_int, -self.i_limit, self.i_limit)

        output = self.kp * e + self.ki * self.e_int + self.kd * d

        self.e_prev = e.copy()
        self.t_prev = t_now

        return output, self.e_int.copy(), d.copy()


def clamp_vector(v, vmax):
    mag = np.linalg.norm(v)
    if mag > vmax and mag > 1e-12:
        return v * (vmax / mag)
    return v


def run_alignment_case(name, kp, ki, kd, seed=0):
    rng = np.random.default_rng(seed)

    pid = SimVisionPID2D(
        dt=DT,
        kp=kp,
        ki=ki,
        kd=kd,
        i_limit=(0.010, 0.010),
        d_alpha=D_ALPHA,
    )

    # true residual error between current EE position and target
    true_error = INITIAL_ERROR_M.copy()

    # integrated reference correction (like x_ref - x_ref0, y_ref - y_ref0)
    total_corr = np.zeros(2, dtype=float)

    t = 0.0
    history = []

    while t < MAX_TIME_S:
        # Simulate blob loss
        blob_seen = rng.random() > BLOB_DROPOUT_PROB

        if not blob_seen:
            pid.reset()
            history.append({
                "t": t,
                "blob_seen": False,
                "ex": np.nan,
                "ey": np.nan,
                "err": np.nan,
                "ux": 0.0,
                "uy": 0.0,
                "u_mag": 0.0,
                "ix": 0.0,
                "iy": 0.0,
                "dx": 0.0,
                "dy": 0.0,
                "corr_x": total_corr[0],
                "corr_y": total_corr[1],
            })
            t += DT
            continue

        measured_error = true_error + rng.normal(0.0, MEASUREMENT_NOISE_STD_M, size=2)
        err_mag = float(np.linalg.norm(measured_error))

        if err_mag < ALIGN_TOL_M:
            history.append({
                "t": t,
                "blob_seen": True,
                "ex": measured_error[0],
                "ey": measured_error[1],
                "err": err_mag,
                "ux": 0.0,
                "uy": 0.0,
                "u_mag": 0.0,
                "ix": pid.e_int[0],
                "iy": pid.e_int[1],
                "dx": pid.d_filt[0],
                "dy": pid.d_filt[1],
                "corr_x": total_corr[0],
                "corr_y": total_corr[1],
            })
            break

        u, i_term, d_term = pid.update(measured_error, t)

        # clamp local correction step
        u = clamp_vector(u, MAX_STEP_M)

        # clamp total accumulated correction
        proposed_total = total_corr + u
        if np.linalg.norm(proposed_total) > MAX_TOTAL_CORR_M:
            proposed_total = clamp_vector(proposed_total, MAX_TOTAL_CORR_M)
            u = proposed_total - total_corr

        total_corr = proposed_total

        # Simulate plant response:
        # applying correction reduces true error next step
        true_error = true_error - PLANT_RESPONSE * u

        history.append({
            "t": t,
            "blob_seen": True,
            "ex": measured_error[0],
            "ey": measured_error[1],
            "err": err_mag,
            "ux": u[0],
            "uy": u[1],
            "u_mag": np.linalg.norm(u),
            "ix": i_term[0],
            "iy": i_term[1],
            "dx": d_term[0],
            "dy": d_term[1],
            "corr_x": total_corr[0],
            "corr_y": total_corr[1],
        })

        t += DT

    return name, history


def summarise_case(name, history):
    valid = [h for h in history if h["blob_seen"] and not np.isnan(h["err"])]
    if not valid:
        return {
            "case": name,
            "settled": False,
            "settling_time_s": None,
            "final_err_mm": None,
            "max_err_mm": None,
            "n_steps": len(history),
        }

    final_err_mm = 1000.0 * valid[-1]["err"]
    max_err_mm = 1000.0 * max(h["err"] for h in valid)
    settled = valid[-1]["err"] < ALIGN_TOL_M
    settling_time_s = valid[-1]["t"] if settled else None

    return {
        "case": name,
        "settled": settled,
        "settling_time_s": settling_time_s,
        "final_err_mm": final_err_mm,
        "max_err_mm": max_err_mm,
        "n_steps": len(history),
    }


cases = [
    # baseline from your current config: P only
    ("P only",   (0.30, 0.30), (0.00, 0.00), (0.00, 0.00)),
    # mild PI
    ("PI",       (0.30, 0.30), (0.05, 0.05), (0.00, 0.00)),
    # mild PD
    ("PD",       (0.30, 0.30), (0.00, 0.00), (0.01, 0.01)),
    # mild PID
    ("PID",      (0.30, 0.30), (0.03, 0.03), (0.01, 0.01)),
]

results = []
summaries = []

for k, (name, kp, ki, kd) in enumerate(cases):
    result = run_alignment_case(name, kp, ki, kd, seed=100 + k)
    results.append(result)
    summaries.append(summarise_case(*result))

print("\n=== Alignment response summary ===")
print("case     | settled | settling_time_s | final_err_mm | max_err_mm | steps")
print("-" * 72)

for s in summaries:
    settling_str = "-" if s["settling_time_s"] is None else f"{s['settling_time_s']:.2f}"
    final_err_str = "-" if s["final_err_mm"] is None else f"{s['final_err_mm']:.2f}"
    max_err_str = "-" if s["max_err_mm"] is None else f"{s['max_err_mm']:.2f}"

    print(
        f"{s['case']:<8} | "
        f"{str(s['settled']):<7} | "
        f"{settling_str:>15} | "
        f"{final_err_str:>12} | "
        f"{max_err_str:>10} | "
        f"{s['n_steps']:>5}"
    )

# -----------------------------
# Plotting
# -----------------------------
fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

for name, hist in results:
    t = np.array([h["t"] for h in hist])
    err = np.array([h["err"] if not np.isnan(h["err"]) else np.nan for h in hist]) * 1000.0
    ex = np.array([h["ex"] if not np.isnan(h["ex"]) else np.nan for h in hist]) * 1000.0
    ey = np.array([h["ey"] if not np.isnan(h["ey"]) else np.nan for h in hist]) * 1000.0
    u_mag = np.array([h["u_mag"] for h in hist]) * 1000.0

    axes[0].plot(t, err, label=name)
    axes[1].plot(t, ex, label=f"{name} ex")
    axes[1].plot(t, ey, linestyle="--", label=f"{name} ey")
    axes[2].plot(t, u_mag, label=name)

axes[0].axhline(ALIGN_TOL_M * 1000.0, linestyle="--")
axes[0].set_ylabel("error magnitude (mm)")
axes[0].set_title("Visual alignment response comparison")

axes[1].set_ylabel("axis error (mm)")
axes[1].set_title("X and Y error")

axes[2].set_ylabel("command step (mm)")
axes[2].set_xlabel("time (s)")
axes[2].set_title("Applied correction magnitude")

for ax in axes:
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

plt.tight_layout()
plt.show()