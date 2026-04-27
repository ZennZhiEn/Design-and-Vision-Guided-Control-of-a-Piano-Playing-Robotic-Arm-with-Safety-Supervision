"""
tracgen.py
==========
Open-loop IK prototype for the piano-playing robotic arm.

This script was the first working motion pipeline developed during the
project. It establishes the ETS kinematic model, key-position geometry,
inverse kinematics, joint-space trajectory generation, servo mapping, and
serial streaming to the Arduino.

It was used during the open-loop verification stage described in Section 5.2
of the report. The FK visualisation plots shown in Figures 19 and 20 were
produced by plot_robot_motion() and animate_robot() at the bottom of this file.

Note: this is a standalone executable script, not a reusable module. The
classes and functions it defines were later refactored into PianoArmController
(tracgenaruco.py) and WorldBaseTrackerHomography (worldbasetrackerhomography.py)
for the full integrated system.

Dependencies
------------
    roboticstoolbox-python, spatialmath-python, numpy, matplotlib, pyserial
"""

# ── Standard library ──────────────────────────────────────────────────────
import time

# ── Third-party ───────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
import serial
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from spatialmath import SE3


# ============================================================
# 1.  COORDINATE FRAME DEFINITION
# ============================================================

# Static rigid-body transform from the piano/keyboard frame (P) to the
# robot base frame (B).  Established by physical measurement and kept fixed
# throughout open-loop testing.  In the final integrated system this
# transform is computed dynamically by WorldBaseTrackerHomography.
T_B_P = SE3(-0.178, 0.10, -0.015)


def piano_to_base(p_P: np.ndarray) -> np.ndarray:
    """
    Convert a point from the piano frame to the robot base frame.

    Parameters
    ----------
    p_P : np.ndarray (3,)
        Point in piano/keyboard coordinates [x, y, z] (metres).

    Returns
    -------
    np.ndarray (3,)
        The same point expressed in the robot base frame.
    """
    p_P = np.asarray(p_P, dtype=float).reshape(3,)
    p_P_h = np.r_[p_P, 1.0]               # homogeneous form
    p_B_h = T_B_P.A @ p_P_h               # apply 4×4 transform
    return p_B_h[:3]


# ============================================================
# 2.  KEYBOARD GEOMETRY
# ============================================================

N_KEYS = 22           # number of playable target keys
KEY_SPACING = 0.0167  # white-key centre-to-centre spacing (metres)

# Generate nominal key-centre coordinates in the piano frame.
# Keys lie along the x-axis; y and z are zero in the piano plane.
keys_P = np.array([[i * KEY_SPACING, 0.0, 0.0] for i in range(N_KEYS)])

# Convert all key positions to the robot base frame.
keys_B = np.array([piano_to_base(p) for p in keys_P], dtype=float)

# Sanity-check a few key positions.
for i in [0, 10, 14]:
    print(f"Key {i} (base frame): {keys_B[i]}")


# ============================================================
# 3.  KINEMATIC MODEL  (ETS — Elementary Transform Sequence)
# ============================================================

# Measured link lengths (metres).
# Final calibrated values used in the report; earlier trial values removed.
L1 = 0.0425   # base-to-shoulder vertical offset
L2 = 0.065    # shoulder-to-elbow link
L3 = 0.147    # elbow-to-wrist link
L4 = 0.022    # wrist lateral offset
L5 = 0.001    # wrist longitudinal offset (minor)

# Build the 3-DOF ETS chain: Rz(base) — Rx(shoulder) — Rx(elbow).
# tx(-L4) and ty(L5) capture the measured wrist offset geometry.
ets = (
    rtb.ET.Rz()      *   # joint 1: base rotation about Z
    rtb.ET.tz(L1)    *   # fixed: vertical rise to shoulder
    rtb.ET.Rx()      *   # joint 2: shoulder pitch about X
    rtb.ET.tz(L2)    *   # fixed: shoulder-to-elbow link
    rtb.ET.Rx()      *   # joint 3: elbow pitch about X
    rtb.ET.tz(L3)    *   # fixed: elbow-to-wrist link
    rtb.ET.tx(-L4)   *   # fixed: wrist lateral offset
    rtb.ET.ty(L5)        # fixed: wrist longitudinal offset
)

robot = rtb.ERobot(ets, name="PianoRoboticArm")

# Allow full ±π rotation on all three joints for IK solver flexibility.
robot.qlim = np.array(
    [[-np.pi, np.pi],
     [-np.pi, np.pi],
     [-np.pi, np.pi]],
    dtype=float
).T


# ============================================================
# 4.  INVERSE KINEMATICS
# ============================================================

def solve_ik(p_B: np.ndarray, q0: np.ndarray, n_restarts: int = 10) -> np.ndarray:
    """
    Solve inverse kinematics for a Cartesian position target.

    Uses the Levenberg–Marquardt solver (ikine_LM) with multiple random
    restart seeds to reduce the risk of converging to a local minimum.
    Only the translational part of the target pose is constrained
    (mask=[1,1,1,0,0,0]), so the solver finds a joint configuration that
    reaches the desired position without enforcing a specific orientation.

    Parameters
    ----------
    p_B : np.ndarray (3,)
        Desired end-effector position in the robot base frame (metres).
    q0 : np.ndarray (3,)
        Initial joint-angle seed (radians).
    n_restarts : int
        Number of additional random seeds to try if q0 fails.

    Returns
    -------
    np.ndarray (3,)
        Joint angles (radians) that place the end-effector at p_B.

    Raises
    ------
    RuntimeError
        If no valid IK solution is found across all seeds.
    """
    p_B = np.asarray(p_B, dtype=float).reshape(3,)
    q0  = np.asarray(q0,  dtype=float).reshape(3,)

    # Target pose: position only — orientation is unconstrained.
    T_target = SE3(p_B[0], p_B[1], p_B[2])

    # Build seed list: provided seed first, then random samples.
    qmin = robot.qlim[0, :]
    qmax = robot.qlim[1, :]
    seeds = [q0] + [qmin + (qmax - qmin) * np.random.rand(3)
                    for _ in range(n_restarts)]

    best_q   = None
    best_err = np.inf

    for qs in seeds:
        sol = robot.ikine_LM(
            T_target,
            q0=qs,
            mask=[1, 1, 1, 0, 0, 0],   # constrain x, y, z only
            joint_limits=True
        )
        if sol.success:
            return sol.q                 # return immediately on first success

        # Track best partial solution in case all seeds fail.
        if hasattr(sol, "residual") and sol.residual < best_err:
            best_err = sol.residual
            best_q   = sol.q

    raise RuntimeError(
        f"IK failed for target {p_B}. "
        f"Best residual = {best_err:.6f}, best_q = {best_q}"
    )


# ============================================================
# 5.  KEY-PRESS MOTION TARGETS
# ============================================================

HOVER_HEIGHT = 0.02    # metres above the key surface for the approach pose
PRESS_DEPTH  = -0.01   # metres below the key surface for the press pose


def key_targets(p_key_B: np.ndarray) -> tuple:
    """
    Compute hover and press target positions for a given key.

    Parameters
    ----------
    p_key_B : np.ndarray (3,)
        Key-centre position in the robot base frame (metres).

    Returns
    -------
    hover : np.ndarray (3,)
        Position HOVER_HEIGHT above the key — the approach pose.
    press : np.ndarray (3,)
        Position PRESS_DEPTH below the key — the contact pose.
    """
    p_key_B = np.asarray(p_key_B, dtype=float).reshape(3,)
    hover = p_key_B + np.array([0.0, 0.0,  HOVER_HEIGHT])
    press = p_key_B + np.array([0.0, 0.0,  PRESS_DEPTH])
    return hover, press


# ============================================================
# 6.  TRAJECTORY GENERATION
# ============================================================

def joint_trajectory(q_start: np.ndarray, q_goal: np.ndarray,
                     steps: int = 60) -> np.ndarray:
    """
    Generate a smooth joint-space trajectory between two configurations.

    Uses rtb.jtraj(), which applies a quintic (degree-5) time scaling to
    give zero velocity and acceleration at both endpoints.

    Parameters
    ----------
    q_start : np.ndarray (3,)
        Starting joint configuration (radians).
    q_goal : np.ndarray (3,)
        Goal joint configuration (radians).
    steps : int
        Number of discrete time steps in the trajectory.

    Returns
    -------
    np.ndarray (steps, 3)
        Array of joint configurations along the path.
    """
    q_start = np.asarray(q_start, dtype=float).reshape(3,)
    q_goal  = np.asarray(q_goal,  dtype=float).reshape(3,)
    traj = rtb.jtraj(q_start, q_goal, steps)
    return traj.q


# ============================================================
# 7.  SERVO MAPPING
# ============================================================

# Per-axis calibration constants for mapping joint angles to servo PWM angles.
# These values were determined empirically during hardware bring-up and are
# the precursors to the calibrated parameters in PianoArmController.
_SERVO_OFFSET_DEG = np.array([84.0,  38.0, 180.0], dtype=float)
_SERVO_SIGN       = np.array([ 1.0,  -1.0,   1.0], dtype=float)
_SERVO_MIN_DEG    = np.array([ 0.0,   0.0,   0.0], dtype=float)
_SERVO_MAX_DEG    = np.array([180.0, 180.0, 180.0], dtype=float)


def qrad_to_servo_deg(q: np.ndarray) -> np.ndarray:
    """
    Map joint angles (radians) to servo command angles (degrees).

    Applies the per-axis sign convention and mechanical zero-offset, then
    clips the result to the safe servo range [0°, 180°].

    Parameters
    ----------
    q : np.ndarray (3,)
        Joint angles in radians.

    Returns
    -------
    np.ndarray (3,)
        Servo command angles in degrees, clipped to [0°, 180°].
    """
    q = np.asarray(q, dtype=float).reshape(3,)
    deg = _SERVO_OFFSET_DEG + _SERVO_SIGN * np.degrees(q)
    return np.clip(deg, _SERVO_MIN_DEG, _SERVO_MAX_DEG)


# ============================================================
# 8.  SERIAL STREAMING
# ============================================================

def send_q_path(port: str, baud: int, q_path: np.ndarray, dt: float = 0.02) -> None:
    """
    Stream a joint-space trajectory to the Arduino over serial.

    Each row of q_path is converted to servo degrees and sent as a
    comma-separated ASCII line terminated with '\\n'. A brief sleep
    between lines gives the Arduino time to parse and execute each command.

    Parameters
    ----------
    port : str
        Serial port identifier (e.g. 'COM8' on Windows, '/dev/ttyUSB0' on Linux).
    baud : int
        Baud rate (must match the Arduino firmware setting — 115200).
    q_path : np.ndarray (N, 3)
        Joint trajectory to stream. Shape must be (N, 3).
    dt : float
        Delay between successive commands in seconds.

    Raises
    ------
    ValueError
        If q_path does not have shape (N, 3).
    """
    q_path = np.asarray(q_path, dtype=float)
    if q_path.ndim != 2 or q_path.shape[1] != 3:
        raise ValueError(f"q_path must have shape (N, 3), got {q_path.shape}")

    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(2)   # allow Arduino to reset after serial connection is opened

    try:
        for q in q_path:
            angles = qrad_to_servo_deg(q)
            line = f"{angles[0]:.1f},{angles[1]:.1f},{angles[2]:.1f}\n"
            ser.write(line.encode("ascii"))
            time.sleep(dt)
    finally:
        ser.close()


def send_to_arduino(q_path: np.ndarray, port: str,
                    baud: int = 115200, dt: float = 0.02) -> None:
    """
    Convenience wrapper around send_q_path with default baud rate.

    Parameters
    ----------
    q_path : np.ndarray (N, 3)
        Joint trajectory to stream.
    port : str
        Serial port identifier.
    baud : int
        Baud rate (default 115200).
    dt : float
        Inter-command delay in seconds (default 0.02 s).
    """
    send_q_path(port, baud, q_path, dt=dt)


# ============================================================
# 9.  FULL MOTION PIPELINE  (key 11 demo)
# ============================================================

# Initial joint configuration: arm approximately upright.
q_current = np.radians([0.0, 0.0, 180.0])

all_traj = []

for key_idx in [11]:

    p_key = keys_B[key_idx]
    hover, press = key_targets(p_key)

    print(f"\n=== Key {key_idx} ===")
    print(f"  key centre (base frame) : {p_key}")
    print(f"  hover target             : {hover}")
    print(f"  press target             : {press}")

    # Solve IK for hover and press poses.
    q_hover = solve_ik(hover, q_current)
    q_press = solve_ik(press, q_hover)

    print(f"\n  q_hover (rad)       : {q_hover}")
    print(f"  q_press (rad)       : {q_press}")
    print(f"  q_hover (servo deg) : {qrad_to_servo_deg(q_hover)}")
    print(f"  q_press (servo deg) : {qrad_to_servo_deg(q_press)}")

    # Generate three trajectory segments:
    #   1. current → hover  (approach)
    #   2. hover   → press  (descend and press)
    #   3. press   → hover  (lift and release)
    traj_approach = joint_trajectory(q_current, q_hover, steps=60)
    traj_press    = joint_trajectory(q_hover,   q_press, steps=30)
    traj_lift     = joint_trajectory(q_press,   q_hover, steps=30)

    all_traj.extend([traj_approach, traj_press, traj_lift])

    # Update current pose for the next iteration.
    q_current = q_hover

# Concatenate all segments into one continuous trajectory.
full_traj = np.vstack(all_traj)

# Stream the trajectory to the Arduino and allow time for completion.
send_to_arduino(full_traj, port="COM8", baud=115200, dt=0.02)
q_current = q_hover
time.sleep(0.4)


# ============================================================
# 10.  VISUALISATION
# ============================================================

def plot_robot_motion(robot, q_path: np.ndarray, step: int = 5) -> None:
    """
    Plot the robot arm geometry at sampled configurations along a trajectory.

    Forward kinematics is computed for every step-th row of q_path, and each
    resulting link-endpoint sequence is drawn as a connected line in 3D. This
    produces Figure 19 in the report.

    Parameters
    ----------
    robot : rtb.ERobot
        ETS-based robot model.
    q_path : np.ndarray (N, 3)
        Full joint trajectory.
    step : int
        Plot every Nth configuration to reduce visual clutter.
    """
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection="3d")

    for q in q_path[::step]:
        Ts     = robot.fkine_all(q)
        points = np.array([T.t for T in Ts])
        ax.plot(points[:, 0], points[:, 1], points[:, 2], marker="o")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Robot Arm Motion (FK from ETS)")
    ax.set_box_aspect([1, 1, 1])
    plt.show()


def animate_robot(robot, q_path: np.ndarray, interval: int = 40) -> FuncAnimation:
    """
    Animate the robot arm moving through a joint-space trajectory.

    Each frame updates the 3D link geometry using forward kinematics,
    producing a real-time animation of the arm motion. This produces
    Figure 20 in the report.

    The returned FuncAnimation object **must be kept alive** by the caller
    (assigned to a variable) for the animation to display. If the object is
    discarded, Python's garbage collector destroys it before plt.show() runs
    and nothing animates. blit=False is required because matplotlib's blit
    optimisation is not supported on 3D axes.

    Parameters
    ----------
    robot : rtb.ERobot
        ETS-based robot model.
    q_path : np.ndarray (N, 3)
        Joint trajectory to animate.
    interval : int
        Time between animation frames in milliseconds.

    Returns
    -------
    FuncAnimation
        The animation object. Assign the return value to a variable so it
        is not garbage-collected before plt.show() is called.
    """
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection="3d")

    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.25, 0.25)
    ax.set_zlim( 0.00, 0.30)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Robot Arm Motion (Animated FK)")

    line, = ax.plot([], [], [], "o-", lw=3)

    def _update(frame: int):
        q   = q_path[frame]
        Ts  = robot.fkine_all(q)
        pts = np.array([T.t for T in Ts])
        line.set_data(pts[:, 0], pts[:, 1])
        line.set_3d_properties(pts[:, 2])
        return (line,)

    # blit=False: blit is unsupported on 3D axes and silently breaks animation.
    # The return value must be stored — if discarded, the animation is
    # immediately garbage-collected and will not play.
    ani = FuncAnimation(fig, _update, frames=len(q_path),
                        interval=interval, blit=False)
    return ani


# ── Run visualisations ────────────────────────────────────────────────────
# All figures are created first, then plt.show() is called once at the end.
# This avoids each plt.show() blocking the script until the window is closed.

plot_robot_motion(robot, full_traj, step=5)       # static multi-pose plot (Figure 19)

# Assign animations to variables — required to prevent garbage collection.
ani_press = animate_robot(robot, traj_press,  interval=50)  # press segment (Figure 20)
ani_full  = animate_robot(robot, full_traj,   interval=30)  # full trajectory

plt.show()   # blocks here; all open figure windows display simultaneously