# Design and Vision-Guided Control of a Piano-Playing Robotic Arm

**MEng Final Year Project — University of Southampton, ECS**  
**Author:** Zhi En Tee | **Supervisor:** Dr. Bing Chu

A vision-guided robotic piano-playing system built on a low-cost robotic arm,
camera-based workspace localisation, and safety-supervised note execution.
The system uses ArUco markers to establish a shared reference frame, homography
to map between image space and the keyboard plane, and a hybrid two-stage
control framework that combines open-loop Cartesian motion with closed-loop
visual alignment before each key press.

---

## Demo

> The system playing *Hot Cross Buns* at 35 BPM.

![System overview](aruco_40mm/aruco_10.png)

---

## Table of Contents

- [Features](#features)
- [Hardware](#hardware)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Physical Setup](#physical-setup)
- [Configuration](#configuration)
- [Running the System](#running-the-system)
- [Operator Controls](#operator-controls)
- [Adding Songs](#adding-songs)
- [Development and Simulation Scripts](#development-and-simulation-scripts)
- [Project Report](#project-report)

---

## Features

- **ArUco-based workspace registration** — three printed markers establish a shared coordinate frame for the keyboard and robot base, recomputed every camera frame via homography
- **Hybrid two-stage key execution** — Stage 1 uses inverse-kinematics Cartesian motion to approach the key region; Stage 2 uses closed-loop proportional visual alignment (green end-effector ring) to correct residual XY error before pressing
- **Motion-class-aware playback** — notes are classified as `phrase_start`, `repeat`, `step`, `near`, or `far`; each class uses different hover heights, alignment strategy, and timing prediction
- **Configurable hybrid press profiles** — staged preload → servo press → hold → release sequence; four profiles: `soft`, `medium`, `hard`, `song_fast`
- **INA219 electrical press monitoring** — current logging on every press provides objective press validation and launch-to-contact timing estimates
- **Per-key calibration** — individual (dx, dy) strike-target offsets with learned correction suggestions persisted between sessions
- **Three operating modes** — `PLAY` for manual and song playback, `CALIBRATE` for offset tuning and sweep tests, `EVALUATE` for structured validation and timing analysis
- **Safety supervision** — safe-stop on serial failure, marker loss, or blob loss; neutral-pose recovery on shutdown

---

## Hardware

| Item | Detail | Cost (approx.) |
|---|---|---|
| Adeept 5-DOF Robotic Arm Kit | Arduino Uno, 5× hobby servos | £45.77 |
| 22-key toy piano/keyboard | Any small keyboard with regular white-key spacing | £15.99 |
| Smartphone + tripod | Overhead camera via DroidCam | £14.05 |
| INA219 current sensor module | I2C, replaces the kit OLED | £3.99 |
| **Total** | | **£79.80** |

The INA219 replaces the OLED display on the Adeept kit. It connects to the Arduino I2C bus (SDA/SCL) and shares the same serial link used for motion commands.

---

## Repository Structure

```
├── main.py                          # Entry point — run this to start
├── config.py                        # All tunable parameters (edit before running)
├── app.py                           # Main application loop, modes, UI
├── tracgenaruco.py                  # PianoArmController: IK, motion, hybrid press
├── worldbasetrackerhomography.py    # ArUco detection, homography, coordinate frames
├── greencircle.py                   # HSV end-effector blob detector
├── pianokeymodel.py                 # Geometric keyboard model (22 key targets)
├── session.py                       # Calibration, stats, timing model
├── unified_controller.py            # Unified sensor-gated controller (conceptual)
├── pidsim.py                        # Simulation: unified controller variants
├── arucomarker.py                   # Generates ArUco marker PDFs (run once)
│
├── tracgen.py                       # [Dev] Open-loop IK prototype (Section 5.2)
├── pidsim_1.py                      # [Dev] P/PI/PD/PID comparison (Section 5.4.3)
├── tracgen.ino                      # Arduino firmware (serial servo + INA219)
│
├── songs/
│   ├── hot_cross_buns.json          # 17-note test melody (used in evaluation)
│   ├── timing_test.json             # 6-note timing diagnostic sequence
│   └── you_are_my_sunshine.json     # 37-note extended melody
│
├── aruco_40mm/
│   ├── aruco_10.png                 # Marker ID10 — keyboard left / world origin
│   ├── aruco_11.png                 # Marker ID11 — keyboard right / +X direction
│   └── aruco_20.png                 # Marker ID20 — robot base reference
│
├── key_target_offsets.json          # Per-key calibration offsets (auto-generated)
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ZennZhiEn/Design-and-Control-of-a-Piano-Playing-Robotic-Hand.git
cd Design-and-Control-of-a-Piano-Playing-Robotic-Hand
```

### 2. Install Python dependencies

Python 3.11 is recommended.

```bash
pip install -r requirements.txt
```

### 3. Flash the Arduino firmware

Open `tracgen.ino` in the Arduino IDE and upload it to the Adeept Arduino Uno.
The firmware expects a 115200 baud serial connection. After flashing, open the
Serial Monitor briefly to confirm it prints `ARDUINO_READY`.

### 4. Install DroidCam (optional)

If using a smartphone as the overhead camera, install
[DroidCam](https://www.dev47apps.com/) on both the phone and the PC. Connect
via USB for the lowest latency. The app will appear as camera index `1` in
OpenCV (the default in `config.py`).

---

## Physical Setup

### 1. Print the ArUco markers

Run the marker generator once to produce correctly sized PDFs:

```bash
python arucomarker.py
```

Print each marker at **100% scale** (no scaling/fit-to-page). Each marker
should measure exactly **40 mm × 40 mm** after printing. Verify with a ruler
before proceeding — incorrect marker size is the most common cause of
calibration errors.

### 2. Place the markers

| Marker | File | Placement |
|---|---|---|
| ID10 | `aruco_40mm/aruco_10.png` | Left side of the keyboard. This is the **world origin**. |
| ID11 | `aruco_40mm/aruco_11.png` | Right side of the keyboard. Defines the **+X direction** along the keys. |
| ID20 | `aruco_40mm/aruco_20.png` | Attached near the **robot base**. |

All three markers must be visible in the camera frame simultaneously during
operation. Arrange the keyboard and robot so both fit within the overhead
camera's field of view.

### 3. Mount the camera

Position the smartphone directly overhead (or at a slight angle) so that the
entire keyboard and the robot arm are visible. The camera should be stable —
vibration during motion will disrupt marker detection.

### 4. Measure and configure offsets

Two physical measurements must be taken and entered into `config.py`:

- **`baseline_m`** — the centre-to-centre distance between ID10 and ID11 in metres (typically around 0.415 m)
- **`mr_to_base_xyz`** — the (dx, dy, dz) offset in metres from the centre of marker ID20 to the actual robot base origin

---

## Configuration

All parameters are in `config.py`. The minimum set to edit before first run:

```python
# Hardware
com_port   = "COM8"      # Windows serial port — change to "/dev/ttyUSB0" on Linux
cam_index  = 1           # 1 = DroidCam USB, 0 = built-in webcam

# Workspace geometry (measure physically)
baseline_m       = 0.415          # ID10 → ID11 distance (metres)
mr_to_base_xyz   = (-0.057, 0.070, 0.023)   # marker → robot base offset (metres)
mr_to_base_yaw_deg = 0.0          # yaw angle if marker is not aligned with base

# Keyboard geometry
x_id10_to_f1 = 0.036     # X distance from ID10 to the first key (metres)
key_pitch    = 0.0162     # white-key centre-to-centre spacing (metres)
n_keys       = 22         # number of playable keys
```

All other parameters (PID gains, press profiles, safety thresholds, hover
heights) are documented with inline comments in `config.py` and use the
validated values from the project evaluation. They do not need to be changed
for basic operation.

---

## Running the System

Ensure the Arduino is connected, DroidCam is running, and the markers are
visible in the camera view. Then:

```bash
python main.py
```

A window opens showing the live camera feed with overlaid key targets, marker
detections, and status information. The status bar at the bottom of the window
shows:

```
mode=PLAY  typed=  marker=OK  blob=OK  serial=OK  stop=NO
```

All four status indicators should show `OK` / `NO` before attempting any key
press or playback. If `marker=LOST`, reposition the camera or markers. If
`serial=FAIL`, check the COM port setting in `config.py`.

---

## Operator Controls

### Global (all modes)

| Key | Action |
|---|---|
| `TAB` | Cycle between PLAY, CALIBRATE, and EVALUATE modes |
| `F` | Request safe stop immediately |
| `G` | Clear safe stop |
| `0`–`9` then `ENTER` | Type a key index (0–21) and execute |
| `q` / `ESC` | Quit and move arm to neutral pose |

### PLAY mode

| Key | Action |
|---|---|
| `ENTER` | Press the currently typed key index |
| `m` | Start song playback |
| `[` / `]` | Previous / next song in library |
| `p` | Print song library to console |
| `-` / `=` | Cycle press profile down / up |
| `k` | Toggle automatic press profile selection |

### CALIBRATE mode

| Key | Action |
|---|---|
| `a` / `d` | Nudge key offset left / right (−1 mm / +1 mm in X) |
| `w` / `x` | Nudge key offset up / down (−1 mm / +1 mm in Y) |
| `z` | Reset selected key offset to zero |
| `v` | Save all offsets to `key_target_offsets.json` |
| `b` | Reload offsets from file |
| `g` | Run automated white-key sweep (keys 4–17) |
| `l` | Print ranked white-key performance report |
| `.` | Run repeated-press test on selected key |
| `j` | Auto-select worst-performing key |
| `y` | Apply learned correction suggestion |

### EVALUATE mode

| Key | Action |
|---|---|
| `r` | Run press profile validation (soft / medium / hard × 5 reps) |
| `A` | Run repeat timing test |
| `B` | Print timing error statistics |
| `D` | Print session evaluation summary |
| `E` | Save session evaluation summary to CSV |

---

## Adding Songs

Songs are stored as JSON files in the `songs/` directory. The format is:

```json
{
  "name": "Song Name",
  "bpm": 35,
  "events": [
    { "key": 13, "beat": 0.0, "dur": 1.0, "phrase_start": true },
    { "key": 12, "beat": 1.0, "dur": 1.0 },
    { "key": 11, "beat": 2.0, "dur": 2.0, "phrase_end": true }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `key` | int | Key index (0–21). Key 11 = E4, 12 = F4, 13 = G4 on the physical keyboard used in testing. |
| `beat` | float | Note onset time in beats from the start of the song. |
| `dur` | float | Note duration in beats. |
| `phrase_start` | bool | Optional. Marks the first note of a phrase — triggers a fresh full alignment. |
| `phrase_end` | bool | Optional. Informational marker, not used by the playback engine. |

Key indices map to physical piano keys based on the geometry configured in
`config.py`. At 35 BPM, one beat = 1.714 seconds.

---

## Development and Simulation Scripts

Two additional scripts capture earlier development stages and are included for
reproducibility. They are not part of the main application.

### `tracgen.py` — Open-Loop IK Prototype

The first working motion script, written before the full `PianoArmController`
class existed. Implements the ETS kinematic chain, IK solving, trajectory
generation, servo mapping, and serial streaming directly. The FK visualisation
plots in the report (Figures 19–20) were produced here.

```bash
python tracgen.py
```

> **Note:** this script attempts to connect to `COM8` and stream motion to the
> Arduino. Comment out the `send_to_arduino(...)` call if you want to run the
> visualisations only.

### `pidsim_1.py` — P/PI/PD/PID Alignment Comparison

Standalone simulation used to justify the choice of proportional-only control
for Stage 2 visual alignment (Section 5.4.3, Figure 22). No hardware required.

```bash
python pidsim_1.py
```

### `pidsim.py` — Unified Sensor-Gated Controller Simulation

Extended simulation comparing fixed-P vs gain-scheduled alignment with and
without a blind approach phase (Sections 5.4.4, Figures 23–24). No hardware required.

```bash
python pidsim.py
```

---

## Project Report

The full project report is available in the repository. It covers the
theoretical background, system design, implementation details, testing
methodology, and experimental results in detail.

**Key sections:**
- Chapter 4: Computer Vision and Workspace Localisation
- Chapter 5: Motor Control and Key Actuation (hybrid framework, visual alignment, press profiles)
- Chapter 6: Testing, Evaluation, and Results (single-key accuracy, white-key sweep, timing model, full-song playback)
- Appendix E: Full software structure documentation

---

## Acknowledgements

Supervisor: **Dr. Bing Chu**, University of Southampton  
Second Examiner: **Dr. Igor Golosnoy**, University of Southampton
