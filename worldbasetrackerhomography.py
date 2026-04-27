"""
worldbasetrackerhomography.py
==============================
Homography-based workspace tracker for the piano robot.

Purpose
-------
This module establishes a shared geometric coordinate frame for the
keyboard and the robot base using three printed ArUco markers:

  ID10  — world-frame origin (keyboard left side)
  ID11  — defines the +x axis direction along the keyboard
  ID_BASE — attached near the robot base; used to estimate base position

How it works
------------
1. Detect all three ArUco markers in the camera frame.
2. Compute a homography H between the keyboard plane (world, metres) and
   the camera image (pixels) using the known corners of ID10 and ID11.
3. Use this homography to convert between pixel and world coordinates:
     world -> pixel  : for overlaying key targets on the live display
     pixel -> world  : for converting the detected end-effector position
                       into world-frame coordinates for the controller
4. Map the base marker (ID_BASE) through the same homography to estimate
   the robot base position in the world frame, then build B_T_W (the
   world-to-base rigid transform used by the IK pipeline).

Z handling
----------
The key surface is treated as planar (Z=0).  A separate z_lock_keyplane()
call adjusts the Z component of B_T_W so that world point (xW_ref, yW_ref, 0)
maps to the desired Z height in the base frame.  This avoids needing full
3D pose estimation from the camera.
"""

import cv2
import numpy as np


def _invert_homogeneous(T: np.ndarray) -> np.ndarray:
    """Efficiently invert a 4x4 homogeneous transform using R^T."""
    R = T[:3, :3]
    t = T[:3,  3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3,  3] = -R.T @ t
    return Ti


class WorldBaseTrackerHomography:
    """
    Tracks keyboard plane and robot base position using ArUco homography.

    After each successful update(), the following attributes are valid:
      H_w2i  — 3x3 homography from world (metres) to image (pixels)
      H_i2w  — inverse homography: image to world
      B_T_W  — 4x4 rigid transform from world frame to robot base frame
    """

    def __init__(
        self,
        id10: int,
        id11: int,
        id_base: int,
        marker_size_m: float,
        baseline_m: float,
        mr_to_base_xyz_m: tuple = (0.0, 0.0, 0.0),
        mr_to_base_yaw_deg: float = 0.0,
        aruco_dict_id=cv2.aruco.DICT_4X4_50,
    ):
        """
        Parameters
        ----------
        id10, id11      : ArUco IDs of the two keyboard reference markers
        id_base         : ArUco ID of the base marker on the robot
        marker_size_m   : physical side length of each marker in metres
        baseline_m      : measured centre-to-centre distance ID10 -> ID11 (metres)
        mr_to_base_xyz_m: (dx, dy, dz) offset from the base marker centre to the
                          actual robot base origin, in the marker frame
        mr_to_base_yaw_deg: rotation angle between marker and base frames (degrees)
        """
        self.id10    = int(id10)
        self.id11    = int(id11)
        self.id_base = int(id_base)

        self.S = float(marker_size_m)
        self.D = float(baseline_m)

        self.mr_to_base = np.array(mr_to_base_xyz_m, dtype=np.float64)
        self.mr_yaw     = np.deg2rad(float(mr_to_base_yaw_deg))

        # NOTE: Dictionary_get() is the OpenCV <4.7 API.
        # In OpenCV >=4.7 use cv2.aruco.getPredefinedDictionary() instead.
        self.aruco_dict   = cv2.aruco.Dictionary_get(aruco_dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # Cached outputs from the most recent update()
        self.last_corners = None
        self.last_ids     = None
        self.H_w2i = None
        self.H_i2w = None
        self.B_T_W = None

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _marker_corners_world_xy(self, cx: float, cy: float) -> np.ndarray:
        """
        Return the four corners of a marker centred at (cx, cy) in world metres.
        Order: top-left, top-right, bottom-right, bottom-left.
        """
        h = self.S / 2.0
        return np.array(
            [
                [cx - h, cy + h],
                [cx + h, cy + h],
                [cx + h, cy - h],
                [cx - h, cy - h],
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Main update — call once per camera frame
    # ------------------------------------------------------------------

    def update(self, frame_bgr: np.ndarray) -> bool:
        """
        Detect markers in frame_bgr and update H_w2i, H_i2w, and B_T_W.

        Returns True if all three markers were found and the homography
        was computed successfully; False otherwise.
        """
        corners, ids, _ = cv2.aruco.detectMarkers(
            frame_bgr, self.aruco_dict, parameters=self.aruco_params
        )
        self.last_corners = corners
        self.last_ids     = ids

        if ids is None:
            return False

        ids = ids.flatten()
        i10 = np.where(ids == self.id10)[0]
        i11 = np.where(ids == self.id11)[0]
        ib  = np.where(ids == self.id_base)[0]

        if len(i10) == 0 or len(i11) == 0 or len(ib) == 0:
            return False  # at least one marker is missing

        img10 = corners[i10[0]].reshape(4, 2).astype(np.float32)
        img11 = corners[i11[0]].reshape(4, 2).astype(np.float32)
        imgB  = corners[ib[0]].reshape(4, 2).astype(np.float32)

        # ---- Step 1: compute homography from keyboard world plane to image ----
        # ID10 is placed at world origin (0, 0); ID11 is at (D, 0)
        world10 = self._marker_corners_world_xy(0.0, 0.0)
        world11 = self._marker_corners_world_xy(self.D, 0.0)

        world_pts = np.vstack([world10, world11])
        img_pts   = np.vstack([img10, img11])

        H, _ = cv2.findHomography(world_pts, img_pts, method=0)
        if H is None:
            return False

        self.H_w2i = H
        self.H_i2w = np.linalg.inv(H)

        # ---- Step 2: map base marker corners into world frame ----
        # This gives us the marker centre and orientation without needing
        # camera intrinsics or 3D pose estimation.
        wB = np.array(
            [self.pixel_to_world(u, v) for (u, v) in imgB],
            dtype=np.float64,
        )  # shape (4, 2)

        xm, ym = wB.mean(axis=0)   # marker centre in world

        # Estimate marker yaw from the top edge (corner 0 → corner 1)
        dx_edge = wB[1, 0] - wB[0, 0]
        dy_edge = wB[1, 1] - wB[0, 1]
        theta_m = np.arctan2(dy_edge, dx_edge)

        # Apply the known yaw offset from the marker frame to the base frame
        theta_b = theta_m + self.mr_yaw

        # Rotate the marker-to-base offset into the world frame
        dx_m, dy_m, _dz = self.mr_to_base
        xb = xm + np.cos(theta_b) * dx_m - np.sin(theta_b) * dy_m
        yb = ym + np.sin(theta_b) * dx_m + np.cos(theta_b) * dy_m

        # ---- Step 3: build B_T_W (world -> base) as a planar rigid transform ----
        W_T_B = np.eye(4, dtype=np.float64)
        R = np.array(
            [
                [np.cos(theta_b), -np.sin(theta_b), 0.0],
                [np.sin(theta_b),  np.cos(theta_b), 0.0],
                [0.0,              0.0,             1.0],
            ],
            dtype=np.float64,
        )
        W_T_B[:3, :3] = R
        W_T_B[:3,  3] = np.array([xb, yb, 0.0], dtype=np.float64)

        self.B_T_W = _invert_homogeneous(W_T_B)
        return True

    # ------------------------------------------------------------------
    # Coordinate conversion helpers
    # ------------------------------------------------------------------

    def pixel_to_world(self, u: float, v: float) -> tuple:
        """Convert a pixel coordinate (u, v) to world (x, y) in metres."""
        if self.H_i2w is None:
            raise RuntimeError("Homography not ready — call update() first.")
        p = np.array([u, v, 1.0], dtype=np.float64)
        q = self.H_i2w @ p
        q /= q[2]
        return float(q[0]), float(q[1])

    def world_to_pixel(self, xW: float, yW: float) -> tuple:
        """Convert a world position (xW, yW) in metres to pixel (u, v)."""
        if self.H_w2i is None:
            raise RuntimeError("Homography not ready — call update() first.")
        p = np.array([xW, yW, 1.0], dtype=np.float64)
        q = self.H_w2i @ p
        q /= q[2]
        return int(q[0]), int(q[1])

    def z_lock_keyplane(
        self,
        xW_ref: float,
        yW_ref: float,
        z_des_base: float = -0.015,
    ) -> bool:
        """
        Adjust B_T_W so that world point (xW_ref, yW_ref, 0) maps to
        z_des_base in the robot base frame.

        This effectively 'locks' the keyboard plane to a known Z height,
        avoiding the need for full 3D camera pose estimation.

        Warning: mutates B_T_W in-place.  Call once per frame before IK.
        """
        if self.B_T_W is None:
            return False

        pW = np.array([xW_ref, yW_ref, 0.0, 1.0], dtype=np.float64)
        pB = self.B_T_W @ pW

        # Shift the Z translation so the reference point lands at z_des_base
        dz = float(z_des_base - pB[2])
        self.B_T_W[2, 3] += dz
        return True

    # ------------------------------------------------------------------
    # Display helper
    # ------------------------------------------------------------------

    def draw_markers(self, frame: np.ndarray) -> None:
        """Draw detected ArUco marker outlines on the display frame."""
        if self.last_ids is None:
            return
        cv2.aruco.drawDetectedMarkers(frame, self.last_corners, self.last_ids)
