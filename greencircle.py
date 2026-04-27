"""
greencircle.py
==============
Green blob detector for end-effector visual tracking.

The robot's end-effector tip is wrapped with a green ring so the camera
can track its position.  A ring shape was chosen deliberately: unlike a
sticker on one face of the tip, a ring remains visible from a wide range
of viewing angles, which is important as the wrist rotates during motion.

Detection pipeline
------------------
  1. Convert the frame from BGR to HSV colour space.
  2. Threshold the HSV image between the configured lower and upper bounds
     to produce a binary mask.
  3. Apply morphological opening (remove small noise) then closing (fill
     small gaps in the ring contour).
  4. Find contours in the cleaned mask.
  5. Select the largest valid contour by area.
  6. Estimate the blob centre using a minimum enclosing circle.
"""

import cv2
import numpy as np


class GreenBlobDetector:
    """
    Detect the green end-effector marker in a BGR camera frame.

    Parameters
    ----------
    lower_hsv : (H, S, V) lower bound for the green threshold
    upper_hsv : (H, S, V) upper bound for the green threshold
    min_area  : minimum contour area in pixels (rejects tiny noise blobs)
    """

    def __init__(
        self,
        lower_hsv: tuple = (35, 70, 70),
        upper_hsv: tuple = (85, 255, 255),
        min_area: int = 6,
    ):
        self.lower   = np.array(lower_hsv, np.uint8)
        self.upper   = np.array(upper_hsv, np.uint8)
        self.min_area = int(min_area)

    def detect(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple:
        """
        Detect the green blob in a BGR frame.

        Returns
        -------
        (pixel_xy, mask) where pixel_xy is (int x, int y) or None if not found,
        and mask is the binary threshold image (useful for debugging).
        """
        # Step 1: convert colour space
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Step 2: threshold to isolate green pixels
        mask = cv2.inRange(hsv, self.lower, self.upper)

        # Step 3: morphological clean-up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Step 4: find contours
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, mask

        # Step 5: keep the largest contour above the minimum area threshold
        c    = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self.min_area:
            return None, mask

        # Step 6: estimate blob centre
        (x, y), _radius = cv2.minEnclosingCircle(c)
        return (int(x), int(y)), mask
