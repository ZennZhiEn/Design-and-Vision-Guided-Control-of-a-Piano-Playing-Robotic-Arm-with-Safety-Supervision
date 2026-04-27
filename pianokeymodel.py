"""
pianokeymodel.py
================
Geometric model of the piano keyboard in the world coordinate frame.

The world frame has its origin at the centre of ArUco marker ID10,
with the +x axis pointing toward marker ID11 along the keyboard.

Key layout
----------
Key index 0 is the leftmost playable key (closest to ID10).
Key pitch is fixed at 16.2 mm for this keyboard.
All 22 keys are modelled as point targets in the world plane.

These nominal key centres are used as the starting point for IK-based
coarse motion.  Per-key offset corrections (stored in KeyOffsetStore)
are then applied on top to account for calibration adjustments.
"""

import numpy as np


class PianoKeyModel:
    """
    Generates nominal key-centre coordinates in the world frame.

    Parameters
    ----------
    n_keys          : total number of playable keys
    x_id10_to_f1_m  : distance in metres from marker ID10 to the first key centre
    key_pitch_m     : centre-to-centre spacing between adjacent keys (metres)
    y_key_m         : world-frame Y position of the key row
    """

    def __init__(
        self,
        n_keys: int = 22,
        x_id10_to_f1_m: float = 0.037,
        key_pitch_m: float = 0.016,
        y_key_m: float = 0.0,
    ):
        self.n_keys = int(n_keys)
        self.x0     = float(x_id10_to_f1_m)   # x-offset to first key
        self.pitch  = float(key_pitch_m)        # key-to-key spacing
        self.y      = float(y_key_m)            # shared Y for all keys

    def key_world_xy(self, idx: int) -> tuple:
        """
        Return the (x, y) world-frame position of key at index idx.

        idx=0 is the first key (leftmost), idx=21 is the last.
        """
        x = self.x0 + idx * self.pitch
        return float(x), float(self.y)

    def all_keys_world_xy(self) -> list:
        """Return a list of (x, y) positions for all keys."""
        return [self.key_world_xy(i) for i in range(self.n_keys)]

    def __repr__(self) -> str:
        return (
            f"PianoKeyModel(n_keys={self.n_keys}, "
            f"x0={self.x0:.4f} m, pitch={self.pitch*1000:.1f} mm, "
            f"y={self.y:.4f} m)"
        )
