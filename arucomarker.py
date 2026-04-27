"""
arucomarker.py
==============
Utility script for generating the ArUco marker images and PDF sheet
used in the physical robot workspace.

Run this script once to produce the printed marker sheet:
    python arucomarker.py

Output
------
  aruco_40mm/aruco_10.png   — keyboard left marker (world origin)
  aruco_40mm/aruco_11.png   — keyboard right marker (defines +X axis)
  aruco_40mm/aruco_20.png   — base marker (attached near robot)
  aruco_40mm.pdf            — printable A4 sheet with all three markers

Print at exactly 100% scale.  Each marker should measure 40 mm per side.
Verify with a ruler before deploying — scale errors shift the world frame.

Note: this file only needs to be run once.  The generated PNG files
are already included in the aruco_40mm/ directory.
"""

import os
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader

# ---- Settings ----
DICT       = cv2.aruco.DICT_4X4_50
MARKER_IDS = [10, 11, 20]
MARKER_MM  = 40      # printed side length in millimetres
PNG_PX     = 1000    # PNG image resolution in pixels
OUT_DIR    = "aruco_40mm"
PDF_NAME   = "aruco_40mm.pdf"

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    # NOTE: Dictionary_get() is the OpenCV <4.7 API.
    # In OpenCV >=4.7 use cv2.aruco.getPredefinedDictionary() instead.
    aruco_dict = cv2.aruco.Dictionary_get(DICT)

    # Generate and save PNG images for each marker
    png_paths = []
    for marker_id in MARKER_IDS:
        img  = cv2.aruco.drawMarker(aruco_dict, marker_id, PNG_PX)
        path = os.path.join(OUT_DIR, f"aruco_{marker_id}.png")
        cv2.imwrite(path, img)
        png_paths.append((marker_id, path))
        print(f"Saved {path}")

    # Lay out all markers on an A4 PDF sheet for printing
    c = canvas.Canvas(PDF_NAME, pagesize=A4)
    page_w, page_h = A4

    margin = 15 * mm
    size   = MARKER_MM * mm
    gap    = 15 * mm

    x = margin
    y = page_h - margin - size

    for marker_id, path in png_paths:
        c.drawImage(ImageReader(path), x, y, width=size, height=size, mask="auto")
        c.setFont("Helvetica", 10)
        c.drawString(x, y - 4 * mm, f"DICT_4X4_50  ID={marker_id}  SIZE={MARKER_MM}mm")

        x += size + gap
        if x + size > page_w - margin:
            x  = margin
            y -= size + 25 * mm

    c.showPage()
    c.save()
    print(f"\nSaved {PDF_NAME} and PNGs in {OUT_DIR}/")
    print("Print at 100% scale — verify each marker measures exactly 40 mm per side.")
