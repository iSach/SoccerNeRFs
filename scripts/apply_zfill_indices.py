"""
Replaces the indices of the files in the given folder with indices padded with zeros.
"""
import os
from pathlib import Path

folder = Path("1x")

for filename in os.listdir(folder):
    # cam_step.png -> cam_step.png with zfill
    if filename.endswith(".png"):
        cam = filename.rsplit("_", 1)[0]
        if "depth" in filename:
            step = filename.rsplit("_", 1)[1].split("-depth.")[0]
            new_filename = cam + "_" + str(step).zfill(6) + "-depth.png"
        else:
            step = filename.rsplit("_", 1)[1].split(".")[0]
            new_filename = cam + "_" + str(step).zfill(6) + ".png"
        os.rename(folder / filename, folder / new_filename)
