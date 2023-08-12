"""
The empty stadium scene follows the dynamic format and requires a frame per time step.
This code simply duplicates static images T times.
"""
import os
from pathlib import Path
from tqdm import tqdm

folder = Path("/workspace/data/synth_paderborn_empty/depth_full/")

t_max = 201

pbar = tqdm(total=t_max * len(os.listdir(folder)))
for filename in os.listdir(folder):
    # cam_step.png -> cam_step.png with zfill
    cam = filename.rsplit("_", 1)[0]
    if "depth" in filename:
        step = filename.rsplit("_", 1)[1].split("-depth.")[0]
    else:
        step = filename.rsplit("_", 1)[1].split(".")[0]
    pbar.update(1)
    for t in range(1, 201):
        new_filename = filename.replace(step, str(t).zfill(6))
        # Copy file
        os.system(f"cp {folder}/{filename} {folder}/{new_filename}")
        pbar.update(1)
