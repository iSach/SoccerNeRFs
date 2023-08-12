"""
Process depth data from Blender into a format that can be used by Nerfstudio
"""

import torch
import numpy as np
import os
import pathlib
import shutil
import PIL
from PIL import Image
import cv2
from tqdm import tqdm

device = "cuda"

data_dir = pathlib.PosixPath("/workspace/data/synth_paderborn_empty/")
depth_maps = "depth_field"
width = 1920
height = 1080

pbar = tqdm(total=len(os.listdir(data_dir / depth_maps)))
# Iterate over npz files in data_dir / "depth"
for npz_file in (data_dir / depth_maps).iterdir():
    # Check if npz
    if not npz_file.suffix == ".npz":
        continue

    # Load depth data
    depth_data = np.load(npz_file)
    depth_map = depth_data["dmap"]  # float32, meters

    # Convert 6.5504000e+04 to 0
    depth_map[depth_map > 65000] = 0

    # Convert to centimeters and uint16
    depth_map = (depth_map * 100).astype(np.uint16)

    # Save depth map
    # Write png instead of npz
    out_path = data_dir / "depth-maps_field" / (npz_file.stem + ".png")
    cv2.imwrite(str(out_path.absolute()), depth_map)
    pbar.update(1)
