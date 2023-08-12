"""
Manual computation of IST maps in a dataset without the need to run a model.
Useful for debugging and creating figures.
"""
from typing import Literal, Dict
import time

import torch
from torchtyping import TensorType

from nerfstudio.data.datasets.base_dataset import InputDataset

from nerfstudio.data.utils.data_utils import get_depth_image_from_path
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import PosixPath

file_path = "/workspace/data/synth_paderborn/images/2x/Left_%06d.png"
method = "kplanes"  # "kplanes" or "custom_mine"

online = False
online_ratio = 0.25
IST_TIME_RANGE = 1.0
isg = True
isg_gamma = 1e-1

if isg:
    map_folder = PosixPath(file_path).parent.parent / ("isg_maps_online" if online else "isg_maps_offline")
else:
    map_folder = PosixPath(file_path).parent.parent / ("ist_maps_online" if online else "ist_maps_offline")
map_folder.mkdir(exist_ok=True, parents=True)

# Load all images
batch = []
fnames = []
i = 0
print("Loading images...")
pbar = tqdm(total=-1, position=0, leave=True)
while PosixPath(file_path % i).exists():
    # while i < 10:
    path = PosixPath(file_path % i)
    img = Image.open(path)
    fnames.append(path.name)
    i += 1
    batch.append(np.array(img) / 255.0)
    pbar.update(1)
    pbar.unit = "images"


def compute_ist_map(img, t):
    """
    img: [H, W, C]
    """

    # Time differences
    time_diffs = np.abs(times - t)  # [N]

    # Filter by time difference > 0.01 to avoid comparing to the same image.
    # Filter by time difference < 0.25 to avoid comparing to images too far in the future.
    close_imgs = batch[np.where((time_diffs <= IST_TIME_RANGE) & (time_diffs > 0.01))[0]]  # [M, H, W, C]

    # Simulate online
    if online:
        idx = np.random.choice(len(close_imgs), int(len(close_imgs) * online_ratio), replace=False)
        close_imgs = close_imgs[idx]

    ist_map = np.zeros_like(img)

    # If no images are found, make a uniform map.
    if len(close_imgs) == 0:
        return np.ones_like(img)

    # Find image with biggest absolute difference among close images.
    max_diff_value = 0.0
    for close_img in close_imgs:
        diff = np.abs(img - close_img)  # [H, W, 3]
        current_max_diff_value = np.max(diff)  # [1]

        if current_max_diff_value <= 0.001:
            continue

        if method == "kplanes":
            ist_map = np.maximum(ist_map, diff)  # [H, W, 3]
        elif method == "custom_mine":
            if current_max_diff_value > max_diff_value:
                ist_map = diff
                max_diff_value = current_max_diff_value
        else:
            raise NotImplementedError(f"Unknown IST computing method: {computing_method}")

    ist_map = np.mean(ist_map, axis=2)  # [H, W]
    ist_map = (ist_map >= th) * ist_map

    return ist_map


def compute_isg_map(img, median_img):
    """
    img: [H, W, C]
    median_img: [H, W, C]
    """
    sq_residual = np.square(img - median_img)  # [H, W, C]
    psidiff = sq_residual / (sq_residual + isg_gamma**2)
    psidiff = (1.0 / 3) * np.sum(psidiff, axis=-1)
    psidiff[psidiff < 0.3] = 0
    return psidiff


# Value is pretty arbitrary, but the goal is to remove parasite areas such as
# camera shaking that makes the whole map non-zero.
# 0.25 seems good for Paderborn but too high for Stadium.
# 0.1 seems good, a bit too low for paderborn but better too low than too high.
th = 0.1

colormap = plt.get_cmap("turbo")

print("\nt is between 0 and", i - 1)

batch = np.array(batch)  # [N, H, W, C]
times = np.array([k / (i - 1) for k in range(i)])  # [N]

if isg:
    # Compute median:
    if online:
        # Do median on a subset of the images.
        idx = np.random.choice(len(batch), int(len(batch) * online_ratio), replace=False)
        median_img = np.median(batch[idx], axis=0)  # [H, W, C]
    else:
        median_img = np.median(batch, axis=0)  # [H, W, C]

pbar = tqdm(total=len(times))
for i, t in enumerate(times):
    if t < 0.56:
        continue
    img = batch[i]  # [H, W, C]

    if isg:
        weights_map = compute_isg_map(img, median_img)
    else:
        weights_map = compute_ist_map(img, t)

    # Save
    colored_ist_map = colormap(weights_map)[..., :3]
    side_by_side = np.concatenate((img, colored_ist_map), axis=1)
    pil_image = Image.fromarray((side_by_side * 255).astype(np.uint8))
    pil_image.save(map_folder / fnames[i])

    pbar.update(1)
