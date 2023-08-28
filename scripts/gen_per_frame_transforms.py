"""
Convert per-camera transforms to per-frame transforms.

The Blender script generates one entry per camera, but Nerfstudio
needs one for each frame file.
"""
import json
import os
from tqdm import tqdm
from pathlib import Path

data_folder = Path("/Users/sach/Library/Mobile Documents/com~apple~CloudDocs/Cours/PhD/soccernerfs_paper/data/stadiumwide/")
include_depth = False

# Read per-cam transforms file
# Specify encoding below:
with open(data_folder / "per_cam_transforms.json", "r", encoding="utf-8") as f:
    per_cam_transforms = json.load(f)
cam_transforms = {}
for cam_dict in per_cam_transforms["frames"]:
    cam_name = cam_dict["file_path"].split(".")[0]
    cam_transforms[cam_name] = cam_dict
frames = []

pbar = tqdm(total=len(os.listdir(data_folder / "images/2x/")), position=0, leave=True)
# Read frames files in "images/1x/" folder
for filename in os.listdir(data_folder / "images/2x/"):
    if filename.endswith(".png"):
        cam_name = filename.rsplit("_", 1)[0]
        frame_dict = cam_transforms[cam_name].copy()
        # images/hbg_000000.png
        frame_dict["file_path"] = "images/" + filename
        # depth-maps/hbg_000000-depth.png
        if include_depth:
            frame_dict["depth_file_path"] = "depth-maps/" + filename.replace(".png", "-depth.png")
        frames.append(frame_dict)
        pbar.update(1)
        pbar.set_description(f"Processing {filename}")

pbar.close()
print("Processed frames:", len(frames))

# Write frames to "transforms.json" file
with open(data_folder / "transforms.json", "w", encoding="utf-8") as f:
    json.dump({"frames": frames}, f, indent=4)
