# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" 
Scene 2: Broadcast-style views
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Optional, Type

import numpy as np
import torch
from PIL import Image
from rich.console import Console
from typing_extensions import Literal

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json

CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 1600

CAM_IDS = {
    "Camera_1": 0,
    "Camera_2": 1,
    "Camera_3": 2,
    "Camera_4": 3,
    "Camera_5": 4,
    "Camera_6": 5,
    "Camera_7": 6,
    "Camera_8": 7,
    "Camera_9": 8,
    "Camera_10": 9,
    "Camera_11": 10,
    "Camera_12": 11,
    "Camera_13": 12,
    "Camera_14": 13,
    "Camera_15": 14,
    "Camera_16": 15,
    "Camera_17": 16,
    "Camera_18": 17,
    "Camera_19": 18,
    "Camera_20": 19,
}

"""
List of possible train/eval split setups.
"""
SETUPS = {
    # Real data setup, 4 cameras.
    # Eval on 3 cameras covered by training, and 1 not.
    # Train: 804.
    # Val: 804.
    "real": {
        "train": [
            "HBG",
            "Left",
            "Right",
            "Main",
        ],
        "eval": [
            "Inter_1",
        ],
    },
    # Real data setup + opposite corresponding cameras.
    # Eval cameras around field.
    "real+opp": {
        "train": [
            "HBG",
            "Left",
            "Right",
            "Main",
            "HBG_opp",
            "Left_opp",
            "Right_opp",
            "Main_opp",
            "Inter_4",
            "Inter_6",
            "Inter_7",
            "Inter_9",
            "Inter_11",
        ],
        "eval": [
            "Inter_1",
            "Inter_2",
            "Inter_3",
            "Inter_5",
            "Inter_8",
        ],
    },
    # Most low cameras, and 1 for eval.
    "low": {
        "train": [
            "HBG",
            "Left",
            "Right",
            "Main",
            "HBG_opp",
            "Left_opp",
            "Right_opp",
            "Main_opp",
            "Inter_1",
            "Inter_2",
            "Inter_3",
            "Inter_4",
            "Inter_5",
            "Inter_6",
            "Inter_7",
            "Inter_9",
            "Inter_10",
            "Inter_11",
            "Inter_12",
        ],
        "eval": [
            "Inter_8",
        ],
    },
    # Train on all high cameras, eval on some low cameras.
    "global": {
        "train": [
            "global_1",
            "global_2",
            "global_3",
            "global_4",
            "global_5",
            "global_6",
            "global_7",
            "global_8",
        ],
        "eval": [
            "Inter_2",
            "Inter_5",
            "Inter_8",
            "Inter_11",
        ],
    },
    "all": {
        "train": [
            "Camera_1",
            "Camera_2",
            "Camera_3",
            "Camera_4",
            "Camera_5",
            "Camera_6",
            "Camera_7",
            "Camera_8",
            "Camera_9",
            "Camera_10",
            "Camera_11",
            "Camera_12",
            "Camera_13",
            "Camera_14",
            "Camera_15",
            "Camera_16",
            "Camera_17",
            "Camera_18",
            "Camera_19",
        ],
        "eval": [
            "Camera_20",
        ],
    },
}


@dataclass
class BroadcaststyleDataParserConfig(DataParserConfig):
    """Broadcast-Style dataset config"""

    _target: Type = field(default_factory=lambda: Broadcaststyle)
    """target class to instantiate"""
    data: Path = Path("data/broadcaststyle/")
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = 2
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.5
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "none"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "none"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    depth_unit_scale_factor: float = 0.01
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    depth_maps: Literal["depth-maps", "depth-maps_field", "none"] = "none"
    """Depth maps to use. Default is full, but can also use field only."""
    depth_mask: Literal["none", "od", "od_below", "ist", "mask", "mask_below", "field"] = "mask"
    """Which depth maps mask to use."""
    cam_split_setup: Literal["real", "real+opp", "low", "global", "all"] = "all"
    """Which setup to use for train/eval split."""
    cap_box_floor: bool = False
    """Whether to use a rectangular scene box by setting floor to -0.01"""
    static: bool = False
    """Whether to use static views."""
    static_allimgs: bool = False
    """If static&empty(step=-1), whether or not to use time indices. If false, only use first images (1 per cam)."""
    static_timestep: int = -1
    """If static, which time step to use. If -1, use empty field"""
    fps_downsample: float = 3.0
    """How much to downsample the fps by. 1.0 is no downsample, 2.0 is half the fps."""


@dataclass
class Broadcaststyle(DataParser):
    """Broadcaststyle DatasetParser"""

    config: BroadcaststyleDataParserConfig
    downscale_factor: Optional[int] = None

    def __get_frame_metadata(self, fname: Path) -> tuple[int, int]:
        """
        Extracts the frame metadata from the frame file.

        Args:
            fname (Path): The file's path to extract the metadata from.

        Returns:
            frame_loc (str): The location of the camera.
            cam_id (int): The camera ID in the cluster. (0-9)
            cam_global_id (int): The unique camera ID. (0-109)
            time_step (int): The time step. (0-99)
        """
        fname_split = fname.name.rsplit("_", 1)
        cam_id = int(CAM_IDS[fname_split[0]])
        time_step = int(fname_split[1].split(".")[0])

        return cam_id, time_step

    def _generate_dataparser_outputs(self, split="train"):
        # pylint: disable=too-many-statements

        if self.config.static and self.config.static_timestep == -1:
            self.config.data = self.config.data.parent / "broadcaststyle_empty/"

        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
            data_dir = self.config.data.parent
        else:
            meta = load_from_json(self.config.data / "transforms.json")
            data_dir = self.config.data

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []
        num_skipped_image_filenames = 0

        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
            if distort_key in meta:
                distort_fixed = True
                break
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []
        times = []

        cam_uids = []

        setup_split = "train" if split == "train" else "eval"
        other_split = "eval" if setup_split == "train" else "train"
        split_cams = SETUPS[self.config.cam_split_setup][setup_split]
        split_cams = [CAM_IDS[cam] for cam in split_cams]
        other_split_cams = SETUPS[self.config.cam_split_setup][other_split]
        other_split_cams = [CAM_IDS[cam] for cam in other_split_cams]

        for frame in meta["frames"]:
            filepath = PurePath(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)
            if not fname.exists():
                num_skipped_image_filenames += 1
                continue

            # Parse file name
            cam_id, time_step = self.__get_frame_metadata(fname)

            if cam_id not in split_cams and cam_id not in other_split_cams:
                # Better camera scaling when ignoring other cameras (global here)
                continue

            if self.config.static and not self.config.static_allimgs:
                if self.config.static_timestep == -1:
                    if time_step != 0:
                        continue
                elif time_step != self.config.static_timestep:
                    continue

            cam_uids.append(cam_id)
            # Append camera time
            times.append(time_step)

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx.append(float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy.append(float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx.append(float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy.append(float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            if not distort_fixed:
                distort.append(
                    camera_utils.get_distortion_params(
                        k1=float(frame["k1"]) if "k1" in frame else 0.0,
                        k2=float(frame["k2"]) if "k2" in frame else 0.0,
                        k3=float(frame["k3"]) if "k3" in frame else 0.0,
                        k4=float(frame["k4"]) if "k4" in frame else 0.0,
                        p1=float(frame["p1"]) if "p1" in frame else 0.0,
                        p2=float(frame["p2"]) if "p2" in frame else 0.0,
                    )
                )

            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            if "mask_path" in frame:
                mask_filepath = PurePath(frame["mask_path"])
                mask_fname = self._get_fname(
                    mask_filepath,
                    data_dir,
                    downsample_folder_prefix="masks_",
                )
                mask_filenames.append(mask_fname)

            if "depth_file_path" in frame and self.config.depth_maps != "none":
                depth_mask = self.config.depth_mask
                depthfilepath = frame["depth_file_path"]
                if depth_mask != "none":
                    depthfilepath = depthfilepath.replace("depth-maps", "depth-maps-" + depth_mask)
                if self.config.depth_maps != "depth-maps":
                    depthfilepath = depthfilepath.replace("depth-maps", self.config.depth_maps)
                depth_filepath = PurePath(depthfilepath)
                depth_fname = self._get_fname(depth_filepath, data_dir, downsample_folder_prefix="depths_")
                depth_filenames.append(depth_fname)

        if num_skipped_image_filenames >= 0:
            CONSOLE.log(f"Skipping {num_skipped_image_filenames} files in dataset split {split}.")
        assert (
            len(image_filenames) != 0
        ), """
        No image files found. 
        You should check the file_paths in the transforms.json file to make sure they are correct.
        """
        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (
            len(depth_filenames) == len(image_filenames)
        ), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """

        # Filter FPS downsample
        times_filter = np.arange(max(times) + 1)
        if self.config.fps_downsample > 1:
            base_duration = max(times) + 1  # Starts at 0
            new_duration = int(base_duration / self.config.fps_downsample)
            times_filter = np.linspace(0, base_duration - 1, new_duration).astype(np.int32)

        # Select frame indices that correspond to the selected cameras
        indices = []
        for i in range(len(image_filenames)):
            if cam_uids[i] in split_cams and times[i] in times_filter:
                indices.append(i)

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []
        poses = poses[indices]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        if self.config.cap_box_floor:
            # 0 gives artifacts.
            scene_box = SceneBox(
                aabb=torch.tensor(
                    [[-aabb_scale, -aabb_scale, -0.1], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
                )
            )
        else:
            scene_box = SceneBox(
                aabb=torch.tensor(
                    [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
                )
            )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        fx = float(meta["fl_x"]) if fx_fixed else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = float(meta["fl_y"]) if fy_fixed else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = float(meta["cx"]) if cx_fixed else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = float(meta["cy"]) if cy_fixed else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = int(meta["h"]) if height_fixed else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = int(meta["w"]) if width_fixed else torch.tensor(width, dtype=torch.int32)[idx_tensor]
        if max(times) != 0:
            times = torch.tensor(times, dtype=torch.float32)[idx_tensor] / max(times)  # Include time in Camera
        else:
            times = torch.tensor(times, dtype=torch.float32)[idx_tensor]
        ids = torch.tensor(cam_uids, dtype=torch.float32)[idx_tensor]  # Include id in Camera
        if distort_fixed:
            distortion_params = camera_utils.get_distortion_params(
                k1=float(meta["k1"]) if "k1" in meta else 0.0,
                k2=float(meta["k2"]) if "k2" in meta else 0.0,
                k3=float(meta["k3"]) if "k3" in meta else 0.0,
                k4=float(meta["k4"]) if "k4" in meta else 0.0,
                p1=float(meta["p1"]) if "p1" in meta else 0.0,
                p2=float(meta["p2"]) if "p2" in meta else 0.0,
            )
        else:
            distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        """
        For pre-training on static scenes, we should intuitively not give any time step information to the network.
        However, K-Planes will then not initialize time planes, therefore instead it is preferable to give time
        in order to have 6 planes, but freeze them (model option).
        """
        # if not self.config.static or (self.config.static and self.config.static_timestep == -1 and self.config.static_allimgs):
        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
            times=times,
            ids=ids,
        )

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                "static": self.config.static,
            },
        )
        return dataparser_outputs

    def _get_fname(self, filepath: PurePath, data_dir: PurePath, downsample_folder_prefix="images_") -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxiliary image data, e.g. masks

        filepath: the base file name of the transformations.
        data_dir: the directory of the data that contains the transform file
        downsample_folder_prefix: prefix of the newly generated downsampled images
        """

        self.downscale_factor = self.config.downscale_factor

        old_path = data_dir / filepath
        fname = old_path.name

        # Now supports downsampling
        new_path = old_path.parent / f"{self.config.downscale_factor}x" / fname

        return new_path
