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
Adapted data parser for dynamic nerfstudio-based dataset. 

EXPERIMENT 4: REAL DATA
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
    "hbg": 0,
    "main": 1,
    "left": 2,
    "right": 3,
}


@dataclass
class PaderbornDataParserConfig(DataParserConfig):
    """Paderborn dataset config"""

    _target: Type = field(default_factory=lambda: Paderborn)
    """target class to instantiate"""
    data: Path = Path("data/paderborn/")
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = 2
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "none"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "none"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_percentage: float = 1.0  # Corresponds to 105 cameras
    """The percent of images to use for training. The remaining images are for eval."""
    depth_unit_scale_factor: float = 0.01
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    depth_maps_to_use: Literal[
        "od", "od_below", "ist", "mask", "mask_below", "old_mask", "old_mask_below", "field"
    ] = "mask"
    """What depth map mask to use."""
    cap_box_floor: bool = True
    """Whether to use a rectangular scene box by setting floor to -0.01"""


@dataclass
class Paderborn(DataParser):
    """Paderborn DatasetParser"""

    config: PaderbornDataParserConfig
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

        for frame in meta["frames"]:
            filepath = PurePath(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)
            if not fname.exists():
                num_skipped_image_filenames += 1
                continue

            # Parse file name
            cam_id, time_step = self.__get_frame_metadata(fname)

            # Check for the camera location
            # Split fname until last hyphen
            """
            if fname_split != self.config.camera_location:
                continue
            """
            """
             if cam_id not in [5]:
                continue
            """
            """
            if frame_loc in ["Right-Ext Right", "Op Right-Op Middle"]:
                continue
            if frame_loc == "High Behind Right-Ext Op Right":
                if cam_id not in [0, 7]:
                    continue
            elif frame_loc in ["Left-Middle", "Op Left-Ext Op Left"]:
                if cam_id != 4:
                    continue
            else:
                if cam_id not in [0, 6]:
                    continue
            """
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

            if "depth_file_path" in frame:
                depthfilepath = frame["depth_file_path"].replace(
                    "depth-maps", "depth-maps-" + self.config.depth_maps_to_use
                )
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

        # Select frame indices that correspond to the selected cameras
        indices = []
        for i in range(len(image_filenames)):
            # 0 * 201 for hbg
            # 1 * 201 for left
            # 2 * 201 for main
            # 3 * 201 for right
            eval_is = [
                0 * 201 + 0,
                1 * 201 + 10,
                2 * 201 + 20,
                3 * 201 + 30,
                0 * 201 + 40,
                1 * 201 + 50,
                2 * 201 + 60,
                3 * 201 + 70,
                0 * 201 + 80,
                1 * 201 + 90,
                2 * 201 + 100,
                3 * 201 + 110,
                0 * 201 + 120,
                1 * 201 + 130,
                2 * 201 + 140,
                3 * 201 + 150,
                0 * 201 + 160,
                1 * 201 + 170,
                2 * 201 + 180,
                3 * 201 + 190,
            ]
            if i in eval_is and split in ["test", "val"]:
                indices.append(i)
            elif i not in eval_is and split in ["train"]:
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
            scene_box = SceneBox(
                aabb=torch.tensor(
                    [[-aabb_scale, -aabb_scale, -0.01], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
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
        times = torch.tensor(times, dtype=torch.float32)[idx_tensor] / max(times)  # Include time in Camera
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
