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


@dataclass
class DynamicDataParserConfig(DataParserConfig):
    """Generic dynamic dataset config"""

    _target: Type = field(default_factory=lambda: Dynamic)
    """target class to instantiate"""
    data: Path = Path("data/dyn3dprinter/")
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = 1
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.5  # let some space for the bleachers :^)
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "focus"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_percentage: float = 1.0
    """The percent of images to use for training. The remaining images are for eval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    static: bool = False
    """Whether to use the static dataset instead of the dynamic dataset."""
    static_timestep: int = 0
    """The timestep to use for the static dataset."""


@dataclass
class Dynamic(DataParser):
    """Generic dynamic DatasetParser"""

    config: DynamicDataParserConfig
    downscale_factor: Optional[int] = None

    def __get_frame_metadata(self, fname: Path) -> tuple[int, int]:
        """
        Extracts the frame metadata from the frame file.

        camid_step.png

        Args:
            fname (Path): The file's path to extract the metadata from.

        Returns:
            cam_id (int): The camera ID.
            time_step (int): The time step. (0-99)
        """
        cam_id = int(fname.stem.split("_")[0])
        time_step = int(fname.stem.split("_")[1]) - 1

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

        cam_ids = []

        for frame in meta["frames"]:
            filepath = PurePath(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)
            if not fname.exists():
                num_skipped_image_filenames += 1
                continue

            # Parse file name
            cam_id, time_step = self.__get_frame_metadata(fname)

            if self.config.static and time_step != self.config.static_timestep:
                continue

            if cam_id == 13:
                continue

            # Check for the camera location
            # Split fname until last hyphen
            cam_ids.append(cam_id)

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
                depth_filepath = PurePath(frame["depth_file_path"])
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

        # filter image_filenames and poses based on train/eval split percentage
        num_cams = len(np.unique(cam_ids))
        num_train_cams = math.ceil(num_cams * self.config.train_split_percentage)
        num_eval_cams = num_cams - num_train_cams
        i_all = np.arange(num_cams) + 1
        i_train = np.linspace(0, num_cams - 1, num_train_cams, dtype=int) + 1  # equally spaced training cams.
        i_eval = np.setdiff1d(i_all, i_train)  # eval cams are the remaining cams
        assert len(i_eval) == num_eval_cams
        if split == "train":
            cam_indices = i_train
        elif split in ["val", "test"]:
            cam_indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        # Select frame indices that correspond to the selected cameras
        indices = []
        for i in range(len(image_filenames)):
            if cam_ids[i] in cam_indices:
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
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        print("Camera Type:", camera_type)

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        fx = float(meta["fl_x"]) if fx_fixed else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = float(meta["fl_y"]) if fy_fixed else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = float(meta["cx"]) if cx_fixed else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = float(meta["cy"]) if cy_fixed else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = int(meta["h"]) if height_fixed else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = int(meta["w"]) if width_fixed else torch.tensor(width, dtype=torch.int32)[idx_tensor]
        times = torch.tensor(times, dtype=torch.float32)[idx_tensor] / max(times)  # Include time in Camera
        ids = torch.tensor(cam_ids, dtype=torch.float32)[idx_tensor]  # Include id in Camera
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

        if self.downscale_factor > 1:
            return data_dir / f"{downsample_folder_prefix}{self.downscale_factor}" / filepath.name
        return data_dir / filepath.with_suffix(".png")
