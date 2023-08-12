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

Images are the the following format:
    loc-camid_step.png
Where:
    loc: location of the camera (see all 11 in CAMERA_LOCATIONS)
    camid: camera id ([[0, 9]]]])
    step: step in the sequence ([[0, 99]])
    
TODO:
    * Time IDs
    * Camera IDs
    * Camera Location selection (one or all?)
    * Train/Eval split (between cameras, must change splitting to be per camera, and wrt time)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path, PurePath, PosixPath
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
class HyperNeRFDataParserConfig(DataParserConfig):
    """hypernerf dataset config"""

    _target: Type = field(default_factory=lambda: HyperNeRF)
    """target class to instantiate"""
    data: Path = Path("data/hypernerf/chicken/")
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
    auto_scale_poses: bool = False
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_percentage: float = 0.95  # Corresponds to 105 cameras
    """The percent of images to use for training. The remaining images are for eval."""


@dataclass
class HyperNeRF(DataParser):
    """HyperNeRF DatasetParser"""

    config: HyperNeRFDataParserConfig
    downscale_factor: Optional[int] = None

    def __get_frame_metadata(self, fname: str) -> tuple[int, int]:
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

        # File format: cam1_000xyz.png
        fname_split = fname.split("_")
        cam_id = 0 if fname_split[0][:-1] == "left" else 1
        time_step = int(fname_split[-1].split(".")[0])

        return cam_id, time_step

    def _generate_dataparser_outputs(self, split="train"):
        # pylint: disable=too-many-statements

        data_dir = self.config.data

        image_filenames = []
        poses = []
        num_skipped_image_filenames = 0

        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []
        times = []

        cam_uids = []

        scene_dict = load_from_json(data_dir / "scene.json")
        center = np.array(scene_dict["center"], dtype=np.float32)
        scale = scene_dict["scale"]
        near = scene_dict["near"]
        far = scene_dict["far"]

        # Iterate over json files in data_dir/camera
        # write code
        for cam_json in (data_dir / "camera").glob("*.json"):
            frame = load_from_json(cam_json)
            rgb_path = data_dir / "rgb" / "1x" / (cam_json.name.split(".")[0] + ".png")
            fname = self._get_fname(rgb_path, data_dir)

            # Parse file name
            cam_id, time_step = self.__get_frame_metadata(fname.name)

            cam_uids.append(cam_id)
            times.append(time_step)

            assert "focal_length" in frame, "focal length not specified in frame"
            fx.append(float(frame["focal_length"]))
            fy.append(float(frame["focal_length"]))

            assert "principal_point" in frame, "principal point not specified in frame"
            cx.append(float(frame["principal_point"][0]))
            cy.append(float(frame["principal_point"][1]))

            assert "image_size" in frame, "resolution not specified in frame"
            width.append(int(frame["image_size"][0]))
            height.append(int(frame["image_size"][1]))

            assert "radial_distortion" in frame, "radial distortion not specified in frame"
            assert "tangential_distortion" in frame, "tangential distortion not specified in frame"
            """
            distort.append(camera_utils.get_distortion_params())
            """
            distort.append(
                camera_utils.get_distortion_params(
                    k1=float(frame["radial_distortion"][0]),
                    k2=float(frame["radial_distortion"][1]),
                    k3=float(frame["radial_distortion"][2]),
                    p1=float(frame["tangential_distortion"][0]),
                    p2=float(frame["tangential_distortion"][1]),
                )
            )

            image_filenames.append(fname)
            # frame["orientation"] = R (world to cam, w2c)
            # "position" = t
            """
            c2w = torch.as_tensor(frame["orientation"]).T
            c2w = c2w * (torch.eye(3) * torch.tensor([1, -1, -1]))  # switch cam coord x,y,z
            position = torch.as_tensor(frame["position"])
            position -= center  # some scenes look weird (wheel)
            position *= scale * self.config.scale_factor
            pose = np.zeros([3, 4])
            pose[:3, :3] = c2w
            pose[:3, 3] = position
            # from opencv coord to opengl coord (used by nerfstudio)
            pose[0:3, 1:3] *= -1  # switch cam coord x,y
            pose = pose[[1, 0, 2], :]  # switch world x,y
            pose[2, :] *= -1  # invert world z
            # for aabb bbox usage
            pose = pose[[1, 2, 0], :]  # switch world xyz to zxy
            """
            Rt = np.array(frame["orientation"]).T
            p = np.array(frame["position"])
            p -= center
            p *= scale * self.config.scale_factor
            pose = np.zeros([3, 4])
            pose[0, 0] = Rt[0, 0]
            pose[0, 1] = -Rt[0, 1]
            pose[0, 2] = -Rt[0, 2]
            pose[1, 0] = -Rt[1, 0]
            pose[1, 1] = Rt[1, 1]
            pose[1, 2] = Rt[1, 2]
            pose[2, 0] = -Rt[2, 0]
            pose[2, 1] = Rt[2, 1]
            pose[2, 2] = Rt[2, 2]
            pose[0, 3] = p[0]
            pose[1, 3] = -p[1]
            pose[2, 3] = -p[2]
            # pose[0:3, 1:3] *= -1  # switch cam coord x,y
            pose = pose[[1, 0, 2], :]  # switch world x,y
            pose[2, :] *= -1  # invert world z
            # for aabb bbox usage
            pose = pose[[1, 2, 0], :]  # switch world xyz to zxy
            poses.append(pose)

        if num_skipped_image_filenames >= 0:
            CONSOLE.log(f"Skipping {num_skipped_image_filenames} files in dataset split {split}.")
        assert (
            len(image_filenames) != 0
        ), """
        No image files found. 
        You should check the file_paths in the transforms.json file to make sure they are correct.
        """

        # Training:
        #   left and even time steps
        #   right and odd time steps
        # Evaluation:
        #   left and odd time steps
        #   right and even time steps
        indices = []
        for i in range(len(image_filenames)):
            cam_id = cam_uids[i]
            time_step = times[i]
            if split == "train":
                if cam_id == 0 and time_step % 2 == 0:
                    indices.append(i)
                if cam_id == 1 and time_step % 2 == 1:
                    indices.append(i)
            elif split in ["val", "test"]:
                if cam_id == 0 and time_step % 2 == 1:
                    indices.append(i)
                if cam_id == 1 and time_step % 2 == 0:
                    indices.append(i)
            """
            if split in ["val", "test"] and cam_id == 0 and time_step == 0:
                indices.append(i)
            elif split == "train":
                indices.append(i)
            """

        poses = torch.from_numpy(np.array(poses).astype(np.float32))

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        """
        poses[:, :3, 3] *= scale_factor
        """

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        poses = poses[indices]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        camera_type = CameraType.PERSPECTIVE
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        fx = torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = torch.tensor(width, dtype=torch.int32)[idx_tensor]
        times = torch.tensor(times, dtype=torch.float32)[idx_tensor] / max(times)  # Include time in Camera
        ids = torch.tensor(cam_uids, dtype=torch.float32)[idx_tensor]  # Include id in Camera
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
            dataparser_scale=scale_factor,
        )
        return dataparser_outputs

    def _get_fname(self, filepath: PurePath, data_dir: PurePath) -> PurePath:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxiliary image data, e.g. s

        filepath: the base file name of the transformations.
        data_dir: the directory of the data that contains the transform file
        downsample_folder_prefix: prefix of the newly generated downsampled images
        """

        if self.downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(data_dir / filepath)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) < MAX_AUTO_RESOLUTION:
                        break
                    if not (data_dir / f"{downsample_folder_prefix}{2**(df+1)}" / filepath.name).exists():
                        break
                    df += 1

                self.downscale_factor = 2**df
                CONSOLE.log(f"Auto image downscale factor of {self.downscale_factor}")
            else:
                self.downscale_factor = self.config.downscale_factor

        return data_dir / "rgb" / f"{self.downscale_factor}x" / filepath.name
