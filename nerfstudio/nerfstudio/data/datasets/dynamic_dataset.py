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
dynamic dataset.
"""

from typing import Literal, Dict
import time

import torch
import torchvision
from torchtyping import TensorType

from nerfstudio.data.datasets.base_dataset import InputDataset

from nerfstudio.data.utils.data_utils import get_depth_image_from_path
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


DEBUG_IST_MAPS = False
# Save ist maps every 10 iterations
DEBUG_FREQUENCY = 10


class DynamicDataset(InputDataset):
    """Dataset that stores importance sampling weights.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """

    def __init__(
        self,
        dataparser_outputs: DataparserOutputs,
        scale_factor: float = 1.0,
        use_importance_sampling: bool = True,
        is_pixel_ratio: float = 0.03,
        ist_range: float = 0.25,
        iters_to_start_is: int = 2000,  # Could be replaced by automatic static convergence detection.
        isg: bool = False,
        isg_gamma: float = 5e-2,
        eval_dataset: bool = False,  # Disable ist weights for eval dataset, it's not needed.
        pick_mode: Literal["normal", "randsteps", "lowfps"] = "randsteps",
    ):
        super().__init__(dataparser_outputs, scale_factor)

        self.use_importance_sampling = use_importance_sampling
        self.is_pixel_ratio = is_pixel_ratio
        self.ist_range = ist_range
        self.isg = isg
        self.isg_gamma = isg_gamma
        self.iters_to_start_ist = iters_to_start_is
        self.eval_dataset = eval_dataset
        self.pick_mode = pick_mode

        self.depth_enabled = False
        if (
            "depth_filenames" in dataparser_outputs.metadata.keys()
            and dataparser_outputs.metadata["depth_filenames"] is not None
        ):
            self.depth_filenames = self.metadata["depth_filenames"]
            self.depth_unit_scale_factor = self.metadata["depth_unit_scale_factor"]
            self.depth_enabled = True

    def get_metadata(self, data: Dict) -> Dict:
        if not self.depth_enabled:
            return {}

        filepath = self.depth_filenames[data["image_idx"]]
        height = int(self._dataparser_outputs.cameras.height[data["image_idx"]])
        width = int(self._dataparser_outputs.cameras.width[data["image_idx"]])

        # Scale depth images to meter units and also by scaling applied to cameras
        scale_factor = self.depth_unit_scale_factor * self._dataparser_outputs.dataparser_scale
        depth_image = get_depth_image_from_path(
            filepath=filepath, height=height, width=width, scale_factor=scale_factor
        )

        return {"depth_image": depth_image}

    def compute_is(
        self,
        batch,
        device,
        offline=False,
    ) -> TensorType["batch_size", "image_width", "image_height"]:
        if "static" in self._dataparser_outputs.metadata and self._dataparser_outputs.metadata["static"]:
            return self.compute_static_is(batch, device, offline=offline)

        if self.isg:
            return self.compute_isg(batch, device, offline=offline)

        return self.compute_ist(batch, device, offline=offline)

    @torch.no_grad()
    def compute_static_is(
        self,
        batch,
        device,
        offline=False,
    ) -> TensorType["batch_size", "image_width", "image_height"]:
        """
        Compute Importance Sampling for static images.
        For simplicity, this simply loads RetinaNet, detects players/balls and use the square directly.

        Args:
            batch: batch of data, with images and image indices.

        Returns:
            ISS weights, a BxHxW tensor that contains for each image
                         of the batch the weights for each pixel considering
                         dynamic neibourghing content.
        """

        # Image shape
        N, H, W = batch["image"].shape[:3]
        N, H, W = str(N), str(H), str(W)
        split_str = "eval" if self.eval_dataset else "train"
        file_name = f"iss-weights-{split_str}-{N}-{H}p.pt"

        if offline:
            print("[ISS]: Checking for pre-computed weights.")
            # Check for pre-computed weights file (images folder (for the given resolution) / ist_weights.npy)
            weights_file = self._dataparser_outputs.image_filenames[0].absolute().parent / file_name
            if weights_file.exists() and not DEBUG_IST_MAPS:
                print("[ISS]: Loading pre-computed weights...")
                weights = torch.load(weights_file)
                # Check shape in case train/eval splits were changed.
                if weights.shape[0] != batch["image"].shape[0]:
                    del weights
                    torch.cuda.empty_cache()
                    print("Invalid pre-computed weights shape. Recomputing...")
                else:
                    return weights
            else:
                print("[ISS]: No pre-computed weights found. Pre-computing...")

        # Load RetinaNet
        retina = (
            torchvision.models.detection.retinanet_resnet50_fpn_v2(
                weights=torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
            )
            .eval()
            .to("cuda")
        )

        img_idx = batch["image_idx"]
        image_height = batch["image"].size(1)
        image_width = batch["image"].size(2)
        batch_size = batch["image"].size(0)

        iss_weights = torch.zeros((batch_size, image_height, image_width), device=device)

        batch_range = range(batch_size)

        if DEBUG_IST_MAPS:
            maps_folder = self._dataparser_outputs.image_filenames[0].absolute().parent.parent / "iss_maps"
            maps_folder.mkdir(exist_ok=True)
            colormap = plt.get_cmap("turbo")

        show_progress = offline or DEBUG_IST_MAPS
        if show_progress:
            batch_range = tqdm(batch_range)

        for i, image in enumerate(batch["image"]):
            # Detect players/balls
            res = retina(image.permute(2, 0, 1).unsqueeze(0).to("cuda"))[0]

            # Filter by categories: only 1 and 37 (person and ball)
            # Filter by score: only > 0.6
            indices = torch.where(((res["labels"] == 1) | (res["labels"] == 37)) & (res["scores"] > 0.6))[0]
            res["boxes"] = res["boxes"][indices]

            # Compute weights
            weights = torch.zeros((image_height, image_width)).to("cuda")
            for box in res["boxes"]:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                weights[y1:y2, x1:x2] = 1.0
            iss_weights[i] = weights

            if DEBUG_IST_MAPS:
                colored_ist_map = colormap(weights.cpu().numpy())[..., :3]
                img = batch["image"][i].cpu().numpy()
                side_by_side = np.concatenate((img, colored_ist_map), axis=1)
                pil_image = Image.fromarray((side_by_side * 255).astype(np.uint8))
                img_name = self._dataparser_outputs.image_filenames[img_idx[i]].name
                pil_image.save(maps_folder / img_name)

            if show_progress:
                batch_range.update()

        # Save weights
        if offline:
            print("[IST]: Saving pre-computed weights.")
            torch.save(iss_weights, weights_file)

    def compute_isg(
        self,
        batch,
        device,
        offline=False,
    ) -> TensorType["batch_size", "image_width", "image_height"]:
        """
        Compute Importance Sampling based on Global median (ISG) weights.

        Args:
            batch: batch of data, with images and image indices.

        Returns:
            ISG weights, a BxHxW tensor that contains for each image
                         of the batch the weights for each pixel considering
                         dynamic neibourghing content.
        """

        # Image shape
        N, H, W = batch["image"].shape[:3]
        N, H, W = str(N), str(H), str(W)
        split_str = "eval" if self.eval_dataset else "train"
        file_name = f"isg-weights-{self.isg_gamma}-{split_str}-{N}-{H}p.pt"

        if offline:
            print("[ISG]: Checking for pre-computed weights.")
            # Check for pre-computed weights file (images folder (for the given resolution) / ist_weights.npy)
            weights_file = self._dataparser_outputs.image_filenames[0].absolute().parent / file_name
            if weights_file.exists() and not DEBUG_IST_MAPS:
                print("[ISG]: Loading pre-computed weights...")
                weights = torch.load(weights_file)
                # Check shape in case train/eval splits were changed.
                if weights.shape[0] != batch["image"].shape[0]:
                    del weights
                    torch.cuda.empty_cache()
                    print("Invalid pre-computed weights shape. Recomputing...")
                else:
                    return weights
            else:
                print("[ISG]: No pre-computed weights found. Pre-computing...")

        show_progress = offline or DEBUG_IST_MAPS

        if self.cameras.times is None:
            return None

        img_idx = batch["image_idx"]
        image_height = batch["image"].size(1)
        image_width = batch["image"].size(2)
        image_channels = batch["image"].size(3)
        batch_size = batch["image"].size(0)

        # Corresponding IDs of the cameras for each batch image.
        cam_ids = self.cameras.ids[img_idx]  # [batch_size,]

        isg_weights = torch.zeros((batch_size, image_height, image_width), device=device)

        batch_range = range(batch_size)

        if show_progress:
            batch_range = tqdm(batch_range)

        if DEBUG_IST_MAPS:
            maps_folder = self._dataparser_outputs.image_filenames[0].absolute().parent.parent / "isg_maps"
            maps_folder.mkdir(exist_ok=True)
            debug_count = 0
            colormap = plt.get_cmap("turbo")

        # Pre-compute cameras medians.
        # Find all cameras as list
        all_cam_ids = torch.unique(cam_ids)
        cam_medians = dict()
        for i, cam_id in enumerate(all_cam_ids):
            # Find all images from this camera
            cam_imgs = torch.where(cam_ids == cam_id)[0]
            # Compute median
            cam_medians[cam_id.item()] = torch.median(batch["image"][cam_imgs], dim=0).values.to(device)

        # Compare to the same camera about 25 frames later
        for i in batch_range:
            # Images with the same camera in the batch.
            cam_id = cam_ids[i].item()
            cam_median = cam_medians[cam_id]

            sq_residual = torch.square(batch["image"][i].to(device) - cam_median)  # [H, W, C]
            psidiff = sq_residual.div_(sq_residual + (self.isg_gamma) ** 2)
            psidiff = (1.0 / 3) * torch.sum(psidiff, dim=-1)

            isg_weights[i] = psidiff

            if DEBUG_IST_MAPS:
                if debug_count % 10 == 0:
                    colored_ist_map = colormap(psidiff.cpu().numpy())[..., :3]
                    img = batch["image"][i].cpu().numpy()
                    side_by_side = np.concatenate((img, colored_ist_map), axis=1)
                    pil_image = Image.fromarray((side_by_side * 255).astype(np.uint8))
                    img_name = self._dataparser_outputs.image_filenames[img_idx[i]].name
                    pil_image.save(maps_folder / img_name)
                    debug_count = 0
                debug_count += 1

        # convert it to take less memory:
        isg_weights = isg_weights.to(torch.float16)
        # multinomial is not supported for torch.float16 (half-precision) on CPU
        # but CPU way too slow for multinomiaL...

        if offline:
            print("[IST]: Saving pre-computed weights.")
            torch.save(isg_weights, weights_file)

        # print(f"IST weights computed in {time.time() - start:.2f} seconds.")
        return isg_weights

    def compute_ist(
        self,
        batch,
        device,
        offline=False,
    ) -> TensorType["batch_size", "image_width", "image_height"]:
        """
        Compute Importance Sampling based on Temporal difference (IST) weights.

        This is done on-line for convenience, but it is not the best way to do it.
        Indeed, it will lack adjacent frames information (due to frames missing in the batch)
        but it's faster and way more convenient.
        It works well in practice and is thus kept this way.

        NB: Can't do this in get_metadata because get_metadata does not give access
        to the rest of the batch... unfortunately.

        It is not automatic and thus not very practical, but it could be an inspiration to
        include IST in Nerfstudio, by then loading the IST weights in the data parser per camera id
        and passing them through the metadata.

        Args:
            batch: batch of data, with images and image indices.

        Returns:
            IST weights, a BxHxW tensor that contains for each image
                         of the batch the weights for each pixel considering
                         dynamic neibourghing content.
        """

        # Image shape
        N, H, W = batch["image"].shape[:3]
        N, H, W = str(N), str(H), str(W)
        split_str = "eval" if self.eval_dataset else "train"
        ist_range = str(self.ist_range).replace(".", "_")
        file_name = f"ist-weights-{ist_range}-{split_str}-{N}-{H}p.pt"

        if offline:
            print("[IST]: Checking for pre-computed weights.")
            # Check for pre-computed weights file (images folder (for the given resolution) / ist_weights.npy)
            weights_file = self._dataparser_outputs.image_filenames[0].absolute().parent / file_name
            if weights_file.exists() and not DEBUG_IST_MAPS:
                print("[IST]: Loading pre-computed weights...")
                weights = torch.load(weights_file)
                # Check shape in case train/eval splits were changed.
                if weights.shape[0] != batch["image"].shape[0]:
                    del weights
                    torch.cuda.empty_cache()
                    print("Invalid pre-computed weights shape. Recomputing...")
                else:
                    return weights
            else:
                print("[IST]: No pre-computed weights found. Pre-computing...")

        show_progress = offline or DEBUG_IST_MAPS

        if self.cameras.times is None:
            return None

        # print("Computing Importance Sampling weights...")
        start = time.time()

        img_idx = batch["image_idx"]
        image_height = batch["image"].size(1)
        image_width = batch["image"].size(2)
        batch_size = batch["image"].size(0)

        # Corresponding times of the images for each batch image.
        cam_times = self.cameras.times[img_idx]  # [batch_size, 1]
        # Corresponding IDs of the cameras for each batch image.
        cam_ids = self.cameras.ids[img_idx]  # [batch_size, 1]

        ist_weights = torch.zeros(batch_size, image_height, image_width, device=device)

        batch_range = range(batch_size)

        if show_progress:
            batch_range = tqdm(batch_range)

        if DEBUG_IST_MAPS:
            maps_folder = self._dataparser_outputs.image_filenames[0].absolute().parent.parent / "ist_maps"
            maps_folder.mkdir(exist_ok=True)
            debug_count = 0
            colormap = plt.get_cmap("turbo")

        # Value is pretty arbitrary, but the goal is to remove parasite areas such as
        # camera shaking that makes the whole map non-zero.
        # 0.25 seems good for Paderborn but too high for Stadium.
        # 0.1 seems good, a bit too low for paderborn but better too low than too high.
        alpha = 0.15

        # Compare to the same camera about 25 frames later
        for i in batch_range:
            # Images with the same camera in the batch.
            same_cam_imgs = torch.where(cam_ids == cam_ids[i])[0]

            # Time differences
            time_diffs = torch.abs(cam_times[same_cam_imgs] - cam_times[i])

            # Filter by time difference > 0.01 to avoid comparing to the same image.
            # Filter by time difference < 0.25 to avoid comparing to images too far in the future.
            close_imgs = same_cam_imgs[torch.where((time_diffs <= self.ist_range) & (time_diffs > 0.01))[0]]

            # If no images are found, make a uniform map.
            if len(close_imgs) == 0:
                ist_weights[i] = torch.ones(image_height, image_width, device=device)
                continue

            # Find image with biggest absolute difference among close images.
            current_img = batch["image"][i].to(device)  # [H, W, 3]
            max_diff = torch.zeros_like(current_img).to(device)  # [H, W, 3]
            for batch_id in close_imgs:
                close_img = batch["image"][batch_id].to(device)  # [H, W, 3]
                diff = torch.abs(current_img - close_img)  # [H, W, 3]
                max_diff = torch.maximum(max_diff, diff)

            max_diff = max_diff.mean(dim=2)  # [H, W]
            # Remove values below alpha
            max_diff = torch.where(max_diff > alpha, max_diff, torch.zeros_like(max_diff))

            ist_weights[i] = max_diff

            if DEBUG_IST_MAPS:
                if debug_count % 10 == 0:
                    colored_ist_map = colormap(max_diff.cpu().numpy())[..., :3]
                    img = batch["image"][i].cpu().numpy()
                    side_by_side = np.concatenate((img, colored_ist_map), axis=1)
                    pil_image = Image.fromarray((side_by_side * 255).astype(np.uint8))
                    img_name = self._dataparser_outputs.image_filenames[img_idx[i]].name
                    pil_image.save(maps_folder / img_name)
                    debug_count = 0
                debug_count += 1

        # convert it to take less memory:
        ist_weights = ist_weights.to(torch.float16)
        # multinomial is not supported for torch.float16 (half-precision) on CPU
        # but CPU way too slow for multinomiaL...

        if offline:
            print("[IST]: Saving pre-computed weights.")
            torch.save(ist_weights, weights_file)

        return ist_weights
