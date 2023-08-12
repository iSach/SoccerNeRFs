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
Implementation of NeRFPlayer (https://arxiv.org/abs/2210.15947) with InstantNGP backbone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

import nerfacc
import torch
from nerfacc import ContractionType
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.nerfplayer_ngp_complete_field import NerfplayerField
from nerfstudio.model_components.losses import (
    MSELoss,
    DepthLossType,
    depth_loss,
)
from nerfstudio.model_components.ray_samplers import VolumetricSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
    DecompositionRenderer,
)
from nerfstudio.models.base_model import Model
from nerfstudio.models.instant_ngp import InstantNGPModelConfig, NGPModel
from nerfstudio.utils import colormaps


@dataclass
class NerfplayerNGPcompleteModelConfig(InstantNGPModelConfig):
    """NeRFPlayer Model Config with InstantNGP backbone.
    Tips for tuning the performance:
    1. If the scene is flickering, this is caused by unwanted high-freq on the temporal dimension.
        Try reducing `temporal_dim` first, but don't be too small, otherwise the dynamic object is blurred.
        Then try increasing the `temporal_tv_weight`. This is the loss for promoting smoothness among the
        temporal channels.
    2. If a faster rendering is preferred, then try reducing `log2_hashmap_size`. If more details are
        wanted, try increasing `log2_hashmap_size`.
    3. If the input cameras are of limited numbers, try reducing `num_levels`. `num_levels` is for
        multi-resolution volume sampling, and has a similar behavior to the freq in NeRF. With a small
        `num_levels`, a blurred rendering will be generated, but it is unlikely to overfit the training views.
    """

    _target: Type = field(default_factory=lambda: NerfplayerModel)
    temporal_dim: int = 64
    """Hashing grid parameter. A higher temporal dim means a higher temporal frequency."""
    num_levels: int = 16
    """Hashing grid parameter."""
    features_per_level: int = 2
    """Hashing grid parameter."""
    log2_hashmap_size: int = 15
    """Hashing grid parameter."""
    base_resolution: int = 16
    """Hashing grid parameter."""
    temporal_tv_weight: float = 1
    """Temporal TV loss balancing weight for feature channels."""
    train_background_color: Literal["random", "black", "white"] = "black"
    """The training background color that is given to untrained areas."""
    eval_background_color: Literal["random", "black", "white"] = "black"
    """The training background color that is given to untrained areas."""
    disable_viewing_dependent: bool = True
    """Disable viewing dependent effects."""
    depth_weight: float = 0.05
    """depth loss balancing weight for feature channels."""
    is_euclidean_depth: bool = True
    """Whether input depth maps are Euclidean distances (or z-distances)."""
    depth_sigma: float = 0.01
    """Uncertainty around depth values in meters (defaults to 1cm)."""
    should_decay_sigma: bool = False
    """Whether to exponentially decay sigma."""
    starting_depth_sigma: float = 0.2
    """Starting uncertainty around depth values in meters (defaults to 0.2m)."""
    sigma_decay_rate: float = 0.99985
    """Rate of exponential decay."""
    depth_loss_type: DepthLossType = DepthLossType.DS_NERF
    """Depth loss type."""


class NerfplayerNGPcompleteModel(NGPModel):
    """NeRFPlayer Model with InstantNGP backbone.

    Args:
        config: NeRFPlayer NGP configuration to instantiate model
    """

    config: NerfplayerModelConfig
    field: NerfplayerField

    def populate_modules(self):
        """Set the fields and modules."""
        Model.populate_modules(self)

        self.field = NerfplayerField(
            aabb=self.scene_box.aabb,
            contraction_type=self.config.contraction_type,
            temporal_dim=self.config.temporal_dim,
            num_levels=self.config.num_levels,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            base_resolution=self.config.base_resolution,
            disable_viewing_dependent=self.config.disable_viewing_dependent,
        )

        if self.config.should_decay_sigma:
            self.depth_sigma = torch.tensor([self.config.starting_depth_sigma])
        else:
            self.depth_sigma = torch.tensor([self.config.depth_sigma])

        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)

        # Occupancy Grid
        self.occupancy_grid = nerfacc.OccupancyGrid(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            contraction_type=self.config.contraction_type,
        )

        # Sampler
        vol_sampler_aabb = self.scene_box.aabb if self.config.contraction_type == ContractionType.AABB else None
        self.sampler = VolumetricSampler(
            scene_aabb=vol_sampler_aabb,
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
        )  # need to update the density_fn later during forward (for input time)

        # renderers
        self.renderer_rgb = RGBRenderer()  # will update bgcolor later during forward
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        self.renderer_probs = DecompositionRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.temporal_distortion = True  # for viewer

    def get_outputs(self, ray_bundle: RayBundle):
        num_rays = len(ray_bundle)

        # update the density_fn of the sampler so that the density is time aware
        self.sampler.density_fn = lambda x: self.field.density_fn(x, ray_bundle.times)
        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.config.render_step_size,
                cone_angle=self.config.cone_angle,
            )

        field_outputs = self.field(ray_samples)

        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights = nerfacc.render_weight_from_density(
            packed_info=packed_info,
            sigmas=field_outputs[FieldHeadNames.DENSITY],
            t_starts=ray_samples.frustums.starts,
            t_ends=ray_samples.frustums.ends,
        )

        # update bgcolor in the renderer; usually random color for training and fixed color for inference
        if self.training:
            self.renderer_rgb.background_color = self.config.train_background_color
        else:
            self.renderer_rgb.background_color = self.config.eval_background_color
        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        depth = self.renderer_depth(
            weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays
        )
        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)
        probs = None
        if FieldHeadNames.PROBS in field_outputs:
            probs = self.renderer_probs(
                probs=field_outputs[FieldHeadNames.PROBS],
                weights=weights,
                ray_indices=ray_indices,
                num_rays=num_rays,
            )
        alive_ray_mask = accumulation.squeeze(-1) > 0

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "alive_ray_mask": alive_ray_mask,  # the rays we kept from sampler
            "num_samples_per_ray": packed_info[:, 1],
        }
        if probs is not None:
            outputs["probs"] = probs

        if ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata:
            outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]
        # adding them to outputs for calculating losses
        if self.training and self.config.depth_weight > 0:
            outputs["ray_indices"] = ray_indices
            outputs["ray_samples"] = ray_samples
            outputs["weights"] = weights
            outputs["weights_list"] = [weights.view(num_rays, -1, 1)]
            outputs["ray_samples_list"] = [ray_samples.view(num_rays, -1, 1)]
            outputs["sigmas"] = field_outputs[FieldHeadNames.DENSITY]
        return outputs

    def _get_sigma(self) -> TensorType[0]:
        if not self.config.should_decay_sigma:
            return self.depth_sigma

        self.depth_sigma = torch.maximum(  # pylint: disable=attribute-defined-outside-init
            self.config.sigma_decay_rate * self.depth_sigma, torch.tensor([self.config.depth_sigma])
        )
        return self.depth_sigma

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)

        if "depth_image" in batch.keys() and self.training:
            metrics_dict["depth_loss"] = 0.0
            sigma = self._get_sigma().to(self.device)
            termination_depth = batch["depth_image"].to(self.device)
            # Iterate over networks (proposal + nerf)
            for i in range(len(outputs["weights_list"])):
                metrics_dict["depth_loss"] += depth_loss(
                    weights=outputs["weights_list"][i],
                    ray_samples=outputs["ray_samples_list"][i],
                    termination_depth=termination_depth,
                    predicted_depth=outputs["depth"],
                    sigma=sigma,
                    directions_norm=None,
                    is_euclidean=self.config.is_euclidean_depth,
                    depth_loss_type=self.config.depth_loss_type,
                ) / len(outputs["weights_list"])

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)
        mask = outputs["alive_ray_mask"]
        rgb_loss = self.rgb_loss(image[mask], outputs["rgb"][mask])
        loss_dict = {"rgb_loss": rgb_loss}
            if self.training and "depth_image" in batch.keys() and self.config.depth_weight > 0:
            loss_dict["depth_loss"] = self.config.depth_weight * metrics_dict["depth_loss"]

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)
        if "probs" in outputs:
            # (P_stat, P_deform, P_new)
            areas = "static", "deform", "new"
            print("probs", outputs["probs"].shape)
            for i, area in enumerate(areas):
                images_dict["probs_" + area] = colormaps.apply_colormap(
                    outputs["probs"][..., i : i + 1],
                    cmap="turbo",
                )
        if "depth_image" in batch.keys():
            ground_truth_depth = batch["depth_image"]
            # if not self.config.is_euclidean_depth: # here it is euclidean.
            #    ground_truth_depth = ground_truth_depth * outputs["directions_norm"]

            ground_truth_depth_colormap = colormaps.apply_depth_colormap(ground_truth_depth)
            depth = images_dict["depth"]
            images_dict["depth"] = torch.cat([ground_truth_depth_colormap, depth], dim=1)
        return metrics_dict, images_dict
