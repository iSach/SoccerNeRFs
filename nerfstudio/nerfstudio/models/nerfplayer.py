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
NeRFPlayer (https://arxiv.org/abs/2210.15947) implementation with nerfacto backbone.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Dict, List, Type
from PIL import Image

import numpy as np
import torch
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.nerfplayer_field import (
    NerfplayerField,
    TemporalHashMLPDensityField,
)
from nerfstudio.model_components.losses import (
    DepthLossType,
    MSELoss,
    depth_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DecompositionRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import AABBBoxCollider, NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.dynmetric import DynMetric


@dataclass
class NerfplayerModelConfig(NerfactoModelConfig):
    """Nerfplayer Model Config with Nerfacto backbone"""

    _target: Type = field(default_factory=lambda: NerfplayerModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    train_background_color: Literal["random", "black", "white"] = "random"
    """The training background color that is given to untrained areas."""
    eval_background_color: Literal["random", "black", "white", "last_sample"] = "white"
    """The training background color that is given to untrained areas."""
    num_levels: int = 16
    """Hashing grid parameter."""
    features_per_level: int = 2
    """Hashing grid parameter."""
    log2_hashmap_size: int = 17
    """Hashing grid parameter."""
    temporal_dim: int = 64
    """Hashing grid parameter. A higher temporal dim means a higher temporal frequency."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "temporal_dim": 32, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 64},
            {"hidden_dim": 16, "temporal_dim": 32, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256},
        ]
    )
    """Arguments for the proposal density fields."""
    disable_viewing_dependent: bool = True
    """Disable viewing dependent effects."""
    distortion_loss_mult: float = 1e-3
    """Distortion loss multiplier."""
    temporal_tv_weight: float = 1.0
    """Temporal TV balancing weight for feature channels."""
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
    prob_reg_loss_mult: float = 0.0001  # Paper: 0.1, seems very high compared to experimental results done here.
    """Probability regularization loss multiplier."""


class NerfplayerModel(NerfactoModel):
    """Nerfplayer model with Nerfacto backbone.

    Args:
        config: Nerfplayer configuration to instantiate model
    """

    config: NerfplayerModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        Model.populate_modules(self)

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        if self.config.should_decay_sigma:
            self.depth_sigma = torch.tensor([self.config.starting_depth_sigma])
        else:
            self.depth_sigma = torch.tensor([self.config.depth_sigma])

        # Fields
        self.field = NerfplayerField(
            self.scene_box.aabb,
            temporal_dim=self.config.temporal_dim,
            num_levels=self.config.num_levels,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            disable_viewing_dependent=self.config.disable_viewing_dependent,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = TemporalHashMLPDensityField(
                self.scene_box.aabb, spatial_distortion=scene_contraction, **prop_net_args
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = TemporalHashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
            1,
            self.config.proposal_update_every,
        )
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
        )

        # Collider
        if self.config.disable_scene_contraction:
            self.collider = AABBBoxCollider(self.scene_box)
        else:
            self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        self.background_color = self.config.train_background_color

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.train_background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")  # for depth loss
        self.renderer_normals = NormalsRenderer()
        self.renderer_probs = DecompositionRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.dynmetric = DynMetric(self.psnr, self.ssim, self.lpips, "cuda")
        self.temporal_distortion = True  # for viewer

    def get_outputs(self, ray_bundle: RayBundle):
        assert ray_bundle.times is not None, "Time not provided."
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=[functools.partial(f, times=ray_bundle.times) for f in self.density_fns]
        )
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        if self.training:
            self.renderer_rgb.background_color = self.config.train_background_color
        else:
            self.renderer_rgb.background_color = self.config.eval_background_color
        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }
        if FieldHeadNames.PROBS in field_outputs:
            probs = self.renderer_probs(
                probs=field_outputs[FieldHeadNames.PROBS],
                weights=weights,
            )
            outputs["probs"] = probs

        if self.config.predict_normals:
            outputs["normals"] = self.normals_shader(
                self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            )
            outputs["pred_normals"] = self.normals_shader(
                self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            )

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        if ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata:
            outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]

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
                    directions_norm=outputs["directions_norm"],
                    is_euclidean=self.config.is_euclidean_depth,
                    depth_loss_type=self.config.depth_loss_type,
                ) / len(outputs["weights_list"])

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]

            if "depth_image" in batch.keys() and self.config.depth_weight > 0:
                loss_dict["depth_loss"] = self.config.depth_weight * metrics_dict["depth_loss"]

            if self.config.temporal_tv_weight > 0:
                loss_dict["temporal_tv_loss"] = self.field.newness_field.get_temporal_tv_loss()
                loss_dict["temporal_tv_loss"] += self.field.decomposition_field.get_temporal_tv_loss()
                for net in self.proposal_networks:
                    loss_dict["temporal_tv_loss"] += net.encoding.get_temporal_tv_loss()
                loss_dict["temporal_tv_loss"] *= self.config.temporal_tv_weight
                loss_dict["temporal_tv_loss"] /= (
                    len(self.proposal_networks) + 2
                )  # Average over all networks: 2 for field, 1 for each proposal

            # When IST is toggled on, there is a bias here with the fact that we consider the mean of all probs... :/
            if "probs" in outputs:
                # 0=static, 1=deform, 2=new
                probs = outputs["probs"].view(-1, 3)
                probs_mean = probs.mean(dim=0)
                prob_loss = 0.01 * probs_mean[1] + probs_mean[2]
                loss_dict["prob_loss"] = prob_loss * self.config.prob_reg_loss_mult

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)
        if "probs" in outputs:
            # (P_stat, P_deform, P_new)
            areas = "static", "deform", "new"
            for i, area in enumerate(areas):
                # colored_prob = colormaps.apply_colormap(outputs["probs"][..., i : i + 1], cmap="turbo")
                images_dict["probs_" + area] = outputs["probs"][..., i : (i + 1)]
        if "depth_image" in batch.keys():
            ground_truth_depth = batch["depth_image"]
            if not self.config.is_euclidean_depth:
                ground_truth_depth = ground_truth_depth * outputs["directions_norm"]

            ground_truth_depth_colormap = colormaps.apply_depth_colormap(ground_truth_depth)
            depth = images_dict["depth"]
            images_dict["depth"] = torch.cat([ground_truth_depth_colormap, depth], dim=1)

        image = batch["image"].to(outputs["rgb"].device)
        rgb = outputs["rgb"]
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]
        bbox_img, dpsnr, dssim, dlpips = self.dynmetric(image, rgb)
        metrics_dict["dpsnr"] = float(dpsnr)
        metrics_dict["dssim"] = float(dssim)
        metrics_dict["dlpips"] = float(dlpips)
        images_dict["bbox"] = bbox_img

        if "ist_weights" in batch:
            print("IST WEIGHTS")
            weights = batch["ist_weights"]
            print(weights.shape)
            images_dict["ist_weights"] = colormaps.apply_colormap(weights, cmap="turbo")

        return metrics_dict, images_dict
