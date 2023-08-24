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
K-Planes implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Type, Literal

import functools
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.utils.dynmetric import DynMetric

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.kplanes_field import KPlanesField, KPlanesDensityField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    space_tv_loss,
    time_smoothness_loss,
    sparse_transients_loss,
    DepthLossType,
    depth_loss,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    MedianRGBRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider, AABBBoxCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, misc
import time


@dataclass
class KPlanesModelConfig(ModelConfig):
    """K-Planes model config"""

    _target: Type = field(default_factory=lambda: KPlanesModel)
    """target class to instantiate"""
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    bounded: bool = True
    """If true, uses AABB contraction and collider."""
    spacetime_resolution: Sequence[int] = (64, 64, 64, 50)
    """Desired resolution of the scene at the base scale. Should include 3 or 4 elements depending
       on whether the scene is static or dynamic.
    """
    feature_dim: int = 32
    """Size of the features stored in the k-planes"""
    multiscale_res: Sequence[int] = (1, 2, 4, 8)
    """Multipliers for the spatial resolution of the k-planes. 
        E.g. if equals to (2, 4) and spacetime_resolution is (128, 128, 128, 50), then
        2 k-plane models will be created at resolutions (256, 256, 256, 50) and (512, 512, 512, 50).
    """
    concat_features_across_scales: bool = True
    """Whether to concatenate or sum together the interpolated features at different scales"""
    linear_decoder: bool = False
    """Whether to use a fully linear decoder, or a non-linear MLP for decoding"""
    linear_decoder_layers: Optional[int] = 1
    """Number of layers in the linear decoder"""
    sigma_net_layers: int = 1
    """Number of layers in the sigma network"""
    sigma_net_hidden_dim: int = 64
    """Hidden dimension of the sigma network"""
    rgb_net_layers: int = 2
    """Number of layers in the rgb network"""
    rgb_net_hidden_dim: int = 64
    """Hidden dimension of the rgb network"""
    background_color_train: Literal[
        "black", "last_sample", "white", "random"
    ] = "random"  # Random gives better results than black experimentally. White is worst.
    """The background color that is given to untrained areas."""
    background_color_eval: Literal[
        "black", "last_sample", "white", "random"
    ] = "last_sample"  # Random gives better results than black experimentally. White is worst.
    """The background color that is given to untrained areas."""
    # proposal-sampling arguments
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"feature_dim": 8, "resolution": [128, 128, 128, 150]},
            {"feature_dim": 8, "resolution": [256, 256, 256, 150]},
        ]
    )
    """Arguments for the proposal density fields."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 128)
    """Number of samples per ray for each proposal network."""
    use_single_jitter: bool = False
    """Whether use single jitter or not for the proposal networks."""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    # appearance embedding (phototourism)
    use_appearance_embedding: bool = False
    """Whether to use per-image appearance embeddings"""
    appearance_embedding_dim: int = 0
    """Size of the appearance vectors, only if use_appearance_embedding is True"""
    disable_viewing_dependent: bool = False
    """If true, color is independent of viewing direction."""
    loss_coefficients: Dict[str, float] = to_immutable_dict(
        {
            "rgb_loss": 1.0,  # Reconstruction Loss
            "interlevel_loss": 1.0,  # Online distillation between the proposal and nerf networks.
            "distortion_loss": 0.001,  # Encourage compact ray weights
            "space_tv_loss": 0.0002,  # Encourage sparse gradients.
            "time_smoothness_loss": 0.001,  # Penalize acceleration.
            "sparse_transients_loss": 0.0001,  # Enforce separation of time and space planes.
            "space_tv_proposal_loss": 0.0002,  # Encourage sparse gradients. (for proposal grids)
            "time_smoothness_proposal_loss": 0.00001,  # Penalize acceleration. (for proposal grids)
            "sparse_transients_proposal_loss": 0.0001,  # Enforce separation of time and space planes. (for proposal grids)
            "depth_loss": 0.05,  # Depth reconstruction loss
        }
    )
    """Loss specific weights."""
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
    freeze_time_planes: bool = False
    """Whether to use freeze time planes."""
    freeze_space_planes: bool = False
    """Whether to use freeze space planes."""


class KPlanesModel(Model):
    """K-Planes Model
    Args:
        config: K-Planes configuration to instantiate model
    """

    config: KPlanesModelConfig

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        linear_decoder = self.config.linear_decoder

        scene_contraction = None if self.config.bounded else SceneContraction(order=float("inf"))

        self.field = KPlanesField(
            self.scene_box.aabb,
            feat_dim=self.config.feature_dim,
            spacetime_resolution=self.config.spacetime_resolution,
            concat_features_across_scales=self.config.concat_features_across_scales,
            multiscale_res=self.config.multiscale_res,
            use_appearance_embedding=self.config.use_appearance_embedding,
            appearance_dim=self.config.appearance_embedding_dim,
            spatial_distortion=scene_contraction,
            linear_decoder=linear_decoder,
            linear_decoder_layers=self.config.linear_decoder_layers,
            num_images=self.num_train_data,
            disable_viewing_dependent=self.config.disable_viewing_dependent,
            sigma_net_layers=self.config.sigma_net_layers,
            sigma_net_hidden_dim=self.config.sigma_net_hidden_dim,
            rgb_net_layers=self.config.rgb_net_layers,
            rgb_net_hidden_dim=self.config.rgb_net_hidden_dim,
            freeze_time_planes=self.config.freeze_time_planes,
            freeze_space_planes=self.config.freeze_space_planes,
        )

        if self.config.should_decay_sigma:
            self.depth_sigma = torch.tensor([self.config.starting_depth_sigma])
        else:
            self.depth_sigma = torch.tensor([self.config.depth_sigma])

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = KPlanesDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                linear_decoder=linear_decoder,
                freeze_time_planes=self.config.freeze_time_planes,
                freeze_space_planes=self.config.freeze_space_planes,
                **prop_net_args,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = KPlanesDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    linear_decoder=linear_decoder,
                    freeze_time_planes=self.config.freeze_time_planes,
                    freeze_space_planes=self.config.freeze_space_planes,
                    **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler to uniform if bounded.
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.bounded:
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        if self.config.bounded:
            self.collider = AABBBoxCollider(scene_box=self.scene_box)
        else:
            self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color_train)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.medianrgb_renderer = MedianRGBRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        self.dynmetric = DynMetric(self.psnr, self.ssim, self.lpips, "cuda")

        # Toggle dynamic viewer
        self.temporal_distortion = len(self.config.spacetime_resolution) == 4

        num_parameters = sum(p.numel() for p in self.field.parameters() if p.requires_grad)
        num_parameters += sum(p.numel() for p in self.proposal_networks.parameters() if p.requires_grad)
        param_size = 0
        buffer_size = 0
        for m in [self.field, self.proposal_networks]:
            for param in m.parameters():
                param_size += param.nelement() * param.element_size()
            for buffer in m.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
        total_size = (param_size + buffer_size) / 1024**2
        print("K-Planes Model initialized. Parameter count: ", num_parameters, "({:.3f}MB)".format(total_size))

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {
            "proposal_networks": list(self.proposal_networks.parameters()),
            "fields": list(self.field.parameters()),
        }
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        density_fns = self.density_fns
        if ray_bundle.times is not None:
            density_fns = [functools.partial(f, times=ray_bundle.times) for f in density_fns]
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=density_fns
        )
        field_out = self.field(ray_samples)

        weights = ray_samples.get_weights(field_out[FieldHeadNames.DENSITY])
        weights_list.append(weights)  # list: [weights_prop1, weights_prop2, weights_nerf]
        ray_samples_list.append(ray_samples)

        if self.training:
            self.renderer_rgb.background_color = self.config.background_color_train
        else:
            self.renderer_rgb.background_color = self.config.background_color_eval
        rgb = self.renderer_rgb(rgb=field_out[FieldHeadNames.RGB], weights=weights)
        accumulation = self.renderer_accumulation(weights)
        depth = self.renderer_depth(weights, ray_samples)

        median_rgb = self.medianrgb_renderer(rgb=field_out[FieldHeadNames.RGB], weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "median_rgb": median_rgb,
        }
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        if ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata:
            outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)

        if "depth_image" in batch.keys() and self.training and self.config.loss_coefficients["depth_loss"] > 0:
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

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        device = outputs["rgb"].device
        image = batch["image"].to(device)

        rgb_loss = self.rgb_loss(image, outputs["rgb"])

        loss_dict = {"rgb_loss": rgb_loss}
        loss_coef = self.config.loss_coefficients

        if self.training:
            if "distortion_loss" in loss_coef:
                loss_dict["distortion_loss"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
            if "interlevel_loss" in loss_coef:
                loss_dict["interlevel_loss"] = interlevel_loss(outputs["weights_list"], outputs["ray_samples_list"])

            # Field and proposal grids.
            ms_grids_nerf = self.field.grids
            ms_grids_prop = [p.grids for p in self.proposal_networks]

            if "space_tv_loss" in loss_coef:
                loss_dict["space_tv_loss"] = space_tv_loss(ms_grids_nerf)
            if "space_tv_proposal_loss" in loss_coef:
                loss_dict["space_tv_proposal_loss"] = space_tv_loss(ms_grids_prop)
            # Time losses
            if len(self.config.spacetime_resolution) > 3 and not self.config.freeze_time_planes:
                if "sparse_transients_loss" in loss_coef:
                    loss_dict["sparse_transients_loss"] = sparse_transients_loss(ms_grids_nerf)
                if "sparse_transients_proposal_loss" in loss_coef:
                    loss_dict["sparse_transients_proposal_loss"] = sparse_transients_loss(ms_grids_prop)
                if "time_smoothness_loss" in loss_coef:
                    loss_dict["time_smoothness_loss"] = time_smoothness_loss(ms_grids_nerf)
                if "time_smoothness_proposal_loss" in loss_coef:
                    loss_dict["time_smoothness_proposal_loss"] = time_smoothness_loss(ms_grids_prop)

            if "depth_image" in batch.keys() and loss_coef["depth_loss"] > 0:
                loss_dict["depth_loss"] = metrics_dict["depth_loss"]

        loss_dict = misc.scale_dict(loss_dict, loss_coef)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])

        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)
        bbox_img, dpsnr, dssim, dlpips = self.dynmetric(image, rgb)

        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim.item()),
            "lpips": float(lpips.item()),
            "dpsnr": float(dpsnr),
            "dssim": float(dssim),
            "dlpips": float(dlpips),
        }
        images_dict = {"img": combined_rgb, "accumulation": acc, "depth": depth, "bbox": bbox_img}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        # Add ground truth depth if present.
        if "depth_image" in batch.keys():
            ground_truth_depth = batch["depth_image"]
            if not self.config.is_euclidean_depth:
                ground_truth_depth = ground_truth_depth * outputs["directions_norm"]

            ground_truth_depth_colormap = colormaps.apply_depth_colormap(ground_truth_depth)
            images_dict["depth"] = torch.cat([ground_truth_depth_colormap, depth], dim=1)

        images_dict["median_rgb"] = outputs["median_rgb"]

        return metrics_dict, images_dict

    def _get_sigma(self) -> TensorType[0]:
        if not self.config.should_decay_sigma:
            return self.depth_sigma

        self.depth_sigma = torch.maximum(  # pylint: disable=attribute-defined-outside-init
            self.config.sigma_decay_rate * self.depth_sigma, torch.tensor([self.config.depth_sigma])
        )
        return self.depth_sigma
