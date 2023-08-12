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
Field implementations for NeRFPlayer (https://arxiv.org/abs/2210.15947) implementation with nerfacto backbone
"""


from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
)
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.field_components.temporal_grid import TemporalGridEncoder
from nerfstudio.fields.base_field import Field, shift_directions_for_tcnn

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


class TemporalHashMLPDensityField(Field):
    """A lightweight temporal density field module.

    Args:
        aabb: Parameters of scene aabb bounds
        temporal_dim: Hashing grid parameter. A higher temporal dim means a higher temporal frequency.
        num_layers: Number of hidden layers
        hidden_dim: Dimension of hidden layers
        spatial_distortion: Spatial distortion module
        num_levels: Hashing grid parameter. Used for initialize TemporalGridEncoder class.
        max_res: Hashing grid parameter. Used for initialize TemporalGridEncoder class.
        base_res: Hashing grid parameter. Used for initialize TemporalGridEncoder class.
        log2_hashmap_size: Hashing grid parameter. Used for initialize TemporalGridEncoder class.
        features_per_level: Hashing grid parameter. Used for initialize TemporalGridEncoder class.
    """

    def __init__(
        self,
        aabb: TensorType,
        temporal_dim: int = 64,
        num_layers: int = 2,
        hidden_dim: int = 64,
        spatial_distortion: Optional[SpatialDistortion] = None,
        num_levels: int = 8,
        max_res: int = 1024,
        base_res: int = 16,
        log2_hashmap_size: int = 18,
        features_per_level: int = 2,
    ) -> None:
        super().__init__()
        # from .temporal_grid import test; test() # DEBUG
        self.aabb = Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))

        self.encoding = TemporalGridEncoder(
            input_dim=3,
            temporal_dim=temporal_dim,
            num_levels=num_levels,
            level_dim=features_per_level,
            per_level_scale=growth_factor,
            base_resolution=base_res,
            log2_hashmap_size=log2_hashmap_size,
        )
        self.linear = tcnn.Network(
            n_input_dims=num_levels * features_per_level,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

    # pylint: disable=arguments-differ
    def density_fn(self, positions: TensorType["bs":..., 3], times: TensorType["bs", 1]) -> TensorType["bs":..., 1]:
        """Returns only the density. Used primarily with the density grid.

        Args:
            positions: the origin of the samples/frustums
            times: the time of rays
        """
        if len(positions.shape) == 3 and len(times.shape) == 2:
            # position is [ray, sample, 3]; times is [ray, 1]
            times = times[:, None]  # RaySamples can handle the shape
        # Need to figure out a better way to descibe positions with a ray.
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=positions,
                directions=torch.ones_like(positions),
                starts=torch.zeros_like(positions[..., :1]),
                ends=torch.zeros_like(positions[..., :1]),
                pixel_area=torch.ones_like(positions[..., :1]),
            ),
            times=times,
        )
        density, _ = self.get_density(ray_samples)
        return density

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, None]:
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(ray_samples.frustums.get_positions())
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions_flat = positions.view(-1, 3)
        time_flat = ray_samples.times.reshape(-1, 1)
        x = self.encoding(positions_flat, time_flat).to(positions)
        density_before_activation = self.linear(x).view(*ray_samples.frustums.shape, -1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation)
        return density, None

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None) -> dict:
        return {}


class NerfplayerField(Field):
    """NeRFPlayer (https://arxiv.org/abs/2210.15947) field with nerfacto backbone.

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        num_layers_transient: number of hidden layers for transient network
        hidden_dim_color: dimension of hidden layers for color network
        hidden_dim_transient: dimension of hidden layers for transient network
        appearance_embedding_dim: dimension of appearance embedding
        transient_embedding_dim: dimension of transient embedding
        use_transient_embedding: whether to use transient embedding
        use_semantics: whether to use semantic segmentation
        num_semantic_classes: number of semantic classes
        use_pred_normals: whether to use predicted normals
        use_average_appearance_embedding: whether to use average appearance embedding or zeros for inference
        spatial_distortion: spatial distortion to apply to the scene
    """

    def __init__(
        self,
        aabb: TensorType,
        num_images: int,
        num_layers: int = 3,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        temporal_dim: int = 64,
        num_levels: int = 16,
        features_per_level: int = 2,
        base_resolution: int = 16,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 4,
        num_layers_transient: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        disable_viewing_dependent: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()

        self.aabb = Parameter(aabb, requires_grad=False)
        self.geo_feat_dim = geo_feat_dim

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_transient_embedding = use_transient_embedding
        self.use_semantics = use_semantics
        self.use_pred_normals = use_pred_normals

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.position_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={"otype": "Frequency", "n_frequencies": 2},
        )

        feature_dim = num_levels * features_per_level

        # deformation_field
        self.deformation_field = tcnn.Network(
            n_input_dims=3,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 128,
                "n_hidden_layers": 3,
            },
        )

        # Explicit encoding for the stationary field. Does not depend on time.
        self.stationary_field = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": 1.4472692012786865,  # base_res * scale ** (level), base level = 0
            },
        )

        # MLP for the stationary field.
        # (features, t) -> (features)
        self.stationary_field_mlp = tcnn.Network(
            n_input_dims=feature_dim + 1,
            n_output_dims=feature_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

        self.newness_field = TemporalGridEncoder(
            input_dim=3,
            temporal_dim=temporal_dim,
            num_levels=num_levels,
            level_dim=features_per_level,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size,
            desired_resolution=1024 * (self.aabb.max() - self.aabb.min()),
        )

        self.decomposition_field = TemporalGridEncoder(
            input_dim=3,
            temporal_dim=temporal_dim,
            num_levels=num_levels,
            level_dim=features_per_level,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size,
            desired_resolution=1024 * (self.aabb.max() - self.aabb.min()),
        )

        self.decomposition_mlp = tcnn.Network(
            n_input_dims=feature_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        self._probs = None

        # Radiance Field (first component for density, the rest for color)
        self.mlp_base_decode = tcnn.Network(
            n_input_dims=feature_dim,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        in_dim = self.direction_encoding.n_output_dims + self.geo_feat_dim
        if disable_viewing_dependent:
            in_dim = self.geo_feat_dim
            self.direction_encoding = None
        self.mlp_head = tcnn.Network(
            n_input_dims=in_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, TensorType]:
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions_flat = positions.view(-1, 3)
        assert ray_samples.times is not None, "Time should be included in the input for NeRFPlayer"
        times_flat = ray_samples.times.reshape(-1, 1)

        # 1. Get the deformation field
        deformation = self.deformation_field(positions_flat)

        # Deform the positions
        deformed_positions = positions_flat + deformation

        # 2. Get the stationary field
        v_stat = self.stationary_field(positions_flat)
        v_deform = self.stationary_field(deformed_positions)
        v_stat = self.stationary_field_mlp(torch.cat([v_stat, times_flat], dim=-1))
        v_deform = self.stationary_field_mlp(torch.cat([v_deform, times_flat], dim=-1))

        # 3. Get the newness field
        v_new = self.newness_field(positions_flat, times_flat)

        # 4. Get the decomposition field
        v_decomp = self.decomposition_field(positions_flat, times_flat)
        probs = self.decomposition_mlp(v_decomp)
        probs = torch.softmax(probs, dim=-1)
        self._probs = probs

        # Mix features
        # Sizes:
        # probs: (batch_size, 3)
        # v: (batch_size, 32)
        v = (
            probs[:, 0].unsqueeze(-1) * v_stat
            + probs[:, 1].unsqueeze(-1) * v_deform
            + probs[:, 2].unsqueeze(-1) * v_new
        )

        h = self.mlp_base_decode(v).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        assert density_embedding is not None
        directions = shift_directions_for_tcnn(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)

        if self.direction_encoding is not None:
            d = self.direction_encoding(directions_flat)
            if density_embedding is None:
                positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
                h = torch.cat([d, positions.view(-1, 3)], dim=-1)
            else:
                h = torch.cat([d, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)
        else:
            # viewing direction is disabled
            if density_embedding is None:
                positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
                h = positions.view(-1, 3)
            else:
                h = density_embedding.view(-1, self.geo_feat_dim)

        rgb = self.mlp_head(h).view(*ray_samples.frustums.directions.shape[:-1], -1).to(directions)

        outputs = {FieldHeadNames.RGB: rgb}

        if self._probs is not None:
            outputs[FieldHeadNames.PROBS] = self._probs.view(*ray_samples.frustums.directions.shape[:-1], -1).to(
                directions
            )
            self._probs = None

        return outputs
