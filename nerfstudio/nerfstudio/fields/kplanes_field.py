# Copyright 2022 The Plenoptix Team. All rights reserved.
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

"""KPlanes Field"""


from gc import freeze
import itertools
from typing import Collection, Iterable, Optional, Sequence

import tinycudann as tcnn
import torch
from torch import is_grad_enabled, nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field
from nerfstudio.utils.interpolation import grid_sample_wrapper
import time


def get_normalized_directions(directions: TensorType["bs":..., 3]):
    """SH encoding must be in the range [0, 1]
    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def init_kplanes_field(out_dim: int, reso: Sequence[int], a: float = 0.1, b: float = 0.5) -> nn.ParameterList:
    """Initialize feature planes at a single scale.

    This functions creates k-choose-2 planes, where k is the number of input coordinates (4 for
    video, 3 for static scenes). k is inferred from the length of the `resolution` sequence.

    Args:
        out_dim: feature size at every point of the planes
        reso: the resolution of the planes, must be of length 3 or 4
        a: the spatial planes are initialized uniformly at random between `a` and `b`
        b: the spatial planes are initialized uniformly at random between `a` and `b`
    """
    in_dim = len(reso)
    has_time_planes = in_dim == 4
    coo_combs = list(itertools.combinations(range(in_dim), 2))
    grid_coefs = nn.ParameterList()
    #   0        1       2       3       4       5
    # (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    #  XY,      XZ,     XT,     YZ,     YT,     ZT
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty([1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:  # Initialize spatial planes as uniform[a, b]
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


def interpolate_kplanes(
    pts: torch.Tensor,
    ms_grids: Collection[Iterable[nn.Module]],
    concat_features: bool,
    freeze_time_planes: bool,
    freeze_space_planes: bool,
) -> torch.Tensor:
    """K-Planes: query multi-scale planes at given points
    Args:
        pts: 3D or 4D points at which the planes are queries
        ms_grids: Multi-scale k-plane grids
        concat_features: If true, the features from each scale are concatenated.
            Otherwise they are summed together.
    """
    coo_combs = list(itertools.combinations(range(pts.shape[-1]), 2))
    multi_scale_interp = [] if concat_features else 0.0
    grid: nn.ParameterList
    for grid in ms_grids:  # type: ignore
        interp_space = 1.0
        for ci, coo_comb in enumerate(coo_combs):
            # Check for freezing time planes
            if freeze_time_planes:
                if pts.shape[-1] == 4 and 3 in coo_comb:
                    continue

            disable_grad = False
            if freeze_space_planes:
                if pts.shape[-1] == 3 or 3 not in coo_comb:
                    disable_grad = True

            if not torch.is_grad_enabled():
                disable_grad = True

            with torch.set_grad_enabled(not disable_grad):
                # interpolate in plane
                feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
                interp_out_plane = grid_sample_wrapper(grid[ci], pts[..., coo_comb]).view(-1, feature_dim)

                # compute product over planes
                interp_space = interp_space * interp_out_plane

        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)  # type: ignore
        else:
            multi_scale_interp = multi_scale_interp + interp_space  # type: ignore

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)  # type: ignore
    return multi_scale_interp  # type: ignore


class KPlanesField(Field):
    """KPlanes Field
    Args:
        aabb: scene aabb bounds
        spacetime_resolution: desired resolution of the scene at the base scale
        feat_dim: size of features stored in the k-planes
        appearance_dim: size of the appearance vectors, only if use_appearance_embedding is True
        spatial_distortion: spatial distortion to apply to the scene
        num_images: number of images in the dataset. Used for appearance embedding
        multiscale_res: list of multipliers for the spatial resolution of the k-planes
        concat_features_across_scales: if True, features from different scales will be concatenated
            before passing to the MLP/linear decoders. Otherwise they will be summed together
        linear_decoder: whether to use a fully linear decoder, or a non-linear MLP for decoding
        linear_decoder_layers: number of layers in the linear decoder
        use_appearance_embedding: Whether to use per-image appearance embeddings
        disable_viewing_dependent: Do not use any view-dependent effects
    """

    def __init__(
        self,
        aabb,
        spacetime_resolution: Sequence[int] = (256, 256, 256, 150),
        feat_dim: int = 16,
        appearance_dim: int = 27,
        spatial_distortion: Optional[SpatialDistortion] = None,
        num_images: int = 0,
        multiscale_res: Optional[Sequence[int]] = None,
        concat_features_across_scales: bool = False,
        linear_decoder: bool = True,
        linear_decoder_layers: Optional[int] = None,
        use_appearance_embedding: bool = False,
        disable_viewing_dependent: bool = False,
        sigma_net_layers: int = 1,
        sigma_net_hidden_dim: int = 64,
        rgb_net_layers: int = 2,
        rgb_net_hidden_dim: int = 64,
        freeze_time_planes: bool = False,
        freeze_space_planes: bool = False,
    ) -> None:
        super().__init__()

        self.aabb = Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.multiscale_res_multipliers: Sequence[int] = multiscale_res or [1]
        self.concat_features_across_scales = concat_features_across_scales
        self.linear_decoder = linear_decoder
        self.has_time_planes = len(spacetime_resolution) == 4
        self.feature_dim = (
            feat_dim * len(self.multiscale_res_multipliers) if self.concat_features_across_scales else feat_dim
        )
        self.freeze_time_planes = freeze_time_planes
        self.freeze_space_planes = freeze_space_planes

        # 1. Init planes
        self.grids = nn.ModuleList()
        for res in self.multiscale_res_multipliers:
            resolution = [r * res for r in spacetime_resolution[:3]]
            if len(spacetime_resolution) > 3:  # Time does not get multi-scale treatment
                resolution.append(spacetime_resolution[3])
            self.grids.append(
                init_kplanes_field(
                    out_dim=feat_dim,
                    reso=resolution,
                )
            )

        # Initialize appearance code-related parameters
        self.use_appearance_embedding = use_appearance_embedding
        self.appearance_embedding = None
        self.appearance_embedding_dim = 0
        if use_appearance_embedding:
            assert num_images is not None
            self.appearance_embedding_dim = appearance_dim
            # this will initialize as normal_(0.0, 1.0)
            self.appearance_embedding = Embedding(num_images, self.appearance_embedding_dim)

        # Initialize direction encoder
        self.disable_viewing_dependent = disable_viewing_dependent

        if not self.disable_viewing_dependent:
            # Init direction encoder
            self.direction_encoder = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )

        # Initialize decoder network
        if self.linear_decoder:
            assert linear_decoder_layers is not None
            # The NN learns a basis that is used instead of spherical harmonics
            # Input is an encoded view direction, output is weights for
            # combining the color features into RGB. This architecture is based on instant-NGP
            self.color_basis = tcnn.Network(
                n_input_dims=3 + self.appearance_embedding_dim,
                n_output_dims=3 * self.feature_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": linear_decoder_layers,
                },
            )
            # sigma_net just does a linear transformation on the features to get density
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=1,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "None",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": 0,
                },
            )
        else:
            self.geo_feat_dim = 15
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=self.geo_feat_dim + 1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": sigma_net_hidden_dim,
                    "n_hidden_layers": sigma_net_layers,
                },
            )
            self.in_dim_color = self.geo_feat_dim + self.appearance_embedding_dim
            if not disable_viewing_dependent:
                self.in_dim_color += self.direction_encoder.n_output_dims
            self.color_net = tcnn.Network(
                n_input_dims=self.in_dim_color,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": rgb_net_hidden_dim,
                    "n_hidden_layers": rgb_net_layers,
                },
            )

    def get_density(self, ray_samples: RaySamples):
        """Computes and returns the densities."""
        positions = ray_samples.frustums.get_positions()
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            positions = positions / 2.0  # From [-2, 2] to [-1, 1]
        else:
            # Equivalent to NerfAcc's AABB contraction.
            positions = SceneBox.get_normalized_positions(positions, self.aabb)
            positions = positions * 2.0 - 1.0  # from [0, 1] to [-1, 1]
        n_rays, n_samples = positions.shape[:2]

        timestamps = ray_samples.times
        if self.has_time_planes and timestamps is not None:
            # Normalize timestamps from [0, 1] to [-1, 1]
            timestamps = (timestamps * 2) - 1
            positions = torch.cat((positions, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        positions = positions.reshape(-1, positions.shape[-1])

        features = interpolate_kplanes(
            positions,
            ms_grids=self.grids,  # type: ignore
            concat_features=self.concat_features_across_scales,
            freeze_time_planes=self.freeze_time_planes,
            freeze_space_planes=self.freeze_space_planes,
        )

        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)
        if self.linear_decoder:
            density_before_activation = self.sigma_net(features)  # [batch, 1]
        else:
            features = self.sigma_net(features)
            features, density_before_activation = torch.split(features, [self.geo_feat_dim, 1], dim=-1)

        density = trunc_exp(density_before_activation.to(positions)).view(n_rays, n_samples, 1)  # type: ignore
        return density, features

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None) -> TensorType:
        assert density_embedding is not None
        n_rays, n_samples = ray_samples.frustums.shape

        directions = ray_samples.frustums.directions.reshape(-1, 3)
        if self.linear_decoder or self.disable_viewing_dependent:
            color_features = [density_embedding]
        else:
            directions = get_normalized_directions(directions)
            encoded_directions = self.direction_encoder(directions)
            color_features = [encoded_directions, density_embedding]

        if self.use_appearance_embedding:
            assert ray_samples.camera_indices is not None
            camera_indices = ray_samples.camera_indices.squeeze()
            if self.training:
                embedded_appearance = self.appearance_embedding(camera_indices)
            else:
                # Average of appearance embeddings for test data
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                ) * self.appearance_embedding.mean(dim=0)

            # expand embedded_appearance from n_rays, dim to n_rays*n_samples, dim
            ea_dim = embedded_appearance.shape[-1]
            embedded_appearance = (
                embedded_appearance.view(-1, 1, ea_dim).expand(n_rays, n_samples, -1).reshape(-1, ea_dim)
            )
            if self.linear_decoder:
                directions = torch.cat((directions, embedded_appearance), dim=-1)
            else:
                color_features.append(embedded_appearance)

        color_features = torch.cat(color_features, dim=-1)

        if self.linear_decoder:
            basis_values = self.color_basis(directions)  # [batch, color_feature_len * 3]
            basis_values = basis_values.view(color_features.shape[0], 3, -1)  # [batch, 3, color_feature_len]
            rgb = torch.sum(color_features[:, None, :] * basis_values, dim=-1)  # [batch, 3]
            rgb = rgb.to(directions)
            rgb = torch.sigmoid(rgb).view(n_rays, n_samples, 3)
        else:
            rgb = self.color_net(color_features).to(directions).view(n_rays, n_samples, 3)

        return rgb

    def forward(
        self,
        ray_samples: RaySamples,
        compute_normals: bool = False,
        mask: Optional[TensorType] = None,
        bg_color: Optional[TensorType] = None,
    ):
        density, density_features = self.get_density(ray_samples)
        rgb = self.get_outputs(ray_samples, density_features)  # type: ignore

        return {FieldHeadNames.DENSITY: density, FieldHeadNames.RGB: rgb}


class KPlanesDensityField(Field):
    """K-Planes Density Field"""

    def __init__(
        self,
        aabb,
        resolution,
        feature_dim,
        spatial_distortion: Optional[SpatialDistortion] = None,
        linear_decoder: bool = True,
        freeze_time_planes: bool = False,
        freeze_space_planes: bool = False,
    ) -> None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.has_time_planes = len(resolution) == 4
        self.freeze_time_planes = freeze_time_planes
        self.freeze_space_planes = freeze_space_planes
        activation = "ReLU"
        if linear_decoder:
            activation = "None"

        self.grids = init_kplanes_field(out_dim=feature_dim, reso=resolution, a=0.1, b=0.15)
        self.sigma_net = tcnn.Network(
            n_input_dims=feature_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": activation,
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
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

    def get_density(self, ray_samples: RaySamples):
        positions = ray_samples.frustums.get_positions()
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            positions = positions / 2.0  # From [-2, 2] to [-1, 1]
        else:
            positions = SceneBox.get_normalized_positions(positions, self.aabb)

        n_rays, n_samples = positions.shape[:2]

        timestamps = ray_samples.times
        if self.has_time_planes and timestamps is not None:
            # Normalize timestamps from [0, 1] to [-1, 1]
            timestamps = (timestamps * 2) - 1
            positions = torch.cat((positions, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        positions = positions.reshape(-1, positions.shape[-1])

        features = interpolate_kplanes(
            positions,
            ms_grids=[self.grids],
            concat_features=False,
            freeze_time_planes=self.freeze_time_planes,
            freeze_space_planes=self.freeze_space_planes,
        )
        density = trunc_exp(self.sigma_net(features).to(positions)).view(n_rays, n_samples, 1)
        return density, features

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None):
        return {}
