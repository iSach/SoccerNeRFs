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
Put all the method implementations in one location.
"""

from __future__ import annotations

from typing import Dict

import tyro
from nerfacc import ContractionType

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.datamanagers.depth_datamanager import DepthDataManagerConfig
from nerfstudio.data.datamanagers.dynamic_datamanager import DynamicDataManagerConfig
from nerfstudio.data.datamanagers.sdf_datamanager import SDFDataManagerConfig
from nerfstudio.data.datamanagers.semantic_datamanager import SemanticDataManagerConfig
from nerfstudio.data.datamanagers.variable_res_datamanager import (
    VariableResDataManagerConfig,
)
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.data.dataparsers.dycheck_dataparser import DycheckDataParserConfig
from nerfstudio.data.dataparsers.instant_ngp_dataparser import (
    InstantNGPDataParserConfig,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.phototourism_dataparser import (
    PhototourismDataParserConfig,
)
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
from nerfstudio.data.dataparsers.sitcoms3d_dataparser import Sitcoms3DDataParserConfig
from nerfstudio.data.dataparsers.stadium_dataparser import StadiumDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.models.depth_nerfacto import DepthNerfactoModelConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.models.kplanes import KPlanesModelConfig
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.nerfplayer import NerfplayerModelConfig
from nerfstudio.models.nerfplayer_nerfacto import NerfplayerNerfactoModelConfig
from nerfstudio.models.nerfplayer_ngp import NerfplayerNGPModelConfig
from nerfstudio.models.neus import NeuSModelConfig
from nerfstudio.models.semantic_nerfw import SemanticNerfWModelConfig
from nerfstudio.models.tensorf import TensoRFModelConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
from nerfstudio.plugins.registry import discover_methods

method_configs: Dict[str, TrainerConfig] = {}
descriptions = {
    "nerfacto": "Recommended real-time model tuned for real captures. This model will be continually updated.",
    "depth-nerfacto": "Nerfacto with depth supervision.",
    "volinga": "Real-time rendering model from Volinga. Directly exportable to NVOL format at https://volinga.ai/",
    "instant-ngp": "Implementation of Instant-NGP. Recommended real-time model for unbounded scenes.",
    "instant-ngp-bounded": "Implementation of Instant-NGP. Recommended for bounded real and synthetic scenes",
    "mipnerf": "High quality model for bounded scenes. (slow)",
    "semantic-nerfw": "Predicts semantic segmentations and filters out transient objects.",
    "vanilla-nerf": "Original NeRF model. (slow)",
    "tensorf": "tensorf",
    "dnerf": "Dynamic-NeRF model. (slow)",
    "phototourism": "Uses the Phototourism data.",
    "nerfplayer-nerfacto": "NeRFPlayer with nerfacto backbone.",
    "nerfplayer-ngp": "NeRFPlayer with InstantNGP backbone.",
    "neus": "Implementation of NeuS. (slow)",
    "nerfplayer": "NeRFPlayer",
    "k-planes": "K-Planes model.",
    "k-planes-static": "Static K-Planes model (3-Planes).",
}

method_configs["nerfacto"] = TrainerConfig(
    method_name="nerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["depth-nerfacto"] = TrainerConfig(
    method_name="depth-nerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=DepthDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=DepthNerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["volinga"] = TrainerConfig(
    method_name="volinga",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NerfactoModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            hidden_dim=32,
            hidden_dim_color=32,
            hidden_dim_transient=32,
            num_nerf_samples_per_ray=24,
            proposal_net_args_list=[
                {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": True},
                {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": True},
            ],
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["instant-ngp"] = TrainerConfig(
    method_name="instant-ngp",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=DynamicBatchPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=NerfstudioDataParserConfig(), train_num_rays_per_batch=8192),
        model=InstantNGPModelConfig(eval_num_rays_per_chunk=8192),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        }
    },
    viewer=ViewerConfig(num_rays_per_chunk=64000),
    vis="viewer",
)


method_configs["instant-ngp-bounded"] = TrainerConfig(
    method_name="instant-ngp-bounded",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=DynamicBatchPipelineConfig(
        datamanager=DynamicDataManagerConfig(
            dataparser=InstantNGPDataParserConfig(),
            train_num_rays_per_batch=8192,
            use_importance_sampling=True,
            iters_to_start_is=500,
            is_pixel_ratio=0.15,
        ),
        model=InstantNGPModelConfig(
            eval_num_rays_per_chunk=8192,
            contraction_type=ContractionType.AABB,
            render_step_size=0.001,
            max_num_samples_per_ray=48,
            near_plane=0.01,
            background_color="black",
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        }
    },
    viewer=ViewerConfig(num_rays_per_chunk=64000),
    vis="viewer",
)


method_configs["mipnerf"] = TrainerConfig(
    method_name="mipnerf",
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=NerfstudioDataParserConfig(), train_num_rays_per_batch=1024),
        model=VanillaModelConfig(
            _target=MipNerfModel,
            loss_coefficients={"rgb_loss_coarse": 0.1, "rgb_loss_fine": 1.0},
            num_coarse_samples=128,
            num_importance_samples=128,
            eval_num_rays_per_chunk=1024,
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        }
    },
)

method_configs["semantic-nerfw"] = TrainerConfig(
    method_name="semantic-nerfw",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=SemanticDataManagerConfig(
            dataparser=Sitcoms3DDataParserConfig(), train_num_rays_per_batch=4096, eval_num_rays_per_batch=8192
        ),
        model=SemanticNerfWModelConfig(eval_num_rays_per_chunk=1 << 16),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 16),
    vis="viewer",
)

method_configs["vanilla-nerf"] = TrainerConfig(
    method_name="vanilla-nerf",
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=BlenderDataParserConfig(),
        ),
        model=VanillaModelConfig(_target=NeRFModel),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
        "temporal_distortion": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
    },
)

method_configs["tensorf"] = TrainerConfig(
    method_name="tensorf",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=False,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=BlenderDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
        ),
        model=TensoRFModelConfig(),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=0.001),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
        },
        "encodings": {
            "optimizer": AdamOptimizerConfig(lr=0.02),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.002, max_steps=30000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["dnerf"] = TrainerConfig(
    method_name="dnerf",
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=DNeRFDataParserConfig()),
        model=VanillaModelConfig(
            _target=NeRFModel,
            enable_temporal_distortion=True,
            temporal_distortion_params={"kind": TemporalDistortionKind.DNERF},
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
        "temporal_distortion": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
    },
)

method_configs["phototourism"] = TrainerConfig(
    method_name="phototourism",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VariableResDataManagerConfig(  # NOTE: one of the only differences with nerfacto
            dataparser=PhototourismDataParserConfig(),  # NOTE: one of the only differences with nerfacto
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["k-planes-static"] = TrainerConfig(
    method_name="k-planes-static",
    steps_per_eval_batch=1000,
    steps_per_save=5000,
    save_only_latest_checkpoint=False,
    steps_per_eval_all_images=100000,
    steps_per_eval_image=500,
    max_num_iterations=20000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=DynamicDataManagerConfig(
            dataparser=StadiumDataParserConfig(),
            train_num_rays_per_batch=8192,
            eval_num_rays_per_batch=1024,
            train_num_images_to_sample_from=1000,
            train_num_times_to_repeat_images=2000,
            eval_num_images_to_sample_from=50,
            eval_num_times_to_repeat_images=5000,
            use_importance_sampling=True,
            is_pixel_ratio=0.15,
            isg=True,
            ist_range=0.25,
            iters_to_start_is=2000,
            camera_optimizer=CameraOptimizerConfig(
                mode="off",
                optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                # orientation_noise_std=0.01,
            ),
        ),
        model=KPlanesModelConfig(
            eval_num_rays_per_chunk=1 << 16,
            multiscale_res=(
                1,
                2,
                4,
                8,
                16,
            ),
            spacetime_resolution=(64, 64, 64),
            feature_dim=32,
            concat_features_across_scales=True,
            disable_viewing_dependent=True,
            proposal_net_args_list=[
                {"feature_dim": 8, "resolution": (128, 128, 128)},
                {"feature_dim": 8, "resolution": (256, 256, 256)},
            ],
            sigma_net_layers=1,
            sigma_net_hidden_dim=64,
            rgb_net_layers=2,
            rgb_net_hidden_dim=64,
            num_proposal_samples_per_ray=(256, 128),
            num_nerf_samples_per_ray=64,
            bounded=True,
            loss_coefficients={
                "rgb_loss": 1.0,  # Reconstruction Loss
                "interlevel_loss": 1.0,  # Online distillation between the proposal and nerf networks.
                "distortion_loss": 0.001,  # Encourage compact ray weights
                "space_tv_loss": 0.02,  # Encourage sparse gradients.
                "time_smoothness_loss": 1.0,  # Penalize acceleration.
                "sparse_transients_loss": 0.001,  # Enforce separation of time and space planes.
                "space_tv_proposal_loss": 0.02,  # Encourage sparse gradients. (for proposal grids)
                "time_smoothness_proposal_loss": 1.0,  # Penalize acceleration. (for proposal grids)
                "sparse_transients_proposal_loss": 0.001,  # Enforce separation of time and space planes. (for proposal grids)
                "depth_loss": 0.05,  # 0.1,  # Depth reconstruction loss
            },
            depth_sigma=0.01,
            is_euclidean_depth=False,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-8),
            "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=20000, learning_rate_alpha=0),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-8),
            "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=20000, learning_rate_alpha=0),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 16),
    vis="wandb",
)

method_configs["k-planes"] = TrainerConfig(
    method_name="k-planes",
    steps_per_eval_batch=1000,
    steps_per_save=10000,
    save_only_latest_checkpoint=False,
    steps_per_eval_all_images=100000,
    steps_per_eval_image=500,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=DynamicDataManagerConfig(
            dataparser=StadiumDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=512,
            train_num_images_to_sample_from=2500,
            train_num_times_to_repeat_images=1000,
            eval_num_images_to_sample_from=100,
            eval_num_times_to_repeat_images=5000,
            use_importance_sampling=True,
            is_pixel_ratio=0.15,
            # is_pixel_ratio=0.075,  # lower ratio for ISG.
            isg=False,
            ist_range=1.0,
            isg_gamma=5e-2,
            iters_to_start_is=2000,
            camera_optimizer=CameraOptimizerConfig(
                mode="off",
                optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                # orientation_noise_std=0.01,
            ),
        ),
        model=KPlanesModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            multiscale_res=(1, 2, 4, 8, 16),
            spacetime_resolution=(64, 64, 64, 100),
            feature_dim=32,
            concat_features_across_scales=True,
            disable_viewing_dependent=True,
            proposal_net_args_list=[
                {"feature_dim": 8, "resolution": (128, 128, 128, 100)},
                {"feature_dim": 8, "resolution": (256, 256, 256, 100)},
            ],
            sigma_net_layers=1,
            sigma_net_hidden_dim=128,
            rgb_net_layers=2,
            rgb_net_hidden_dim=64,
            num_proposal_samples_per_ray=(256, 128),
            num_nerf_samples_per_ray=64,
            bounded=True,
            loss_coefficients={
                "rgb_loss": 1.0,  # Reconstruction Loss
                "interlevel_loss": 1.0,  # Online distillation between the proposal and nerf networks.
                "distortion_loss": 0.001,  # Encourage compact ray weights
                "space_tv_loss": 0.02,  # Encourage sparse gradients.
                "time_smoothness_loss": 1.0,  # Penalize acceleration.
                "sparse_transients_loss": 0.001,  # Enforce separation of time and space planes.
                "space_tv_proposal_loss": 0.02,  # Encourage sparse gradients. (for proposal grids)
                "time_smoothness_proposal_loss": 1.0,  # Penalize acceleration. (for proposal grids)
                "sparse_transients_proposal_loss": 0.001,  # Enforce separation of time and space planes. (for proposal grids)
                "depth_loss": 0.05,  # 0.1,  # Depth reconstruction loss
            },
            depth_sigma=0.01,
            is_euclidean_depth=False,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-12),
            "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000, learning_rate_alpha=0),
            # "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-12),
            "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000, learning_rate_alpha=0),
            # "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 16),
    vis="wandb",
)

method_configs["nerfplayer"] = TrainerConfig(
    method_name="nerfplayer",
    steps_per_eval_batch=1000,
    steps_per_eval_all_images=0,
    steps_per_eval_image=500,
    steps_per_save=10000,
    save_only_latest_checkpoint=False,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=DynamicDataManagerConfig(
            dataparser=StadiumDataParserConfig(),
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=1024,
            train_num_images_to_sample_from=3000,
            train_num_times_to_repeat_images=1000,
            eval_num_images_to_sample_from=50,
            eval_num_times_to_repeat_images=5000,
            use_importance_sampling=True,
            is_pixel_ratio=0.1,
            isg=False,
            ist_range=0.25,
            iters_to_start_is=3000,
        ),
        model=NerfplayerModelConfig(
            disable_scene_contraction=True,
            eval_num_rays_per_chunk=1 << 15,
            log2_hashmap_size=18,
            temporal_dim=64,
            depth_weight=0.0,
            depth_sigma=0.01,
            prob_reg_loss_mult=0.1,  # .01,
            distortion_loss_mult=0.001,
            temporal_tv_weight=1.0,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-6),
            "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000, learning_rate_alpha=0),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-6),
            "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000, learning_rate_alpha=0),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=64000),
    vis="wandb",
)


method_configs["nerfplayer-nerfacto"] = TrainerConfig(
    method_name="nerfplayer-nerfacto",
    steps_per_eval_batch=1000,
    steps_per_eval_all_images=0,
    steps_per_eval_image=500,
    steps_per_save=10000,
    save_only_latest_checkpoint=False,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=DynamicDataManagerConfig(
            dataparser=StadiumDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=1024,
            train_num_images_to_sample_from=3000,
            train_num_times_to_repeat_images=1000,
            eval_num_images_to_sample_from=50,
            eval_num_times_to_repeat_images=5000,
            use_importance_sampling=True,
            is_pixel_ratio=0.15,
            isg=False,
            ist_range=1.0,
            iters_to_start_is=3000,
        ),
        model=NerfplayerNerfactoModelConfig(
            disable_scene_contraction=True,
            eval_num_rays_per_chunk=1 << 15,
            log2_hashmap_size=19,
            temporal_dim=64,
            temporal_tv_weight=1.0,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-12),
            "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000, learning_rate_alpha=0),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-12),
            "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000, learning_rate_alpha=0),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=65536),
    vis="wandb",
)

method_configs["nerfplayer-ngp"] = TrainerConfig(
    method_name="nerfplayer-ngp",
    steps_per_eval_batch=1000,
    steps_per_eval_image=500,
    steps_per_eval_all_images=0,
    steps_per_save=5000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=DynamicBatchPipelineConfig(
        datamanager=DynamicDataManagerConfig(
            dataparser=StadiumDataParserConfig(),
            train_num_rays_per_batch=8192,
            eval_num_rays_per_batch=4096,
            train_num_images_to_sample_from=500,
            train_num_times_to_repeat_images=2000,
            eval_num_images_to_sample_from=50,
            eval_num_times_to_repeat_images=5000,
            use_importance_sampling=True,
        ),
        model=NerfplayerNGPModelConfig(
            eval_num_rays_per_chunk=8192,
            contraction_type=ContractionType.AABB,
            render_step_size=0.001,
            max_num_samples_per_ray=48,
            near_plane=0.01,
            temporal_tv_weight=0.05,
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-12),
            "scheduler": None,
        }
    },
    viewer=ViewerConfig(num_rays_per_chunk=64000),
    vis="viewer",
)

external_methods, external_descriptions = discover_methods()
method_configs.update(external_methods)
descriptions.update(external_descriptions)

AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=method_configs, descriptions=descriptions)
    ]
]
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
