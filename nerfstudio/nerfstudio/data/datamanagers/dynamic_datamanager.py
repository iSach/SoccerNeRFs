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
Dynamic datamanager.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Type

from nerfstudio.cameras.cameras import CameraType
from nerfstudio.data.datamanagers import base_datamanager
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.datasets.dynamic_dataset import DynamicDataset
from nerfstudio.data.pixel_samplers import (
    DynamicBasedPixelSampler,
    EquirectangularPixelSampler,
    PatchPixelSampler,
    PixelSampler,
)


@dataclass
class DynamicDataManagerConfig(base_datamanager.VanillaDataManagerConfig):
    """A dynamic datamanager - required to use with .setup()"""

    _target: Type = field(default_factory=lambda: DynamicDataManager)

    use_importance_sampling: bool = True
    """Whether to use importance sampling for the dynamic dataset."""

    is_pixel_ratio: float = 0.1
    """The ratio of pixels to sample using importance sampling per iteration."""

    ist_range: float = 0.25
    """The range of time differences to use for importance sampling."""

    isg: bool = False
    """Use ISG (true) or IST (False)."""

    isg_gamma: float = 5e-2
    """ISG gamma."""

    iters_to_start_is: int = 5000
    """Iterations before starting IST sampling."""

    pick_mode: Literal["normal", "randsteps", "lowfps"] = "normal"
    """Method for picking images in random loading."""


class DynamicDataManager(base_datamanager.VanillaDataManager):  # pylint: disable=abstract-method
    """Data manager implementation for data that also requires processing depth data.
    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: DynamicDataManagerConfig

    def create_train_dataset(self) -> DynamicDataset:
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="train")
        return DynamicDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            use_importance_sampling=self.config.use_importance_sampling,
            is_pixel_ratio=self.config.is_pixel_ratio,
            ist_range=self.config.ist_range,
            isg=self.config.isg,
            isg_gamma=self.config.isg_gamma,
            iters_to_start_is=self.config.iters_to_start_is,
            eval_dataset=False,
            pick_mode=self.config.pick_mode,
        )

    def create_eval_dataset(self) -> DynamicDataset:
        return DynamicDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            use_importance_sampling=self.config.use_importance_sampling,
            is_pixel_ratio=self.config.is_pixel_ratio,
            ist_range=self.config.ist_range,
            isg=self.config.isg,
            isg_gamma=self.config.isg_gamma,
            iters_to_start_is=self.config.iters_to_start_is,
            eval_dataset=True,
            pick_mode=self.config.pick_mode,
        )

    def _get_pixel_sampler(  # pylint: disable=no-self-use
        self, dataset: InputDataset, *args: Any, **kwargs: Any
    ) -> PixelSampler:
        """Infer pixel sampler to use."""
        if self.config.patch_size > 1:
            return PatchPixelSampler(*args, **kwargs, patch_size=self.config.patch_size)

        # If all images are equirectangular, use equirectangular pixel sampler
        is_equirectangular = dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value
        if is_equirectangular.all():
            return EquirectangularPixelSampler(*args, **kwargs)
        # Otherwise, use the default pixel sampler

        if self.config.use_importance_sampling:
            return DynamicBasedPixelSampler(*args, **kwargs, dataset=dataset)

        return PixelSampler(*args, **kwargs)
