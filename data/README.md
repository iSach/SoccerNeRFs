# **Dynamic NeRFs for Soccer Scenes: Datasets**

## Scenes

The data consists of 3 synthetic scenes built using Blender. Each scene contains from 20 to 30 synchronized and calibrated cameras that capture a 4-second long scene at 25 FPS. For each scene, the downloadable data consists of images and corresponding poses. These are ready to use either with our code, either with Nerfstudio using the slightly different data parsers available in this folder.

### Close-up

This synthetic environment features a single player placed at the center of the field, shooting a ball. This first camera setup is composed of 30 close-up views around the player and resembles typical conditions of benchmarks like DyNeRF. Models should work pretty fine with no changes in this setup.

[[Download (Soon)](#)] [[Video Results](https://soccernerfs.isach.be/assets/closeup.mp4)]

### Broadcast-style

Within the same environment as close-up, we consider a second camera configuration that features 20 views placed around the field, whose field of view is close to broadcast conditions. The player represents only a tiny portion of the images. Satisfying results are much harder to obtain in this setup. 1080p training and ray importance sampling substantially improve results.

[[Download (Soon)](#)] [[Video Results]()]

### Stadium-wide

This more complex environment features several players and balls interacting all over the field, captured by 30 wide-angle cameras placed high up in the bleachers and are thus much more distant from the field. Six additional cameras, used exclusively for evaluation, are placed near the players for more meaningful results. In this setup, training views cover the whole field at all times but cover very few details about the players and balls due to their large distance. This scene is extremely challenging, and finding more advanced strategies to accurately reconstruct them would be interesting.

[[Download (Soon)](#)] [[Video Results]()]

## Stadium Model

The stadium model is freely available (CC0) on [BlendSwap](https://www.blendswap.com/blend/7488). 

## Player Models

The player models we use are from [Adobe Mixamo](https://mixamo.com). Unfortunately, it is not allowed to redistribute the Blender files that contain these player models. To modify the scenes, you therefore need to use them by yourself in Blender along with the stadium model. For the ball, we recommend using the [Projectile](https://github.com/natecraddock/projectile) Blender add-on. For exporting images and poses from Blender, see the [Blender script](blender-export.py).

## Usage

We detail here two ways of using this data, either using our code, a modified version of Nerfstudio that notably includes ray importance sampling and is therefore recommended, or directly using Nerfstudio.

### Using our Code (Recommended)

For this, follow the instructions [here](../README.md) to install the code. Then, you can use Nerfstudio as in the original version (see [here](https://docs.nerf.studio)) and use any of the following data:

`ns-train k-planes [...model params...] <scene>-data [...scene params...]`

Where you should replace `<scene>` with one of the following:
* `closeup`
* `broadcaststyle`
* `stadiumwide`

To see the parameters, you can use the autocompletion of Nerfstudio or use --help as a scene parameter (e.g., `ns-train k-planes closeup-data --help`).

### Using Nerfstudio

Nerfstudio lacks dynamic dataparsers for the format used here, which we haven't converted yet to D-NeRF because we recommend using our code benefiting from importance sampling. Nonetheless, you can easily use them with Nerfstudio.

For this, install the three python scripts in [this folder](dataparsers/) corresponding to each scene in the Nerfstudio's `data/dataparsers` folder.

Then, you need to register them in [this file](https://github.com/nerfstudio-project/nerfstudio/blob/48135ee4c8e0fb9ac3ab0ea80d2d71042dfb0b41/nerfstudio/configs/dataparser_configs.py#L55), by importing the three parsers then adding them to the list.

```python
from nerfstudio.data.dataparsers.closeup_dataparser import CloseupDataParserConfig
from nerfstudio.data.dataparsers.broadcaststyle_dataparser import BroadcaststyleDataParserConfig
from nerfstudio.data.dataparsers.stadiumwide_dataparser import StadiumwideDataParserConfig

dataparsers = {
	...
	"closeup-data": CloseupDataParserConfig(),
	"broadcaststyle-data": BroadcaststyleDataParserConfig(),
	"stadiumwide-data": StadiumwideDataParserConfig(),
```

Afterward, you can use them as detailed above.
