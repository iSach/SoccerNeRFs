# **Dynamic NeRFs for Soccer Scenes**

This repository contains the code for the paper "Dynamic NeRFs for Soccer Scenes", by Sacha Lewin, Maxime Vandegar, Thomas Hoyoux, Olivier Barnich, and Gilles Louppe.

This paper was accepted at the [http://mmsports.multimedia-computing.de/mmsports2023/index.html](6th Int. Workshop on Multimedia Content Analysis in Sports (MMSports'23) @ ACM Multimedia 2023), and will be presented on 29th October.

## **Abstract**
The long-standing problem of novel view synthesis has many applications, notably in sports broadcasting. Photorealistic novel view synthesis of soccer actions, in particular, is of enormous interest to the broadcast industry. Yet only a few industrial solutions have been proposed, and even fewer that achieve near-broadcast quality of the synthetic replays. Except for their setup of multiple static cameras around the playfield, the best proprietary systems disclose close to no information about their inner workings. Leveraging multiple static cameras for such a task indeed presents a challenge rarely tackled in the literature, for a lack of public datasets: the reconstruction of a large-scale, mostly static environment, with small, fast-moving elements. Recently, the emergence of neural radiance fields has induced stunning progress in many novel view synthesis applications, leveraging deep learning principles to produce photorealistic results in the most challenging settings. In this work, we investigate the feasibility of basing a solution to the task on dynamic NeRFs, i.e., neural models purposed to reconstruct general dynamic content. We compose synthetic soccer environments and conduct multiple experiments using them, identifying key components that help reconstruct soccer scenes with dynamic NeRFs. We show that, although this approach cannot fully meet the quality requirements for the target application, it suggests promising avenues toward a cost-efficient, automatic solution. We also make our work dataset and code publicly available, with the goal to encourage further efforts from the research community on the task of novel view synthesis for dynamic soccer scenes.

## Data

Please see [data/README.md](here).

## Running the code

All experiments were carried out in a Docker container. We recommend following the instructions on https://docs.nerf.studio

Otherwise, you can install Python 3.8, PyTorch 1.13.1, and CUDA 11.7.
Next, install [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), and the bindings for Torch:
```
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
Install other requirements:
```
cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .
pip install -e .[dev]
```
Install the CLI completion:
```
ns-install-cli
```

Nerfstudio should be ready to use, otherwise we again recommend to have a look at the [official documentation](https://docs.nerf.studio/).

Running a model is very easy. For example, running K-Planes with a new scale (32x), modified importance sampling, and downscaling FPS by 4 on the "Broadcast-style" Scene, and visualizing through the online viewer:
```bash
ns-train k-planes \                                   # Train K-Planes
    --vis viewer \                                    # Use the viewer
    --pipeline.model.multiscale-res 1 2 4 8 16 32 \   # New scale (32x)
    --pipeline.datamanager.ist-range 0.75 \           # Modified IST Range
    broadcaststyle-data \                             # Dataset
    --fps-downsample 4                                # Read 1 frame every 4
```

Default models can require large amounts of VRAM for running, so feel free to tune down the default settings if you have a GPU with less than 12GB.

## Models

### K-Planes

A slightly modified version of K-Planes is implemented. Please have a look at `nerfstudio/nerfstudio/kplanes.py` for more information on the settings.

### NeRFPlayer

NeRFPlayer is originally implemented in Nerfstudio as its truncated version, which we consider (`nerfplayer-nerfacto`). Additionally, we reimplemented the full version of NeRFplayer, with the decomposition, which was not really helpful here.

Please have a look at `nerfstudio/nerfstudio/nerfplayer[-nerfacto].py` for more information on the settings for both versions.

## **Repository structure**

### **Nerfstudio**
Some contributions were already made public, but most of the work was done separately in this repository, to save time and avoid conflicts. Indeed, Nerfstudio is still in _active development_. Due to time constraints, we were not able to make it stable and well documented to be ready for pull requests.

However, the code was implemented to follow the structure of Nerfstudio as much as possible! This means that the code is very modular and should be easily adapted to new Nerfstudio versions, which might be done in the future. Nonetheless, we tried to document the code as much as possible.

The code was last merged with the main branch of Nerfstudio with the version `0.1.19`. The `nerfstudio` folder corresponds to the actual repository (i.e., [Official Nerfstudio Repo](https://github.com/nerfstudio-project/nerfstudio)), with all changes being in `nerfstudio/nerfstudio`.

Main modifications include:
* __`configs`__: In `method_configs.py`, default configs for K-Planes and NeRFPlayer are added. These are used when running `ns-train`.
* __`models` & `fields`__: **Implementations of the full NeRFPlayer model**: `nerfplayer.py` and `nerfplayer_field.py`, based on the truncated versions (e.g., `nerfplayer_nerfacto.py`). **Implementation of K-Planes**: `kplanes.py` and `kplanes_field.py`, based on [this repo](https://github.com/akristoffersen/nerfstudio_kplanes) which began including in Nerfstudio the [original code](https://github.com/sarafridov/K-Planes) but was still quite buggy and missing components (e.g., bounded scenes, all losses, correct input encoding).
* __`model_components`__: Added all losses (`losses.py`) from K-Planes based on original code. Space TV loss was [wrongly implemented](https://github.com/nerfstudio-project/nerfstudio/pull/1584#issuecomment-1466680500) and thus fixed (almost no difference). Implemented a probability renderer (`renderers.py`) for the decomposition in NeRFPlayer.
* __`utils`__: New dynamic metric in `dynmetric.py`, can be easily integrated by any model inside of the `get_image_metrics_and_images` function. 
* __`data/datasets`__: New dynamic dataset class (`dynamic_dataset.py`) for dynamic scenes with optional depth. This class computes and caches importance sampling maps, more details below.
* __`data/datamanagers`__: Dynamic datamanager for generating dynamic datasets from any data parser that outputs time and possibly depth. Simply replace the vanilla manager by the dynamic one in a method config to use it.
* __`data/dataparsers`__: New data parsers for the experiment environments: `closeup_`, `broadcaststyle_`, `stadiumwide_dataparser.py`, plus a more generic one (`dynamic_dataparser.py`) as an example. They were tuned for the experiments but it is possible to make a single generic parser. :) Also, a data parser for HyperNeRF is included (`hypernerf_dataparser.py`), as experiments were planned on it, but not done in the end.
* __`data`__: In `pixel_samplers.py`, importance sampling is implemented using the pre-computed weight maps in `dynamic_dataset.py`. This supports any method as long as the dynamic data manager is used, which makes it very easy to use and follows Nerfstudio's format.

### **Importance Sampling**

An important component that is added is (ray) importance sampling, based on DyNeRF. For better support within Nerfstudio, it was implemented slightly differently, and is far from optimal. Indeed, weights are computed at training resolution (and not at x4 downscale like DyNeRF), and the use of different keyframes during training is not done. However, it provides minor compute overhead and performs pretty well. Depth maps are compared to the ones of DyNeRF and are very similar. This short message just suggests it can be improved substantially!

### **Experiments**

The `experiments` folder includes code that was used to automate batches of experiments. Some code is available for automated benchmarks in Nerfstudio (e.g., for Blender) but is barely usable. New experiments can be implemented very easily by importing `ns_experiment.py`, creating an experiment with desired hyperparameters to tune, and run it. Runs are automatically grouped in Weights&Biases (by `experiment_name`) with automatically computed run names (called `timestamp` in Nerfstudio).

Feel free to check any experiment file to see how it works.

### **Scripts**

The `scripts` folder contains various scripts that were used to manipulate datasets and create figures in the thesis. Each file has a header which documents what it does but remember these are simple scripts, and are not intended as clean and documented code.

## Citation

```bib
@inproceedings{lewin2023dynamic,
    title={Dynamic Ne{RF}s for Soccer Scenes},
    author={Sacha Lewin and Maxime Vandegar and Thomas Hoyoux and Olivier Barnich and Gilles Louppe},
    booktitle={6th Int. Workshop on Multimedia Content Analysis in Sports (MMSports'23) @ ACM Multimedia 2023},
    year={2023},
}
```
