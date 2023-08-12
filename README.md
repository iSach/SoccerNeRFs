# **Exploring Dynamic NeRFs for Reconstructing Soccer Scenes**

This repository contains the code of my master's thesis, titled "Exploring Dynamic NeRFs for Reconstructing Soccer Scenes", conducted from 6th Feb to 9th June 2023.

## **Abstract**
In computer vision, novel view synthesis refers to generating new views of an environment from unseen viewpoints, given only a set of images and possibly associated camera poses. This long-standing problem is often tackled by building an underlying 3D model that can then be rendered from new angles. Neural radiance fields (NeRFs), a recent neural method using this approach, brought groundbreaking results compared to previous techniques. The problem can be extended to _4D scene reconstruction_ where the scene is _dynamic_. Solving this even more complex problem could lead to many applications, especially in the considered context of broadcast sequences, such as special effects, bullet time, and more. Sports environments are rarely tackled yet exhibit particular characteristics as they are often composed of small dynamic parts in a large static environment (e.g., players in a stadium). This thesis explores how current state-of-the-art dynamic NeRF models perform in such environments, seeking to assess their applicability and identify crucial components that should be the focus of future work. Various experiments are performed in four environments, including real-world conditions. Although satisfactory performance can be reached in synthetic environments, we show they currently drastically fail in real-world conditions and require additional assumptions to work. However, promising directions are identified toward practical applications.

## **Running the code**

All experiments were carried out in a Docker container, with a custom image. If you're from EVS, the image is available at `evs-innovation-snapshot-docker.artifactory.evs.tv/nerfstudio:slew-latest`. 

Otherwise, install Python 3.8, PyTorch 1.13.1, and CUDA 11.7.
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

Nerfstudio should be ready to use, otherwise have a look at the [official documentation](https://docs.nerf.studio/).

Running a model is very easy. For example, running K-Planes with a new scale (32x), modified importance sampling, and downscaling FPS by 4 on the "Synthetic Player" Scene, and visualizing through the online viewer:
```bash
ns-train k-planes \                                   # Train K-Planes
    --vis viewer \                                    # Use the viewer
    --pipeline.model.multiscale-res 1 2 4 8 16 32 \   # New scale (32x)
    --pipeline.datamanager.ist-range 0.75 \           # Modified IST Range
    synthpaderborn-data \                             # Dataset
    --fps-downsample 4                                # Read 1 frame every 4
```

Default models can require large amounts of VRAM for running, so feel free to tune down the default settings if you have a GPU with less than 12GB.

## **Repository structure**

### **Omitted files**
Some files are omitted from the repository, mainly due to their size. If you're from EVS, they should still be stored in my personal folder on ahl01 (`students/slew`).
* `blender` folder: contains the Blender projects used to generate the synthetic data. It is omitted due to its size (~200GB).
* `data` folder: contains the data used in the experiments. It is omitted due to its size (~800GB).
* `outputs` folder: contains the models checkpoints. It is omitted due to its size (~1.5TB).
* `renders` folder: contains many renders. It is omitted due to its size (600MB).
* `assets/vid` folder: contains the renders in the YT videos. Same as renders.

### **Nerfstudio**
Some contributions were already made public, but most of the work was done in this private repository. Indeed, Nerfstudio is still in _active development_. Due to time constraints, I was not able to make it stable and well documented to be ready for pull requests. I decided it was not worth spending weeks of time during the thesis for that. Additionally, I wanted to be sure new Nerfstudio updates (like 0.2.0) would not break my code.

However, the code was implemented to follow the structure of Nerfstudio as much as possible! This means that the code is very modular and should be easily adapted to new Nerfstudio versions, which might be done in the future. I still tried to document the code as I could for easier understand in case somebody wants to use it! :)

The code was last merged with the main branch of Nerfstudio with the version `0.1.19`. The `nerfstudio` folder corresponds to the actual repository (i.e., [Official Nerfstudio Repo](https://github.com/nerfstudio-project/nerfstudio)), with all changes being in `nerfstudio/nerfstudio`.

Main modifications include:
* __`configs`__: In `method_configs.py`, default configs for K-Planes and NeRFPlayer are added. These are used when running `ns-train`.
* __`models` & `fields`__: **Implementations of the full NeRFPlayer model**: `nerfplayer.py` and `nerfplayer_field.py`, based on the truncated versions (e.g., `nerfplayer_nerfacto.py`). **Implementation of K-Planes**: `kplanes.py` and `kplanes_field.py`, based on [this repo](https://github.com/akristoffersen/nerfstudio_kplanes) which began including in Nerfstudio the [original code](https://github.com/sarafridov/K-Planes) but was still quite buggy and missing components (e.g., bounded scenes, all losses, correct input encoding).
* __`model_components`__: Added all losses (`losses.py`) from K-Planes based on original code. Space TV loss was [wrongly implemented](https://github.com/nerfstudio-project/nerfstudio/pull/1584#issuecomment-1466680500) and thus fixed (almost no difference). Implemented a probability renderer (`renderers.py`) for the decomposition in NeRFPlayer.
* __`utils`__: New dynamic metric in `dynmetric.py`, can be easily integrated by any model inside of the `get_image_metrics_and_images` function. 
* __`data/datasets`__: New dynamic dataset class (`dynamic_dataset.py`) for dynamic scenes with optional depth. This class computes and caches importance sampling maps, more details below.
* __`data/datamanagers`__: Dynamic datamanager for generating dynamic datasets from any data parser that outputs time and possibly depth. Simply replace the vanilla manager by the dynamic one in a method config to use it.
* __`data/dataparsers`__: New data parsers for the experiment environments: `stadium_`, `stadiumplayers_`, `synthpaderborn_`, `paderborn_dataparser.py`, plus a more generic one (`dynamic_dataparser.py`) as an example. They were tuned for the experiments but it is possible to make a single generic parser. :) Also, a data parser for HyperNeRF is included (`hypernerf_dataparser.py`), as experiments were planned on it, but not done in the end.
* __`data`__: In `pixel_samplers.py`, importance sampling is implemented using the pre-computed weight maps in `dynamic_dataset.py`. This supports any method as long as the dynamic data manager is used, which makes it very easy to use and follows Nerfstudio's format.

### **Importance Sampling**

An important component that is added is (ray) importance sampling, based on DyNeRF. For better support within Nerfstudio, it was implemented slightly differently, and is far from optimal. Indeed, weights are computed at training resolution (and not at x4 downscale like DyNeRF), and the use of different keyframes during training is not done. However, it provides minor compute overhead and performs pretty well. Depth maps are compared to the ones of DyNeRF and are very similar. This short message just suggests it can be improved substantially!

### **Experiments**

The `experiments` folder includes code that was used to automate batches of experiments. Some code is available for automated benchmarks in Nerfstudio (e.g., for Blender) but is barely usable. New experiments can be implemented very easily by importing `ns_experiment.py`, creating an experiment with desired hyperparameters to tune, and run it. Runs are automatically grouped in Weights&Biases (by `experiment_name`) with automatically computed run names (called `timestamp` in Nerfstudio).

Feel free to check any experiment file to see how it works.

### **Scripts**

The `scripts` folder contains various scripts that were used to manipulate datasets and create figures in the thesis. Each file has a header which documents what it does but remember these are simple scripts, and are not intended as clean and documented code.
