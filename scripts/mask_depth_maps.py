"""
This script processes depth maps from the Paderborn dataset
and hides (or interpolate) the player and the ball.

This one uses the Segment-Anything (https://github.com/facebookresearch/segment-anything)
model with a RetinaNet head to detect the player and the ball.

The underlying masks are a lot better than using e.g. DeepLabV3 for direct segmentation.

Experimentally, three heads for detecting the player and the ball
have been tested: RetinaNet, FasterRCNN and YOLO.

Accuracy was the sole metric used to determine the best model.
The best model was RetinaNet, which is used in this script.
YOLO was worse, expected considering its bad performance on small objects.
FasterRCNN was better than YOLO, but worse than RetinaNet.

ISG was implemented after all this, but would probably be even better.
"""

from typing import List, Tuple
import torch
import numpy as np
import os
import pathlib
from pathlib import Path
import shutil
import PIL
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
import argparse

from torchvision import transforms

DOWNSCALE_ENABLED = True
NB_DOWNSCALES = 2

"""
Possible modes:
  * 'od': Object detection (rectangle mask)
  * 'od_below': Object detection (rectangle mask). 
                Instead of masking, interpolates depth 
                with its bottom neighboring depth value.
  * 'mask': More precise mask with object detection + Segment-Anything.
  * 'mask_below': OD+SAM, but interpolates depth with its bottom neighboring depth value.
  * 'mask_old': Direct segmentation using DeepLabV3, poor results.
  * 'mask_old_below': mask_old with interpolation.
  * 'ist': masks with importance sampling based on temporal difference.

  This list contains modes to run.
"""
MODES = [
    "od",
    "od_below",
    "mask",
    "mask_below",
    # "mask_old",
    # "mask_old_below",
    # "ist",
]


class DepthMaskingMode:
    """
    Abstract class for depth masking modes.


    Args:
        below: Whether to interpolate the depth with the bottom neighboring depth value.
        name: The name of the mode.
    """

    def __init__(self, below, name):
        self.below = below
        self.name = name

    def mask(self, image, depth_map):
        """
        Processes the depth map to mask the player and the ball.

        Args:
            image: The image corresponding to the depth map. [3, H, W]
            depth_map: The depth map to process. [H, W]
        """
        raise NotImplementedError()

    def post_process(self, depth_map, original_depth_map):
        """
        Post-processes the depth map.
        """
        depth_map[original_depth_map == 0] = 0
        return depth_map.cpu().numpy().astype(np.uint16)

    def get_dir(self, path, maps_to_mask):
        """
        Returns the directory where the depth maps should be saved (without downscale).

        e.g. /workspace/data/synth_paderborn/depth-maps-od_below/
        """
        name = maps_to_mask + "-" + self.name
        if self.below:
            name += "_below"
        return path / name

    def save(self, depth_map, base_dir, maps_to_mask, cam, timestep):
        """
        Saves the depth map to a file.
        """
        new_dir = self.get_dir(base_dir, maps_to_mask)
        cv2.imwrite(
            str((new_dir / "1x" / (cam + "_" + str(timestep).zfill(6) + "-depth.png")).absolute()),
            depth_map,
        )


class ODMode(DepthMaskingMode):
    """
    Masks the player and the ball using object detection.

    Masks are filled rectangles.
    """

    def __init__(self, below, name="od"):
        super().__init__(below=below, name=name)
        self.od_model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
            weights=torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        )
        self.od_model = self.od_model.eval().to("cuda")

    @torch.no_grad()
    def mask(self, image, depth_map):
        res = self.od_model(image.unsqueeze(0))[0]

        # Filter by categories: only 1 and 37 (person and ball)
        # Filter by score: only > 0.6
        indices = torch.where(((res["labels"] == 1) | (res["labels"] == 37)) & (res["scores"] > 0.6))[0]

        res["boxes"] = res["boxes"][indices].cpu().numpy()

        for box in res["boxes"]:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            new_val = 0
            if self.below and y2 + 1 < image.shape[1]:
                new_val = depth_map[y2 + 1, x1:x2].mean()
            depth_map[y1:y2, x1:x2] = new_val

        return depth_map


SAM_CKPTS = {
    "B": "/workspace/sam_ckpts/sam_vit_b_01ec64.pth",
    "L": "/workspace/sam_ckpts/sam_vit_l_0b3195.pth",
    "H": "/workspace/sam_ckpts/sam_vit_h_4b8939.pth",
}
SAM_MODELS = {
    "B": "vit_b",
    "L": "vit_l",
    "H": "default",
}


class MaskMode(ODMode):
    """
    Combines object detection and SAM (Segment Anything Model).

    Masks are more precise than ODMode.

    Args:
        sam_model: The SAM model to use. Can be "B", "L" or "H".
                     "B" is the smallest, "H" is the largest.
    """

    def __init__(self, below, sam_model="H"):
        super().__init__(below=below, name="mask")

        self.sam = sam_model_registry[SAM_MODELS[sam_model]](checkpoint=SAM_CKPTS[sam_model]).to("cuda")
        self.predictor = SamPredictor(self.sam)

    @torch.no_grad()
    def mask(self, image, depth_map):
        res = self.od_model(image.unsqueeze(0))[0]

        # Filter by categories: only 1 and 37 (person and ball)
        # Filter by score: only > 0.6
        indices = torch.where(((res["labels"] == 1) | (res["labels"] == 37)) & (res["scores"] > 0.6))[0]

        res["boxes"] = res["boxes"][indices].cpu().numpy()

        # img to numpy
        img = image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        img = (img * 255).astype(np.uint8)
        self.predictor.set_image(img)
        for box in res["boxes"]:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Can probably save time by predicting once per image with all boxes
            masks, _, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],
                multimask_output=False,
            )
            masks = masks[0]

            # Set depth map to 0 where mask is True
            new_val = 0
            if self.below and y2 + 1 < image.shape[1]:
                new_val = depth_map[y2 + 1, x1:x2].mean()
            depth_map[masks] = new_val

        return depth_map


class MaskOldMode(DepthMaskingMode):
    """
    Old masks with DeepLabV3. Deprecated.
    """

    def __init__(self, below):
        super().__init__(below=below, name="mask_old")
        self.model = torch.hub.load("pytorch/vision:v0.10.0", "deeplabv3_resnet101", pretrained=True)
        self.model = self.model.eval().to("cuda")
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @torch.no_grad()
    def mask(self, image, depth_map):
        img_batch = self.norm(image).unsqueeze(0)

        output = self.model(img_batch)["out"][0]
        output_predictions = output.argmax(0)

        output_predictions[depth_map == 0] = 0
        value_to_set = 0
        W = torch.where(output_predictions == 15)
        idx = torch.argmax(W[0])
        lowest_person_y, lowest_person_x = W[0][idx], W[1][idx]
        if lowest_person_y < image.shape[1] - 1:
            value_to_set = depth_map[lowest_person_y + 1, lowest_person_x]

        depth_map[output_predictions == 15] = value_to_set  # person = 15

        return depth_map


# IST mode
class ISTMode(DepthMaskingMode):
    """
    Masks regions with IST.
    """

    def __init__(self):
        super().__init__(below=False, name="ist")
        self.model = torch.hub.load("facebookresearch/detr", "detr_resnet101", pretrained=True)
        self.model = self.model.eval().to("cuda")
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @torch.no_grad()
    def mask(self, image, depth_map):

        return depth_map


def load_image(images_dir: Path, cam: str, time_step: int) -> torch.Tensor:
    """
    Loads an image from the dataset, given a camera and time step.

    Args:
        cam: The camera to load the image from.
        time_step: The time step to load the image from.

    Returns:
        The image as a torch tensor. [3, H, W]

    """
    img = Image.open(images_dir / (cam + "_" + str(time_step).zfill(6) + ".png"))  # [H, W, 3]
    img = torch.from_numpy(np.array(img, dtype="uint8").astype("float32") / 255.0)
    return img.to("cuda").permute(2, 0, 1)


def load_depth_map(depth_dir: Path, cam: str, time_step: int = -1):
    """
    Loads a depth map from the dataset, given a camera and time step.

    Args:
        depth_dir: The directory to load the depth map from.
        cam: The camera to load the depth map from.
        time_step: The time step to load the depth map from.

    Returns:
        The depth map as a torch tensor. [H, W]
    """
    filepath = depth_dir / (cam + "_" + str(time_step).zfill(6) + "-depth.png")
    if not filepath.exists():
        filepath = depth_dir / (cam + "-depth.png")

    if not filepath.exists():
        raise FileNotFoundError(f"Could not find depth map for camera {cam} at time step {time_step}")

    image = cv2.imread(str(filepath.absolute()), cv2.IMREAD_ANYDEPTH)
    image = image.astype(np.float64)
    # image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(image).to("cuda")


def retrieve_cams_time(maps_dir: Path) -> Tuple[List[str], int]:
    """
    Retrieves the camera names and the maximum time step from the images folder.

    Args:
        images_folder: Path to the images folder.
    """
    cams = set()
    max_t = 0
    for filename in os.listdir(maps_dir):
        if filename.endswith(".png"):
            cam = filename.rsplit("_", 1)[0]
            cams.add(cam)
            step = int(filename.rsplit("_", 1)[1].split("-depth.")[0])
            max_t = max(max_t, step)
    return list(cams), max_t


def process_depth_maps(
    images_dir: Path,
    maps_dir: Path,
    maps_to_mask: str,
    mode_name: str,
    cams: List[str],
    max_t: int,
):
    """
    Processes the depth maps.

    Args:
        data_dir: Path to the data folder.
        downscale: Whether to downscale the depth maps.
        below: Whether to set the depth to 0 below the human.
        mode: The mode to use to mask the depth maps.
    """
    below = mode_name.endswith("_below")
    if below:
        mode_name = mode_name[:-6]

    mode: DepthMaskingMode
    if mode_name == "od":
        mode = ODMode(below=below)
    elif mode_name == "mask":
        mode = MaskMode(below=below)
    elif mode_name == "mask_old":
        mode = MaskOldMode(below=below)
    elif mode_name == "ist":
        mode = ISTMode()
    else:
        raise ValueError(f"Invalid mode {mode_name}!")

    base_dir = maps_dir.parent.parent
    new_dir = mode.get_dir(base_dir, maps_to_mask) / "1x"

    if not new_dir.exists():
        new_dir.mkdir(parents=True)

    progress_bar = tqdm(total=(max_t + 1) * len(cams))

    for cam in cams:
        for time_step in range(0, max_t + 1):
            img = load_image(images_dir, cam, time_step)
            depth_map = load_depth_map(maps_dir, cam, time_step)
            original_depth_map = depth_map.clone()

            depth_map = mode.mask(img, depth_map)
            depth_map = mode.post_process(depth_map, original_depth_map)
            mode.save(depth_map, base_dir, maps_to_mask, cam, time_step)

            progress_bar.update(1)

    return mode.get_dir(base_dir, maps_to_mask)


def downscale_depth_maps(new_maps_dir: Path, num_downscales: int = 2):
    """
    Downscale the depth maps.

    Args:
        new_maps_dir: Path to the new depth maps folder. (parent of 1x)
        num_downscales: Number of times to downscale the depth maps.
    """
    x1_dir = new_maps_dir / "1x"
    for power in range(num_downscales):
        downscale_factor = 2 ** (power + 1)
        output_folder_dx = new_maps_dir / (str(downscale_factor) + "x")
        os.system(f"mkdir -p {output_folder_dx}")

        # Set up FFmpeg command
        ffmpeg_cmd = "ffmpeg -y -hide_banner -loglevel error -i {} -q:v 2 -vf scale={}:-1 {}"

        # Loop through all files in input folder
        pbar = tqdm(total=len(os.listdir(x1_dir)), position=0, leave=True)
        pbar.set_description(f"{downscale_factor}x")
        for filename in os.listdir(x1_dir):
            if filename.endswith(".png"):
                # Set up paths to input and output files
                input_file = x1_dir / filename
                output_file_dx = output_folder_dx / filename
                os.system(ffmpeg_cmd.format(input_file, "iw/" + str(downscale_factor), output_file_dx))
                pbar.update(1)


def main():
    """
    Main function of the script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps_dir", type=str, help="Path to the depth maps folder.")
    args = parser.parse_args()

    if not args.maps_dir:
        parser.print_help()
        return

    maps_dir = Path(args.maps_dir)
    maps_to_mask = maps_dir.name
    data_dir = maps_dir.parent

    images_dir = data_dir / "images" / "1x"
    maps_dir = maps_dir / "1x"

    print(images_dir.absolute())
    print(maps_dir.absolute())

    # Find cameras
    cams, max_t = retrieve_cams_time(maps_dir)

    print("Found", len(cams), "cameras:", cams)
    print(f"Found time range: [[0, {max_t}]]")

    for mode in MODES:
        print(f"Processing depth maps with mode {mode}...")
        new_maps_dir = process_depth_maps(images_dir, maps_dir, maps_to_mask, mode, cams, max_t)

        if DOWNSCALE_ENABLED:
            print("Downscaling depth maps...")
            downscale_depth_maps(new_maps_dir, NB_DOWNSCALES)
        print()

    print("\U0001f389 Done!")


# Write a main function to run the script
# that requires one argument: the depth maps to mask folder
if __name__ == "__main__":
    main()
