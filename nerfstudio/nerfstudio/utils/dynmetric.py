"""
This module contains the DynMetric metric.

The DynMetric metric is a modified version of the existing metrics, which is
used to measure the quality of a compressed image. The DynMetric metric
is used to measure the quality of a frame from a dynamic scene.

It was designed to be used with dynamic datasets in order to better measure
the reconstruction around dynamic content.

Although the train and eval losses are indeed biased towards these dynamic regions
once IST is toggled on, the DynMetric is meant to be a separate metrics that is focused
around these zones and not only on sampled pixels, to be more representative.

It was designed specifically to have a better quantitative analysis between different measures,
as PSNR, SSIM and LPIPS are not good metrics for comparing small dynamic content in the frames.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from torchtyping import TensorType


class DynMetric:
    def __init__(self, psnr, ssim, lpips, device, w_factor=7, h_factor=2.5):
        """
        The underyling metric can be changed here.
        """
        self.psnr = psnr
        self.ssim = ssim
        self.lpips = lpips
        self.w_factor = w_factor
        self.h_factor = h_factor
        # FasterRCNN is used because it detects small objects better than YOLO.
        # self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        # RetinaNet performs better than FasterRCNN, verified experimentally on Paderborn and its synthetic version.
        self.model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
            weights=torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        )
        self.model = self.model.eval()
        self.model = self.model.to(device)

    @torch.no_grad()
    def __call__(self, true_image: TensorType["1", "C", "H", "W"], pred_image: TensorType["1", "C", "H", "W"]):
        """Computes the DynMetric metric.

        The way it works is by detecting dynamic regions using an object detection method.

        Specifically, it detects humans and balls, which is obviously a limitation of this method.

        The underlying backbone is then computed in these regions only.

        Args:
            true_image: The ground truth image. [1, C, H, W]
            pred_image: The predicted image. [1, C, H, W]
            max_val: The maximum value of the image.
            eps: A small value to avoid division by zero.

        Returns:
            new_img: Edited image with boxes shown [H, W, C] for wandb
            dynmetric: The metric value.
        """
        _, C, H, W = true_image.shape
        res = self.model(true_image)[0]  #  yields [C, H, W]

        # Filter by categories: only 1 and 37 (person and ball)
        # Filter by score
        indices = torch.where(((res["labels"] == 1) | (res["labels"] == 37)) & (res["scores"] > 0.6))[0]

        res["boxes"] = res["boxes"][indices]  # [N, 4]
        res["labels"] = res["labels"][indices]  # [N]

        # return 0 if no boxes are found
        if len(res["boxes"]) == 0:
            return true_image[0].permute(1, 2, 0), np.nan, np.nan, np.nan

        ball_boxes = []
        person_boxes = []
        for i in range(len(res["boxes"])):
            box = res["boxes"][i]
            label = res["labels"][i]

            if label == 1:
                person_boxes.append(box)
            else:
                ball_boxes.append(box)

        # Filter human boxes to only keep the one closest to the center of the image
        # using euclidean distance
        #
        # Ideally, this should be improved but requires to detect only players :-)
        # Otherwise it makes the metric useless because side persons are considered important
        # but are pretty much static (e.g., in Paderborn scene)
        if len(person_boxes) > 1:
            person_boxes = [
                min(
                    person_boxes,
                    key=lambda box: ((box[0] + box[2]) / 2 - W / 2) ** 2 + ((box[1] + box[3]) / 2 - H / 2) ** 2,
                )
            ]

        boxes = []
        for box in person_boxes + ball_boxes:
            # Rescale the bounding boxes
            boxes.append(rescale_bbox(box, self.w_factor, self.h_factor, true_image.shape[3], true_image.shape[2]))

        # Compute DynMetric
        box_sizes = []
        lpips_sizes = []
        psnrs = []
        ssims = []
        lpipss = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            box_size = (x2 - x1) * (y2 - y1)

            # Compute backbone in the box
            psnr = self.psnr(true_image[:, :, y1:y2, x1:x2], pred_image[:, :, y1:y2, x1:x2])
            psnrs.append(psnr.cpu())
            ssim = self.ssim(true_image[:, :, y1:y2, x1:x2], pred_image[:, :, y1:y2, x1:x2])
            ssims.append(ssim.cpu())
            box_sizes.append(box_size)
            if min(x2 - x1, y2 - y1) >= 32:
                lpips = self.lpips(true_image[:, :, y1:y2, x1:x2], pred_image[:, :, y1:y2, x1:x2])
                lpipss.append(lpips.cpu())
                lpips_sizes.append(box_size)

        # Draw boxes on image
        img_uint8 = (true_image[0] * 255).to(torch.uint8)
        new_img = draw_bounding_boxes(img_uint8, torch.tensor(boxes), width=2, colors="black")
        new_img = new_img.to(torch.float32) / 255

        # C, H, W to H, W, C
        new_img = new_img.permute(1, 2, 0)

        # Compute weighted average of values
        dpsnr = np.average(psnrs, weights=box_sizes)
        dssim = np.average(ssims, weights=box_sizes)
        dlpips = 0
        if len(lpipss) > 0:
            dlpips = np.average(lpipss, weights=lpips_sizes)

        # Replace 0 by NaN, easier to ignore visually on w&b graphs
        if dpsnr < 1e-4:
            dpsnr = np.nan
        if dssim < 1e-4:
            dssim = np.nan
        if dlpips < 1e-4:
            dlpips = np.nan

        return new_img, dpsnr, dssim, dlpips


def rescale_bbox(bbox, w_factor, h_factor, img_width, img_height):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    new_width = int(width * w_factor)
    new_height = int(height * h_factor)

    # Calculate the difference between new and old dimensions
    width_diff = (new_width - width) / 2
    height_diff = (new_height - height) / 2

    # Adjust the bounding box coordinates
    x1 = max(0, x1 - width_diff)
    x2 = x1 + new_width
    y1 = max(0, y1 - height_diff)
    y2 = y1 + new_height

    # Check if x2 or y2 is out of image boundaries
    if x2 > img_width:
        x1 -= x2 - img_width
        x2 = img_width
    if y2 > img_height:
        y1 -= y2 - img_height
        y2 = img_height

    return x1, y1, x2, y2
