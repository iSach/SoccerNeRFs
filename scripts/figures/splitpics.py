"""
W&B images: GT|Pred

This script splits a set {GT|Pred1, GT|Pred2, .., GT|PredN}
into images {GT, Pred1, Pred2, ..., PredN}.
"""
from PIL import Image
import os
os.environ["OPENCV_LOG_LEVEL"]="SILENT"
import cv2
import tkinter as tk
import tkinter.filedialog as fd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    imgs = fd.askopenfilenames(
        title='Predictions',
        filetypes=[("Images", ".png")],
        initialdir=CURRENT_DIR)
    root.destroy()

    image = cv2.imread(imgs[0], cv2.IMREAD_COLOR)
    image = image[:, :image.shape[1]//2]
    cv2.imwrite(os.path.join(os.path.dirname(imgs[0]), "groundtruth.png"), image)

    for img in imgs:
        image = cv2.imread(img, cv2.IMREAD_COLOR)
        image = image[:, image.shape[1]//2:]
        cv2.imwrite(img, image)