"""
Applies the Turbo colormap to a selected list of grayscale images.

Weights&Biases seems to affect colored grayscale images. Modified the code to log grayscale images to W&B.
Then they are turbo colored here.
"""
from matplotlib import cm
import torch
import cv2
import numpy as np
import tkinter as tk
import tkinter.filedialog as fd
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    root = tk.Tk()
    root.withdraw()
    filez = fd.askopenfilenames(
        title='Choose a file', 
        filetypes=[("Images", ".png")],
        initialdir=CURRENT_DIR)
    root.destroy()

    normalize = False
    if len(sys.argv) > 1:
        normalize = sys.argv[1]

    let_black = False
    if len(sys.argv) > 2:
        let_black = sys.argv[2]
    
    for file in filez:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        if normalize:
            img = cv2.normalize(img,  img, 0, 255, cv2.NORM_MINMAX)

        turbo_img = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)

        if let_black:
            turbo_img[img == 0] = 0

        cv2.imwrite(file[:-4] + "_turbo.png", turbo_img)

if __name__ == "__main__":
    main()