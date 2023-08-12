"""
Crops a list of pics (GT+Preds), and saves the GT with a rectangle in it plus extracted rectangles from predictions.

PyGame code from: https://stackoverflow.com/questions/6136588/image-cropping-using-python
"""
import pygame, sys
from PIL import Image
import os
os.environ["OPENCV_LOG_LEVEL"]="SILENT"
import cv2
import argparse
import tkinter as tk
import tkinter.filedialog as fd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def displayImage(screen, px, topleft, prior) -> tuple[int, int, int, int]:
    x, y = topleft
    width =  pygame.mouse.get_pos()[0] - topleft[0]
    height = pygame.mouse.get_pos()[1] - topleft[1]
    if width < 0:
        x += width
        width = abs(width)
    if height < 0:
        y += height
        height = abs(height)

    current = x, y, width, height
    if not (width and height):
        return current
    if current == prior:
        return current

    screen.blit(px, px.get_rect())
    im = pygame.Surface((width, height))
    im.fill((128, 128, 128))
    pygame.draw.rect(im, (32, 32, 32), im.get_rect(), 1)
    im.set_alpha(128)
    screen.blit(im, (x, y))
    pygame.display.flip()

    return (x, y, width, height)

def setup(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    px = pygame.image.frombuffer(image.tobytes(), image.shape[1::-1], "BGR")
    screen = pygame.display.set_mode( px.get_rect()[2:] )
    screen.blit(px, px.get_rect())
    pygame.display.flip()
    return screen, px

def mainLoop(screen, px) -> tuple[int, int, int, int]:
    topleft = bottomright = prior = None
    n=0
    x, y, width, height = 0, 0, 0, 0
    while n!=1:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                if not topleft:
                    topleft = event.pos
                else:
                    bottomright = event.pos
                    n=1
        if topleft:
            x, y, width, height = displayImage(screen, px, topleft, prior)
    return x, y, width, height

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    ground_truth = fd.askopenfilename(
        title='Ground Truth',
        filetypes=[("Images", ".png")],
        initialdir=CURRENT_DIR)

    root = tk.Tk()
    root.withdraw()
    preds = fd.askopenfilenames(
        title='Predictions',
        filetypes=[("Images", ".png")],
        initialdir=os.path.dirname(ground_truth))
    root.destroy()

    pygame.init()
    screen, px = setup(ground_truth)
    x, y, width, height = mainLoop(screen, px)

    image = cv2.imread(ground_truth, cv2.IMREAD_COLOR)
    cv2.rectangle(image, (x, y), (x+width, y+height), (0, 0, 255), 2)
    cv2.imwrite(ground_truth[:-4] + "_rect.png", image)

    for pred in preds:
        image = cv2.imread(pred, cv2.IMREAD_COLOR)
        image = image[y:y+height, x:x+width]
        cv2.imwrite(pred[:-4] + "_crop.png", image)

    image = cv2.imread(ground_truth, cv2.IMREAD_COLOR)
    image = image[y:y+height, x:x+width]
    cv2.imwrite(ground_truth[:-4] + "_crop.png", image)

    pygame.display.quit()