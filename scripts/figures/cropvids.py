"""
Crops a set of videos and outputs the entire frames with a rectangle + the extracted rectangle image.

Allows to select a region graphically.

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

def setup(path, frame_number):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    _, image = cap.read()
    cap.release()
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
    # Command is "python3 cropvids.py frame_nb vid1.mp4 vid2.mp4 ... vidN.mp4"
    parser = argparse.ArgumentParser(description='Crop videos')
    parser.add_argument('frame', type=float, help='Time code (in seconds)')
    parser.add_argument('input', type=str, help='Input video', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    
    time_code = args.frame
    vids_list = args.input

    if vids_list == []:
        root = tk.Tk()
        root.withdraw()
        filez = fd.askopenfilenames(
            title='Choose a file', 
            filetypes=[("Videos", ".mp4")],
            initialdir=CURRENT_DIR)
        root.destroy()

        vids_list = list(filez)

    common_folder = os.path.commonpath(vids_list)
    if common_folder.endswith('.mp4'):
        common_folder = common_folder.rsplit('/', 1)[0]

    cap = cv2.VideoCapture(vids_list[0])
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    frame_nb = int(time_code * fps)
    print(f"Time code {time_code}s = frame n°{frame_nb} ({fps}fps)")

    pygame.init()
    screen, px = setup(vids_list[0], frame_nb)
    x, y, width, height = mainLoop(screen, px)

    print(f"Frame {frame_nb} : x={x} y={y} W={width} H={height}")
    print(f"Saving to folder {common_folder}")
    print(f"Extracting first video with rectangle... [{vids_list[0].split('/')[-1].split('.')[0]}_{frame_nb}_rect.png]")
    cap = cv2.VideoCapture(vids_list[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nb)
    _, image = cap.read()
    cap.release()
    cv2.rectangle(image, (x, y), (x+width, y+height), (0, 0, 255), 2)
    cv2.imwrite(f"{common_folder}/{vids_list[0].split('/')[-1].split('.')[0]}_{frame_nb}_rect.png", image)
    
    for vid in vids_list:
        frame_name = vid.split('/')[-1].split('.')[0]
        print(f"Extracting {frame_name} with rectangle... [{frame_name}_{frame_nb}_crop.png]")
        cap = cv2.VideoCapture(vid)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nb)
        _, image = cap.read()
        cap.release()
        image = image[y:y+height, x:x+width]
        cv2.imwrite(f"{common_folder}/{vid.split('/')[-1].split('.')[0]}_{frame_nb}_crop.png", image)

    pygame.display.quit()
