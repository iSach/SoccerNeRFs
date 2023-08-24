import json
from dataclasses import asdict, dataclass
from os import path
from typing import Any, Optional
from os.path import join
import bpy
from contextlib import contextmanager
import pathlib
import numpy as np
from pprint import pprint

CAM_WIDTH = 1920
CAM_HEIGHT = 1080


@dataclass
class CamParams:
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float


def matrix_to_list(matrix):
    return [[e for e in row] for row in matrix]


def transform_matrix(cam, scale: float = 1.):
    c2w = cam.matrix_world.copy()
    return c2w


def cam_params(cam) -> CamParams:
    f, fx, fy, cx, cy, k1, k2, p1, p2 = 0, 0, 0, 0, 0, 0, 0, 0, 0

    assert cam.data.type == 'PERSP', 'Only perspective cameras are supported'

    f = cam.data.lens  # mm
    fx = fy = f * CAM_WIDTH / cam.data.sensor_width  # px
    cx = CAM_WIDTH / 2
    cy = CAM_HEIGHT / 2

    return CamParams(fx, fy, cx, cy, k1, k2, p1, p2)


DEPTH_EXT = '.png'


def name_to_depth_path(name: str, depth_dir: str) -> str:
    return join(depth_dir, name + '-depth' + DEPTH_EXT)

def name_to_image_path(name: str, image_dir: str) -> str:
    return join(image_dir, name + '.png')

def generate_transforms(
    cameras: list[bpy.types.Object],
    scale: float = 1.,
    image_dir: str = 'images',
    depth_dir: Optional[str] = None,
) -> dict[str, Any]:

    transforms: dict[str, Any] = {
        'frames': [],
    }

    for i, cam in enumerate(cameras):
        params = cam_params(cam)
        frame: dict[str, Any] = {
            'file_path': name_to_image_path(cam.name, image_dir),
            'transform_matrix': matrix_to_list(transform_matrix(cam, scale)),
            'w': CAM_WIDTH,
            'h': CAM_HEIGHT,
        }

        if depth_dir is not None:
            frame['depth_path'] = name_to_depth_path(cam.name, depth_dir)

        for k, v in asdict(params).items():
            if v != 0.:
                frame[k] = v

        transforms['frames'].append(frame)
    return transforms


def get_cameras(collection_name: str) -> list[bpy.types.Object]:
    """
    Get all cameras in the collection
    """
    cameras = []
    for obj in bpy.data.collections[collection_name].objects:
        if obj.type == 'CAMERA':
            cameras.append(obj)

    return cameras

def setup_frames(num_frames: int):
    """Set the number of frames to render"""
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = num_frames

def bind_camera_to_frame(camera: bpy.types.Object, frame: int):
    """Bind the camera to the frame"""
    bpy.context.scene.frame_set(frame)
    bpy.context.scene.camera = camera
    bpy.ops.marker.camera_bind()

def render_camera(camera: bpy.types.Object, output: str):
    """Render the camera"""
    bpy.context.scene.camera = camera
    bpy.context.scene.render.filepath = output
    bpy.ops.render.render(write_still=True)

def generate_frames(camera_collection):
    """
    Script that generate a frame for each camera in the dataset
    """

    cameras = get_cameras(camera_collection)
    setup_frames(len(cameras))
    
    T = 100
    for i, cam in enumerate(cameras):
        for t in range(T):
            bpy.context.scene.frame_set(t)
            formatted_step = str(t).zfill(6)
            render_camera(cam, f'images/{cam.name}_{formatted_step}.png')

def generate_depth(camera_collection):
    """
    Script that generate a depth map for each camera in the dataset
    """

    cameras = get_cameras(camera_collection)
    setup_frames(len(cameras))
    tree = bpy.context.scene.node_tree

    # Depth maps are euclidean (="diagonal" distance, *not* z-distance)
    
    T = 100
    for i, cam in enumerate(cameras):
        for t in range(T):
            formatted_step = str(t).zfill(6)
            bpy.context.scene.frame_set(t)
            bpy.context.scene.camera = cam
            bpy.context.scene.render.filepath = f'images/{cam.name}_{formatted_step}.png'
            bpy.ops.render.render(write_still=True)
            z = bpy.data.images['Viewer Node']
            w, h = z.size
            dmap = np.array(z.pixels[:], dtype=np.float32) # convert to numpy array
            dmap = np.reshape(dmap, (h, w, 4))[:,:,0]
            dmap = np.rot90(dmap, k=2)
            dmap = np.fliplr(dmap)
            np.savez_compressed(f'depth_maps/{cam.name}_{formatted_step}-depth.npz', dmap=dmap)


def main():
    camera_collection = 'Your Blender Collection Name'

    # ------------------------------------------------------
    #                      Camera Poses
    # ------------------------------------------------------
    # Replace by the name of Blender collection containing your cameras (it's like a folder in the layers)
    cameras = get_cameras(camera_collection)
    transforms = generate_transforms(cameras, image_dir='')
    with open('transforms.json', 'w') as f:
        json.dump(transforms, f, indent=2)

    # ------------------------------------------------------
    #                    Generating Frames
    # ------------------------------------------------------

    generate_frames(camera_collection)
    # generate_depth(camera_collection) # In case you need the depth, can help for sparser setups.

if __name__ == "__main__":
    main()