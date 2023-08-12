"""
This script plots the planes of a trained K-Planes model.

python3 plot_kplane.py /outputs/unnamed/k-planes/.../ <res>

where res = 2 means 64*2^2 = 256, res = 3 means 64*2^3 = 512, etc.
"""
import torch
import matplotlib.pyplot as plt
import sys
import os
import glob

PLANES = {
    'xy': 0,
    'xz': 1,
    'xt': 2,
    'yz': 3,
    'yt': 4,
    'zt': 5,
}

def res(scale_factor):
    return 64*2**scale_factor

model_path = sys.argv[1]
plane_res = 2 # 256
if len(sys.argv) > 2:
    plane_res = int(sys.argv[2])

# load ckpt
model_path = model_path + "nerfstudio_models/" if model_path.endswith('/') else model_path + "/nerfstudio_models/"
# Find most recent .ckpt file in model_path (e.g., step-000029999.ckpt)
latest_file = max(glob.glob(model_path + "*.ckpt"), key=os.path.getctime)
state_dict = torch.load(latest_file)
print(f"Loaded step {latest_file.split('-')[-1].split('.')[0]}.")

# mkdir planes folder in model_path
planes_path = model_path.split('nerfstudio_models')[0] + 'planes/'
if not os.path.exists(planes_path):
    os.makedirs(planes_path)

pipeline = state_dict['pipeline']
for plane_name in PLANES:
    G = pipeline[f'_model.field.grids.{plane_res}.{PLANES[plane_name]}'].squeeze().mean(dim=0)
    print(plane_name, pipeline[f'_model.field.grids.{plane_res}.{PLANES[plane_name]}'].shape)

    # Plot with colormap and show the bar
    # If space-time, plot it as a rectangle of 4/3 aspect ratio 
    if 't' in plane_name:
        plt.figure(figsize=(8,4.5))
   #else:
    #    plt.figure(figsize=(8, 8))
    plt.imshow(G.cpu())
    plt.colorbar()
    plt.xlabel(plane_name[0])
    plt.ylabel(plane_name[1])
    plt.gca().set_aspect('auto')
    plt.savefig(planes_path + f"{plane_name}_{res(plane_res)}.png")
    
    # Close
    plt.close()