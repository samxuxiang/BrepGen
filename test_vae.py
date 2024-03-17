import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import pdb 
import pickle 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.nn import functional as F
from diffusers import AutoencoderKL
from network import AutoencoderKL1D


def compute_bbox_center_and_size(min_corner, max_corner):
    # Calculate the center
    center_x = (min_corner[0] + max_corner[0]) / 2
    center_y = (min_corner[1] + max_corner[1]) / 2
    center_z = (min_corner[2] + max_corner[2]) / 2
    center = np.array([center_x, center_y, center_z])
    # Calculate the size
    size_x = max_corner[0] - min_corner[0]
    size_y = max_corner[1] - min_corner[1]
    size_z = max_corner[2] - min_corner[2]
    size = max(size_x, size_y, size_z)
    return center, size


@torch.no_grad()
def sample():

    model = AutoencoderKL(in_channels=3,
        out_channels=3,
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types= ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        act_fn='silu',
        latent_channels=3,
        norm_num_groups=32,
        sample_size=512,
    )
    model.load_state_dict(torch.load(f'proj_log/abc_surfvae/epoch_100.pt'))
    model = model.cuda().eval()


    model2 = AutoencoderKL1D(
        in_channels=3,
        out_channels=3,
        down_block_types=['DownBlock1D', 'DownBlock1D', 'DownBlock1D'],
        up_block_types=['UpBlock1D', 'UpBlock1D', 'UpBlock1D'],
        block_out_channels=[128, 256, 512],  
        layers_per_block=2,
        act_fn='silu',
        latent_channels=3,
        norm_num_groups=32,
        sample_size=512)
    model2.load_state_dict(torch.load(f'proj_log/abc_edgevae/epoch_300.pt'))
    model2 = model2.cuda().eval()

    with open('data_process/deepcad_data_split_6bit.pkl', "rb") as file:
        data_list = pickle.load(file)['val']

    # vis = []
    for path_idx, path in tqdm(enumerate(data_list)):
        if path_idx > 50: break
        path = os.path.join('data_process', path)
        with open(path, "rb") as file:
            data = pickle.load(file)  
        
        with torch.cuda.amp.autocast():
            _, _, surf_ncs, edge_ncs, _, _, _, _, surf_bbox, edge_bbox, _, _ = data.values()

            surf_uv = torch.FloatTensor(surf_ncs).cuda().permute(0,3,1,2)
            posterior = model.encode(surf_uv).latent_dist
            z = posterior.sample()
            # vis.append(z.detach().cpu().numpy().flatten())
            surf_uv = model.decode(z).sample.permute(0,2,3,1).detach().cpu().numpy()

            edge_u = torch.FloatTensor(edge_ncs).cuda().permute(0,2,1)
            posterior = model2.encode(edge_u).latent_dist
            z = posterior.sample()
            edge_u = model2.decode(z).sample.permute(0,2,1).detach().cpu().numpy()
        
        # ncs + global bbox
        colors = cm.rainbow(np.linspace(0, 1, len(surf_uv)))
        np.random.shuffle(colors)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-1.1, 1.1])  
        ax.set_ylim([-1.1, 1.1])  
        ax.set_zlim([-1.1, 1.1]) 
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        for surf, color, bbox in zip(surf_uv, colors, surf_bbox):
            center, size = compute_bbox_center_and_size(bbox[0:3], bbox[3:])
            surf = surf.reshape(-1,3)
            wcs = surf*(size/2) + center
            ax.scatter(wcs[:,0], wcs[:,1], wcs[:,2], c=color, s=0.5)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-1.1, 1.1])  
        ax.set_ylim([-1.1, 1.1])  
        ax.set_zlim([-1.1, 1.1]) 
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        for edge, bbox in zip(edge_u, edge_bbox):
            center, size = compute_bbox_center_and_size(bbox[0:3], bbox[3:])
            edge = edge.reshape(-1,3)
            wcs = edge*(size/2) + center
            ax.scatter(wcs[:,0], wcs[:,1], wcs[:,2], c='black', s=2)
            
        plt.show()

      
if __name__ == "__main__":
    sample()