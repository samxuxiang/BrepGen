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
from network import SurfPosNet
from utils import *
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers import DDPMScheduler, DDIMScheduler, PNDMScheduler


FAST_INFERENCE = True

@torch.no_grad()
def sample():

    surfPos_model = SurfPosNet(sincos=False)
    surfPos_model.load_state_dict(torch.load(f'proj_log/abc_ldm_surfpos_nosincos/epoch_200.pt'))
    surfPos_model = surfPos_model.cuda().eval()

    noise_scheduler = PNDMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start = 0.0001,
        beta_end = 0.02,
    )

    noise_scheduler2 = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start = 0.0001,
        beta_end = 0.02,
        clip_sample = True,
        clip_sample_range=3
    )  


    for gen_idx in range(1000000):
        ###########################################
        # STEP 1-1: generate the surface position #
        ###########################################
        surfPos = randn_tensor((1,80,6)).cuda()  

        # Fast inference speed, 200 total time steps 
        if FAST_INFERENCE:
            noise_scheduler.set_timesteps(100)  
            for t in tqdm(noise_scheduler.timesteps[:-25]):
                timesteps = t.reshape(-1).cuda()
                pred = surfPos_model(surfPos, timesteps)
                surfPos = noise_scheduler.step(pred, t, surfPos).prev_sample
            noise_scheduler2.set_timesteps(500)  
            for t in tqdm(noise_scheduler2.timesteps[-125:]):
                timesteps = t.reshape(-1).cuda()
                pred = surfPos_model(surfPos, timesteps)
                surfPos = noise_scheduler2.step(pred, t, surfPos).prev_sample

        # Slow inference speed, 400 total time steps (as in paper)
        else:
            noise_scheduler.set_timesteps(200)  
            for t in tqdm(noise_scheduler.timesteps[:-50]):
                timesteps = t.reshape(-1).cuda()
                pred = surfPos_model(surfPos, timesteps)
                surfPos = noise_scheduler.step(pred, t, surfPos).prev_sample
            noise_scheduler2.set_timesteps(1000)  
            for t in tqdm(noise_scheduler2.timesteps[-250:]):
                timesteps = t.reshape(-1).cuda()
                pred = surfPos_model(surfPos, timesteps)
                surfPos = noise_scheduler2.step(pred, t, surfPos).prev_sample

        #######################################
        # STEP 1-2: remove duplicate surfaces #
        #######################################
        bboxes = np.round(surfPos[0].unflatten(-1,torch.Size([2,3])).detach().cpu().numpy(), 4)   
        non_repeat = bboxes[:1]
        for bbox_idx, bbox in enumerate(bboxes):
            diff = np.max(np.max(np.abs(non_repeat - bbox),-1),-1)
            same = diff < 0.05
            bbox_rev = bbox[::-1]  # also test reverse bbox for matching
            diff_rev = np.max(np.max(np.abs(non_repeat - bbox_rev),-1),-1)
            same_rev = diff_rev < 0.05
            if same.sum()>=1 or same_rev.sum()>=1:
                continue # repeat value
            else:
                non_repeat = np.concatenate([non_repeat, bbox[np.newaxis,:,:]],0)
        bboxes = non_repeat.reshape(len(non_repeat),-1)

        num_surf = len(bboxes)
        surf_mask = (torch.zeros((1, num_surf)) == 0).cuda()
        surf_mask[:, :] = False
        surfPos = torch.FloatTensor(bboxes).cuda().unsqueeze(0)
       
        #####################
        ### Visualization ###
        #####################
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-1.1, 1.1])  
        ax.set_ylim([-1.1, 1.1])  
        ax.set_zlim([-1.1, 1.1]) 
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        bboxes = surfPos[0].detach().cpu().numpy()/3

        colors = cm.rainbow(np.linspace(0, 1, len(bboxes)))
        np.random.shuffle(colors)

        for bbox, color in zip (bboxes, colors):
            plot_3d_bbox(ax, bbox[0:3], bbox[3:6], color)
        plt.tight_layout()
        plt.show()  

      
if __name__ == "__main__":
    sample()