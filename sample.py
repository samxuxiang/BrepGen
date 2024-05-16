import os
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from network import *
from diffusers import DDPMScheduler, PNDMScheduler
from OCC.Extend.DataExchange import write_stl_file, write_step_file
from utils import (
    randn_tensor,
    compute_bbox_center_and_size,
    generate_random_string,
    construct_brep,
    detect_shared_vertex,
    detect_shared_edge,
    joint_optimize
)


text2int = {'uncond':0, 
            'bathtub':1, 
            'bed':2, 
            'bench':3, 
            'bookshelf':4,
            'cabinet':5, 
            'chair':6, 
            'couch':7, 
            'lamp':8, 
            'sofa':9, 
            'table':10
            }


def sample(eval_args):

    # Inference configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = eval_args['batch_size']
    z_threshold = eval_args['z_threshold']
    bbox_threshold =eval_args['bbox_threshold']
    save_folder = eval_args['save_folder']
    num_surfaces = eval_args['num_surfaces'] 
    num_edges = eval_args['num_edges']

    if eval_args['use_cf']:
        class_label = torch.LongTensor([text2int[eval_args['class_label']]]*batch_size + \
                                       [text2int['uncond']]*batch_size).cuda().reshape(-1,1) 
        w = 0.6
    else:
        class_label = None

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    surfPos_model = SurfPosNet(eval_args['use_cf'])
    surfPos_model.load_state_dict(torch.load(eval_args['surfpos_weight']))  
    surfPos_model = surfPos_model.to(device).eval()

    surfZ_model = SurfZNet(eval_args['use_cf'])
    surfZ_model.load_state_dict(torch.load(eval_args['surfz_weight']))
    surfZ_model = surfZ_model.to(device).eval()

    edgePos_model = EdgePosNet(eval_args['use_cf'])
    edgePos_model.load_state_dict(torch.load(eval_args['edgepos_weight']))
    edgePos_model = edgePos_model.to(device).eval()

    edgeZ_model = EdgeZNet(eval_args['use_cf'])
    edgeZ_model.load_state_dict(torch.load(eval_args['edgez_weight']))
    edgeZ_model = edgeZ_model.to(device).eval()

    surf_vae = AutoencoderKLFastDecode(in_channels=3,
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
    surf_vae.load_state_dict(torch.load(eval_args['surfvae_weight']), strict=False)
    surf_vae = surf_vae.to(device).eval()

    edge_vae = AutoencoderKL1DFastDecode(
        in_channels=3,
        out_channels=3,
        down_block_types=['DownBlock1D', 'DownBlock1D', 'DownBlock1D'],
        up_block_types=['UpBlock1D', 'UpBlock1D', 'UpBlock1D'],
        block_out_channels=[128, 256, 512],  
        layers_per_block=2,
        act_fn='silu',
        latent_channels=3,
        norm_num_groups=32,
        sample_size=512
    )
    edge_vae.load_state_dict(torch.load(eval_args['edgevae_weight']), strict=False)
    edge_vae = edge_vae.to(device).eval()

    pndm_scheduler = PNDMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start = 0.0001,
        beta_end = 0.02,
    )

    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start = 0.0001,
        beta_end = 0.02,
        clip_sample = True,
        clip_sample_range=3
    ) 


    with torch.no_grad():
        with torch.cuda.amp.autocast():
        
            ###########################################
            # STEP 1-1: generate the surface position #
            ###########################################
            surfPos = randn_tensor((batch_size, num_surfaces, 6)).to(device)

            pndm_scheduler.set_timesteps(200)  
            for t in tqdm(pndm_scheduler.timesteps[:158]):#
                timesteps = t.reshape(-1).cuda()
                if class_label is not None:
                    _surfPos_ = surfPos.repeat(2,1,1)
                    pred = surfPos_model(_surfPos_, timesteps, class_label)
                    pred = pred[:batch_size] * (1+w) - pred[batch_size:] * w
                else:
                    pred = surfPos_model(surfPos, timesteps, class_label)
                surfPos = pndm_scheduler.step(pred, t, surfPos).prev_sample
           
            # Late increase for ABC/DeepCAD (slightly more efficient)
            if not eval_args['use_cf']:
                surfPos = surfPos.repeat(1,2,1)
                num_surfaces *= 2

            ddpm_scheduler.set_timesteps(1000)  
            for t in tqdm(ddpm_scheduler.timesteps[-250:]):   
                timesteps = t.reshape(-1).cuda()
                if class_label is not None:
                    _surfPos_ = surfPos.repeat(2,1,1)
                    pred = surfPos_model(_surfPos_, timesteps, class_label)
                    pred = pred[:batch_size] * (1+w) - pred[batch_size:] * w
                else:
                    pred = surfPos_model(surfPos, timesteps, class_label)
                surfPos = ddpm_scheduler.step(pred, t, surfPos).prev_sample
           

            #######################################
            # STEP 1-2: remove duplicate surfaces #
            #######################################
            surfPos_deduplicate = []
            surfMask_deduplicate = []
            for ii in range(batch_size):
                bboxes = np.round(surfPos[ii].unflatten(-1,torch.Size([2,3])).detach().cpu().numpy(), 4)   
                non_repeat = bboxes[:1]
                for bbox_idx, bbox in enumerate(bboxes):
                    diff = np.max(np.max(np.abs(non_repeat - bbox),-1),-1)
                    same = diff < bbox_threshold
                    bbox_rev = bbox[::-1]  # also test reverse bbox for matching
                    diff_rev = np.max(np.max(np.abs(non_repeat - bbox_rev),-1),-1)
                    same_rev = diff_rev < bbox_threshold
                    if same.sum()>=1 or same_rev.sum()>=1:
                        continue # repeat value
                    else:
                        non_repeat = np.concatenate([non_repeat, bbox[np.newaxis,:,:]],0)
                bboxes = non_repeat.reshape(len(non_repeat),-1)

                surf_mask = torch.zeros((1, len(bboxes))) == 1
                bbox_padded = torch.concat([torch.FloatTensor(bboxes), torch.zeros(num_surfaces-len(bboxes),6)])
                mask_padded = torch.concat([surf_mask, torch.zeros(1, num_surfaces-len(bboxes))==0], -1)
                surfPos_deduplicate.append(bbox_padded)
                surfMask_deduplicate.append(mask_padded)

            surfPos = torch.stack(surfPos_deduplicate).cuda()
            surfMask = torch.vstack(surfMask_deduplicate).cuda()


            #################################
            # STEP 1-3:  generate surface z #
            #################################
            surfZ = randn_tensor((batch_size, num_surfaces, 48)).to(device)
            
            pndm_scheduler.set_timesteps(200)   
            for t in tqdm(pndm_scheduler.timesteps): 
                timesteps = t.reshape(-1).cuda()
                if class_label is not None:
                    _surfZ_ = surfZ.repeat(2,1,1)
                    _surfPos_ = surfPos.repeat(2,1,1)
                    _surfMask_ = surfMask.repeat(2,1)
                    pred = surfZ_model(_surfZ_, timesteps, _surfPos_, _surfMask_, class_label)
                    pred = pred[:batch_size] * (1+w) - pred[batch_size:] * w
                else:
                    pred = surfZ_model(surfZ, timesteps, surfPos, surfMask, class_label)
                surfZ = pndm_scheduler.step(pred, t, surfZ).prev_sample
        

            ########################################
            # STEP 2-1: generate the edge position #
            ########################################
            edgePos = randn_tensor((batch_size, num_surfaces, num_edges, 6)).cuda()
          
            pndm_scheduler.set_timesteps(200)  
            for t in tqdm(pndm_scheduler.timesteps[:158]):  
                timesteps = t.reshape(-1).cuda()   
                if class_label is not None:
                    _surfZ_ = surfZ.repeat(2,1,1)
                    _surfPos_ = surfPos.repeat(2,1,1)
                    _surfMask_ = surfMask.repeat(2,1)
                    _edgePos_ = edgePos.repeat(2,1,1,1)
                    noise_pred = edgePos_model(_edgePos_, timesteps, _surfPos_, _surfZ_, _surfMask_, class_label)
                    noise_pred = noise_pred[:batch_size] * (1+w) - noise_pred[batch_size:] * w
                else:
                    noise_pred = edgePos_model(edgePos, timesteps, surfPos, surfZ, surfMask, class_label)
                edgePos = pndm_scheduler.step(noise_pred, t, edgePos).prev_sample

            ddpm_scheduler.set_timesteps(1000)  
            for t in tqdm(ddpm_scheduler.timesteps[-250:]):
                timesteps = t.reshape(-1).cuda()   
                if class_label is not None:
                    _surfZ_ = surfZ.repeat(2,1,1)
                    _surfPos_ = surfPos.repeat(2,1,1)
                    _surfMask_ = surfMask.repeat(2,1)
                    _edgePos_ = edgePos.repeat(2,1,1,1)
                    noise_pred = edgePos_model(_edgePos_, timesteps, _surfPos_, _surfZ_, _surfMask_, class_label)
                    noise_pred = noise_pred[:batch_size] * (1+w) - noise_pred[batch_size:] * w
                else:
                    noise_pred = edgePos_model(edgePos, timesteps, surfPos, surfZ, surfMask, class_label)
                edgePos = ddpm_scheduler.step(noise_pred, t, edgePos).prev_sample


            ####################################################
            # STEP 2-2: remove duplicate edges per face (bbox) #
            ####################################################
            edgeM = surfMask.unsqueeze(-1).repeat(1, 1, num_edges)

            for ii in range(batch_size):
                edge_bboxs = edgePos[ii][~surfMask[ii]].detach().cpu().numpy()

                for surf_idx, bboxes in enumerate(edge_bboxs):
                    bboxes = bboxes.reshape(len(bboxes),2,3)
                    valid_bbox = bboxes[0:1]
                    for bbox_idx, bbox in enumerate(bboxes):
                        diff = np.max(np.max(np.abs(valid_bbox - bbox),-1),-1)
                        bbox_rev = bbox[::-1]  # also test reverse bbox for matching
                        diff_rev = np.max(np.max(np.abs(valid_bbox - bbox_rev),-1),-1)
                        same = diff < bbox_threshold
                        same_rev = diff_rev < bbox_threshold
                        if same.sum()>=1 or same_rev.sum()>=1:
                            edgeM[ii, surf_idx, bbox_idx] = True
                            continue # repeat value
                        else:
                            valid_bbox = np.concatenate([valid_bbox, bbox[np.newaxis,:,:]],0)
                    edgeM[ii, surf_idx, 0] = False  # set first one to False  


            ##############################
            # STEP 2-3: generate edge zv #
            ##############################   
            edgeZV = randn_tensor((batch_size, num_surfaces, num_edges, 18)).cuda()

            pndm_scheduler.set_timesteps(200)   
            for t in tqdm(pndm_scheduler.timesteps):
                timesteps = t.reshape(-1).cuda()   
                if class_label is not None:
                    _surfZ_ = surfZ.repeat(2,1,1)
                    _surfPos_ = surfPos.repeat(2,1,1)
                    _edgePos_ = edgePos.repeat(2,1,1,1)
                    _edgeM_ = edgeM.repeat(2,1,1)
                    _edgeZV_ = edgeZV.repeat(2,1,1,1)
                    noise_pred = edgeZ_model(_edgeZV_, timesteps, _edgePos_, _surfPos_, _surfZ_, _edgeM_, class_label)
                    noise_pred = noise_pred[:batch_size] * (1+w) - noise_pred[batch_size:] * w
                else:
                    noise_pred = edgeZ_model(edgeZV, timesteps, edgePos, surfPos, surfZ, edgeM, class_label)
                edgeZV = pndm_scheduler.step(noise_pred, t, edgeZV).prev_sample

            edgeZV[edgeM] = 0 # set removed data to 0
            edge_z = edgeZV[:,:,:,:12]
            edgeV = edgeZV[:,:,:,12:].detach().cpu().numpy()

            # Decode the surfaces
            surf_ncs = surf_vae(surfZ.unflatten(-1,torch.Size([16,3])).flatten(0,1).permute(0,2,1).unflatten(-1,torch.Size([4,4])))
            surf_ncs = surf_ncs.permute(0,2,3,1).unflatten(0, torch.Size([batch_size, num_surfaces])).detach().cpu().numpy()
            
            # Decode the edges
            edge_ncs = edge_vae(edge_z.unflatten(-1,torch.Size([4,3])).reshape(-1,4,3).permute(0,2,1))
            edge_ncs = edge_ncs.permute(0,2,1).reshape(batch_size, num_surfaces, num_edges, 32, 3).detach().cpu().numpy()


            edge_mask = edgeM.detach().cpu().numpy()     
            edge_pos = edgePos.detach().cpu().numpy() / 3.0
            surfPos = surfPos.detach().cpu().numpy()  / 3.0


    #############################################
    ### STEP 3: Post-process (per-single CAD) ###
    #############################################
    for batch_idx in range(batch_size):
        # Per cad (not including invalid faces)
        surfMask_cad = surfMask[batch_idx].detach().cpu().numpy()
        edge_mask_cad = edge_mask[batch_idx][~surfMask_cad]
        edge_pos_cad = edge_pos[batch_idx][~surfMask_cad]
        edge_ncs_cad = edge_ncs[batch_idx][~surfMask_cad]
        edgeV_cad = edgeV[batch_idx][~surfMask_cad]
        edge_z_cad = edge_z[batch_idx][~surfMask[batch_idx]].detach().cpu().numpy()[~edge_mask_cad]
        surf_z_cad = surfZ[batch_idx][~surfMask[batch_idx]].detach().cpu().numpy()
        surf_pos_cad = surfPos[batch_idx][~surfMask_cad]

        # Retrieve vertices based on edge start/end
        edgeV_bbox = []
        for bbox, ncs, mask in zip(edge_pos_cad, edge_ncs_cad, edge_mask_cad):
            epos = bbox[~mask]
            edge = ncs[~mask]
            bbox_startends = []
            for bb, ee in zip(epos, edge): 
                bcenter, bsize = compute_bbox_center_and_size(bb[0:3], bb[3:])
                wcs = ee*(bsize/2) + bcenter
                bbox_start_end = wcs[[0,-1]]
                bbox_start_end = bbox_start_end.reshape(2,3)
                bbox_startends.append(bbox_start_end.reshape(1,2,3))
            bbox_startends = np.vstack(bbox_startends)
            edgeV_bbox.append(bbox_startends)
        
        ### 3-1: Detect shared vertices ###
        try:
            unique_vertices, new_vertex_dict = detect_shared_vertex(edgeV_cad, edge_mask_cad, edgeV_bbox)
        except Exception as e:
            print('Vertex detection failed...')
            continue
        
        ### 3-2: Detect shared edges ###
        try:
            unique_faces, unique_edges, FaceEdgeAdj, EdgeVertexAdj = detect_shared_edge(unique_vertices, new_vertex_dict, edge_z_cad, surf_z_cad, z_threshold, edge_mask_cad)
        except Exception as e:
            print('Edge detection failed...')
            continue
        
        # Decode unique faces / edges
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                surf_ncs_cad = surf_vae(torch.FloatTensor(unique_faces).cuda().unflatten(-1,torch.Size([16,3])).permute(0,2,1).unflatten(-1,torch.Size([4,4])))
                surf_ncs_cad = surf_ncs_cad.permute(0,2,3,1).detach().cpu().numpy()
                edge_ncs_cad = edge_vae(torch.FloatTensor(unique_edges).cuda().unflatten(-1,torch.Size([4,3])).permute(0,2,1))
                edge_ncs_cad = edge_ncs_cad.permute(0,2,1).detach().cpu().numpy()

        #### 3-3: Joint Optimize ###
        num_edge = len(edge_ncs_cad)
        num_surf = len(surf_ncs_cad)
        surf_wcs, edge_wcs = joint_optimize(surf_ncs_cad, edge_ncs_cad, surf_pos_cad, unique_vertices, EdgeVertexAdj, FaceEdgeAdj, num_edge, num_surf)
        
        #### 3-4: Build the B-rep ###
        try:
            solid = construct_brep(surf_wcs, edge_wcs, FaceEdgeAdj, EdgeVertexAdj)
        except Exception as e:
            print('B-rep rebuild failed...')
            continue

        # Save CAD model
        random_string = generate_random_string(15)
        write_step_file(solid, f'{save_folder}/{random_string}_{batch_idx}.step')
        write_stl_file(solid, f'{save_folder}/{random_string}_{batch_idx}.stl', linear_deflection=0.001, angular_deflection=0.5)
    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=['abc', 'deepcad', 'furniture'], default='abc', 
                        help="Choose between evaluation mode [abc/deepcad/furniture] (default: abc)")
    args = parser.parse_args()    

    # Load evaluation config 
    with open('eval_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    eval_args = config[args.mode]

    while(True):
        sample(eval_args)