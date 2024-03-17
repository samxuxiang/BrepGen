import os
import pickle 
import torch
import numpy as np 
from tqdm import tqdm
import random
from multiprocessing.pool import Pool
from utils import *
import pdb 
        

def filter_data(data):
    """ 
    Helper function to check is a brep need to be incuded 
        in the training or not 
    """
    data_path, max_face, max_edge, scaled_value, threshold_value = data
    # Load data 
    with open(os.path.join('data_process',data_path), "rb") as tf:
        data = pickle.load(tf)
    _, _, _, _, _, _, _, faceEdge_adj, surf_bbox, edge_bbox, _, _ = data.values()   
    
    skip = False

    # Skip over max size data
    if len(surf_bbox)>max_face:
        skip = True

    for surf_edges in faceEdge_adj:
        if len(surf_edges)>max_edge:
            skip = True 
    
    # Skip surfaces too close to each other
    surf_bbox = surf_bbox * scaled_value  # make bbox difference larger

    _surf_bbox_ = surf_bbox.reshape(len(surf_bbox),2,3)
    non_repeat = _surf_bbox_[:1]
    for bbox in _surf_bbox_:
        diff = np.max(np.max(np.abs(non_repeat - bbox),-1),-1)
        same = diff < threshold_value
        if same.sum()>=1:
            continue # repeat value
        else:
            non_repeat = np.concatenate([non_repeat, bbox[np.newaxis,:,:]],0)
    if len(non_repeat) != len(_surf_bbox_):
        skip = True

    # Skip edges too close to each other
    se_bbox = []
    for adj in faceEdge_adj:
        se_bbox.append(edge_bbox[adj] * scaled_value)

    for bbb in se_bbox:
        _edge_bbox_ = bbb.reshape(len(bbb),2,3)
        non_repeat = _edge_bbox_[:1]
        for bbox in _edge_bbox_:
            diff = np.max(np.max(np.abs(non_repeat - bbox),-1),-1)
            same = diff < threshold_value
            if same.sum()>=1:
                continue # repeat value
            else:
                non_repeat = np.concatenate([non_repeat, bbox[np.newaxis,:,:]],0)
        if len(non_repeat) != len(_edge_bbox_):
            skip = True

    if skip: 
        return None 
    else: 
        return data_path


class SURFData(torch.utils.data.Dataset):
    """ Surface VAE Dataloader """
    def __init__(self, input_data, validate=False, aug=False): 
        self.validate = validate
        self.aug = aug
        # Load validation data
        if self.validate: 
            print('Loading validation data...')
            with open(input_data, "rb") as tf:
                data_list = pickle.load(tf)['val']
            datas = []
            for path in tqdm(data_list):
                with open(os.path.join('data_process',path), "rb") as tf:
                    data = pickle.load(tf)
                _, _, surf_uv, _, _, _, _, _, _, _, _, _ = data.values() 
                datas.append(surf_uv)
            self.data = np.vstack(datas)
            print(len(self.data))
        # Load training data (deduplicated)
        else:
            print('Loading training data...')
            with open(input_data, "rb") as tf:
                self.data = pickle.load(tf)  
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        surf_uv = self.data[index]
        if np.random.rand()>0.5 and self.aug:
            for axis in ['x', 'y', 'z']:
                angle = random.choice([90, 180, 270])
                surf_uv = rotate_point_cloud(surf_uv.reshape(-1, 3), angle, axis).reshape(32, 32, 3)
        return torch.FloatTensor(surf_uv)


class EDGEData(torch.utils.data.Dataset):
    """ Edge VAE Dataloader """
    def __init__(self, input_data, validate=False, aug=False): 
        self.validate = validate
        self.aug = aug
        # Load validation data
        if self.validate: 
            print('Loading validation data...')
            with open(input_data, "rb") as tf:
                data_list = pickle.load(tf)['val']
            datas = []
            for path in tqdm(data_list):
                with open(os.path.join('data_process',path), "rb") as tf:
                    data = pickle.load(tf)
                _, _, _, edge_u, _, _, _, _, _, _, _, _ = data.values() 
                datas.append(edge_u)
            self.data = np.vstack(datas)
            print(len(self.data))
        # Load training data (deduplicated)
        else:
            print('Loading training data...')
            with open(input_data, "rb") as tf:
                self.data = pickle.load(tf)          
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        edge_u = self.data[index]
        # Data augmention, randomly rotate 50% of the times
        if np.random.rand()>0.5 and self.aug:
            for axis in ['x', 'y', 'z']:
                angle = random.choice([90, 180, 270])
                edge_u = rotate_point_cloud(edge_u, angle, axis)   
        return torch.FloatTensor(edge_u)
        

class SURFPosData(torch.utils.data.Dataset):
    """ Surface position (3D bbox) Dataloader """
    def __init__(self, input_data, validate=False, aug=False, args=None): 
        self.max_face = args.max_face
        self.max_edge = args.max_edge
        self.scaled = args.scaled
        self.aug = aug
        self.data = []

        # Filter data list (only required once)
        with open(input_data, "rb") as tf:
            if validate:
                data_list = pickle.load(tf)['val']
            else:
                data_list = pickle.load(tf)['train']

        # Filter data in parallel
        params = zip(data_list, [args.max_face]*len(data_list), [args.max_edge]*len(data_list), 
                     [args.scaled]*len(data_list), [args.threshold]*len(data_list))
        convert_iter = Pool(os.cpu_count()).imap(filter_data, params) 
        for data_path in tqdm(convert_iter, total=len(data_list)):
            if data_path is not None:
                self.data.append(data_path) 

        print(f'Processed {len(self.data)}/{len(data_list)}')
        return      

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load data
        with open(os.path.join('data_process', self.data[index]), "rb") as tf:
            data = pickle.load(tf)
        _, _, _, _, _, _, _, _, surf_pos, _, _, _ = data.values()

        # Data augmentation
        random_num = np.random.rand()

        if random_num>0.25 and self.aug:
            # Get all eight corners
            surfpos_corners = bbox_corners(surf_pos)

            # Random rotation
            for axis in ['x', 'y', 'z']:
                angle = random.choice([90, 180, 270]) 
                surfpos_corners = rotate_bbox(surfpos_corners, angle, axis)
            
            # Re-compute the bottom left and top right corners 
            surf_pos = get_bbox(surfpos_corners)

            if random_num > 0.5:
                # Randomly rescale the bboxes (100%~80%)
                scale = np.random.uniform(0.8, 1.0)
                surf_pos = rescale_bbox(surf_pos, scale)

                if random_num > 0.6:
                    # Randomly translate the bboxes (within the unit cube)
                    surf_pos = translate_bbox(surf_pos)

            surf_pos = surf_pos.reshape(len(surf_pos),6)

        # Make bbox range larger
        surf_pos = surf_pos * self.scaled  

        # Randomly shuffle the sequence
        random_indices = np.random.permutation(surf_pos.shape[0])
        surf_pos = surf_pos[random_indices]
    
        # Padding
        surf_pos = pad_repeat(surf_pos, self.max_face)
        
        # Randomly shuffle the sequence
        random_indices = np.random.permutation(surf_pos.shape[0])
        surf_pos = surf_pos[random_indices]

        return torch.FloatTensor(surf_pos)
    

class SURFZData(torch.utils.data.Dataset):
    """ Surface latent geometry Dataloader """
    def __init__(self, input_data, validate=False, aug=False, args=None): 
        self.max_face = args.max_face
        self.max_edge = args.max_edge
        self.scaled = args.scaled
        self.aug = aug
        self.data = []

        # Filter data list (only required once)
        with open(input_data, "rb") as tf:
            if validate:
                data_list = pickle.load(tf)['val']
            else:
                data_list = pickle.load(tf)['train']

        # Filter data in parallel
        params = zip(data_list, [args.max_face]*len(data_list), [args.max_edge]*len(data_list), 
                     [args.scaled]*len(data_list), [args.threshold]*len(data_list))
        convert_iter = Pool(os.cpu_count()).imap(filter_data, params) 
        for data_path in tqdm(convert_iter, total=len(data_list)):
            if data_path is not None:
                self.data.append(data_path) 

        print(f'Processed {len(self.data)}/{len(data_list)}')
        return      

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load data
        with open(os.path.join('data_process', self.data[index]), "rb") as tf:
            data = pickle.load(tf)
        _, _, _, _, _, _, _, _, surf_pos, _, _, _ = data.values()

        # Data augmentation
        random_num = np.random.rand()

        if random_num>0.25 and self.aug:
            # Get all eight corners
            surfpos_corners = bbox_corners(surf_pos)

            # Random rotation
            for axis in ['x', 'y', 'z']:
                angle = random.choice([90, 180, 270]) 
                surfpos_corners = rotate_bbox(surfpos_corners, angle, axis)
            
            # Re-compute the bottom left and top right corners 
            surf_pos = get_bbox(surfpos_corners)

            if random_num > 0.5:
                # Randomly rescale the bboxes (100%~80%)
                scale = np.random.uniform(0.8, 1.0)
                surf_pos = rescale_bbox(surf_pos, scale)

                if random_num > 0.6:
                    # Randomly translate the bboxes (within the unit cube)
                    surf_pos = translate_bbox(surf_pos)

            surf_pos = surf_pos.reshape(len(surf_pos),6)

        # Make bbox range larger
        surf_pos = surf_pos * self.scaled  

        # Randomly shuffle the sequence
        random_indices = np.random.permutation(surf_pos.shape[0])
        surf_pos = surf_pos[random_indices]
    
        # Padding
        surf_pos = pad_repeat(surf_pos, self.max_face)
        
        # Randomly shuffle the sequence
        random_indices = np.random.permutation(surf_pos.shape[0])
        surf_pos = surf_pos[random_indices]

        return torch.FloatTensor(surf_pos)