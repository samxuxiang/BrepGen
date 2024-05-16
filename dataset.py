import os
import math
import pickle 
import torch
import numpy as np 
from tqdm import tqdm
import random
from multiprocessing.pool import Pool
from utils import (
    rotate_point_cloud,
    bbox_corners,
    rotate_axis,
    get_bbox,
    pad_repeat,
    pad_zero,
)

# furniture class labels
text2int = {'bathtub':0, 'bed':1, 'bench':2, 'bookshelf':3,'cabinet':4, 'chair':5, 'couch':6, 'lamp':7, 'sofa':8, 'table':9}


def filter_data(data):
    """ 
    Helper function to check if a brep needs to be included
        in the training data or not 
    """
    data_path, max_face, max_edge, scaled_value, threshold_value, data_class = data
    # Load data 
    with open(data_path, "rb") as tf:
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
        if len(edge_bbox[adj]) == 0: 
            skip = True
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
        return None, None 
    else: 
        return data_path, data_class


def load_data(input_data, input_list, validate, args):
    # Filter data list
    with open(input_list, "rb") as tf:
        if validate:
            data_list = pickle.load(tf)['val']
        else:
            data_list = pickle.load(tf)['train']

    data_paths = []
    data_classes = []
    for uid in data_list:
        try:
            path = os.path.join(input_data, str(math.floor(int(uid.split('.')[0])/10000)).zfill(4), uid)
            class_label = -1  # unconditional generation (abc/deepcad)
        except Exception:
            path = os.path.join(input_data, uid)  
            class_label = text2int[uid.split('/')[0]]  # conditional generation (furniture)
        data_paths.append(path)
        data_classes.append(class_label)
    
    # Filter data in parallel
    loaded_data = []
    params = zip(data_paths, [args.max_face]*len(data_list), [args.max_edge]*len(data_list), 
                    [args.bbox_scaled]*len(data_list), [args.threshold]*len(data_list), data_classes)
    convert_iter = Pool(os.cpu_count()).imap(filter_data, params) 
    for data_path, data_class in tqdm(convert_iter, total=len(data_list)):
        if data_path is not None:
            if data_class<0: # abc or deepcad
                loaded_data.append(data_path) 
            else:   # furniture
                loaded_data.append((data_path,data_class)) 

    print(f'Processed {len(loaded_data)}/{len(data_list)}')
    return loaded_data


class SurfData(torch.utils.data.Dataset):
    """ Surface VAE Dataloader """
    def __init__(self, input_data, input_list, validate=False, aug=False): 
        self.validate = validate
        self.aug = aug

        # Load validation data
        if self.validate: 
            print('Loading validation data...')
            with open(input_list, "rb") as tf:
                data_list = pickle.load(tf)['val']
            
            datas = [] 
            for uid in data_list:
                try:
                    path = os.path.join(input_data, str(math.floor(int(uid.split('.')[0])/10000)).zfill(4), uid)
                except Exception:
                    path = os.path.join(input_data, uid)
                
                with open(path, "rb") as tf:
                    data = pickle.load(tf)
                _, _, surf_uv, _, _, _, _, _, _, _, _, _ = data.values() 
                datas.append(surf_uv)
            self.data = np.vstack(datas)

        # Load training data (deduplicated)
        else:
            print('Loading training data...')
            with open(input_list, "rb") as tf:
                self.data = pickle.load(tf)  
                
        print(len(self.data))
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


class EdgeData(torch.utils.data.Dataset):
    """ Edge VAE Dataloader """
    def __init__(self, input_data, input_list, validate=False, aug=False): 
        self.validate = validate
        self.aug = aug

        # Load validation data
        if self.validate: 
            print('Loading validation data...')
            with open(input_list, "rb") as tf:
                data_list = pickle.load(tf)['val']

            datas = []
            for uid in tqdm(data_list):
                try:
                    path = os.path.join(input_data, str(math.floor(int(uid.split('.')[0])/10000)).zfill(4), uid)
                except Exception:
                    path = os.path.join(input_data, uid)

                with open(path, "rb") as tf:
                    data = pickle.load(tf)

                _, _, _, edge_u, _, _, _, _, _, _, _, _ = data.values() 
                datas.append(edge_u)
            self.data = np.vstack(datas)

        # Load training data (deduplicated)
        else:
            print('Loading training data...')
            with open(input_list, "rb") as tf:
                self.data = pickle.load(tf)   

        print(len(self.data))       
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        edge_u = self.data[index]
        # Data augmentation, randomly rotate 50% of the times
        if np.random.rand()>0.5 and self.aug:
            for axis in ['x', 'y', 'z']:
                angle = random.choice([90, 180, 270])
                edge_u = rotate_point_cloud(edge_u, angle, axis)   
        return torch.FloatTensor(edge_u)
        

class SurfPosData(torch.utils.data.Dataset):
    """ Surface position (3D bbox) Dataloader """
    def __init__(self, input_data, input_list, validate=False, aug=False, args=None): 
        self.max_face = args.max_face
        self.max_edge = args.max_edge
        self.bbox_scaled = args.bbox_scaled
        self.aug = aug
        # Load data 
        self.data = load_data(input_data, input_list, validate, args)
        # Inflate furniture x50 times for training
        if len(self.data)<2000 and not validate:
            self.data = self.data*50
        return      

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load data
        data_class = None
        if isinstance(self.data[index], tuple):
            data_path, data_class = self.data[index]
        else:
            data_path = self.data[index]

        with open(data_path, "rb") as tf:
            data = pickle.load(tf)
        _, _, _, _, _, _, _, _, surf_pos, _, _, _ = data.values()

        # Data augmentation
        random_num = np.random.rand()

        if random_num>0.5 and self.aug:  
            # Get all eight corners
            surfpos_corners = bbox_corners(surf_pos)

            # Random rotation
            for axis in ['x', 'y', 'z']:
                angle = random.choice([90, 180, 270]) 
                surfpos_corners = rotate_axis(surfpos_corners, angle, axis, normalized=True)
            
            # Re-compute the bottom left and top right corners 
            surf_pos = get_bbox(surfpos_corners)
            surf_pos = surf_pos.reshape(len(surf_pos),6)

        # Make bbox range larger
        surf_pos = surf_pos * self.bbox_scaled  

        # Randomly shuffle the sequence
        random_indices = np.random.permutation(surf_pos.shape[0])
        surf_pos = surf_pos[random_indices]
    
        # Padding
        surf_pos = pad_repeat(surf_pos, self.max_face)
        
        # Randomly shuffle the sequence
        random_indices = np.random.permutation(surf_pos.shape[0])
        surf_pos = surf_pos[random_indices]

        if data_class is not None:
            return (
                torch.FloatTensor(surf_pos), 
                torch.LongTensor([data_class+1]) # add 1, class 0 = uncond (furniture)
            )  
        else:
            return torch.FloatTensor(surf_pos) # abc or deepcad
    

class SurfZData(torch.utils.data.Dataset):
    """ Surface latent geometry Dataloader """
    def __init__(self, input_data, input_list, validate=False, aug=False, args=None): 
        self.max_face = args.max_face
        self.max_edge = args.max_edge
        self.bbox_scaled = args.bbox_scaled
        self.aug = aug
        # Load data 
        self.data = load_data(input_data, input_list, validate, args)
        # Inflate furniture x50 times for training
        if len(self.data)<2000 and not validate:
            self.data = self.data*50
        return  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load data
        data_class = None
        if isinstance(self.data[index], tuple):
            data_path, data_class = self.data[index]
        else:
            data_path = self.data[index]

        with open(data_path, "rb") as tf:
            data = pickle.load(tf)
        _, _, surf_ncs, _, _, _, _, _, surf_pos, _, _, _ = data.values()

        # Data augmentation
        random_num = np.random.rand()

        if random_num>0.5 and self.aug:
            # Get all eight corners
            surfpos_corners = bbox_corners(surf_pos)

            # Random rotation
            for axis in ['x', 'y', 'z']:
                angle = random.choice([90, 180, 270]) 
                surfpos_corners = rotate_axis(surfpos_corners, angle, axis, normalized=True)
                surf_ncs = rotate_axis(surf_ncs, angle, axis, normalized=False)
            
            # Re-compute the bottom left and top right corners 
            surf_pos = get_bbox(surfpos_corners)
            surf_pos = surf_pos.reshape(len(surf_pos),6)

        # Make bbox range larger
        surf_pos = surf_pos * self.bbox_scaled  

        # Randomly shuffle the sequence
        random_indices = np.random.permutation(surf_pos.shape[0])
        surf_pos = surf_pos[random_indices]
        surf_ncs = surf_ncs[random_indices]
    
        # Pad data
        surf_pos, surf_mask = pad_zero(surf_pos, self.max_face, return_mask=True)
        surf_ncs = pad_zero(surf_ncs, self.max_face)

        if data_class is not None:
            return (
                torch.FloatTensor(surf_pos),
                torch.FloatTensor(surf_ncs),
                torch.BoolTensor(surf_mask),
                torch.LongTensor([data_class+1]) # add 1, class 0 = uncond (furniture)
            )  
        else:
            return (
                torch.FloatTensor(surf_pos),
                torch.FloatTensor(surf_ncs),
                torch.BoolTensor(surf_mask),
            ) # abc or deepcad
    

class EdgePosData(torch.utils.data.Dataset):
    """ Edge Position (3D bbox) Dataloader """
    def __init__(self, input_data, input_list, validate=False, aug=False, args=None): 
        self.max_face = args.max_face
        self.max_edge = args.max_edge
        self.bbox_scaled = args.bbox_scaled
        self.aug = aug
        self.data = []
        # Load data 
        self.data = load_data(input_data, input_list, validate, args)
        # Inflate furniture x50 times for training
        if len(self.data)<2000 and not validate:
            self.data = self.data*50
        return  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load data
        data_class = None
        if isinstance(self.data[index], tuple):
            data_path, data_class = self.data[index]
        else:
            data_path = self.data[index]

        with open(data_path, "rb") as tf:
            data = pickle.load(tf)
        
        _, _, surf_ncs, _, _, _, _, faceEdge_adj, surf_pos, edge_pos, _, _ = data.values()

        # Data augmentation
        random_num = np.random.rand()
        if random_num > 0.5 and self.aug:
            # Get all eight corners
            surfpos_corners = bbox_corners(surf_pos)
            edgepos_corners = bbox_corners(edge_pos)

            # Random rotation
            for axis in ['x', 'y', 'z']:
                angle = random.choice([90, 180, 270]) 
                surfpos_corners = rotate_axis(surfpos_corners, angle, axis, normalized=True)
                edgepos_corners = rotate_axis(edgepos_corners, angle, axis, normalized=True)
                surf_ncs = rotate_axis(surf_ncs, angle, axis, normalized=False)
            
            # Re-compute the bottom left and top right corners 
            surf_pos = get_bbox(surfpos_corners)
            surf_pos = surf_pos.reshape(len(surf_pos),6)
            edge_pos = get_bbox(edgepos_corners)
            edge_pos = edge_pos.reshape(len(edge_pos),6)

        # Increase bbox value range
        surf_pos = surf_pos * self.bbox_scaled 
        edge_pos = edge_pos * self.bbox_scaled 

        # Mating duplication
        edge_pos_duplicated = []
        for adj in faceEdge_adj:
            edge_pos_duplicated.append(edge_pos[adj])

        # Randomly shuffle the edges per face
        edge_pos_new = []
        for pos in edge_pos_duplicated:
            random_indices = np.random.permutation(pos.shape[0])
            pos = pos[random_indices]
            pos = pad_repeat(pos, self.max_edge) #make sure some values are always repeated
            random_indices = np.random.permutation(pos.shape[0])
            pos = pos[random_indices]
            edge_pos_new.append(pos)
        edge_pos = np.stack(edge_pos_new)

        # Randomly shuffle the face sequence
        random_indices = np.random.permutation(surf_pos.shape[0])
        surf_pos = surf_pos[random_indices]
        edge_pos = edge_pos[random_indices]
        surf_ncs = surf_ncs[random_indices]
    
        # Padding
        surf_pos, surf_mask = pad_zero(surf_pos, self.max_face, return_mask=True)
        surf_ncs = pad_zero(surf_ncs, self.max_face)
        edge_pos = pad_zero(edge_pos, self.max_face)
    
        if data_class is not None:
            return (
                torch.FloatTensor(edge_pos), 
                torch.FloatTensor(surf_ncs), 
                torch.FloatTensor(surf_pos), 
                torch.BoolTensor(surf_mask), 
                torch.LongTensor([data_class+1]) # add 1, class 0 = uncond (furniture)
            )  
        else:
            return (
                torch.FloatTensor(edge_pos), 
                torch.FloatTensor(surf_ncs), 
                torch.FloatTensor(surf_pos), 
                torch.BoolTensor(surf_mask), 
            )# abc or deepcad
    
    

class EdgeZData(torch.utils.data.Dataset):
    """ Edge Latent z Dataloader """
    def __init__(self, input_data, input_list, validate=False, aug=False, args=None): 
        self.max_face = args.max_face
        self.max_edge = args.max_edge
        self.bbox_scaled = args.bbox_scaled
        self.aug = aug
        self.data = []
        # Load data 
        self.data = load_data(input_data, input_list, validate, args)
        # Inflate furniture x50 times for training
        if len(self.data)<2000 and not validate:
            self.data = self.data*50
        return 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load data
        data_class = None
        if isinstance(self.data[index], tuple):
            data_path, data_class = self.data[index]
        else:
            data_path = self.data[index]

        with open(data_path, "rb") as tf:
            data = pickle.load(tf)

        _, _, surf_ncs, edge_ncs, corner_wcs, _, _, faceEdge_adj, surf_pos, edge_pos, _, _ = data.values()

        # Data augmentation
        random_num = np.random.rand()
        if random_num > 0.5 and self.aug:
            # Get all eight corners
            surfpos_corners = bbox_corners(surf_pos)
            edgepos_corners = bbox_corners(edge_pos)

            # Random rotation
            for axis in ['x', 'y', 'z']:
                angle = random.choice([90, 180, 270]) 
                surfpos_corners = rotate_axis(surfpos_corners, angle, axis, normalized=True)
                edgepos_corners = rotate_axis(edgepos_corners, angle, axis, normalized=True)
                corner_wcs = rotate_axis(corner_wcs, angle, axis, normalized=True)
                surf_ncs = rotate_axis(surf_ncs, angle, axis, normalized=False)
                edge_ncs = rotate_axis(edge_ncs, angle, axis, normalized=False)
            
            # Re-compute the bottom left and top right corners 
            surf_pos = get_bbox(surfpos_corners)
            surf_pos = surf_pos.reshape(len(surf_pos),6)
            edge_pos = get_bbox(edgepos_corners)
            edge_pos = edge_pos.reshape(len(edge_pos),6)

        # Increase value range
        surf_pos = surf_pos * self.bbox_scaled 
        edge_pos = edge_pos * self.bbox_scaled 
        corner_wcs = corner_wcs * self.bbox_scaled 

        # Mating duplication
        edge_pos_duplicated = []
        vertex_pos_duplicated = []
        edge_ncs_duplicated = []
        for adj in faceEdge_adj:
            edge_ncs_duplicated.append(edge_ncs[adj])
            edge_pos_duplicated.append(edge_pos[adj])
            corners = corner_wcs[adj]
            corners_sorted = []
            for corner in corners:
                sorted_indices = np.lexsort((corner[:, 2], corner[:, 1], corner[:, 0])) 
                corners_sorted.append(corner[sorted_indices].flatten()) # 1 x 6 corner pos
            corners = np.stack(corners_sorted)
            vertex_pos_duplicated.append(corners)

        # Edge Shuffle and Padding
        edge_pos_new = []
        edge_ncs_new = []
        vert_pos_new = []
        edge_mask = []
        for pos, ncs, vert in zip(edge_pos_duplicated, edge_ncs_duplicated, vertex_pos_duplicated):
            random_indices = np.random.permutation(pos.shape[0])
            pos = pos[random_indices]
            ncs = ncs[random_indices]
            vert = vert[random_indices]

            pos, mask = pad_zero(pos, self.max_edge, return_mask=True)
            ncs = pad_zero(ncs, self.max_edge)
            vert = pad_zero(vert, self.max_edge)
            
            edge_pos_new.append(pos)
            edge_ncs_new.append(ncs)
            edge_mask.append(mask)
            vert_pos_new.append(vert)

        edge_pos = np.stack(edge_pos_new)
        edge_ncs = np.stack(edge_ncs_new)
        edge_mask = np.stack(edge_mask)
        vertex_pos = np.stack(vert_pos_new)

        # Face Shuffle and Padding
        random_indices = np.random.permutation(surf_pos.shape[0])
        surf_pos = surf_pos[random_indices]
        edge_pos = edge_pos[random_indices]
        surf_ncs = surf_ncs[random_indices]
        edge_ncs = edge_ncs[random_indices]
        edge_mask = edge_mask[random_indices]
        vertex_pos = vertex_pos[random_indices]
    
        # Padding
        surf_pos = pad_zero(surf_pos, self.max_face)
        surf_ncs = pad_zero(surf_ncs, self.max_face)
        edge_pos = pad_zero(edge_pos, self.max_face)
        edge_ncs = pad_zero(edge_ncs, self.max_face)
        vertex_pos = pad_zero(vertex_pos, self.max_face)
        padding = np.zeros((self.max_face-len(edge_mask), *edge_mask.shape[1:]))==0
        edge_mask = np.concatenate([edge_mask, padding], 0)

        if data_class is not None:
            return (
                torch.FloatTensor(edge_ncs), 
                torch.FloatTensor(edge_pos), 
                torch.BoolTensor(edge_mask),
                torch.FloatTensor(surf_ncs), 
                torch.FloatTensor(surf_pos), 
                torch.FloatTensor(vertex_pos), 
                torch.LongTensor([data_class+1]) # add 1, class 0 = uncond (furniture)
            )
        else:
            return (
                torch.FloatTensor(edge_ncs), 
                torch.FloatTensor(edge_pos), 
                torch.BoolTensor(edge_mask),
                torch.FloatTensor(surf_ncs), 
                torch.FloatTensor(surf_pos), 
                torch.FloatTensor(vertex_pos),  # uncond deepcad/abc
            )

      