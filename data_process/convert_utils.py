import json
import os 
import random
import numpy as np
from occwl.uvgrid import ugrid, uvgrid
from occwl.compound import Compound
from occwl.solid import Solid
from occwl.shell import Shell
from occwl.entity_mapper import EntityMapper


def get_bbox(point_cloud):
    """
    Get the tighest fitting 3D bounding box giving a set of points (axis-aligned)
    """
    # Find the minimum and maximum coordinates along each axis
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])

    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])

    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])

    # Create the 3D bounding box using the min and max values
    min_point = np.array([min_x, min_y, min_z])
    max_point = np.array([max_x, max_y, max_z])
    return min_point, max_point


def real2bit(data, n_bits=8, min_range=-1, max_range=1):
    """Convert vertices in [-1., 1.] to discrete values in [0, n_bits**2 - 1]."""
    range_quantize = 2**n_bits - 1
    data_quantize = (data - min_range) * range_quantize / (max_range - min_range)
    data_quantize = np.clip(data_quantize, a_min=0, a_max=range_quantize) # clip values
    return data_quantize.astype(int) 


def load_abc_pkl(root_dir, use_deepcad):
    """
    Recursively searches through a given parent directory and its subdirectories
    to find the paths of all ABC .pkl files.

    Args:
    - root_dir (str): Path to the root directory where the search begins.
    - use_deepcad (bool): Process deepcad or not

    Returns:
    - train [str]: A list containing the paths to all .pkl train data
    - val [str]: A list containing the paths to all .pkl validation data
    - test [str]: A list containing the paths to all .pkl test data
    """
    # Load DeepCAD UID 
    if use_deepcad:
        with open('train_val_test_split.json', 'r') as json_file:
            deepcad_data = json.load(json_file)
        train_uid = set([uid.split('/')[1] for uid in deepcad_data['train']])
        val_uid = set([uid.split('/')[1] for uid in deepcad_data['validation']])
        test_uid = set([uid.split('/')[1] for uid in deepcad_data['test']])

    # Load ABC UID
    else:
        full_uids = []
        dirs = [f'{root_dir}/{str(i).zfill(4)}' for i in range(100)]
        for folder in dirs:
            files = os.listdir(folder)
            full_uids += files 
        # 90-5-5 random split, same as deepcad
        random.shuffle(full_uids) # randomly shuffle data 
        train_uid = full_uids[0:int(len(full_uids)*0.9)]    
        val_uid = full_uids[int(len(full_uids)*0.9):int(len(full_uids)*0.95)]  
        test_uid = full_uids[int(len(full_uids)*0.95):]
        train_uid = set([uid.split('.')[0] for uid in train_uid])
        val_uid = set([uid.split('.')[0] for uid in val_uid])
        test_uid = set([uid.split('.')[0] for uid in test_uid])

    train = []
    val = []
    test = []
    dirs = [f'{root_dir}/{str(i).zfill(4)}' for i in range(100)]
    for folder in dirs:
        files = os.listdir(folder)
        for file in files:
            key_id = file.split('.')[0]
            if key_id in train_uid:
                train.append(file)
            elif key_id in val_uid:
                val.append(file)
            elif key_id in test_uid:
                test.append(file)
            else:
                print('unknown uid...')
                assert False
    return train, val, test


def load_furniture_pkl(root_dir):
    """
    Recursively searches through a given parent directory and its subdirectories
    to find the paths of all furniture .pkl files.

    Args:
    - root_dir (str): Path to the root directory where the search begins.

    Returns:
    - train [str]: A list containing the paths to all .pkl train data
    - val [str]: A list containing the paths to all .pkl validation data
    - test [str]: A list containing the paths to all .pkl test data
    """
    full_uids = []
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            # Check if the file ends with the specified prefix
            if filename.endswith('.pkl'):
                file_path = os.path.join(root, filename)
                full_uids.append(file_path)
                
    # 90-5-5 random split, similary to deepcad
    random.shuffle(full_uids) # randomly shuffle data 
    train_uid = full_uids[0:int(len(full_uids)*0.9)]    
    val_uid = full_uids[int(len(full_uids)*0.9):int(len(full_uids)*0.95)]  
    test_uid = full_uids[int(len(full_uids)*0.95):]
    
    train_uid = ['/'.join(uid.split('/')[-2:]) for uid in train_uid]
    val_uid = ['/'.join(uid.split('/')[-2:]) for uid in val_uid]
    test_uid = ['/'.join(uid.split('/')[-2:]) for uid in test_uid]

    return train_uid, val_uid, test_uid


def load_abc_step(root_dir, use_deepcad):
    """
    Recursively searches through a given parent directory and its subdirectories
    to find the paths of all ABC STEP files.

    Args:
    - root_dir (str): Path to the root directory where the search begins.
    - use_deepcad (bool): Process deepcad or not

    Returns:
    - step_dirs [str]: A list containing the paths to all STEP parent directory
    """
    # Load DeepCAD UID 
    if use_deepcad:
        with open('train_val_test_split.json', 'r') as json_file:
            deepcad_data = json.load(json_file)
        deepcad_data = deepcad_data['train'] + deepcad_data['validation'] + deepcad_data['test']
        deepcad_uid = set([uid.split('/')[1] for uid in deepcad_data])

    # Create STEP file folder path (based on the default ABC STEP format)
    dirs_nested = [[f'{root_dir}/abc_{str(i).zfill(4)}_step_v00']*10000 for i in range(100)]
    dirs = [item for sublist in dirs_nested for item in sublist]
    subdirs = [f'{str(i).zfill(8)}' for i in range(1000000)]
    
    if use_deepcad:
        step_dirs = [root + '/' + sub for root, sub in zip(dirs, subdirs) if sub in deepcad_uid]
    else:
        step_dirs = [root + '/' + sub for root, sub in zip(dirs, subdirs)]

    return step_dirs


def load_furniture_step(root_dir):
    """
    Recursively searches through a given parent directory and its subdirectories
    to find the paths of all Furniture STEP files.

    Args:
    - root_dir (str): Path to the root directory where the search begins.

    Returns:
    - data_files [str]: A list containing the paths to all STEP parent directory
    """
    data_files = []
    # Walk through the directory tree starting from the root folder
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            # Check if the file ends with the specified prefix
            if filename.endswith('.step'):
                file_path = os.path.join(root, filename)
                data_files.append(file_path)
    return data_files
    

def update_mapping(data_dict):
    """
    Remove unused key index from data dictionary.
    """
    dict_new = {}
    mapping = {}
    max_idx = max(data_dict.keys())
    skipped_indices = np.array(sorted(list(set(np.arange(max_idx)) - set(data_dict.keys()))))
    for idx, value in data_dict.items():
        skips = (skipped_indices < idx).sum()
        idx_new = idx - skips
        dict_new[idx_new] = value
        mapping[idx] = idx_new
    return dict_new, mapping


def face_edge_adj(shape):
    """
    *** COPY AND MODIFIED FROM THE ORIGINAL OCCWL SOURCE CODE ***
    Extract face/edge geometry and create a face-edge adjacency 
    graph from the given shape (Solid or Compound)

    Args:
    - shape (Shell, Solid, or Compound): Shape
        
    Returns:
    - face_dict: Dictionary of occwl faces, with face ID as the key
    - edge_dict: Dictionary of occwl edges, with edge ID as the key
    - edgeFace_IncM: Edge ID as the key, Adjacent faces ID as the value 
    """
    assert isinstance(shape, (Shell, Solid, Compound))
    mapper = EntityMapper(shape)
   
    ### Faces ###
    face_dict = {}
    for face in shape.faces():
        face_idx = mapper.face_index(face)
        face_dict[face_idx] = (face.surface_type(), face)

    ### Edges and IncidenceMat ###
    edgeFace_IncM = {}
    edge_dict = {}
    for edge in shape.edges():
        if not edge.has_curve():
            continue    

        connected_faces = list(shape.faces_from_edge(edge))
        if len(connected_faces) == 2 and not edge.seam(connected_faces[0]) and not edge.seam(connected_faces[1]):
            left_face, right_face = edge.find_left_and_right_faces(connected_faces)
            if left_face is None or right_face is None:
                continue
            edge_idx = mapper.edge_index(edge) 
            edge_dict[edge_idx] = edge 
            left_index = mapper.face_index(left_face)
            right_index = mapper.face_index(right_face)

            if edge_idx in edgeFace_IncM:
                edgeFace_IncM[edge_idx] += [left_index, right_index]
            else:
                edgeFace_IncM[edge_idx] = [left_index, right_index]
        else:
            pass # ignore seam

    return face_dict, edge_dict, edgeFace_IncM


def extract_primitive(solid):
    """
    Extract all primitive information from splitted solid

    Args:
    - solid (occwl.Solid): A single b-rep solid in occwl format

    Returns:
    - face_pnts (N x 32 x 32 x 3): Sampled uv-grid points on the bounded surface region (face)
    - edge_pnts (M x 32 x 3): Sampled u-grid points on the boundged curve region (edge)
    - edge_corner_pnts (M x 2 x 3): Start & end vertices per edge
    - edgeFace_IncM (M x 2): Edge-Face incident matrix, every edge is connect to two face IDs
    - faceEdge_IncM: A list of N sublist, where each sublist represents the adjacent edge IDs to a face
    """
    assert isinstance(solid, Solid)

    # Retrieve face, edge geometry and face-edge adjacency
    face_dict, edge_dict, edgeFace_IncM = face_edge_adj(solid)

    # Skip unused index key, and update the adj 
    face_dict, face_map = update_mapping(face_dict)
    edge_dict, edge_map = update_mapping(edge_dict)
    edgeFace_IncM_update = {}
    for key, value in edgeFace_IncM.items():
        new_face_indices = [face_map[x] for x in value]
        edgeFace_IncM_update[edge_map[key]] = new_face_indices 
    edgeFace_IncM = edgeFace_IncM_update
    
    # Face-edge adj
    num_faces = len(face_dict)
    edgeFace_IncM = np.stack([x for x in edgeFace_IncM.values()])
    faceEdge_IncM = []
    for surf_idx in range(num_faces):
        surf_edges, _ = np.where(edgeFace_IncM == surf_idx)
        faceEdge_IncM.append(surf_edges)
    
    # Sample uv-grid from surface (32x32)
    graph_face_feat = {}
    for face_idx, face_feature in face_dict.items():
        _, face = face_feature
        points = uvgrid(
            face, method="point", num_u=32, num_v=32
        )
        visibility_status = uvgrid(
            face, method="visibility_status", num_u=32, num_v=32
        )
        mask = np.logical_or(visibility_status == 0, visibility_status == 2)  # 0: Inside, 1: Outside, 2: On boundary
        # Concatenate channel-wise to form face feature tensor
        face_feat = np.concatenate((points, mask), axis=-1)
        graph_face_feat[face_idx] = face_feat
    face_pnts = np.stack([x for x in graph_face_feat.values()])[:,:,:,:3]
    
    # sample u-grid from curve (1x32)
    graph_edge_feat = {}
    graph_corner_feat = {}
    for edge_idx, edge in edge_dict.items():
        points = ugrid(edge, method="point", num_u=32)
        graph_edge_feat[edge_idx] = points
        #### edge corners as start/end vertex ###
        v_start = points[0]  
        v_end = points[-1]
        graph_corner_feat[edge_idx] = (v_start, v_end)
    edge_pnts = np.stack([x for x in graph_edge_feat.values()])
    edge_corner_pnts = np.stack([x for x in graph_corner_feat.values()])

    return [face_pnts, edge_pnts, edge_corner_pnts, edgeFace_IncM, faceEdge_IncM]