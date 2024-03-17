import argparse
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Optional, Tuple, Union
import numpy as np 
import math
import pdb
import torch


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


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """This is a helper function that allows to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators one can seed each batched size individually. If CPU generators are passed the tensor
    will always be created on CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


def pad_repeat(x, max_len):
    repeat_times = math.floor(max_len/len(x))
    sep = max_len-repeat_times*len(x)
    sep1 = np.repeat(x[:sep], repeat_times+1, axis=0)
    sep2 = np.repeat(x[sep:], repeat_times, axis=0)
    x_repeat = np.concatenate([sep1, sep2], 0)
    return x_repeat


def pad_zero(x, max_len, return_mask=False):
    keys = np.ones(len(x))
    padding = np.zeros((max_len-len(x))).astype(int)  
    mask = 1-np.concatenate([keys, padding]) == 1  
    padding = np.zeros((max_len-len(x), *x.shape[1:]))
    x_padded = np.concatenate([x, padding], axis=0)
    if return_mask:
        return x_padded, mask
    else:
        return x_padded


def plot_3d_bbox(ax, min_corner, max_corner, color='r'):
    """
    Helper function for plotting 3D bounding boxese
    """
    vertices = [
        (min_corner[0], min_corner[1], min_corner[2]),
        (max_corner[0], min_corner[1], min_corner[2]),
        (max_corner[0], max_corner[1], min_corner[2]),
        (min_corner[0], max_corner[1], min_corner[2]),
        (min_corner[0], min_corner[1], max_corner[2]),
        (max_corner[0], min_corner[1], max_corner[2]),
        (max_corner[0], max_corner[1], max_corner[2]),
        (min_corner[0], max_corner[1], max_corner[2])
    ]
    # Define the 12 triangles composing the box
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]], 
        [vertices[0], vertices[1], vertices[5], vertices[4]], 
        [vertices[2], vertices[3], vertices[7], vertices[6]], 
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[4], vertices[7], vertices[3], vertices[0]]
    ]
    ax.add_collection3d(Poly3DCollection(faces, facecolors='blue', linewidths=1, edgecolors=color, alpha=0))
    return


def get_args_vae():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='data_process/deepcad_data_split_6bit_surface.pkl', 
                        help='Path to training data folder')  
    parser.add_argument('--val_data', type=str, default='data_process/deepcad_data_split_6bit.pkl', 
                        help='Path to validation data folder')  
    # Training parameters
    parser.add_argument("--option", type=str, choices=['surface', 'edge'], default='surface', 
                        help="Choose between option surface or edge (default: surface)")
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size')  
    parser.add_argument('--train_nepoch', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--save_nepoch', type=int, default=20, help='number of epochs to save model')
    parser.add_argument('--test_nepoch', type=int, default=10, help='number of epochs to test model')
    parser.add_argument("--data_aug",  action='store_true', help='Use data augmentation')
    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help="GPU IDs to use for training (default: [0])")
    # Save dirs and reload
    parser.add_argument('--env', type=str, default="surface_vae", help='environment')
    parser.add_argument('--dir_name', type=str, default="proj_log", help='name of the log folder.')
    args = parser.parse_args()
    # saved folder
    args.save_dir = f'{args.dir_name}/{args.env}'
    return args


def get_args_ldm():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='data_process/deepcad_data_split_6bit_surface.pkl', 
                        help='Path to data folder')  
    parser.add_argument('--val_data', type=str, default='data_process/deepcad_data_split_6bit.pkl', 
                        help='Path to validation data folder')  
    parser.add_argument("--option", type=str, choices=['surfpos', 'edgepos'], default='surfpos', 
                        help="Choose between option [surfpos,edgepos] (default: surfpos)")
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size')  
    parser.add_argument('--train_nepoch', type=int, default=3000, help='number of epochs to train for')
    parser.add_argument('--test_nepoch', type=int, default=25, help='number of epochs to test model')
    parser.add_argument('--save_nepoch', type=int, default=50, help='number of epochs to save model')
    parser.add_argument('--max_face', type=int, default=50, help='maximum number of faces')
    parser.add_argument('--max_edge', type=int, default=30, help='maximum number of edges per face')
    parser.add_argument('--threshold', type=float, default=0.05, help='minimum threshold between two faces')
    parser.add_argument('--scaled', type=float, default=3, help='wcs from -n ~ n')
    parser.add_argument("--gpu", type=int, nargs='+', default=[0, 1], help="GPU IDs to use for training (default: [0, 1])")
    parser.add_argument("--data_aug",  action='store_true', help='Use data augmentation')
    parser.add_argument("--sincos",  action='store_true', help='Use sincos positional encoding')
    # Save dirs and reload
    parser.add_argument('--env', type=str, default="surface_pos", help='environment')
    parser.add_argument('--dir_name', type=str, default="proj_log", help='name of the log folder.')
    args = parser.parse_args()
    # saved folder
    args.save_dir = f'{args.dir_name}/{args.env}'
    return args


def rotate_point_cloud(point_cloud, angle_degrees, axis):
    """
    Rotate a point cloud around its center by a specified angle in degrees along a specified axis.

    Args:
    - point_cloud: Numpy array of shape (N, 3) representing the point cloud.
    - angle_degrees: Angle of rotation in degrees.
    - axis: Axis of rotation. Can be 'x', 'y', or 'z'.

    Returns:
    - rotated_point_cloud: Numpy array of shape (N, 3) representing the rotated point cloud.
    """

    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Compute rotation matrix based on the specified axis
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(angle_radians), -np.sin(angle_radians)],
                                    [0, np.sin(angle_radians), np.cos(angle_radians)]])
    elif axis == 'y':
        rotation_matrix = np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)],
                                    [0, 1, 0],
                                    [-np.sin(angle_radians), 0, np.cos(angle_radians)]])
    elif axis == 'z':
        rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                                    [np.sin(angle_radians), np.cos(angle_radians), 0],
                                    [0, 0, 1]])
    else:
        raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")

    # Center the point cloud
    center = np.mean(point_cloud, axis=0)
    centered_point_cloud = point_cloud - center

    # Apply rotation
    rotated_point_cloud = np.dot(centered_point_cloud, rotation_matrix.T)

    # Translate back to original position
    rotated_point_cloud += center

    # Find the maximum absolute coordinate value
    max_abs_coord = np.max(np.abs(rotated_point_cloud))

    # Scale the point cloud to fit within the -1 to 1 cube
    normalized_point_cloud = rotated_point_cloud / max_abs_coord

    return normalized_point_cloud


def get_bbox(bboxes):
    """
    Get the tighest fitting 3D bounding box giving a set of points (axis-aligned)
    """
    bbox_corners = []
    for point_cloud in bboxes:
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
        bbox_corners.append([min_point, max_point])
    return np.array(bbox_corners)


def bbox_corners(bboxes):
    """
    Given the bottom-left and top-right corners of the bbox
    Return all eight corners 
    """
    bboxes_all_corners = []
    for bbox in bboxes:
        bottom_left, top_right = bbox[:3], bbox[3:]
        # Bottom 4 corners
        bottom_front_left = bottom_left
        bottom_front_right = (top_right[0], bottom_left[1], bottom_left[2])
        bottom_back_left = (bottom_left[0], top_right[1], bottom_left[2])
        bottom_back_right = (top_right[0], top_right[1], bottom_left[2])

        # Top 4 corners
        top_front_left = (bottom_left[0], bottom_left[1], top_right[2])
        top_front_right = (top_right[0], bottom_left[1], top_right[2])
        top_back_left = (bottom_left[0], top_right[1], top_right[2])
        top_back_right = top_right

        # Combine all coordinates
        all_corners = [
            bottom_front_left,
            bottom_front_right,
            bottom_back_left,
            bottom_back_right,
            top_front_left,
            top_front_right,
            top_back_left,
            top_back_right,
        ]
        bboxes_all_corners.append(np.vstack(all_corners))
    bboxes_all_corners = np.array(bboxes_all_corners)
    return bboxes_all_corners


def rotate_bbox(bboxes, angle_degrees, axis):
    """
    Rotate a point cloud around its center by a specified angle in degrees along a specified axis.

    Args:
    - point_cloud: Numpy array of shape (N, 3) representing the point cloud.
    - angle_degrees: Angle of rotation in degrees.
    - axis: Axis of rotation. Can be 'x', 'y', or 'z'.

    Returns:
    - rotated_point_cloud: Numpy array of shape (N, 3) representing the rotated point cloud.
    """

    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)
    
    # Convert bounding boxes to homogeneous coordinates
    bboxes_homogeneous = np.concatenate((bboxes, np.ones((len(bboxes), 8, 1))), axis=2)

    # Compute rotation matrix based on the specified axis
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians), 0],
            [0, np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_radians), 0, np.sin(angle_radians), 0],
            [0, 1, 0, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0, 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")

    # Apply rotation to each bounding box
    rotated_bboxes_homogeneous = np.dot(bboxes_homogeneous, rotation_matrix.T)

    # Extract rotated bounding boxes
    rotated_bboxes = rotated_bboxes_homogeneous[:, :, :3]

    # Find the maximum absolute coordinate value
    max_abs_coord = np.max(np.abs(rotated_bboxes))

    # Scale the point cloud to fit within the -1 to 1 cube
    normalized_bbox = rotated_bboxes / max_abs_coord

    return normalized_bbox


def rescale_bbox(bboxes, scale):
    # Apply scaling factors to bounding boxes
    scaled_bboxes = bboxes*scale
    return scaled_bboxes


def translate_bbox(bboxes):
    """
    Randomly move object within the cube (x,y,z direction)
    """
    point_cloud = bboxes.reshape(-1,3)
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])
    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])
    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])
    x_offset = np.random.uniform( np.min(-1-min_x,0), np.max(1-max_x,0) )
    y_offset = np.random.uniform( np.min(-1-min_y,0), np.max(1-max_y,0) )
    z_offset = np.random.uniform( np.min(-1-min_z,0), np.max(1-max_z,0) )
    random_translation = np.array([x_offset,y_offset,z_offset])
    bboxes_translated = bboxes + random_translation
    return bboxes_translated