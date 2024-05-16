import os
import argparse
from tqdm import tqdm
import multiprocessing
from pathlib import Path
import trimesh
from trimesh.sample import sample_surface
from plyfile import PlyData, PlyElement
import numpy as np

def write_ply(points, filename, text=False):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    with open(filename, mode='wb') as f:
        PlyData([el], text=text).write(f)


def find_files(folder, extension):
    return sorted([Path(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith(extension)])


def load_data_with_prefix(root_folder, prefix):
    data_files = []

    # Walk through the directory tree starting from the root folder
    for root, dirs, files in os.walk(root_folder):
        for filename in files:
            # Check if the file ends with the specified prefix
            if filename.endswith(prefix):
                file_path = os.path.join(root, filename)
                data_files.append(file_path)

    return data_files

class SamplePoints:
    """
    Perform sampleing of points.
    """

    def __init__(self):
        """
        Constructor.
        """
        parser = self.get_parser()
        self.options = parser.parse_args()


    def get_parser(self):
        """
        Get parser of tool.

        :return: parser
        """
        parser = argparse.ArgumentParser(description='Scale a set of meshes stored as OFF files.')
        parser.add_argument('--in_dir', type=str, help='Path to input directory.')
        parser.add_argument('--out_dir', type=str, help='Path to output directory; files within are overwritten!')
        return parser


    def run_parallel(self, path):
        fileName =  os.path.join(self.options.out_dir, path.split('/')[-1][:-4])

        N_POINTS = 2000
        out_mesh = trimesh.load(path)
        out_pc, _ = sample_surface(out_mesh, N_POINTS)
        save_path = os.path.join(fileName+'.ply')
        write_ply(out_pc, save_path)
        return


    def run(self):
        """
        Run simplification.
        """
        if not os.path.exists(self.options.out_dir):
            os.makedirs(self.options.out_dir)
            
        shape_paths = load_data_with_prefix(self.options.in_dir, '.stl') #+ load_data_with_prefix(self.options.in_dir, '.obj')
        # for path in shape_paths:
        #     self.run_parallel(path)
        num_cpus = multiprocessing.cpu_count()
        convert_iter =  multiprocessing.Pool(num_cpus).imap(self.run_parallel, shape_paths) 
        for _ in tqdm(convert_iter, total=len(shape_paths)):
            pass
       

if __name__ == '__main__':
    app = SamplePoints()
    app.run()
