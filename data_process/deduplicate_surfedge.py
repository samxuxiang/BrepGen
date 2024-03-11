import pdb 
import pickle 
import argparse
from tqdm import tqdm
from hashlib import sha256
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="CAD .pkl file", required=True)
parser.add_argument("--edge",  action='store_true', help='Process edge instead of surface')
parser.add_argument("--bit",  type=int, help='Deduplicate precision')
args = parser.parse_args()


with open(args.input, "rb") as file:
    data_list = pickle.load(file)['train']

unique_data = []
unique_hash = set()
total = 0

for path_idx, path in tqdm(enumerate(data_list)):
    with open(path, "rb") as file:
        data = pickle.load(file)  

    _, _, surf_ncs, edge_ncs, _, _, _, _, _, _, _, _ = data.values()

    if args.edge:
        data = edge_ncs
        save_path = args.input.split('.')[0] + '_edge.pkl'
    else:
        data = surf_ncs
        save_path = args.input.split('.')[0] + '_surface.pkl'
   
    data_bits = real2bit(data, n_bits=args.bit)
    
    for np_bit, np_real in zip(data_bits, data):
        total += 1

        # Reshape the array to a 2D array
        np_bit = np_bit.reshape(-1, 3)
        data_hash = sha256(np_bit.tobytes()).hexdigest()
        
        prev_len = len(unique_hash)
        unique_hash.add(data_hash)   
        
        if prev_len < len(unique_hash):
            unique_data.append(np_real)
        else:
            continue
    
    if path_idx % 2000 == 0:
        print(len(unique_hash)/total)


with open(save_path, "wb") as tf:
    pickle.dump(unique_data, tf)
