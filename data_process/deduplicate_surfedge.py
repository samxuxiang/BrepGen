import math
import pickle 
import argparse
from tqdm import tqdm
from hashlib import sha256
from convert_utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="CAD .pkl file", required=True)
parser.add_argument("--list", type=str, help="UID list", required=True)
parser.add_argument("--edge",  action='store_true', help='Process edge instead of surface')
parser.add_argument("--bit",  type=int, help='Deduplicate precision')
parser.add_argument("--option", type=str, choices=['abc', 'deepcad', 'furniture'], default='abc', 
                    help="Choose between dataset option [abc/deepcad/furniture] (default: abc)")
args = parser.parse_args()


with open(args.list, "rb") as file:
    data_list = pickle.load(file)['train']

unique_data = []
unique_hash = set()
total = 0

for path_idx, uid in tqdm(enumerate(data_list)):
    if args.option == 'furniture':
        path = os.path.join(args.data, uid)
    else:
        path = os.path.join(args.data, str(math.floor(int(uid.split('.')[0])/10000)).zfill(4), uid)
    with open(path, "rb") as file:
        data = pickle.load(file)  

    _, _, surf_ncs, edge_ncs, _, _, _, _, _, _, _, _ = data.values()

    if args.edge:
        data = edge_ncs
    else:
        data = surf_ncs
   
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


if args.edge:
    save_path = args.list.split('.')[0] + '_edge.pkl'
else:
    save_path = args.list.split('.')[0] + '_surface.pkl'

with open(save_path, "wb") as tf:
    pickle.dump(unique_data, tf)
