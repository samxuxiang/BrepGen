import pdb 
import pickle 
import argparse
from tqdm import tqdm
from hashlib import sha256
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="Data folder path", required=True)
parser.add_argument("--deepcad",  action='store_true', help='Process deepcad subset')
parser.add_argument("--bit",  type=int, help='Deduplicate precision')
args = parser.parse_args()


if args.deepcad: 
    OUTPUT = f'deepcad_data_split_{args.bit}bit.pkl'
else:
    OUTPUT = f'abc_data_split_{args.bit}bit.pkl'

# Load all ABC STEP folders
train, val_path, test_path = load_abc_pkl(args.input, args.deepcad)

# Remove duplicate for the training set 
train_path = []
unique_hash = set()
total = 0
for path_idx, path in tqdm(enumerate(train)):
    total += 1
    
    with open(path, "rb") as file:
        data = pickle.load(file) 

    # Hash the surface sampled points
    surfs_wcs = data['surf_wcs']
    surf_hash_total = []
    for surf in surfs_wcs:
        np_bit = real2bit(surf, n_bits=args.bit).reshape(-1, 3)  # bits
        data_hash = sha256(np_bit.tobytes()).hexdigest()
        surf_hash_total.append(data_hash)
    surf_hash_total = sorted(surf_hash_total)
    data_hash = '_'.join(surf_hash_total)

    # Save non-duplicate shapes
    prev_len = len(unique_hash)
    unique_hash.add(data_hash)  
    if prev_len < len(unique_hash):
        train_path.append(path)
    else:
        continue
        
    if path_idx % 2000 == 0:
        print(len(unique_hash)/total)

# save data 
data_path = {
    'train':train_path,
    'val':val_path,
    'test':test_path,
}
with open(OUTPUT, "wb") as tf:
    pickle.dump(data_path, tf)

