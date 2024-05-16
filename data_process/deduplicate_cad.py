import math
import pickle 
import argparse
from tqdm import tqdm
from hashlib import sha256
from convert_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="Data folder path", required=True)
parser.add_argument("--bit",  type=int, help='Deduplicate precision')
parser.add_argument("--option", type=str, choices=['abc', 'deepcad', 'furniture'], default='abc', 
                    help="Choose between dataset option [abc/deepcad/furniture] (default: abc)")
args = parser.parse_args()

if args.option == 'deepcad': 
    OUTPUT = f'deepcad_data_split_{args.bit}bit.pkl'
elif args.option == 'abc': 
    OUTPUT = f'abc_data_split_{args.bit}bit.pkl'
else:
    OUTPUT = f'furniture_data_split_{args.bit}bit.pkl'

# Load all STEP folders
if args.option == 'furniture':
    train, val_path, test_path = load_furniture_pkl(args.data)
else:
    train, val_path, test_path = load_abc_pkl(args.data, args.option=='deepcad')

# Remove duplicate for the training set 
train_path = []
unique_hash = set()
total = 0

for path_idx, uid in tqdm(enumerate(train)):
    total += 1

    # Load pkl data
    if args.option == 'furniture':
        path = os.path.join(args.data, uid)
    else:
        path = os.path.join(args.data, str(math.floor(int(uid.split('.')[0])/10000)).zfill(4), uid)
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
        train_path.append(uid)
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

