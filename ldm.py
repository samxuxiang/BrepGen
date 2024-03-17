import os
import pdb 
from utils import *

# Parse input augments
args = get_args_ldm()

# Set PyTorch to use only the specified GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu))

# Make project directory if not exist
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

from dataset import SURFPosData, SURFZData
from trainer import SurfPosTrainer, SurfZTrainer

def run(args):
    if args.option == 'surfpos':
        # Initialize dataset and trainer
        train_dataset = SURFPosData(args.train_data, validate=False, aug=args.data_aug, args=args)
        val_dataset = SURFPosData(args.val_data, validate=True, aug=False, args=args)
        ldm = SurfPosTrainer(args, train_dataset, val_dataset)

    elif args.option == 'surfz':
        # Initialize dataset and trainer
        train_dataset = SURFZData(args.train_data, validate=False, aug=args.data_aug, args=args)
        val_dataset = SURFZData(args.val_data, validate=True, aug=False, args=args)
        ldm = SurfZTrainer(args, train_dataset, val_dataset)

    else:
        assert False, 'please choose between surfpos, surfz, edgepos, edgez'

    print('Start training...')
    # Main training loop
    for _ in range(args.train_nepoch): 

        # Train for one epoch
        ldm.train_one_epoch()

        # Evaluate model performance on validation set
        if ldm.epoch % args.test_nepoch == 0:
            ldm.test_val()

        # save model
        if ldm.epoch % args.save_nepoch == 0:
            ldm.save_model()

    return


if __name__ == "__main__":
    run(args)