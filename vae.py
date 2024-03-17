import os
from utils import *

# Parse input augments
args = get_args_vae()

# Set PyTorch to use only the specified GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu))

# Make project directory if not exist
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

from trainer import SurfVAETrainer
from dataset import SURFData
from trainer import EdgeVAETrainer
from dataset import EDGEData

def run(args):
    if args.option == 'surface':
        # Initialize dataset
        train_dataset = SURFData(args.train_data, validate=False, aug=args.data_aug)
        val_dataset = SURFData(args.val_data, validate=True, aug=False)
        # Initialize trainer
        vae = SurfVAETrainer(args, train_dataset, val_dataset)
    else:
        assert args.option == 'edge', 'please choose between surface or edge'
        # Initialize dataset loader
        train_dataset = EDGEData(args.train_data, validate=False, aug=args.data_aug)
        val_dataset = EDGEData(args.val_data, validate=True, aug=False)
        # Initialize trainer
        vae = EdgeVAETrainer(args, train_dataset, val_dataset)

    # Main training loop
    print('Start training...')
    for _ in range(args.train_nepoch):  
        # Train for one epoch
        vae.train_one_epoch()

        # Evaluate model performance on validation set
        if vae.epoch % args.test_nepoch == 0:
            vae.test_val()

        # save model
        if vae.epoch % args.save_nepoch == 0:
            vae.save_model()
    return
           

if __name__ == "__main__":
    run(args)