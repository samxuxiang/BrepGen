import os
from utils import get_args_vae

# Parse input augments
args = get_args_vae()

# Set PyTorch to use only the specified GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu))

# Make project directory if not exist
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

from trainer import SurfVAETrainer
from dataset import SurfData
from trainer import EdgeVAETrainer
from dataset import EdgeData

def run(args):
    # Initialize dataset loader and trainer
    if args.option == 'surface':
        train_dataset = SurfData(args.data, args.train_list, validate=False, aug=args.data_aug)
        val_dataset = SurfData(args.data, args.val_list, validate=True, aug=False)
        vae = SurfVAETrainer(args, train_dataset, val_dataset)
    else:
        assert args.option == 'edge', 'please choose between surface or edge'
        train_dataset = EdgeData(args.data, args.train_list, validate=False, aug=args.data_aug)
        val_dataset = EdgeData(args.data, args.val_list, validate=True, aug=False)
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