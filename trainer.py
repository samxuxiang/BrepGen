import os
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn 
from diffusers import AutoencoderKL, DDPMScheduler
from network import AutoencoderKL1D, SurfPosNet


class SurfVAETrainer():
    """ Surface VAE Trainer """
    def __init__(self, args, train_dataset, val_dataset): 
        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = AutoencoderKL(in_channels=3,
            out_channels=3,
            down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
            up_block_types= ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            act_fn='silu',
            latent_channels=3,
            norm_num_groups=32,
            sample_size=512,
        )
        self.model = model.to(self.device).train()

        # Initialize optimizer
        self.network_params = list(self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=5e-4, 
            weight_decay=1e-5
        )
        self.scaler = torch.cuda.amp.GradScaler()

        # Initialize wandb
        wandb.init(project='BrepGen', dir=args.save_dir, name=args.env)

        # Initilizer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                shuffle=True, 
                                                batch_size=args.batch_size,
                                                num_workers=8)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                             shuffle=False, 
                                             batch_size=args.batch_size,
                                             num_workers=8)
        return
    
        
    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()
        loss_fn = nn.MSELoss()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train    
        for surf_uv in self.train_dataloader:
            with torch.cuda.amp.autocast():
                surf_uv = surf_uv.to(self.device).permute(0,3,1,2)
                self.optimizer.zero_grad() # zero gradient

                # Pass through VAE 
                posterior = self.model.encode(surf_uv).latent_dist
                z = posterior.sample()
                dec = self.model.decode(z).sample

                # Loss functions
                kl_loss = posterior.kl().mean()
                mse_loss = loss_fn(dec, surf_uv) 
                total_loss = mse_loss + 1e-6*kl_loss

                # Update model
                self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=5.0)  
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 10 == 0:
                wandb.log({"Loss-mse": mse_loss, "Loss-kl": kl_loss}, step=self.iters)

            self.iters += 1
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1 
        return 
    

    def test_val(self):
        """
        Test the model on validation set
        """
        self.model.eval() # set to eval
        total_loss = 0
        total_count = 0
        mse_loss = nn.MSELoss(reduction='none')
        with torch.no_grad():
            for surf_uv in self.val_dataloader:
                surf_uv = surf_uv.to(self.device).permute(0,3,1,2)
                
                posterior = self.model.encode(surf_uv).latent_dist
                z = posterior.sample()
                dec = self.model.decode(z).sample

                loss = mse_loss(dec, surf_uv).mean((1,2,3)).sum().item()
                total_loss += loss
                total_count += len(surf_uv)

        mse = total_loss/total_count
        self.model.train() # set to train
        wandb.log({"Val-mse": mse}, step=self.iters)
        return mse
    

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.args.save_dir,'epoch_'+str(self.epoch)+'.pt'))
        return


class EdgeVAETrainer():
    """ Edge VAE Trainer """
    def __init__(self, args, train_dataset, val_dataset): 
        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoencoderKL1D(
            in_channels=3,
            out_channels=3,
            down_block_types=['DownBlock1D', 'DownBlock1D', 'DownBlock1D'],
            up_block_types=['UpBlock1D', 'UpBlock1D', 'UpBlock1D'],
            block_out_channels=[128, 256, 512],  
            layers_per_block=2,
            act_fn='silu',
            latent_channels=3,
            norm_num_groups=32,
            sample_size=512
        )
        self.model = model.to(self.device).train()

        # Initialize optimizer
        self.network_params = list(self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=5e-4, 
            weight_decay=1e-5
        )
        self.scaler = torch.cuda.amp.GradScaler()

        # Initialize wandb
        wandb.init(project='BrepGen', dir=args.save_dir, name=args.env)

        # Initilizer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                shuffle=True, 
                                                batch_size=args.batch_size,
                                                num_workers=8)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                             shuffle=False, 
                                             batch_size=args.batch_size,
                                             num_workers=8)
        return
    

    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()
        loss_fn = nn.MSELoss()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train    
        for edge_u in self.train_dataloader:
            with torch.cuda.amp.autocast():
                edge_u = edge_u.to(self.device).permute(0,2,1)
                self.optimizer.zero_grad() # zero gradient

               # Pass through VAE 
                posterior = self.model.encode(edge_u).latent_dist
                z = posterior.sample()
                dec = self.model.decode(z).sample
                
                # Loss functions
                kl_loss =  0.5 * torch.sum(
                    torch.pow(posterior.mean, 2) + posterior.var - 1.0 - posterior.logvar,
                    dim=[1, 2],
                ).mean()            
                mse_loss = loss_fn(dec, edge_u) 
                total_loss = mse_loss + 1e-6*kl_loss

                # Update model
                self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=5.0)  
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 10 == 0:
                wandb.log({"Loss-mse": mse_loss, "Loss-kl": kl_loss}, step=self.iters)

            self.iters += 1
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1 
        return 
    

    def test_val(self):
        """
        Test the model on validation set
        """
        self.model.eval() # set to eval
        total_loss = 0
        total_count = 0
        mse_loss = nn.MSELoss(reduction='none')
        with torch.no_grad():
            for edge_u in self.val_dataloader:
                edge_u = edge_u.to(self.device).permute(0,2,1)
                
                posterior = self.model.encode(edge_u).latent_dist
                z = posterior.sample()
                dec = self.model.decode(z).sample

                loss = mse_loss(dec, edge_u).mean((1,2)).sum().item()
                total_loss += loss
                total_count += len(edge_u)

        mse = total_loss/total_count
        self.model.train() # set to train
        wandb.log({"Val-mse": mse}, step=self.iters)
        return mse
    

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.args.save_dir,'epoch_'+str(self.epoch)+'.pt'))
        return
    

class SurfPosTrainer():
    """ Surface Position Trainer (3D bbox) """
    def __init__(self, args, train_dataset, val_dataset): 
        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize network
        model = SurfPosNet(self.args.sincos)
        model = nn.DataParallel(model) # distributed training 
        self.model = model.to(self.device).train()

        self.loss_fn = nn.MSELoss()

        # Initialize diffusion scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='linear',
            prediction_type='epsilon',
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False,
        )

        # Initialize optimizer
        self.network_params = list(self.model.parameters())
        
        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=5e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )

        self.scaler = torch.cuda.amp.GradScaler()

        # Initialize wandb
        wandb.init(project='BrepGen', dir=args.save_dir, name=args.env)

        # Initilizer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                shuffle=True, 
                                                batch_size=args.batch_size,
                                                num_workers=16)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                             shuffle=False, 
                                             batch_size=args.batch_size,
                                             num_workers=16)
        return
    

    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train    
        for surfPos in self.train_dataloader:
            with torch.cuda.amp.autocast():
                surfPos = surfPos.to(self.device)
                bsz = len(surfPos)
                
                self.optimizer.zero_grad() # zero gradient

                # Add noise
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()  # [batch,]
                surfPos_noise = torch.randn(surfPos.shape).to(self.device)  
                surfPos_diffused = self.noise_scheduler.add_noise(surfPos, surfPos_noise, timesteps)

                # Predict noise
                surfPos_pred = self.model(surfPos_diffused, timesteps)
                
                total_loss = self.loss_fn(surfPos_pred, surfPos_noise)
             
                # Update model
                self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0) # clip gradient
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 10 == 0:
                wandb.log({"Loss-noise": total_loss}, step=self.iters)

            self.iters += 1
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1 
        return 
    

    def test_val(self):
        """
        Test the model on validation set
        """
        self.model.eval() # set to eval
        total_loss1, total_loss2, total_loss3, total_loss4, total_loss5 = 0,0,0,0,0
        total_count = 0
        mse_loss = nn.MSELoss(reduction='none')

        with torch.no_grad():
            for surfPos in self.val_dataloader:
                surfPos = surfPos.to(self.device)
                bsz = len(surfPos)
                
                # Evaluate at timestep = 10
                timesteps = torch.randint(9, 10, (bsz,), device=self.device).long()  # [batch,]
                surfPos_noise = torch.randn(surfPos.shape).to(self.device)  
                surfPos_diffused = self.noise_scheduler.add_noise(surfPos, surfPos_noise, timesteps)
                surfPos_pred = self.model(surfPos_diffused, timesteps)
                loss = mse_loss(surfPos_pred, surfPos_noise).mean((1,2)).sum().item()
                total_loss1 += loss

                # Evaluate at timestep = 50
                timesteps = torch.randint(49, 50, (bsz,), device=self.device).long()  # [batch,]
                surfPos_noise = torch.randn(surfPos.shape).to(self.device)  
                surfPos_diffused = self.noise_scheduler.add_noise(surfPos, surfPos_noise, timesteps)
                surfPos_pred = self.model(surfPos_diffused, timesteps)
                loss = mse_loss(surfPos_pred, surfPos_noise).mean((1,2)).sum().item()
                total_loss2 += loss

                # Evaluate at timestep = 100
                timesteps = torch.randint(99, 100, (bsz,), device=self.device).long()  # [batch,]
                surfPos_noise = torch.randn(surfPos.shape).to(self.device)  
                surfPos_diffused = self.noise_scheduler.add_noise(surfPos, surfPos_noise, timesteps)
                surfPos_pred = self.model(surfPos_diffused, timesteps)
                loss = mse_loss(surfPos_pred, surfPos_noise).mean((1,2)).sum().item()
                total_loss3 += loss

                # Evaluate at timestep = 200
                timesteps = torch.randint(199, 200, (bsz,), device=self.device).long()  # [batch,]
                surfPos_noise = torch.randn(surfPos.shape).to(self.device)  
                surfPos_diffused = self.noise_scheduler.add_noise(surfPos, surfPos_noise, timesteps)
                surfPos_pred = self.model(surfPos_diffused, timesteps)
                loss = mse_loss(surfPos_pred, surfPos_noise).mean((1,2)).sum().item()
                total_loss4 += loss

                # Evaluate at timestep = 500
                timesteps = torch.randint(499, 500, (bsz,), device=self.device).long()  # [batch,]
                surfPos_noise = torch.randn(surfPos.shape).to(self.device)  
                surfPos_diffused = self.noise_scheduler.add_noise(surfPos, surfPos_noise, timesteps)
                surfPos_pred = self.model(surfPos_diffused, timesteps)
                loss = mse_loss(surfPos_pred, surfPos_noise).mean((1,2)).sum().item()
                total_loss5 += loss
                
                total_count += len(surfPos)

        mse1 = total_loss1/total_count
        mse2 = total_loss2/total_count
        mse3 = total_loss3/total_count
        mse4 = total_loss4/total_count
        mse5 = total_loss5/total_count

        self.model.train() # set to train

        wandb.log({"Val-010": mse1, "Val-050": mse2, "Val-100": mse3, "Val-200": mse4, "Val-500": mse5}, step=self.iters)
        return
    

    def save_model(self):
        torch.save(self.model.module.state_dict(), os.path.join(self.args.save_dir,'epoch_'+str(self.epoch)+'.pt'))
        return
