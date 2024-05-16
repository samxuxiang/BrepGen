#!/bin/bash\


### DeepCAD VAE Training ###
python vae.py --data data_process/deepcad_parsed \
    --train_list data_process/deepcad_data_split_6bit_surface.pkl \
    --val_list data_process/deepcad_data_split_6bit.pkl \
    --option surface --gpu 0 --env deepcad_vae_surf --train_nepoch 400 --data_aug

python vae.py --data data_process/deepcad_parsed \
    --train_list data_process/deepcad_data_split_6bit_edge.pkl \
    --val_list data_process/deepcad_data_split_6bit.pkl \
    --option edge --gpu 0 --env deepcad_vae_edge --train_nepoch 400 --data_aug


### ABC VAE Training ###
python vae.py --data data_process/abc_parsed \
    --train_list data_process/abc_data_split_6bit_surface.pkl \
    --val_list data_process/abc_data_split_6bit.pkl \
    --option surface --gpu 0 --env abc_vae_surf --train_nepoch 200 --data_aug

python vae.py --data data_process/abc_parsed \
    --train_list data_process/abc_data_split_6bit_edge.pkl \
    --val_list data_process/abc_data_split_6bit.pkl \
    --option edge --gpu 0 --env abc_vae_edge --train_nepoch 200 --data_aug


### Furniture VAE Training (fintune) ###
python vae.py --data data_process/furniture_parsed \
    --train_list data_process/furniture_data_split_6bit_surface.pkl \
    --val_list data_process/furniture_data_split_6bit.pkl \
    --option surface --gpu 0 --env furniture_vae_surf --train_nepoch 200 --finetune \
    --weight proj_log/deepcad_vae_surf.pt

python vae.py --data data_process/furniture_parsed \
    --train_list data_process/furniture_data_split_6bit_edge.pkl \
    --val_list data_process/furniture_data_split_6bit.pkl \
    --option edge --gpu 0 --env furniture_vae_edge --train_nepoch 200 --finetune \
    --weight proj_log/deepcad_vae_edge.pt