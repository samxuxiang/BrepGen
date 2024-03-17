#!/bin/bash\

# python vae.py --train_data data_process/deepcad_data_split_6bit_surface.pkl \
#     --val_data data_process/deepcad_data_split_6bit.pkl --option surface --gpu 3 \
#     --env deepcad_surfvae --train_nepoch 400 --data_aug

# python vae.py --train_data data_process/abc_data_split_6bit_surface.pkl \
#     --val_data data_process/abc_data_split_6bit.pkl --option surface --gpu 2 \
#     --env abc_surfvae --train_nepoch 400 --data_aug

# python vae.py --train_data data_process/deepcad_data_split_6bit_edge.pkl \
#     --val_data data_process/deepcad_data_split_6bit.pkl --option edge --gpu 0 \
#     --env deepcad_edgevae --train_nepoch 300 --data_aug

# python vae.py --train_data data_process/abc_data_split_6bit_edge.pkl \
#     --val_data data_process/abc_data_split_6bit.pkl --option edge --gpu 1 \
#     --env abc_edgevae --train_nepoch 300 --data_aug

# python ldm.py --train_data data_process/abc_data_split_6bit.pkl \
#     --val_data data_process/abc_data_split_6bit.pkl --option surfpos --gpu 0 1 \
#     --env abc_ldm_surfpos --train_nepoch 10000 --data_aug


# python ldm.py --train_data data_process/deepcad_data_split_6bit.pkl \
#     --val_data data_process/deepcad_data_split_6bit.pkl --option surfz \
#     --surfvae proj_log/deepcad_surfvae/epoch_400.pt --gpu 0 1 3 \
#     --env deepcad_ldm_surfz --train_nepoch 10000 --data_aug --batch_size 300