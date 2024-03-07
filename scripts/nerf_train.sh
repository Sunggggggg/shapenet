#!/bin/bash

nerf_config=configs/MipNeRF/train.txt
mae_config=configs/MAE/mae.txt

python run_mask_mipnerf.py --nerf_config $nerf_config --mae_config $mae_config