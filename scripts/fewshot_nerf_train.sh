#!/bin/bash

data_dirs=("/home/dev4/data/SKY/datasets/NMR_Dataset/02958343/10247b51a42b41603ffe0e5069bf1eb5"
"/home/dev4/data/SKY/datasets/NMR_Dataset/02958343/1047f2879c7fdcb5709a6634bf501a9e"
"/home/dev4/data/SKY/datasets/NMR_Dataset/02958343/1005ca47e516495512da0dbf3c68e847"
"/home/dev4/data/SKY/datasets/NMR_Dataset/02958343/103d6951a3ed3c0a203f35d9b3b48203"
"/home/dev4/data/SKY/datasets/NMR_Dataset/03001627/1013f70851210a618f2e765c4a8ed3d"
"/home/dev4/data/SKY/datasets/NMR_Dataset/03001627/1007e20d5e811b308351982a6e40cf41"
"/home/dev4/data/SKY/datasets/NMR_Dataset/03001627/1006be65e7bc937e9141f9b58470d646"
"/home/dev4/data/SKY/datasets/NMR_Dataset/03001627/100b18376b885f206ae9ad7e32c4139d")

cats=("car1" "car2" "car3" "car4" "chair1" "chair2" "chair3" "chair4")

for ((idx=0; idx<8; ++idx)) 
do
    expname=${cats[idx]}
    data_dir=${data_dirs[idx]}
    nerf_config=configs/MipNeRF/train.txt
    mae_config=configs/MAE/mae.txt
    
    python run_mask_mipnerf.py --nerf_config $nerf_config --mae_config $mae_config --expname $expname --data_dir $data_dir
done