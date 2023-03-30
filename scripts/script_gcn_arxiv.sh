#!/bin/bash


############
# Usage
############


####################################
# Cora - 4 SEED RUNS OF EACH EXPTS
####################################

seed0=41
seed1=95
seed2=12
seed3=35
code=main.py 
dataset=Arxiv

python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/GCN_Arxiv.json' 
# python $code --dataset $dataset --gpu_id 0 --seed $seed1 --config 'configs/GCN_Arxiv.json' 
# python $code --dataset $dataset --gpu_id 0 --seed $seed2 --config 'configs/GCN_Arxiv.json' 
# python $code --dataset $dataset --gpu_id 0 --seed $seed3 --config 'configs/GCN_Arxiv.json' 

