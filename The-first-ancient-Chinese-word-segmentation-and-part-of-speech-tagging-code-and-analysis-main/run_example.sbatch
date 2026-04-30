#!/bin/bash
#SBATCH -J TorchCNN
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH -t 1:00:00
#SBATCH -o Output-%j.out
cd $SLURM_SUBMIT_DIR/TorchCNN

module load pytorch/2.1.0
python train.py
