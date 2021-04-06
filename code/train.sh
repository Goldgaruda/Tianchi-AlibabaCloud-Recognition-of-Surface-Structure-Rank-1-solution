#!/bin/bash

#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -n 5
#SBATCH -t 48:00:00
##SBATCH  --mem=100GB

# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

module purge
source ~/.bashrc
source activate pytorch-1.7.1

#python ./baseline.py
#python ./upp2.py
#python ./DeepLabV3.py
#python ./linkres50.py
#python ./deepv3efb4.py
#python ./psprxt.py
#python ./unet.py
python train_upp.py