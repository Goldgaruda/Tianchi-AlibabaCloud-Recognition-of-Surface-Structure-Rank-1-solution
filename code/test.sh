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

#python ./refertta1.py
#python ./refer5tta.py
#python ./mixtta3.py
python ./ref_mixtta.py
#python ./mixtta1.py