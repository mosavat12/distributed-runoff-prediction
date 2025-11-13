#!/bin/bash
#SBATCH --job-name=conf
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --exclude=uahpc-gpu003,uahpc-gpu004,uahpc-gpu008
#SBATCH --partition=mh1
#SBATCH --qos=mh1
#SBATCH --output=logs/spatial_%A.out
#SBATCH --error=logs/spatial_%A.err
#SBATCH --array=1

python spatial_dataset.py