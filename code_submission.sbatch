#!/bin/bash
#SBATCH --job-name=dist_ext
#SBATCH --time=01:00:00
#SBATCH --mem=1G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclude=uahpc-gpu003,uahpc-gpu004,uahpc-gpu008
#SBATCH --partition=mh1
#SBATCH --qos=mh1
#SBATCH --output=test.out
#SBATCH --error=test.err

python inspection.py