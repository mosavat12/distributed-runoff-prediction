#!/bin/bash
#SBATCH --job-name=re
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclude=uahpc-gpu003,uahpc-gpu004,uahpc-gpu008
#SBATCH --partition=mh1
#SBATCH --qos=mh1
#SBATCH --array=0-1170%200
#SBATCH --output=logs/basin_%A_%a.out
#SBATCH --error=logs/basin_%A_%a.err

# Run the distributed variables calculator
python final_file_extractor.py $SLURM_ARRAY_TASK_ID