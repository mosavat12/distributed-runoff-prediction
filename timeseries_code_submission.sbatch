#!/bin/bash
#SBATCH --job-name=dist_ext
#SBATCH --time=03:00:00
#SBATCH --mem=2G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclude=uahpc-gpu003,uahpc-gpu004,uahpc-gpu008
#SBATCH --partition=mh1
#SBATCH --qos=mh1
#SBATCH --array=1-1141%500
#SBATCH --output=logs/basin_%A_%a.out
#SBATCH --error=logs/basin_%A_%a.err

python daymet_distributed_timeseries_extractor.py $SLURM_ARRAY_TASK_ID