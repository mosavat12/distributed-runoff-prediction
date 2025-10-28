#!/bin/bash
#SBATCH --job-name=convlstm_train
#SBATCH --partition=mh1
#SBATCH --qos=mh1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=640G
#SBATCH --time=100:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Create log directory
mkdir -p logs

# Activate conda environment (CRITICAL: Use this exact pattern)
eval "$(conda shell.bash hook)"
conda activate modis

# Verify environment
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Set paths
DATA_DIR="/icebox/data/shares/mh2/mosavat/Distributed"
TRAIN_BASINS="/icebox/data/shares/mh2/mosavat/Distributed/temporal_test_basins_sample.txt"
TEST_BASINS="/icebox/data/shares/mh2/mosavat/Distributed/temporal_test_basins_sample.txt"
OUTPUT_DIR="/icebox/data/shares/mh2/mosavat/Distributed/runs/convlstm_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training
python train_convlstm.py \
    --data_dir $DATA_DIR \
    --train_basin_file $TRAIN_BASINS \
    --test_basin_file $TEST_BASINS \
    --output_dir $OUTPUT_DIR \
    --seq_length 365 \
    --mask_channel 31 \
    --hidden_dims 64 128 64 \
    --kernel_sizes 5 3 3 \
    --mlp_hidden_dims 128 64 \
    --dropout 0.2 \
    --batch_size 4 \
    --epochs 100 \
    --learning_rate 0.0001 \
    --patience 15 \
    --norm_samples 100 \
    --num_workers 12 \
    --seed 42

echo "Job finished at: $(date)"