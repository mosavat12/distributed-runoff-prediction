#!/bin/bash
#SBATCH --job-name=cl
#SBATCH --partition=mh1
#SBATCH --qos=mh1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --exclude=uahpc-gpu008
#SBATCH --mem=990G
#SBATCH --time=240:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Print job information
echo "============================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================"

# Create log directory
mkdir -p logs

# Activate conda environment (CRITICAL: Use this exact pattern)
eval "$(conda shell.bash hook)"
conda activate modis

# Verify environment
echo ""
echo "Environment Check:"
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Set paths
DATA_DIR="/icebox/data/shares/mh2/mosavat/Distributed"
TRAIN_BASINS="/icebox/data/shares/mh2/mosavat/Distributed/temporal_test_basins.txt"
TEST_BASINS="/icebox/data/shares/mh2/mosavat/Distributed/temporal_test_basins.txt"
OUTPUT_DIR="/icebox/data/shares/mh2/mosavat/Distributed/runs/convlstm_FIXED_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p $OUTPUT_DIR

# CRITICAL: Verify we're using the FIXED files
echo "Verifying fixed code files..."
if ! grep -q "normalize_targets" train_convlstm.py; then
    echo "ERROR: train_convlstm.py doesn't have target normalization!"
    echo "Please use train_convlstm_FIXED.py"
    exit 1
fi

if ! grep -q "mean(dim=(-2, -1))" convlstm_model.py; then
    echo "WARNING: convlstm_model.py may not have the pooling fix"
    echo "Please verify you're using convlstm_model_FIXED.py"
fi

echo "? Code files verified"
echo ""

# Run training with output logging
echo "Starting training with FIXED code (target normalization + no mask)..."
echo "Output directory: $OUTPUT_DIR"
echo ""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
    --gradient_accumulation 1 \
    --batch_size 8 \
    --epochs 150 \
    --learning_rate 0.00005 \
    --patience 15 \
    --norm_samples 200 \
    --num_workers 12 \
    --seed 42 \
    2>&1 | tee $OUTPUT_DIR/training.log

EXIT_CODE=$?

echo ""
echo "============================================"
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================"

# Quick check of results
if [ -f "$OUTPUT_DIR/checkpoint_best.pt" ]; then
    echo "? Best checkpoint created successfully"
else
    echo "? No best checkpoint found - training may have failed"
fi

exit $EXIT_CODE