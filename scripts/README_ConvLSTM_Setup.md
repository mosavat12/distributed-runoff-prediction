# ConvLSTM Spatial Runoff Prediction - Setup Guide

## Project Structure

```
/icebox/data/shares/mh2/mosavat/Distributed/
+-- train_final_inputs/          # Training input .npy files
+-- test_final_inputs/           # Test input .npy files
+-- train_targets/               # Training target .csv files
+-- test_targets/                # Test target .csv files
+-- train_basins.txt             # List of training basin IDs
+-- test_basins.txt              # List of test basin IDs
+-- scripts/                     # Code directory
¦   +-- spatial_dataset.py
¦   +-- convlstm_model.py
¦   +-- train_convlstm.py
¦   +-- evaluate_convlstm.py
¦   +-- train_convlstm.sh
+-- runs/                        # Output directory for experiments
```

## Quick Start

### Step 1: Prepare Basin Lists

Create text files with basin IDs (one per line):

```bash
# Go to data directory
cd /icebox/data/shares/mh2/mosavat/Distributed

# Create basin lists by listing .npy files
ls train_final_inputs/*.npy | xargs -n 1 basename | sed 's/.npy$//' > train_basins.txt
ls test_final_inputs/*.npy | xargs -n 1 basename | sed 's/.npy$//' > test_basins.txt

# Verify
echo "Train basins: $(wc -l < train_basins.txt)"
echo "Test basins: $(wc -l < test_basins.txt)"
```

### Step 2: Setup Code Directory

```bash
# Create scripts directory
mkdir -p scripts
cd scripts

# Copy the Python scripts here:
# - spatial_dataset.py
# - convlstm_model.py
# - train_convlstm.py
# - evaluate_convlstm.py
# - train_convlstm.sh
```

### Step 3: Update SLURM Script Paths

Edit `train_convlstm.sh` and verify these paths:

```bash
DATA_DIR="/icebox/data/shares/mh2/mosavat/Distributed"
TRAIN_BASINS="/icebox/data/shares/mh2/mosavat/Distributed/train_basins.txt"
TEST_BASINS="/icebox/data/shares/mh2/mosavat/Distributed/test_basins.txt"
```

### Step 4: Test Dataset Loading

Before submitting the full training job, test data loading:

```bash
# Activate environment
eval "$(conda shell.bash hook)"
conda activate modis

# Test script
cd /icebox/data/shares/mh2/mosavat/Distributed/scripts

python -c "
from spatial_dataset import SpatialHydroDataset, get_basin_list

# Load basin lists
train_basins = get_basin_list('../train_basins.txt')
print(f'Loaded {len(train_basins)} training basins')

# Create dataset
dataset = SpatialHydroDataset(
    basin_list=train_basins[:5],  # Test with first 5 basins
    data_dir='/icebox/data/shares/mh2/mosavat/Distributed',
    seq_length=365,
    train=True,
    mask_channel=31
)

# Load one sample
x, y, mask, basin_id = dataset[0]
print(f'Basin: {basin_id}')
print(f'Input shape: {x.shape}')  # Should be (365, 32, 61, 61)
print(f'Target shape: {y.shape}')  # Should be (365,)
print(f'Mask shape: {mask.shape}')  # Should be (61, 61)
print(f'Valid pixels: {mask.sum().item()}')
print('SUCCESS!')
"
```

### Step 5: Submit Training Job

```bash
cd /icebox/data/shares/mh2/mosavat/Distributed/scripts

# Create logs directory
mkdir -p logs

# Submit job
sbatch train_convlstm.sh

# Monitor job
squeue -u $USER
tail -f logs/train_<jobid>.out
```

## Configuration Options

### Model Architecture

Default configuration (in `train_convlstm.sh`):
```bash
--hidden_dims 64 128 64        # ConvLSTM hidden channels
--kernel_sizes 5 3 3           # ConvLSTM kernel sizes
--mlp_hidden_dims 128 64       # MLP hidden dimensions
--dropout 0.2                  # Dropout rate
```

**To modify**: Edit these values in `train_convlstm.sh`

### Training Hyperparameters

```bash
--batch_size 4                 # Number of basins per batch (increase if memory allows)
--seq_length 365               # Temporal window length (days)
--learning_rate 0.0001         # Learning rate (CRITICAL: keep at 0.0001)
--epochs 100                   # Maximum epochs
--patience 15                  # Early stopping patience
```

**Memory considerations**:
- Each sample: (365, 32, 61, 61) ˜ 175 MB
- Batch size 4: ~700 MB input + model activations
- H100 80GB can handle batch_size=8-12

### Data Configuration

```bash
--mask_channel 31              # Which channel contains basin mask (0-indexed)
--norm_samples 100             # Basins to sample for normalization statistics
```

## Monitoring Training

### TensorBoard

```bash
# In a separate terminal/session
cd /icebox/data/shares/mh2/mosavat/Distributed/runs/<your_run>
tensorboard --logdir tensorboard --port 6006

# Forward port to local machine (from your laptop)
ssh -L 6006:localhost:6006 <username>@<hpc_address>

# Open in browser: http://localhost:6006
```

### Training Logs

```bash
# Watch training progress
tail -f logs/train_<jobid>.out

# Check for errors
tail -f logs/train_<jobid>.err
```

## Evaluation

After training completes:

```bash
cd /icebox/data/shares/mh2/mosavat/Distributed/scripts

# Run evaluation
python evaluate_convlstm.py \
    --checkpoint ../runs/<your_run>/checkpoint_best.pt \
    --data_dir /icebox/data/shares/mh2/mosavat/Distributed \
    --test_basin_file ../test_basins.txt \
    --output_dir ../runs/<your_run>/evaluation \
    --batch_size 8 \
    --num_workers 4
```

This generates:
- `predictions.csv`: Per-basin predictions
- `basin_metrics.csv`: Per-basin metrics (NSE, RMSE, etc.)
- `summary.txt`: Aggregate metrics
- Plots: scatter plots, NSE distribution, etc.

## Troubleshooting

### Issue: "FileNotFoundError" for basin files

**Solution**: Check basin IDs in `train_basins.txt` match filenames exactly
```bash
# Check first few lines
head train_basins.txt

# Check if corresponding files exist
head -n 1 train_basins.txt | xargs -I {} ls train_final_inputs/{}.npy
```

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size
```bash
# In train_convlstm.sh, change:
--batch_size 2   # or even 1
```

### Issue: NaN loss during training

**Possible causes**:
1. Learning rate too high ? Keep at 0.0001
2. Gradient explosion ? Already using gradient clipping (max_norm=1.0)
3. Data issues ? Check for NaN/Inf in input data

**Debug**:
```python
# Check for NaN in data
import numpy as np
data = np.load('train_final_inputs/<basin_id>.npy')
print(f"NaN count: {np.isnan(data).sum()}")
print(f"Inf count: {np.isinf(data).sum()}")
```

### Issue: Slow data loading

**Solution**: Increase num_workers
```bash
# In train_convlstm.sh, change:
--num_workers 8   # or 12
```

### Issue: Training too slow

**Checklist**:
1. ? Using GPU? Check with `nvidia-smi` in job output
2. ? Data on fast storage? (not network drive)
3. ? Sufficient num_workers?
4. ? Batch size large enough? (try 8 if memory allows)

## Expected Training Time

With H100 GPU:
- **Single epoch**: ~15-30 minutes (depends on number of basins)
- **Full training** (30-50 epochs with early stopping): 8-15 hours

## Key Implementation Details

### 1. Basin Shuffling
? **Implemented** via `shuffle=True` in DataLoader
- Each epoch, basins are randomly ordered
- Model doesn't learn basin order within batches
- Each batch contains different random basins

### 2. Temporal Window Sampling
? **Implemented** in `SpatialHydroDataset.__getitem__()`
- Each basin sample uses a random 365-day window
- Different window each time basin is loaded
- Increases training diversity

### 3. Masked Pooling
? **Implemented** in `SpatialRunoffModel.forward()`
- Basin mask applied before spatial pooling
- Only valid pixels contribute to basin representation
- Handles stretched basins correctly

### 4. Normalization
? **Implemented** with per-channel statistics
- Computed from 100 random training basins
- Applied to both train and test
- Statistics saved with checkpoint

## Comparison with Lumped Model

| Aspect | Lumped LSTM | ConvLSTM (This Project) |
|--------|-------------|-------------------------|
| Input | (3650, 32) per basin | (365, 32, 61, 61) per sample |
| Spatial info | Averaged | Preserved |
| Architecture | LSTM ? MLP | ConvLSTM ? Pooling ? MLP |
| Training | NeuralHydrology | Custom (inspired by NH) |
| Batch size | 256 basins | 4-8 basins |
| LR | 0.0001 | 0.0001 |
| Output | 1 value/basin | 1 value/basin |

## Next Steps After Initial Training

### 1. Hyperparameter Tuning

Try different configurations:

```bash
# Deeper model
--hidden_dims 64 128 128 64
--kernel_sizes 5 3 3 3

# Wider model
--hidden_dims 128 256 128
--kernel_sizes 5 3 3

# Larger sequence length
--seq_length 730  # 2 years

# Larger batch size (if memory allows)
--batch_size 8
```

### 2. Multi-timestep Prediction

Current model predicts only the last timestep. To predict full sequence:

**Modify in `train_convlstm.py`**:
```python
# In train_epoch() and validate(), change:
# FROM:
target = y[:, -1].unsqueeze(1)  # (batch, 1)

# TO:
target = y  # (batch, seq_len)

# And in model forward pass, return sequence:
# predictions shape should be (batch, seq_len)
```

**Requires model architecture change** - replace MLP with sequence decoder.

### 3. Spatial Output (Optional)

To predict spatially-distributed runoff:

1. Modify model to output (batch, 1, 61, 61) instead of (batch, 1)
2. Use encoder-decoder ConvLSTM architecture
3. Requires spatially-distributed target data

### 4. Transfer Learning

After training on current basins:
- Fine-tune on new basins
- Use learned features for other variables (e.g., soil moisture)

## File Sizes and Storage

Expected storage requirements:

```
Input data:
- Each .npy file: ~1.7 GB
- 1,089 basins × 2 (train+test) = ~3.7 TB total

Model checkpoints:
- Single checkpoint: ~50-100 MB (depends on architecture)
- Best + Latest: ~200 MB

Training outputs:
- TensorBoard logs: ~500 MB
- Evaluation results: ~100 MB
```

## Advanced: Custom Loss Functions

To implement custom losses (e.g., NSE loss):

**Add to `train_convlstm.py`**:
```python
class NSELoss(nn.Module):
    """Nash-Sutcliffe Efficiency as loss (negative NSE)"""
    
    def __init__(self):
        super(NSELoss, self).__init__()
    
    def forward(self, predictions, targets):
        numerator = torch.sum((targets - predictions) ** 2)
        denominator = torch.sum((targets - torch.mean(targets)) ** 2)
        nse = 1 - (numerator / (denominator + 1e-8))
        return -nse  # Negative because we minimize loss

# Use in main():
# criterion = NSELoss()
```

## Contact and Support

If you encounter issues:

1. Check this README first
2. Review error messages in `logs/train_*.err`
3. Test with small subset of basins first
4. Verify data file formats and paths

## References

**ConvLSTM Paper**: 
- Shi et al. (2015) "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
- https://arxiv.org/abs/1506.04214

**Hydrological Applications**:
- Gauch et al. (2021) "Rainfall-Runoff Prediction at Multiple Timescales with a Single Long Short-Term Memory Network"
- NeuralHydrology framework: https://github.com/neuralhydrology/neuralhydrology

## Changelog

**Version 1.0** (Initial Release)
- Complete ConvLSTM implementation
- Dataset with lazy loading
- Training with proper basin shuffling
- Masked pooling for irregular basins
- Normalization pipeline
- Evaluation with metrics and plots