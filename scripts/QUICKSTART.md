# ConvLSTM Runoff Prediction - Quick Start Guide

## ?? Project Overview

**Goal**: Predict basin runoff using spatial-temporal ConvLSTM model

**Input**: 
- Raster time series: (61, 61, 32, 3650) per basin
- 32 channels: 6 dynamic + 26 static variables
- 1,089 basins total

**Output**: 
- One runoff value per basin
- Architecture: ConvLSTM ? Masked Pooling ? MLP

**Key Features**:
- ? Basin shuffling (model doesn't learn basin order)
- ? Lazy loading (handles large files efficiently)
- ? Masked pooling (handles irregular basin shapes)
- ? Proper normalization pipeline

---

## ?? Files You Have

All code is ready to use:

1. **`spatial_dataset.py`** - Dataset class with lazy loading
2. **`convlstm_model.py`** - ConvLSTM model (Architecture A)
3. **`train_convlstm.py`** - Complete training script
4. **`evaluate_convlstm.py`** - Evaluation with metrics
5. **`train_convlstm.sh`** - SLURM submission script
6. **`verify_data.py`** - Data verification utility
7. **`README_ConvLSTM_Setup.md`** - Detailed documentation

---

## ?? Steps to Run (15 Minutes to Training!)

### Step 1: Setup Directory (2 min)

```bash
cd /icebox/data/shares/mh2/mosavat/Distributed

# Create scripts directory
mkdir -p scripts
cd scripts

# Copy all Python files here (spatial_dataset.py, convlstm_model.py, etc.)
```

### Step 2: Verify Data (5 min)

```bash
# Activate environment
eval "$(conda shell.bash hook)"
conda activate modis

# Run verification script
python verify_data.py \
    --data_dir /icebox/data/shares/mh2/mosavat/Distributed \
    --create_lists \
    --verify_train \
    --verify_test \
    --max_check 10  # Quick check of first 10 basins

# Expected output:
# - Creates train_basins.txt and test_basins.txt
# - Reports if all files are OK
```

### Step 3: Test Data Loading (3 min)

```bash
# Quick test
python -c "
from spatial_dataset import SpatialHydroDataset, get_basin_list

basins = get_basin_list('../train_basins.txt')
print(f'Found {len(basins)} basins')

dataset = SpatialHydroDataset(
    basin_list=basins[:3],
    data_dir='/icebox/data/shares/mh2/mosavat/Distributed',
    seq_length=365,
    train=True,
    mask_channel=31
)

x, y, mask, basin_id = dataset[0]
print(f'Input: {x.shape}, Target: {y.shape}, Mask: {mask.shape}')
print('? Data loading works!')
"
```

### Step 4: Submit Training Job (5 min)

```bash
# Review SLURM script settings
nano train_convlstm.sh  # Optional: adjust batch_size, hidden_dims, etc.

# Create logs directory
mkdir -p logs

# Submit!
sbatch train_convlstm.sh

# Check job status
squeue -u $USER

# Monitor training
tail -f logs/train_<jobid>.out
```

**That's it! Training is now running!** ?

---

## ?? Monitoring Training

### Watch Progress

```bash
# Training log
tail -f logs/train_<jobid>.out

# Error log (should be empty)
tail -f logs/train_<jobid>.err
```

### TensorBoard (Optional)

```bash
# Find your run directory
ls /icebox/data/shares/mh2/mosavat/Distributed/runs/

# Start TensorBoard
tensorboard --logdir /icebox/data/shares/mh2/mosavat/Distributed/runs/<your_run>/tensorboard --port 6006

# From your laptop, forward port:
ssh -L 6006:localhost:6006 <user>@<hpc>

# Open: http://localhost:6006
```

---

## ?? After Training Completes

### Evaluate Model

```bash
cd /icebox/data/shares/mh2/mosavat/Distributed/scripts

# Run evaluation
python evaluate_convlstm.py \
    --checkpoint ../runs/<your_run>/checkpoint_best.pt \
    --data_dir /icebox/data/shares/mh2/mosavat/Distributed \
    --test_basin_file ../test_basins.txt \
    --output_dir ../runs/<your_run>/evaluation \
    --batch_size 8
```

**Outputs**:
- `predictions.csv` - All predictions
- `basin_metrics.csv` - Per-basin NSE, RMSE, etc.
- `summary.txt` - Aggregate metrics
- Plots: scatter, NSE distribution, etc.

---

## ?? Key Configuration

### In `train_convlstm.sh`:

```bash
# Model architecture
--hidden_dims 64 128 64        # ConvLSTM layers
--kernel_sizes 5 3 3           # Kernel sizes
--mlp_hidden_dims 128 64       # MLP layers

# Training
--batch_size 4                 # Basins per batch (increase if memory allows)
--seq_length 365               # Days per sequence
--learning_rate 0.0001         # KEEP THIS! (learned from lumped model)
--epochs 100
--patience 15                  # Early stopping

# Data
--mask_channel 31              # Which channel is the mask
```

### Memory Tuning

If you see "CUDA out of memory":
```bash
--batch_size 2   # Reduce batch size
```

If training is slow:
```bash
--batch_size 8   # Increase batch size (H100 can handle it)
--num_workers 8  # More data loading workers
```

---

## ?? Expected Results

### Training Time
- **Per epoch**: 15-30 minutes
- **Full training**: 8-15 hours (with early stopping)

### Performance
Based on similar hydrological models:
- **NSE**: 0.5-0.8 (good models achieve >0.7)
- **RMSE**: Depends on basin characteristics

Your ConvLSTM should outperform the lumped LSTM since it preserves spatial information!

---

## ?? Common Issues & Solutions

### Issue: Can't find basin files

```bash
# Check basin list matches filenames
head train_basins.txt
ls train_final_inputs/ | head
```

### Issue: Training won't start

```bash
# Check SLURM job
squeue -u $USER
scancel <jobid>  # Cancel if needed

# Check error log
cat logs/train_<jobid>.err
```

### Issue: NaN loss

**Solution**: Already handled!
- Learning rate: 0.0001 (proven stable)
- Gradient clipping: max_norm=1.0
- If still happens, check for NaN in data with `verify_data.py`

---

## ?? What Each File Does

| File | Purpose |
|------|---------|
| `spatial_dataset.py` | Loads .npy and .csv files, samples temporal windows |
| `convlstm_model.py` | ConvLSTM + masked pooling + MLP architecture |
| `train_convlstm.py` | Training loop, normalization, checkpointing |
| `evaluate_convlstm.py` | Test set evaluation, metrics, plots |
| `train_convlstm.sh` | SLURM job submission script |
| `verify_data.py` | Check data integrity |

---

## ?? Pro Tips

1. **Start small**: Test with `--max_check 10` basins first
2. **Monitor GPU**: Check `nvidia-smi` in job output
3. **Save checkpoints**: Both best and latest are saved automatically
4. **Use TensorBoard**: Great for debugging and monitoring
5. **Compare to lumped model**: You should see improvement!

---

## ?? Next Steps After First Run

1. **Tune hyperparameters**: Try different hidden_dims, batch_size
2. **Longer sequences**: Try `--seq_length 730` (2 years)
3. **Deeper model**: Add more ConvLSTM layers
4. **Ensemble**: Train multiple models with different seeds

---

## ?? Key Differences from Lumped Model

| Aspect | Lumped LSTM | ConvLSTM (This) |
|--------|-------------|-----------------|
| Input shape | (3650, 32) | (365, 32, 61, 61) |
| Spatial info | ? Averaged | ? Preserved |
| Architecture | LSTM ? MLP | ConvLSTM ? Pool ? MLP |
| Batch size | 256 | 4-8 |
| Parameters | ~100K | ~5M |
| Training time | 2-4 hours | 8-15 hours |

---

## ? Checklist

Before submitting your first job:

- [ ] All Python files copied to `scripts/` directory
- [ ] Ran `verify_data.py` successfully
- [ ] Tested data loading (Step 3)
- [ ] Reviewed SLURM script paths
- [ ] Created `logs/` directory
- [ ] Conda environment `modis` is activated in SLURM script

**Ready to go!** ??

```bash
sbatch train_convlstm.sh
```

---

## ?? Need Help?

1. Check `README_ConvLSTM_Setup.md` for detailed docs
2. Review error logs: `logs/train_*.err`
3. Test with small data subset first
4. Verify all file paths are correct

**Good luck with your ConvLSTM training!** ??