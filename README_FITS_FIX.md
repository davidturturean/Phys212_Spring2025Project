# FITS Data Loading Fix for MCMC Chain Continuation

This document explains the fixes implemented in `run_restart_continue_mki_fixed.sh` to ensure proper FITS data usage when continuing MCMC chains.

## Problem: MCMC Using Synthetic Data Instead of FITS Data

The original MCMC script would sometimes use synthetic model data instead of the actual Planck FITS data, causing incorrect parameter inference results. The error messages indicated import issues with numpy, pandas, matplotlib, and healpy, as well as problems with FITS data access.

## Solution: Runtime Patching

Our solution implements runtime patches that modify Python modules at load time, ensuring:

1. **FITS Data Is Always Used**: We force the data loader to always use FITS data by patching the `load_planck_data` function.

2. **NumPy Compatibility Fixes**: We fix the "concatenate() got an unexpected keyword argument 'dtype'" error by patching NumPy's concatenate function.

3. **Sequential Import Order**: We enforce the correct import order with a wrapper script.

## Implementation Details

### 1. NumPy Fix Patch

```python
# Define a patched version of concatenate
def _patched_concatenate(*args, **kwargs):
    if 'dtype' in kwargs:
        dtype = kwargs.pop('dtype')
        import warnings
        warnings.warn("Removed dtype from np.concatenate for compatibility")
        result = _original_concatenate(*args, **kwargs)
        if dtype is not None:
            result = result.astype(dtype)
        return result
    return _original_concatenate(*args, **kwargs)
```

### 2. Data Loader Patch

```python
# Create a patched version that always uses FITS
def patched_load_planck_data(source=None, **kwargs):
    print("Forcing FITS source for Planck data")
    return original_load_planck_data(source="fits", **kwargs)
```

### 3. Wrapper Script

```python
#!/usr/bin/env python3
# Import patches first
import numpy_fix_patch
print("NumPy patch loaded")
import data_loader_patch
data_loader_patch.patch_data_loader()

# Now run the real script
import sys
import restart_continue_chains
```

### 4. Environment Variables

We set environment variables to enforce FITS data usage:

```bash
export FITS_DATA=1
export FORCE_FITS=1
```

## Data Verification Checks

The `restart_continue_chains.py` script now includes explicit verification to ensure it's using real Planck data:

```python
# Check that we're using real data
from data_loader import load_planck_data
logger.info("Testing data loading to ensure real Planck data is used...")
ell, dl, sigma = load_planck_data(source="fits")
logger.info(f"Successfully loaded {len(ell)} multipoles from FITS file")
```

## Fallback Mechanism

If continuing the chains fails, the script will fall back to just analyzing existing results:

```bash
# Check if the run completed successfully
if [ $? -eq 0 ]; then
    echo "MCMC restart and merge job completed successfully"
else
    echo "ERROR: MCMC restart and merge job failed!"
    
    # Try just the analysis phase as fallback
    echo "Trying just the analysis step on existing results..."
    ./run_with_patches.sh \
        --mode analyze \
        --output_dir /nfs/home2/davidct/FinalProject/mcmc_results_merged \
        --burnin 0.2
fi
```

## How to Use

1. Transfer the necessary files to the cluster:
   ```
   ./transfer_continue_restart.sh
   ```

2. On the cluster, submit the job:
   ```
   sbatch run_restart_continue_mki_fixed.sh
   ```

3. The job will:
   - Apply patches to ensure FITS data usage
   - Continue your MCMC chains from checkpoints
   - Merge the continued chains with previous results
   - Analyze the merged results
   - Compare standard and annealing MCMC methods

## Resources

The job is configured to run on the MIT MKI SLURM partition with:
- 16 CPUs
- 32GB of memory
- 24 hours of runtime

These resources should be sufficient to complete the remaining MCMC steps and analysis.