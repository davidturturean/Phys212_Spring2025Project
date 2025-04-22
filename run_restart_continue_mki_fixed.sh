#!/bin/bash
#SBATCH --job-name=MCMC_Continue
#SBATCH -p sched_mit_mki_r8
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Set environment variables
export NUMEXPR_MAX_THREADS=16

# Change to the project directory
cd /nfs/home2/davidct/FinalProject

# Load Python environment that works with healpy and numpy
source activate myenv39

# Apply the numpy fix patch before importing anything
cat > numpy_fix_patch.py << 'EOF'
#!/usr/bin/env python3
import sys
import numpy

# Store the original concatenate function
if not hasattr(numpy, '_original_concatenate'):
    _original_concatenate = numpy.concatenate
    
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
    
    # Apply the patch
    numpy.concatenate = _patched_concatenate
    numpy._original_concatenate = _original_concatenate
    
    print("NumPy concatenate patched successfully")
EOF

# Create a data loader patch
cat > data_loader_patch.py << 'EOF'
#!/usr/bin/env python3
import sys
import os

def patch_data_loader():
    try:
        import data_loader
        
        # Store original function
        original_load_planck_data = data_loader.load_planck_data
        
        # Create a patched version that always uses FITS
        def patched_load_planck_data(source=None, **kwargs):
            print("Forcing FITS source for Planck data")
            return original_load_planck_data(source="fits", **kwargs)
            
        # Apply the patch
        data_loader.load_planck_data = patched_load_planck_data
        data_loader._original_load_planck_data = original_load_planck_data
        
        print("Data loader patched to always use FITS source")
        return True
    except Exception as e:
        print(f"Error patching data_loader: {e}")
        return False

if __name__ == "__main__":
    success = patch_data_loader()
    sys.exit(0 if success else 1)
EOF

# Make the patch scripts executable
chmod +x numpy_fix_patch.py data_loader_patch.py

# Create a wrapper script to run with patches
cat > wrapper.py << 'EOF'
#!/usr/bin/env python3
# Import patches first
import numpy_fix_patch
print("NumPy patch loaded")
import data_loader_patch
data_loader_patch.patch_data_loader()

# Now run the real script
import sys
import restart_continue_chains

if __name__ == "__main__":
    print("Running restart_continue_chains.py with fixed imports...")
    restart_continue_chains.main()
EOF

# Create execution script
cat > run_with_patches.sh << 'EOF'
#!/bin/bash
# First ensure healpy uses FITS data
export FITS_DATA=1
export FORCE_FITS=1

# Run with patched imports
python wrapper.py "$@"
EOF

chmod +x run_with_patches.sh wrapper.py

# Run the restart script with all our patches
./run_with_patches.sh \
    --source_dir /nfs/home2/davidct/FinalProject/mcmc_results \
    --output_dir /nfs/home2/davidct/FinalProject/mcmc_results_merged \
    --mode both \
    --remaining_steps 19200 \
    --burnin 0.2 \
    --t_initial 5.0 \
    --t_final 1.0 \
    --schedule exponential \
    --backup

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