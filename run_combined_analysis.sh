#!/bin/bash
#SBATCH --job-name=Combined_Analysis
#SBATCH -p sched_mit_mki_r8
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00

# Change to the project directory
cd /nfs/home2/davidct/FinalProject

# Load Python environment
source activate myenv39

# Set environment variables for NumPy
export NUMEXPR_MAX_THREADS=8

# Create a patch script to fix NumPy concatenate
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

# Create a wrapper to patch numpy first
cat > run_analysis.py << 'EOF'
#!/usr/bin/env python3
# Import numpy patch first
import numpy_fix_patch
print("NumPy patch loaded")

# Import other standard modules needed for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    print("Matplotlib loaded")
except ImportError:
    print("WARNING: Could not import matplotlib, plotting will be disabled")

try:
    import pandas as pd
    print("Pandas loaded")
except ImportError:
    print("WARNING: Could not import pandas, table creation will be limited")

try:
    import corner
    print("Corner loaded")
except ImportError:
    print("WARNING: Could not import corner, corner plots will be disabled")

# Now import and run our analysis script
import analyze_combined_results
analyze_combined_results.main()
EOF

# Make scripts executable
chmod +x numpy_fix_patch.py run_analysis.py

# Use only the original results and final results directories
echo "Using only mcmc_results and final results directories..."
mcmc_dirs=()

# Check standard mcmc_results directory
if [ -d "mcmc_results" ]; then
    mcmc_dirs+=("mcmc_results")
    echo "- Using mcmc_results directory"
fi

# Check for specific final results directory
if [ -d "mcmc_results_final_20250422_080408" ]; then
    mcmc_dirs+=("mcmc_results_final_20250422_080408")
    echo "- Using mcmc_results_final_20250422_080408 directory"
fi

# Explicitly exclude merged results
echo "- Explicitly excluding mcmc_results_merged directory"

# Build directories argument
dirs_arg=""
for dir in "${mcmc_dirs[@]}"; do
    dirs_arg+="$dir "
done

if [ -z "$dirs_arg" ]; then
    echo "ERROR: No results directories found!"
    exit 1
fi

echo "Running combined analysis on: $dirs_arg"

# Run the analysis
python run_analysis.py --dirs $dirs_arg --output_dir combined_analysis_results --mode both --burnin 0.2

# Check if analysis completed successfully
if [ $? -eq 0 ]; then
    echo "Combined analysis completed successfully!"
    
    # Create a tar archive of the results
    tar -czf combined_analysis_results.tar.gz combined_analysis_results/
    echo "Results compressed to combined_analysis_results.tar.gz"
    
    # Print summary of key results
    echo ""
    echo "=== ANALYSIS SUMMARY ==="
    if [ -f "combined_analysis_results/README_COMPARISON.md" ]; then
        echo "Comparison of methods found in: combined_analysis_results/README_COMPARISON.md"
    fi
    
    if [ -f "combined_analysis_results/method_comparison.csv" ]; then
        echo ""
        echo "Parameter comparison:"
        head -n 10 combined_analysis_results/method_comparison.csv
    fi
else
    echo "ERROR: Analysis failed!"
fi