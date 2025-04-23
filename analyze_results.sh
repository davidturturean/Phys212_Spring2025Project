#!/bin/bash
#SBATCH --job-name=LCDM_Analysis
#SBATCH -o analysis_output_%j.txt
#SBATCH -e analysis_error_%j.txt
#SBATCH -t 4:00:00
#SBATCH -p sched_mit_mki
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=davidct@mit.edu

# Print commands as they are executed
set -x

echo "Starting ΛCDM Cosmological Parameter Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"

# Load necessary modules
module load deprecated-modules
module load anaconda3/2022.05-x86_64

# Activate your Conda environment
source activate myenv39

# Set the working directory
cd /pool001/davidct/PHYS212/FinalProject

# Create directory for analysis results if it doesn't exist
mkdir -p mcmc_results/constraints
mkdir -p mcmc_results/model_fits

# Verify that MCMC results exist
if [ ! -f "mcmc_results/chains_lambdaCDM.npy" ] || [ ! -f "mcmc_results/posterior_samples_lambdaCDM.npy" ]; then
    echo "ERROR: Required MCMC result files not found!"
    echo "Please ensure the MCMC job has completed successfully."
    exit 1
fi

# Run all the analysis scripts from Phase 3
echo "Running convergence diagnostics..."
python -u run_diagnostics.py

echo "Running parameter constraint visualization..."
python -u visualize_constraints.py

echo "Generating LaTeX table of parameter constraints..."
python -u generate_latex_table.py

echo "Running comparison with Planck 2018 values..."
python -u compare_with_planck.py

echo "Visualizing model fits..."
python -u visualize_model_fits.py

echo "Analysis complete at $(date)"
echo "Total runtime: $SECONDS seconds"

# Summarize the generated files
echo "=== Generated Files ==="
find mcmc_results -type f -name "*.png" | sort
find mcmc_results -type f -name "*.tex" | sort
find mcmc_results -type f -name "*.md" | sort
find mcmc_results -type f -name "*.txt" | sort

# Create a README summarizing all the results
cat > mcmc_results/ANALYSIS_SUMMARY.md << 'EOL'
# ΛCDM Cosmological Parameter Analysis Summary

## Generated Files

### Convergence Diagnostics
- `mcmc_results/running_r_hat.png`: Running R-hat values showing convergence over MCMC steps
- `mcmc_results/autocorrelation.png`: Autocorrelation function for each parameter

### Parameter Constraints
- `mcmc_results/constraints/`: Directory containing parameter constraint plots
  - Individual posterior plots for each parameter with comparison to Planck 2018
  - Correlation matrix visualization
  - Joint posterior plots for strongly correlated parameters
- `mcmc_results/constraints_table.tex`: LaTeX table for scientific report

### Comparison with Planck
- `mcmc_results/planck_comparison.png`: Visual comparison of our constraints with Planck 2018
- `mcmc_results/parameter_interpretation.md`: Detailed interpretation of results

### Model Fits
- `mcmc_results/model_fits/best_fit_model.png`: Best-fit ΛCDM model vs Planck data
- `mcmc_results/model_fits/best_fit_zoomed.png`: Zoomed view of first few acoustic peaks
- `mcmc_results/model_fits/residuals.png`: Normalized residuals between model and data
- `mcmc_results/model_fits/posterior_models.png`: Model uncertainty visualization
- `mcmc_results/model_fits/goodness_of_fit.txt`: Chi-squared statistics

## Next Steps for Phase 4
1. Write the scientific report using these results
2. Implement project extensions (polarization, Bayesian evidence, or extended parameter space)
3. Prepare code documentation and packaging
EOL

echo "Created analysis summary: mcmc_results/ANALYSIS_SUMMARY.md"
echo "All analysis tasks completed successfully!"