#!/bin/bash
#SBATCH --job-name=annealing_analysis
#SBATCH --output=annealing_analysis_%j.log
#SBATCH --error=annealing_analysis_%j.err
#SBATCH --partition=sched_mit_mki
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Use direct path to conda Python instead of activating environment
PYTHON="/home/davidct/.conda/envs/myenv39/bin/python"

# Check if Python exists
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON"
    echo "Available Python versions:"
    find /home/davidct -name "python" | grep "/bin/python"
    exit 1
fi

echo "Using Python: $PYTHON"

# Create output directories
mkdir -p annealing_analysis
mkdir -p annealing_comparison

# Set core count for parallel processing
export OMP_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

# First, analyze all annealing chains together
echo "Analyzing all annealing chains..."
$PYTHON analyze_annealing_chains.py --output-dir ./annealing_analysis

# Then, compare separate annealing runs
echo "Comparing separate annealing runs..."
$PYTHON compare_annealing_runs.py --auto-find --output-dir ./annealing_comparison

echo "Analysis complete. Results saved to annealing_analysis/ and annealing_comparison/"