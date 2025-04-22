# Complete Guide to Restarting and Merging MCMC Runs

This guide explains how to restart both your standard and temperature annealing MCMC runs from checkpoints and merge them with previous results.

## Overview

Your MCMC jobs were running on the MIT Engaging cluster with 8 chains, 30,000 steps planned for each chain. Your annealing MCMC run was terminated after approximately 10,800 steps (36% completion). The `restart_full_mcmc.py` script will:

1. Find the latest checkpoints for all chains (both standard and annealing)
2. Continue each chain from exactly where it left off
3. Merge original and continued chains for both MCMC methods
4. Run analysis on the merged results
5. Compare standard vs. annealing results

## How It Works

### Finding the Latest Checkpoints

The script searches for:
- Standard MCMC checkpoints: `chain_*_checkpoint_*.npz` or `standard_chain_*_checkpoint_*.npz`
- Annealing MCMC checkpoints: `annealing_chain_*_checkpoint_*.npz`

### Continuing Chains

For standard MCMC chains:
1. Loads the last parameter state
2. Uses appropriate proposal scales
3. Runs the remaining steps

For annealing MCMC chains:
1. Loads the last parameter state
2. Calculates the appropriate temperature to restart from
3. Continues with remaining steps using the correct temperature schedule

### Merging and Analysis

1. Original and continued chains are merged for each chain ID
2. Burn-in fraction is applied to the merged chains
3. Analysis is run on the merged chains
4. Standard and annealing results are compared

## Step-by-Step Instructions

### 1. Transfer the Scripts to Engaging

Use the provided transfer script:

```bash
./transfer_full_restart.sh
```

### 2. Submit the Job on Engaging

SSH into the cluster and run:

```bash
ssh davidct@engaging-login.mit.edu
cd /nfs/home2/davidct/FinalProject
chmod +x run_restart_full.sh
sbatch run_restart_full.sh
```

### 3. Monitor Progress

Check the job status with:

```bash
squeue -u davidct
```

Monitor the logs to ensure chains are continuing successfully:

```bash
tail -f mcmc_full_*.log
```

## Job Configuration

The SLURM job is configured with:
- 72 hours run time to ensure completion
- 8 CPUs and 16GB memory
- NumPy compatibility fix automatically loaded
- Backup of original data before processing

## Adjustable Parameters

You can modify these parameters in `run_restart_full.sh` if needed:

- `--remaining_steps`: Number of steps to run (default: 19200)
- `--burnin`: Burn-in fraction for merged chains (default: 0.2)
- `--t_initial`: Initial temperature (default: 5.0)
- `--t_final`: Final temperature (default: 1.0)
- `--schedule`: Temperature schedule (default: exponential)

## Output Structure

The merged results directory will contain:

### Standard MCMC Results
- `standard_chains_lambdaCDM.npy`: Combined chains
- `standard_trimmed_chains_lambdaCDM.npy`: Chains with burn-in removed
- `README_STANDARD.md`: Summary of standard MCMC results

### Annealing MCMC Results
- `annealing_chains_lambdaCDM.npy`: Combined chains
- `annealing_trimmed_chains_lambdaCDM.npy`: Chains with burn-in removed
- `annealing_temps_lambdaCDM.npy`: Temperature profiles
- `README_ANNEALING.md`: Summary of annealing MCMC results

### Comparison Results
- `README_COMPARISON_MERGED.md`: Side-by-side comparison
- `method_comparison.csv`: Parameter constraint table
- Comparison plots showing benefits of temperature annealing

### Individual Chain Data
Each chain will have its own subdirectory with:
- Original chain data
- Continued chain data
- Merged chain data

## Understanding the Output

### Chain Statistics

For each chain, you'll see:
- Original steps completed
- Continued steps added
- Total steps
- Acceptance rate

### Parameter Constraints

Parameter constraints will show:
- Median values
- 68% credible intervals
- Comparison with fiducial values

### Visualization

Various plots will be generated:
- Corner plots showing parameter correlations
- Trace plots for checking convergence
- Comparison plots between standard and annealing methods

## Troubleshooting

- **Job terminates early**: Check the error log for issues
- **Checkpoints not found**: Verify the source directory path
- **Missing temperature files**: The script can synthesize temperature arrays if needed
- **Memory issues**: Adjust the `--mem` parameter in the SLURM script

## Contact for Help

If you encounter issues, please contact me for assistance with restarting your MCMC runs.