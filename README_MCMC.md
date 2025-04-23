# ΛCDM Cosmological Parameter Inference - MCMC Analysis

Scripts for running the MCMC analysis on the MIT Engaging cluster as outlined in Phase3_Phase4_Plan.tex.

## Files Included

### Production MCMC Scripts
- `run_mcmc_engaging.sh`: SLURM job script to run the MCMC production on Engaging
- `submit_mcmc_job.sh`: Helper script to submit and monitor the MCMC job
- `analyze_results.sh`: SLURM job script to run analysis on MCMC results

### Analysis Scripts
- `run_diagnostics.py`: Generates convergence diagnostics (R-hat, autocorrelation, effective sample size)
- `visualize_constraints.py`: Creates parameter constraint visualizations
- `generate_latex_table.py`: Creates publication-ready LaTeX table of constraints
- `compare_with_planck.py`: Compares results with Planck 2018 values
- `visualize_model_fits.py`: Creates comparison plots of model vs. data

## Running the MCMC Analysis

### Step 1: Submit the MCMC Job on Engaging
1. Copy the project files to Engaging
```bash
rsync -avz /Users/davidturturean/Documents/Academics/AcademicsSpring2025/PHYS212/FinalProject/ engaging:/pool001/davidct/PHYS212/FinalProject/
```

2. SSH into Engaging
```bash
ssh engaging
```

3. Navigate to the project directory
```bash
cd /pool001/davidct/PHYS212/FinalProject/
```

4. Submit the MCMC job
```bash
bash submit_mcmc_job.sh
```
This will submit the job and create a monitoring script called `check_progress.sh`.

### Step 2: Monitor Job Progress
While the job is running, you can monitor its progress:

1. Check job status and resource usage
```bash
./check_progress.sh
```

2. Watch the log file in real-time
```bash
tail -f mcmc_output_JOBID.txt
```

3. Monitor the MCMC progress directly
```bash
tail -f mcmc_results/mcmc_production_*.log
```

### Step 3: Analyze Results After Completion
Once the MCMC job completes, run the analysis job:

```bash
sbatch analyze_results.sh
```

This will:
- Generate convergence diagnostics
- Create parameter constraint visualizations
- Compare results with Planck 2018 values
- Visualize model fits to the data
- Create interpretation and summary documents

### Step 4: Copy Results Back to Local Machine
After the analysis job completes, retrieve the results:

```bash
# From your local machine
rsync -avz engaging:/pool001/davidct/PHYS212/FinalProject/mcmc_results/ ./mcmc_results/
```

For large files, you may want to compress them first:
```bash
# On Engaging
cd /pool001/davidct/PHYS212/FinalProject/
tar -czf mcmc_results.tar.gz mcmc_results

# On your local machine
scp engaging:/pool001/davidct/PHYS212/FinalProject/mcmc_results.tar.gz .
tar -xzf mcmc_results.tar.gz
```

## Resource Allocation

The scripts are configured to use:
- **MCMC Job**: 16 CPUs, 64GB memory, 1 day runtime
- **Analysis Job**: 16 CPUs, 32GB memory, 4 hours runtime

These resource allocations are based on the available resources in the `sched_mit_mki` partition, which has 27 idle nodes with 64 CPUs and approximately 385GB memory per node.

## Data Requirements

The script requires the FITS file `COM_CMB_IQU-smica_2048_R3.00_full.fits` to be present in the project directory.

## Expected Results

The MCMC run will produce:
- Raw MCMC chains
- Parameter constraints and credible intervals
- Convergence diagnostics
- Visualizations of parameter posteriors
- Comparisons with Planck 2018 values
- Model fit visualizations with residuals

These results will serve as the basis for your scientific report in Phase 4 of the project.

## Runtime Expectations

The full MCMC analysis with 4 chains of 30,000 steps each will take:
- 8-12 hours on the Engaging cluster with the allocated resources
- Analysis scripts will take approximately 30-45 minutes