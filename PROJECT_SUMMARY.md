# PHYS212 Final Project - MCMC Analysis of CMB Data

## Implementation Improvements
The theoretical model has been updated to properly implement all formulas from Section 1.1 of the report, including:
- Fixed Silk damping to use the correct exponent of 2 (exp(-ℓ²/ℓ_D²))
- Proper calculation of the acoustic scale (ℓ_A = π·d_A/r_s)
- Parameter-dependent calculation of angular diameter distance (d_A)
- Recombination redshift (z*) calculated using Hu & Sugiyama formula
- Matter-radiation equality redshift (z_eq) calculated from the ratio of matter to radiation density
- All key quantities now properly respond to changes in cosmological parameters

## Project Structure

### Core Files
- `CAMB.py`: Power spectrum calculation using CAMB
- `cosmology_model.py`: Implementation of ΛCDM cosmological model
- `data_loader.py`: Functions to load Planck CMB data
- `likelihood.py`: Likelihood function for parameter estimation
- `mcmc_run.py`: Standard MCMC implementation
- `modified_mcmc.py`: Enhanced MCMC with temperature annealing
- `parameters.py`: Cosmological parameter definitions and priors
- `priors.py`: Prior distribution implementations
- `theoretical_lcdm.py`: Theoretical predictions for ΛCDM

### Analysis Scripts
- `analyze_annealing_chains.py`: Analysis of temperature annealing chains
- `analyze_final_steps.py`: Analysis of final steps in annealing chains
- `analyze_standard_chains.py`: Analysis of standard MCMC chains
- `compare_annealing_runs.py`: Comparison between different annealing runs
- `visualize_constraints.py`: Visualization of parameter constraints
- `visualize_model_fits.py`: Visualization of model fits to data
- `compare_with_planck.py`: Comparison with Planck 2018 results

### Result Directories
- `mcmc_results/`: Standard MCMC results
- `mcmc_results_merged/`: Merged MCMC results
- `mcmc_results_annealing_20250422_170412/`: First annealing run results
- `annealing_analysis/`: Analysis of annealing chains
- `annealing_analysis_final_steps/`: Analysis of final portion of annealing
- `annealing_comparison/`: Comparison between annealing runs
- `standard_chains_analysis/`: Analysis of standard MCMC chains
- `output/`: Output visualizations and data

## How to Run

1. Standard MCMC Analysis:
   ```
   python analyze_standard_chains.py
   ```

2. Annealing MCMC Analysis:
   ```
   python analyze_annealing_chains.py
   ```

3. Compare with Planck:
   ```
   python compare_with_planck.py
   ```

4. Final Steps Analysis:
   ```
   python analyze_final_steps.py
   ```

5. Run Combined Analysis:
   ```
   ./run_combined_analysis.sh
   ```
