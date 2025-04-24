# ΛCDM Cosmological Parameter Inference from CMB Data

This repository contains a pipeline for inferring ΛCDM cosmological parameters from Planck CMB data, created for the PHYS212 final project at Harvard University.

## Implementation Features

The codebase implements all theoretical aspects from Section 1.1 of the technical report:

- **Complete ΛCDM Model**: Accurately models the CMB power spectrum including Sachs-Wolfe plateau, acoustic oscillations with proper spacing, and Silk damping
- **Parameter Dependence**: All key quantities respond properly to changes in cosmological parameters
- **Correct Theoretical Implementation**:
  - Acoustic scale calculated as ℓ_A = π·d_A/r_s
  - Silk damping with the proper exponent 2 in exp(-ℓ²/ℓ_D²)
  - Angular diameter distance (d_A) calculated with parameter dependence
  - Recombination redshift (z*) calculated using Hu & Sugiyama formula
  - Matter-radiation equality redshift (z_eq) calculated from cosmological parameters

## Pipeline Overview

The project creates a complete parameter inference pipeline:

1. **Theoretical Model**: Implements the ΛCDM model to predict the CMB power spectrum
2. **Data Handling**: Loads and processes Planck CMB data
3. **Bayesian Inference**: Uses MCMC to sample the posterior distribution of parameters
4. **Advanced Techniques**: Implements temperature annealing for improved exploration
5. **Analysis Tools**: Processes MCMC chains to derive parameter constraints and diagnostics
6. **Visualization**: Creates publication-quality figures of results

## Cosmological Parameters

The code constrains six key ΛCDM parameters:

- **H₀**: Hubble constant (expansion rate) in km/s/Mpc
- **Ω_b h²**: Physical baryon density 
- **Ω_c h²**: Physical cold dark matter density
- **n_s**: Scalar spectral index of primordial fluctuations
- **A_s**: Amplitude of primordial fluctuations
- **τ**: Optical depth to reionization

## Repository Structure

### Core Implementation

- `theoretical_lcdm.py`: Comprehensive ΛCDM model for the CMB power spectrum
- `cosmology_model.py`: Core cosmological calculations and parameter handling
- `CAMB.py`: Simplified interface to CMB physics
- `likelihood.py`: Compares theoretical predictions with observed data
- `parameters.py`: Defines parameter ranges and default values
- `priors.py`: Implements prior distributions for parameters
- `data_loader.py`: Loads Planck CMB data

### MCMC Methods

- `mcmc_run.py`: Standard MCMC implementation
- `modified_mcmc.py`: Temperature annealing MCMC for improved mixing
- `restart_continue_chains.py`: Tools for continuing interrupted MCMC chains

### Analysis Tools

- `analyze_standard_chains.py`: Analysis of standard MCMC results
- `analyze_annealing_chains.py`: Analysis of temperature annealing results
- `analyze_final_steps.py`: Analysis of final (converged) steps in chains
- `compare_annealing_runs.py`: Compares results from different MCMC runs
- `compare_with_planck.py`: Compares results with published Planck values
- `visualize_constraints.py`: Creates constraint plots and parameter tables
- `visualize_model_fits.py`: Shows model fits against observed data

### Execution Scripts

- `run_combined_analysis.sh`: Runs the complete analysis pipeline
- `run_annealing_analysis_engaging.sh`: HPC job script for MIT Engaging cluster
- `analyze_results.sh`: Analyzes MCMC results after completion

## Running the Code

### Prerequisites

1. Create a Python environment with required dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Download Planck CMB data:
   - For full analysis: Obtain `COM_CMB_IQU-smica_2048_R3.00_full.fits` from Planck Legacy Archive
   - For quick testing: Use included binned spectrum data

### Standard MCMC Run

For parameter inference with traditional MCMC:
```bash
python mcmc_run.py --steps 30000 --chains 4 --burn_in 0.2
```

### Temperature Annealing Run

For improved parameter space exploration:
```bash
python modified_mcmc.py --steps 30000 --chains 4 --t_initial 10.0 --t_final 1.0
```

### Analysis

To analyze and visualize MCMC results:
```bash
python analyze_standard_chains.py --chain_dir standard_mcmc_results
python analyze_annealing_chains.py --chain_dir mcmc_results_annealing
python compare_with_planck.py --chain_file posterior_samples_lambdaCDM.npy
```

### Complete Pipeline

Run the entire analysis workflow:
```bash
./run_combined_analysis.sh
```

## Technical Documentation

- `PROJECT_SUMMARY.md`: Overview of implementation and project structure
- `README_MCMC.md`: Details on MCMC implementation and configuration
- `FULL_RESTART_GUIDE.md`: Instructions for continuing interrupted MCMC runs
- `project_technical_report.tex`: Detailed technical report on implementation

## Results

The repository includes results from both standard MCMC and temperature annealing methods:
- Parameter constraints and credible intervals
- Corner plots showing parameter correlations
- Trace plots demonstrating convergence
- Comparison with published Planck 2018 values

## Acknowledgments

- Planck Collaboration for CMB data
- CAMB (Code for Anisotropies in the Microwave Background)
- emcee for MCMC implementation
- Scientific Python ecosystem: NumPy, SciPy, Matplotlib, etc.