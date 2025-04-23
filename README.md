# ΛCDM Cosmological Parameter Inference

This repository contains the code for inferring cosmological parameters from Planck CMB data using a ΛCDM model, created for the PHYS212 course at Harvard University.

## Project Overview

This project implements a complete pipeline for cosmological parameter inference:

1. **Cosmological Model**: Implementing the ΛCDM model using CAMB, which predicts the CMB power spectrum based on cosmological parameters.
   
2. **Bayesian Inference**: Using Markov Chain Monte Carlo (MCMC) methods to sample the posterior distribution of cosmological parameters.
   
3. **Data Analysis**: Working with Planck CMB data to compare theoretical predictions with observational data.
   
4. **Temperature Annealing**: Implementing advanced MCMC techniques for improved parameter constraints.

## Key Cosmological Parameters

The ΛCDM model includes six key parameters that we infer:

- **H₀**: Hubble constant (current expansion rate of the universe)
- **Ωₘh²**: Physical baryon density
- **Ωₖh²**: Physical dark matter density
- **nₛ**: Scalar spectral index (measure of scale-dependence of primordial fluctuations)
- **Aₛ**: Amplitude of primordial fluctuations
- **τ**: Optical depth to reionization

## Repository Structure

- `CAMB.py`: Interface to the CAMB cosmological code
- `theoretical_lcdm.py`: ΛCDM model implementation
- `cosmology_model.py`: Core cosmological model calculations
- `likelihood.py`: Likelihood function for comparing model to data
- `priors.py`: Prior distributions for cosmological parameters
- `parameters.py`: Parameter definitions and fiducial values
- `data_loader.py`: Loads and processes Planck CMB data
- `mcmc_run.py`: Standard MCMC implementation
- `modified_mcmc.py`: Temperature annealing MCMC implementation
- `restart_continue_chains.py`: Tools for continuing interrupted MCMC runs
- `analyze_combined_results.py`: Analysis and visualization of MCMC results

## Running the Code

### Environment Setup

1. Create a Python environment with required dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Download the Planck CMB data file from the Planck Legacy Archive and place it in the project directory.

### Local Execution

For a local MCMC run with standard sampling:
```bash
python mcmc_run.py --steps 5000 --chains 4
```

For a local run with temperature annealing:
```bash
python modified_mcmc.py --steps 5000 --chains 4 --t_initial 5.0 --t_final 1.0
```

### HPC Execution

For running on a high-performance computing cluster (e.g., MIT Engaging):
```bash
sbatch run_restart_continue_mki_fixed.sh
```

## Analysis

Analyze and visualize MCMC results:
```bash
python analyze_combined_results.py --output_dir results
```

## Key Files

- `README_FITS_FIX.md`: Documentation on FITS data loading
- `README_MCMC.md`: Documentation on MCMC implementation
- `FULL_RESTART_GUIDE.md`: Guide for continuing interrupted MCMC runs

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the Planck Legacy Archive data
- CAMB (Code for Anisotropies in the Microwave Background)
- MCMC implementation inspired by statistical inference best practices