# ΛCDM Cosmological Parameter Inference

PHYS 212 Spring 2025 Final Project  
Harvard University  
Project Team: Daria Teodora Harabor & David Turturean

## Project Overview

This project performs Bayesian inference of ΛCDM cosmological parameters using the Planck CMB TT power spectrum data. We extract the CMB temperature power spectrum directly from the Planck FITS maps, implement a physically-motivated model to explain the acoustic oscillations, and use MCMC techniques to sample the posterior distribution of cosmological parameters.

## Scientific Goals

1. Extract and visualize the CMB power spectrum from the Planck FITS data, highlighting the acoustic peaks
2. Implement a semi-analytic model for the CMB power spectrum that captures the key physical features
3. Use Bayesian inference to constrain ΛCDM parameters and estimate their uncertainties
4. Investigate parameter degeneracies and compare results with published Planck values

## Data

All our analysis is based on **real data** from the Planck satellite:
- `COM_CMB_IQU-smica_2048_R3.00_full.fits`: Planck SMICA temperature map from the public data release
- We extract the TT power spectrum directly from the map using spherical harmonic transforms (no simulated data is used)

## Pipeline Overview

The project follows this analysis pipeline:

1. **Data Extraction**: Extract CMB power spectrum from Planck FITS map
2. **Data Processing**: Apply binning to enhance visualization of acoustic peaks
3. **Model Implementation**: Create theoretical ΛCDM model for the power spectrum
4. **Likelihood Analysis**: Calculate likelihood comparing model to data
5. **MCMC Sampling**: Use Metropolis-Hastings to sample parameter space
6. **Parameter Inference**: Analyze MCMC chains to derive constraints

## Requirements

- Python 3.7+
- NumPy, SciPy, Matplotlib
- healpy (for processing Planck FITS maps)
- pandas (for data management)
- corner (for plotting parameter constraints)

## Setup and Running the Analysis

1. **Setup the environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Data Visualization and Model Comparison**:
   ```bash
   python3 final_visualization.py
   ```
   This extracts the CMB power spectrum from the FITS file, bins the data, and creates visualizations comparing the data with our theoretical ΛCDM model. Results are saved in the `output/` directory.

3. **Full Analysis Workflow**:
   ```bash
   python3 run_final.py
   ```
   This runs the complete analysis pipeline, from data extraction to model fitting and comparison, with quantitative evaluation of the model fit.

4. **MCMC Parameter Inference**:
   ```bash
   # For a quick test run to verify the MCMC pipeline
   python3 mcmc_test.py
   
   # For the full production run (takes several hours)
   python3 run_mcmc_production.py
   ```
   The production script runs 4 MCMC chains with 30,000 steps each to robustly constrain cosmological parameters, implementing a 20% burn-in period and regular checkpointing. Results are saved to the `mcmc_results/` directory with comprehensive logging.

5. **Verify the Output**:
   Analysis results are saved to the following directories:
   
   **Visualization Results** (`output/` directory):
   - `cmb_power_spectrum_final.png` - CMB power spectrum with acoustic peaks
   - `cmb_power_spectrum_zoomed.png` - Zoomed view of acoustic peaks
   - `model_vs_data.png` - Comparison of ΛCDM model with Planck data
   - `model_vs_data_zoomed.png` - Zoomed comparison view
   - `cmb_spectrum_scientific.png` - Scientific visualization with feature annotations
   - Parameter sensitivity plots in `output/parameter_sensitivity/`
   
   **MCMC Results** (`mcmc_results/` directory):
   - `chains_lambdaCDM.npy` - Raw MCMC chains for all parameters
   - `posterior_samples_lambdaCDM.npy` - Flattened posterior samples after burn-in
   - `parameter_constraints.csv` - Summary statistics for all parameters
   - `trace_plots.png` - Trace plots showing parameter convergence
   - `corner_plot.png` - Corner plot showing parameter distributions and correlations
   - `README.md` - Detailed summary of MCMC results and parameter constraints
   - Individual chain files and checkpoints for resume capability

## File Descriptions

### Core Files

- `data_loader.py`: Extracts and processes the CMB power spectrum from Planck FITS file
- `theoretical_lcdm.py`: State-of-the-art ΛCDM model with proper acoustic oscillations
- `final_visualization.py`: Creates publication-quality visualizations of data and model
- `run_final.py`: Runs the full analysis pipeline from data extraction to model comparison

### Theoretical Framework

- `CAMB.py`: Basic semi-analytic model for CMB spectrum (Daria's implementation)
- `cosmology_model.py`: Enhanced physical model integrating the improved ΛCDM model
- `parameters.py`: Defines ΛCDM parameters, prior distributions, and parameter ranges

### MCMC Implementation

- `likelihood.py`: Log-likelihood function comparing model predictions to data
- `priors.py`: Implements prior distributions and log-posterior calculation
- `mcmc_run.py`: Metropolis-Hastings MCMC infrastructure with robust logging and checkpointing
- `run_mcmc_production.py`: Production-ready script for running the full MCMC analysis
- `mcmc_test.py`: Quick test script to verify the MCMC pipeline
- `analysis.py`: Tools for analyzing MCMC chains and generating publication-quality plots

### Utilities and Supporting Files

- `check.py`: Diagnostic tools for model validation (Daria's implementation)
- `MCMC.py`: Original MCMC implementation (Daria's implementation)
- `readheader.py`: Helper module for reading FITS headers (Daria's implementation)
- `cleanup_final.py`: Script for organizing output files and cleaning the project
- `requirements.txt`: Python package dependencies

## ΛCDM Parameters

We infer these six standard ΛCDM parameters:

| Parameter | Description | Prior Range |
|-----------|-------------|-------------|
| **H₀** | Hubble constant [km/s/Mpc] | Uniform (50-90) |
| **Ωbₕ²** | Physical baryon density | Gaussian (0.0224 ± 0.0005) |
| **Ωcₕ²** | Physical cold dark matter density | Uniform (0.05-0.25) |
| **nₛ** | Scalar spectral index | Uniform (0.85-1.05) |
| **Aₛ** | Primordial amplitude [10⁻⁹] | Uniform (1.0-4.0) |
| **τ** | Optical depth to reionization | Gaussian (0.0544 ± 0.0073) |

Our MCMC implementation uses multiple chains with different starting points within these ranges, and we apply a 20% burn-in period to remove the initial non-equilibrium samples. We use the Metropolis-Hastings algorithm with adaptive proposal scales, and implement the Gelman-Rubin diagnostic to verify convergence.

## Key Features

- Direct extraction of CMB power spectrum from Planck FITS maps
- ΛCDM model with physically-motivated acoustic oscillations
- Gaussian prior on optical depth to reionization (τ) based on Planck constraints
- Parameter ranges broad enough to allow exploration beyond Planck's best-fit values
- Multiple MCMC chains with different starting points to ensure robust convergence
- Implementation of Gelman-Rubin convergence diagnostics
- Regular checkpointing to protect long runs from system interruptions
- Comprehensive logging of MCMC progress and parameter evolution
- Publication-quality visualization of parameter constraints with corner plots

## Acknowledgments

This project is part of the PHYS 212 course at Harvard University. We thank our instructor for guidance and the Planck Collaboration for making their data publicly available.

## References

- Planck 2018 Results: [arXiv:1807.06209](https://arxiv.org/abs/1807.06209)
- Hu & Sugiyama (1996): [arXiv:astro-ph/9510117](https://arxiv.org/abs/astro-ph/9510117)
- Lewis & Challinor (2000): [arXiv:astro-ph/9911177](https://arxiv.org/abs/astro-ph/9911177)