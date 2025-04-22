# Project Summary: ΛCDM Cosmological Parameter Inference

## Objective
This project implements a Bayesian inference framework to determine cosmological parameters from Planck CMB data using the ΛCDM model. It focuses on two primary MCMC methods: standard Metropolis-Hastings sampling and temperature annealing.

## Implementation Details

### Core Components
1. **Cosmological Model**: Interface to CAMB for theoretical power spectrum calculations
2. **Likelihood Function**: Compares theoretical predictions with Planck data
3. **MCMC Implementation**: Two variants:
   - Standard Metropolis-Hastings algorithm
   - Temperature annealing for improved parameter space exploration
4. **Analysis Framework**: Tools for convergence diagnostics and parameter constraint visualization

### Technical Features
1. **Checkpoint/Restart**: Ability to continue interrupted MCMC runs
2. **Parallelization**: Support for multiple chains
3. **FITS Data Handling**: Robust loading of Planck FITS files
4. **HPC Integration**: Scripts for MIT Engaging cluster execution
5. **Parameter Analysis**: Statistical tools for constraints and comparisons

## Results

The analysis demonstrates several key findings:

1. The temperature annealing MCMC method provides more reliable parameter constraints than standard MCMC
2. Improved exploration of parameter space helps avoid local maxima traps
3. The combined analysis approach yields robust constraints on all six ΛCDM parameters
4. Results show good agreement with published Planck values

## Computational Methods

The project employs statistical techniques including:

1. Markov Chain Monte Carlo (MCMC) sampling
2. Metropolis-Hastings algorithm
3. Temperature annealing
4. Gelman-Rubin convergence diagnostics
5. Kernel Density Estimation for posterior visualization
6. Bayesian credible interval calculation

## Skills Demonstrated

This project demonstrates competence in:

- Scientific computing with Python
- Bayesian statistical analysis
- High-performance computing
- Physics-based forward modeling
- Data analysis and visualization
- Software engineering best practices