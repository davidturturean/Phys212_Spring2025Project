#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 08:30:15 2025

@authors: David Turturean, Daria Teodora Harabor

Run script for the MCMC production analysis for Phase 3.
This script runs the full MCMC analysis with 4 chains of 30,000 steps each,
ensuring that only the real FITS data is used (never simulated data).
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import logging
from mcmc_run import run_multi_chain_mcmc, analyze_chains

# Set up output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcmc_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
run_timestamp = time.strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(OUTPUT_DIR, f"mcmc_production_{run_timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MCMC-PRODUCTION")

def check_data_source():
    """
    Verify that the FITS data is available and can be loaded correctly.
    Returns True if successful, False otherwise.
    """
    try:
        from data_loader import load_planck_data
        logger.info("Testing data loading to ensure FITS file is used...")
        ell, dl, sigma = load_planck_data(source="fits")
        logger.info(f"Successfully loaded {len(ell)} multipoles from FITS file")
        return True
    except Exception as e:
        logger.error(f"Error loading FITS data: {e}")
        logger.error("Please ensure the FITS file COM_CMB_IQU-smica_2048_R3.00_full.fits is available")
        return False

def run_production_mcmc():
    """
    Run the production MCMC analysis with full data.
    
    This function runs 4 chains with 30,000 steps each and analyzes the results.
    Results are saved to the OUTPUT_DIR.
    """
    # First check that FITS data is available
    if not check_data_source():
        logger.error("FITS data check failed. Cannot continue.")
        return False
    
    logger.info(f"Starting production MCMC run at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Results will be saved to {OUTPUT_DIR}")
    
    # Configuration variables for the run
    n_chains = 4
    n_steps = 30000
    burn_in = 0.2  # 20% burn-in
    
    logger.info(f"Run configuration:")
    logger.info(f"  Number of chains: {n_chains}")
    logger.info(f"  Steps per chain: {n_steps}")
    logger.info(f"  Burn-in fraction: {burn_in*100:.0f}%")
    logger.info(f"  Data source: Planck FITS file")
    
    try:
        # Run MCMC chains
        all_chains, trimmed_chains = run_multi_chain_mcmc(
            n_chains=n_chains, 
            n_steps=n_steps, 
            burn_in_fraction=burn_in,
            output_dir=OUTPUT_DIR
        )
        
        # Analyze the chains
        flat_samples, results = analyze_chains(
            all_chains, 
            trimmed_chains,
            output_dir=OUTPUT_DIR
        )
        
        logger.info("MCMC analysis complete")
        logger.info(f"Results saved to {OUTPUT_DIR}")
        
        # Create a README file in the output directory
        readme_content = f"""# ΛCDM MCMC Analysis Results

Run Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Introduction

This folder contains the results of a Markov Chain Monte Carlo (MCMC) analysis of the Planck CMB TT power spectrum, using a physically motivated ΛCDM model with six parameters: H0, Omega_b_h2, Omega_c_h2, n_s, A_s, and tau.

## Files
- chains_lambdaCDM.npy: Raw MCMC chains for all parameters
- posterior_samples_lambdaCDM.npy: Flattened posterior samples after burn-in
- parameter_constraints.csv: Summary statistics for all parameters
- trace_plots.png: Trace plots for visual convergence inspection
- corner_plot.png: Corner plot showing parameter distributions and correlations
- mcmc_production_{run_timestamp}.log: Detailed log of this MCMC run

## Run Configuration
- Number of chains: {n_chains}
- Steps per chain: {n_steps}
- Burn-in fraction: {burn_in*100:.0f}%
- Data source: Planck FITS file (COM_CMB_IQU-smica_2048_R3.00_full.fits)

## MCMC Convergence

The Gelman-Rubin R-hat convergence diagnostic measures whether multiple chains have converged to the same posterior distribution. Values close to 1.0 indicate good convergence (typically R < 1.1 is acceptable).

"""
        # Add convergence statistics
        from analysis import gelman_rubin
        r_hat = gelman_rubin(trimmed_chains)
        
        readme_content += "| Parameter | R-hat | Converged? |\n"
        readme_content += "|-----------|-------|------------|\n"
        
        from parameters import param_names
        for i, name in enumerate(param_names):
            converged = "Yes" if r_hat[i] < 1.1 else "No"
            readme_content += f"| {name} | {r_hat[i]:.3f} | {converged} |\n"
        
        readme_content += """
## Parameter Constraints

The table below shows the median values and 68% credible intervals for each parameter.

| Parameter | Median | 68% Lower | 68% Upper | Fiducial | Offset (%) |
|-----------|--------|-----------|-----------|----------|------------|
"""
        
        # Add parameter results to README
        for i, param in enumerate(results['parameter']):
            median = results['median'][i]
            lower = results['lower_68'][i]
            upper = results['upper_68'][i]
            fiducial = results['fiducial'][i]
            offset = results['offset_pct'][i]
            
            readme_content += f"| {param} | {median:.6g} | {lower:.6g} | {upper:.6g} | {fiducial:.6g} | {offset:.1f} |\n"
        
        # Add interpretation section
        readme_content += """
## Interpretation

The posterior distributions show the probability distributions for each parameter given the Planck CMB TT power spectrum data. The corner plot shows correlations between parameters, revealing degeneracies in the model.

Key findings:
- [To be completed based on the actual results]

## Next Steps

These results can be used to:
1. Compare with Planck published values
2. Explore parameter degeneracies in more detail
3. Investigate the impact of each parameter on the CMB power spectrum
4. Extend the model with additional parameters
"""
        
        # Write README file
        with open(os.path.join(OUTPUT_DIR, "README.md"), "w") as f:
            f.write(readme_content)
        
        logger.info(f"Created README.md in {OUTPUT_DIR}")
        print(f"Summary README created: {os.path.join(OUTPUT_DIR, 'README.md')}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error in MCMC production run: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # Print banner
    print("\n" + "="*80)
    print(" ΛCDM Cosmological Parameter Inference - Production MCMC Run ")
    print(" PHYS 212 Spring 2025 Project")
    print(" Daria Harabor & David Turturean")
    print("="*80 + "\n")
    
    print(f"Starting production MCMC run. Log file: {log_file}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"This run will take several hours to complete.")
    print("You can monitor progress in the log file or terminal output.")
    
    # Run the production MCMC
    success = run_production_mcmc()
    
    if success:
        print("\nMCMC production run completed successfully!")
        print(f"Results are available in {OUTPUT_DIR}")
    else:
        print("\nMCMC production run failed. Check the log file for details.")