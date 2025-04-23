#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize parameter constraints from ΛCDM MCMC analysis

This script creates detailed visualizations of parameter constraints:
1. Individual posterior plots for each parameter with comparison to Planck 2018
2. Parameter correlation matrix
3. Joint posterior plots for strongly correlated parameters

Authors: David Turturean, Daria Teodora Harabor
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import corner

# Create output directory for constraint visualizations
os.makedirs("mcmc_results/constraints", exist_ok=True)

# Load the posterior samples (after burn-in)
try:
    posterior_samples = np.load("mcmc_results/posterior_samples_lambdaCDM.npy")
    from parameters import param_names
    print(f"Loaded posterior samples with shape {posterior_samples.shape}")
except Exception as e:
    print(f"ERROR: Failed to load posterior samples: {e}")
    print("Make sure to run the MCMC simulation first using run_mcmc_production.py")
    exit(1)

# Planck 2018 values for comparison
planck_values = {
    "H0": (67.36, 0.54),
    "Omega_b_h2": (0.02237, 0.00015),
    "Omega_c_h2": (0.1200, 0.0012),
    "n_s": (0.9649, 0.0042),
    "A_s": (2.1e-9, 0.03e-9),
    "tau": (0.0544, 0.0073)
}

print("Generating individual posterior plots with Planck comparison...")

# Generate individual posterior plots with Planck comparison
for i, param in enumerate(param_names):
    samples = posterior_samples[:, i]
    planck_val, planck_err = planck_values[param]
    
    # Create histogram
    plt.figure(figsize=(8, 5))
    n, bins, patches = plt.hist(samples, bins=50, density=True, alpha=0.7, 
                           color='blue', label='Our constraint')
    
    # Add Planck value
    plt.axvline(planck_val, color='red', linestyle='-', linewidth=2, 
               label=f'Planck 2018: {planck_val:.6g}±{planck_err:.6g}')
    
    # Add shaded 1σ region for Planck
    plt.axvspan(planck_val-planck_err, planck_val+planck_err, 
               alpha=0.2, color='red')
    
    # Calculate our constraints
    median = np.percentile(samples, 50)
    lower = np.percentile(samples, 16)
    upper = np.percentile(samples, 84)
    
    # Add our constraints
    plt.axvline(median, color='blue', linestyle='-', linewidth=2)
    plt.axvspan(lower, upper, alpha=0.2, color='blue')
    
    # Add text for our constraint
    plt.text(0.05, 0.95, f'Our constraint: {median:.6g}$^{{+{upper-median:.6g}}}_{{{lower-median:.6g}}}$',
            transform=plt.gca().transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Formatting
    plt.xlabel(param)
    plt.ylabel('Probability density')
    plt.title(f'Posterior distribution: {param}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"mcmc_results/constraints/{param}_posterior.png", dpi=300)
    plt.close()

print("Generated individual posterior plots for all parameters")
    
# Calculate parameter correlations
print("Calculating parameter correlations...")
corr_matrix = np.corrcoef(posterior_samples.T)

# Plot correlation matrix
plt.figure(figsize=(8, 7))
im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, label='Correlation coefficient')

# Add correlation values
for i in range(len(param_names)):
    for j in range(len(param_names)):
        plt.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                ha='center', va='center', color='black')

plt.xticks(np.arange(len(param_names)), param_names, rotation=45)
plt.yticks(np.arange(len(param_names)), param_names)
plt.title('Parameter Correlation Matrix')
plt.tight_layout()
plt.savefig("mcmc_results/constraints/correlation_matrix.png", dpi=300)
plt.close()

print("Saved correlation matrix visualization")

# Create 2D posterior plots for strongly correlated parameters
print("Creating 2D posterior plots for strongly correlated parameters...")

# Find top 3 strongest correlations
corr_abs = np.abs(corr_matrix - np.identity(len(param_names)))
pairs = []
for _ in range(3):
    i, j = np.unravel_index(np.argmax(corr_abs), corr_abs.shape)
    if i != j:  # Avoid diagonal
        pairs.append((i, j))
        corr_abs[i, j] = 0  # Zero out to find next highest
        
for i, j in pairs:
    # Create 2D plot for this parameter pair
    plt.figure(figsize=(8, 7))
    param_i, param_j = param_names[i], param_names[j]
    
    # Extract samples for these parameters
    x = posterior_samples[:, i]
    y = posterior_samples[:, j]
    
    # Create 2D histogram
    plt.hist2d(x, y, bins=50, cmap='Blues')
    
    # Add Planck values
    planck_i, planck_err_i = planck_values[param_i]
    planck_j, planck_err_j = planck_values[param_j]
    plt.errorbar(planck_i, planck_j, xerr=planck_err_i, yerr=planck_err_j,
                marker='o', color='red', label='Planck 2018')
    
    # Add correlation coefficient
    plt.text(0.05, 0.95, f'Correlation: {corr_matrix[i, j]:.3f}',
            transform=plt.gca().transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Formatting
    plt.xlabel(param_i)
    plt.ylabel(param_j)
    plt.title(f'Joint Posterior: {param_i} vs {param_j}')
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"mcmc_results/constraints/{param_i}_{param_j}_joint.png", dpi=300)
    plt.close()
    
print("Created 2D posterior plots for the most strongly correlated parameter pairs")
print("Parameter constraint visualizations saved to mcmc_results/constraints/")
