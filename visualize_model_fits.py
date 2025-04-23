#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize model fits to CMB data from ΛCDM MCMC analysis

This script creates visualizations comparing the best-fit ΛCDM model
and parameter uncertainty with Planck CMB data.

Authors: David Turturean, Daria Teodora Harabor
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from theoretical_lcdm import lcdm_power_spectrum
from data_loader import load_planck_data

# Create output directory
os.makedirs("mcmc_results/model_fits", exist_ok=True)

# Load Planck data
try:
    ell_data, dl_data, sigma_data = load_planck_data(source="fits")
    print(f"Loaded {len(ell_data)} multipoles from FITS file")
    
    # Filter the data (optional, for cleaner visualization)
    mask = (ell_data >= 2) & (ell_data <= 2000)
    ell_data = ell_data[mask]
    dl_data = dl_data[mask]
    sigma_data = sigma_data[mask]
    print(f"Filtered to {len(ell_data)} multipoles for visualization")
except Exception as e:
    print(f"ERROR: Failed to load FITS data: {e}")
    print("Make sure the FITS file COM_CMB_IQU-smica_2048_R3.00_full.fits is available")
    exit(1)

# Load posterior samples
try:
    posterior_samples = np.load("mcmc_results/posterior_samples_lambdaCDM.npy")
    from parameters import param_names
    print(f"Loaded posterior samples with shape {posterior_samples.shape}")
except Exception as e:
    print(f"ERROR: Failed to load posterior samples: {e}")
    print("Make sure to run the MCMC simulation first using run_mcmc_production.py")
    exit(1)

# Get best-fit (median) parameters
median_params = {}
for i, param in enumerate(param_names):
    median_params[param] = np.median(posterior_samples[:, i])

print("Median parameter values:")
for param, value in median_params.items():
    print(f"  {param}: {value:.6g}")

# Generate model with median parameters
print("\nGenerating best-fit model spectrum...")
ell_model = np.arange(2, 2501)
dl_model = lcdm_power_spectrum(ell_model, median_params)

# Plot best-fit model vs data
plt.figure(figsize=(12, 8))
plt.errorbar(ell_data, dl_data, yerr=sigma_data, fmt='o', markersize=3, 
             alpha=0.4, label='Planck Data')
plt.plot(ell_model, dl_model, 'r-', lw=2, label='Best-fit ΛCDM Model')

plt.xscale('log')
plt.xlabel(r'Multipole $\ell$')
plt.ylabel(r'$D_\ell$ [$\mu K^2$]')
plt.title('CMB TT Power Spectrum: Data vs. Best-fit Model')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mcmc_results/model_fits/best_fit_model.png", dpi=300)
plt.close()

print("Saved best-fit model plot")

# Plot zoomed version of first few peaks
plt.figure(figsize=(12, 8))
mask_data = ell_data <= 1000
mask_model = ell_model <= 1000

plt.errorbar(ell_data[mask_data], dl_data[mask_data], yerr=sigma_data[mask_data], 
             fmt='o', markersize=3, alpha=0.4, label='Planck Data')
plt.plot(ell_model[mask_model], dl_model[mask_model], 'r-', lw=2, label='Best-fit ΛCDM Model')

plt.xlabel(r'Multipole $\ell$')
plt.ylabel(r'$D_\ell$ [$\mu K^2$]')
plt.title('CMB TT Power Spectrum: First Few Acoustic Peaks')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mcmc_results/model_fits/best_fit_zoomed.png", dpi=300)
plt.close()

print("Saved zoomed best-fit model plot")

# Calculate residuals
from scipy.interpolate import interp1d
model_interp = interp1d(ell_model, dl_model, kind='linear', bounds_error=False, fill_value=0)
dl_model_at_data = model_interp(ell_data)
residuals = dl_data - dl_model_at_data
normalized_residuals = residuals / sigma_data

# Plot residuals
plt.figure(figsize=(12, 6))
plt.errorbar(ell_data, normalized_residuals, yerr=1, fmt='o', markersize=3, alpha=0.4)
plt.axhline(0, color='r', ls='-')
plt.axhline(1, color='r', ls='--', alpha=0.5)
plt.axhline(-1, color='r', ls='--', alpha=0.5)
plt.axhline(2, color='r', ls=':', alpha=0.5)
plt.axhline(-2, color='r', ls=':', alpha=0.5)

plt.xscale('log')
plt.xlabel(r'Multipole $\ell$')
plt.ylabel(r'Normalized Residuals $(D_\ell^{\rm data} - D_\ell^{\rm model})/\sigma_\ell$')
plt.title('Residuals: Data - Model')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mcmc_results/model_fits/residuals.png", dpi=300)
plt.close()

print("Saved residual plot")

# Calculate chi-squared
chi2 = np.sum((residuals / sigma_data)**2)
dof = len(ell_data) - len(param_names)
reduced_chi2 = chi2 / dof

print(f"\nChi-squared: {chi2:.2f} for {dof} degrees of freedom")
print(f"Reduced chi-squared: {reduced_chi2:.2f}")

# Sample random models from the posterior
n_samples = 50
indices = np.random.choice(len(posterior_samples), n_samples, replace=False)
sample_models = []

plt.figure(figsize=(12, 8))
plt.errorbar(ell_data, dl_data, yerr=sigma_data, fmt='o', markersize=3, 
             alpha=0.2, color='blue', label='Planck Data')

# Plot best-fit model
plt.plot(ell_model, dl_model, 'r-', lw=3, label='Best-fit Model')

# Plot random models from posterior
for idx in indices:
    params = {param_names[i]: posterior_samples[idx, i] for i in range(len(param_names))}
    dl_sample = lcdm_power_spectrum(ell_model, params)
    plt.plot(ell_model, dl_sample, '-', lw=0.5, alpha=0.1, color='green')

# Add a sample model to the legend
plt.plot([], [], '-', lw=1, alpha=0.5, color='green', label='Posterior Samples')

plt.xscale('log')
plt.xlabel(r'Multipole $\ell$')
plt.ylabel(r'$D_\ell$ [$\mu K^2$]')
plt.title('CMB TT Power Spectrum: Model Uncertainty')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mcmc_results/model_fits/posterior_models.png", dpi=300)
plt.close()

print("Saved posterior model uncertainty plot")

# Save chi-squared and goodness-of-fit info
with open("mcmc_results/model_fits/goodness_of_fit.txt", "w") as f:
    f.write(f"Chi-squared: {chi2:.2f} for {dof} degrees of freedom\n")
    f.write(f"Reduced chi-squared: {reduced_chi2:.2f}\n")
    
    if reduced_chi2 < 0.8:
        f.write("\nThe reduced chi-squared < 1 suggests that our error bars might be overestimated or the model has too many free parameters for the available data.\n")
    elif reduced_chi2 < 1.2:
        f.write("\nThe reduced chi-squared ≈ 1 indicates a good fit of the model to the data.\n")
    else:
        f.write("\nThe reduced chi-squared > 1 suggests that either the error bars are underestimated or the model doesn't fully capture all features in the data.\n")
        
    f.write("\nParameter values for best-fit model:\n")
    for param, value in median_params.items():
        f.write(f"  {param}: {value:.6g}\n")

print("Saved goodness-of-fit information")
print("Model fits and diagnostics completed successfully")
