# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 17:42:15 2025

@authors: David Turturean, Daria Teodora Harabor
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from parameters import fiducial_params
from data_loader import load_planck_data, plot_power_spectrum

def main():
    """
    Final analysis script for the PHYS 212 project.
    This script properly analyzes the Planck FITS file and
    demonstrates the ΛCDM model fitting.
    """
    print("ΛCDM Cosmological Parameter Inference - Final Analysis")
    print("====================================================")
    print("PHYS 212 Spring 2025 Project")
    print("Daria Harabor & David Turturean")
    print("\n")
    
    # Load the Planck data from the FITS file
    print("Step 1: Loading Planck CMB data from FITS map...")
    ell, dl, sigma = load_planck_data(source="fits")
    
    # Display data statistics
    print(f"Data summary:")
    print(f"  Multipoles: {len(ell)} from ℓ={min(ell)} to ℓ={max(ell)}")
    print(f"  Mean D_ell: {np.mean(dl):.2f} μK²")
    print(f"  Max D_ell: {np.max(dl):.2f} μK²")
    
    # Plot the power spectrum
    print("\nStep 2: Plotting Planck TT power spectrum...")
    plot_power_spectrum(ell, dl, sigma, title="Planck CMB TT Power Spectrum", 
                       save_path="planck_tt_spectrum.png")
    
    # Bin the data for faster analysis
    print("\nStep 3: Binning the power spectrum for faster analysis...")
    ell_binned, dl_binned, sigma_binned = bin_power_spectrum(ell, dl, sigma, bin_width=30)
    plot_power_spectrum(ell_binned, dl_binned, sigma_binned, 
                       title="Binned Planck CMB TT Power Spectrum",
                       save_path="planck_tt_spectrum_binned.png")
    
    # Save the binned data for future use
    save_spectrum(ell_binned, dl_binned, sigma_binned, 
                 filename="planck_tt_spectrum_binned.csv")
    
    # Test the model against the data
    print("\nStep 4: Testing theoretical model against data...")
    test_model(ell_binned, dl_binned, sigma_binned)
    
    # Apply scaling to model to make it fit better
    print("\nStep 5: Applying scaling to model for better fit...")
    fit_scaled_model(ell_binned, dl_binned, sigma_binned)
    
    print("\nAnalysis complete.")

def bin_power_spectrum(ell, dl, sigma, bin_width=30):
    """
    Bin the power spectrum for faster analysis.
    
    Args:
        ell (array): Multipole values
        dl (array): Power spectrum values
        sigma (array): Error bars
        bin_width (int): Width of each bin
        
    Returns:
        tuple: (ell_binned, dl_binned, sigma_binned) arrays
    """
    # Create bin edges
    bin_edges = np.arange(min(ell), max(ell) + bin_width, bin_width)
    n_bins = len(bin_edges) - 1
    
    # Initialize arrays for binned values
    ell_binned = np.zeros(n_bins)
    dl_binned = np.zeros(n_bins)
    sigma_binned = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    
    # Perform binning
    for i in range(len(ell)):
        bin_idx = int((ell[i] - min(ell)) / bin_width)
        if bin_idx >= n_bins:
            continue
            
        # Add to bin
        ell_binned[bin_idx] += ell[i]
        dl_binned[bin_idx] += dl[i]
        sigma_binned[bin_idx] += sigma[i]**2  # Add variances
        counts[bin_idx] += 1
    
    # Compute averages for each bin
    mask = counts > 0
    ell_binned[mask] /= counts[mask]
    dl_binned[mask] /= counts[mask]
    sigma_binned[mask] = np.sqrt(sigma_binned[mask]) / counts[mask]  # Standard error
    
    # Filter out empty bins
    ell_binned = ell_binned[mask]
    dl_binned = dl_binned[mask]
    sigma_binned = sigma_binned[mask]
    
    print(f"Binned {len(ell)} multipoles into {len(ell_binned)} bins")
    
    return ell_binned, dl_binned, sigma_binned

def save_spectrum(ell, dl, sigma, filename="planck_tt_spectrum.csv"):
    """
    Save the power spectrum to a CSV file.
    
    Args:
        ell (array): Multipole values
        dl (array): Power spectrum values
        sigma (array): Error bars
        filename (str): Output filename
    """
    import pandas as pd
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame({
        "ell": ell,
        "Dl": dl,
        "errDl": sigma
    })
    
    df.to_csv(filename, index=False)
    print(f"Saved power spectrum to {filename}")

def test_model(ell_obs, dl_obs, sigma_obs):
    """
    Test the cosmological model against the observed data.
    
    Args:
        ell_obs (array): Observed multipole values
        dl_obs (array): Observed power spectrum values
        sigma_obs (array): Observed error bars
    """
    # First try the enhanced model
    try:
        from cosmology_model import compute_cl
        print("Using enhanced cosmological model")
        
        # Generate model spectrum
        ell_model = np.arange(2, 2501)
        dl_model = compute_cl(fiducial_params)
    except ImportError:
        # Fall back to simple model
        from CAMB import model_Dl_TT
        print("Using simplified cosmological model")
        
        # Generate model spectrum
        ell_model = np.arange(2, 2501)
        dl_model = model_Dl_TT(ell_model, fiducial_params['A_s'], fiducial_params['n_s'])
        
        # Apply reionization effect
        if 'tau' in fiducial_params:
            tau = fiducial_params['tau']
            dl_model *= np.exp(-2 * tau * (ell_model < 30))
    
    # Interpolate model to match observed multipoles
    from scipy.interpolate import interp1d
    interp_model = interp1d(ell_model, dl_model, kind='linear', bounds_error=False, fill_value=0)
    dl_model_interp = interp_model(ell_obs)
    
    # Find best scaling factor for model to match data
    def chi_squared(scale):
        residuals = dl_obs - scale * dl_model_interp
        return np.sum((residuals / sigma_obs)**2)
    
    # Search for optimal scaling factor
    scales = np.linspace(0.5, 2.0, 100)
    chi2_values = [chi_squared(s) for s in scales]
    best_scale = scales[np.argmin(chi2_values)]
    print(f"Best-fit scaling factor: {best_scale:.4f}")
    
    # Apply best scaling factor
    dl_model = dl_model * best_scale
    dl_model_interp = dl_model_interp * best_scale
    
    # Compute chi-squared with scaled model
    residuals = dl_obs - dl_model_interp
    chi2 = np.sum((residuals / sigma_obs)**2)
    dof = len(ell_obs) - len(fiducial_params) - 1  # -1 for scaling factor
    
    print(f"Model test results:")
    print(f"  χ² = {chi2:.2f} for {dof} degrees of freedom")
    print(f"  Reduced χ² = {chi2/dof:.2f}")
    
    # Save the results to output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot model vs data
    plt.figure(figsize=(10, 8))
    
    # Top panel: spectrum
    plt.subplot(2, 1, 1)
    plt.errorbar(ell_obs, dl_obs, yerr=sigma_obs, fmt='o', markersize=3,
                alpha=0.6, label="Planck data")
    plt.plot(ell_model, dl_model, 'r-', lw=1.5, 
             label=f"ΛCDM model (scaled ×{best_scale:.2f})")
    plt.xlabel(r"Multipole $\ell$")
    plt.ylabel(r"$D_\ell$ [$\mu K^2$]")
    plt.title("CMB TT Power Spectrum: Model vs Data")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Bottom panel: residuals
    plt.subplot(2, 1, 2)
    plt.errorbar(ell_obs, residuals, yerr=sigma_obs, fmt='o', markersize=3,
                alpha=0.6)
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel(r"Multipole $\ell$")
    plt.ylabel(r"Residual $D_\ell$ [$\mu K^2$]")
    plt.title(f"Residuals (χ² = {chi2:.2f})")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "model_vs_data.png")
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.show()
    
    # Create a zoomed version focusing on first few peaks
    plt.figure(figsize=(10, 6))
    
    # Filter to ell < 1000 (first few peaks)
    mask_obs = ell_obs < 1000
    mask_model = ell_model < 1000
    
    plt.errorbar(ell_obs[mask_obs], dl_obs[mask_obs], 
                yerr=sigma_obs[mask_obs], fmt='o', markersize=3,
                alpha=0.6, label="Planck data")
    plt.plot(ell_model[mask_model], dl_model[mask_model], 'r-', lw=1.5, 
             label=f"ΛCDM model (scaled ×{best_scale:.2f})")
    
    plt.xlabel(r"Multipole $\ell$")
    plt.ylabel(r"$D_\ell$ [$\mu K^2$]")
    plt.title("CMB TT Power Spectrum: Zoomed View of Acoustic Peaks")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "model_vs_data_zoomed.png")
    plt.savefig(save_path)
    print(f"Saved zoomed plot to {save_path}")
    plt.show()
    
    return best_scale

# The fit_scaled_model function is redundant now that we've integrated
# the scaling into the test_model function. We'll keep this as a placeholder
# to maintain compatibility, but it just calls test_model.
def fit_scaled_model(ell_obs, dl_obs, sigma_obs):
    """
    Fit a scaled version of the model to the data.
    
    This function is now redundant as scaling is done in test_model.
    Kept for compatibility.
    
    Args:
        ell_obs (array): Observed multipole values
        dl_obs (array): Observed power spectrum values
        sigma_obs (array): Observed error bars
    """
    print("Note: Model scaling is now integrated into the main test_model function.")
    return

if __name__ == "__main__":
    main()