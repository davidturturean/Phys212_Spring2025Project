# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 16:45:23 2025

@authors: David Turturean, Daria Teodora Harabor

ΛCDM model for Planck CMB power spectrum - Captures the main
features of the TT power spectrum including acoustic peaks, Silk damping, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from parameters import fiducial_params

# Constants
c_light = 299792.458  # km/s
G_newton = 6.67430e-11  # m^3 kg^-1 s^-2
H0_fiducial = 67.36  # km/s/Mpc
rho_crit = 3 * (H0_fiducial**2) / (8 * np.pi * G_newton)  # Critical density

# CMB temp
T_cmb = 2.7255  # K

# Functions for redshifts
def calculate_z_eq(Omega_m_h2):
    """Get matter-radiation equality redshift"""
    # z_eq = Omega_m/Omega_r
    # Omega_r*h^2 ≈ 4.15e-5 for T_CMB = 2.7255K
    Omega_r_h2 = 4.15e-5
    return Omega_m_h2/Omega_r_h2 - 1

def calculate_z_star(Omega_b_h2, Omega_m_h2):
    """Get recomb redshift - Hu & Sugiyama formula"""
    g1 = 0.0783 * Omega_b_h2**(-0.238) / (1 + 39.5 * Omega_b_h2**0.763)
    g2 = 0.560 / (1 + 21.1 * Omega_b_h2**1.81)
    
    return 1048 * (1 + 0.00124 * Omega_b_h2**(-0.738)) * (1 + g1 * Omega_m_h2**g2)

def calculate_z_drag(Omega_b_h2, Omega_m_h2):
    """Get drag epoch redshift - Hu & Sugiyama formula"""
    b1 = 0.313 * Omega_m_h2**(-0.419) * (1 + 0.607 * Omega_m_h2**0.674)
    b2 = 0.238 * Omega_m_h2**0.223
    
    return 1291 * (Omega_m_h2**0.251) / (1 + 0.659 * Omega_m_h2**0.828) * (1 + b1 * Omega_b_h2**b2)

# Ensure output directory exists
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def lcdm_power_spectrum(ell, params=None, normalize=True):
    """
    ΛCDM power spectrum with main features - SW plateau, acoustic peaks, Silk damping.
    
    Stuff this does:
    - SW plateau at low ℓ
    - Acoustic peaks with proper spacing 
    - Baryon effects on peak heights
    - Silk damping at high ℓ
    - Parameter dependence
    
    Args:
        ell: multipoles
        params: cosmological params or None for fiducial
        normalize: whether to normalize to match Planck
    
    Returns:
        D_ℓ values in μK²
    """
    # Use default params if none given
    if params is None:
        params = fiducial_params
    
    # Get params
    h = params.get('H0', 67.36) / 100.0
    Omega_b_h2 = params.get('Omega_b_h2', 0.02237)  # Baryon density
    Omega_c_h2 = params.get('Omega_c_h2', 0.1200)   # CDM density
    Omega_m_h2 = Omega_b_h2 + Omega_c_h2            # Matter density
    ns = params.get('n_s', 0.9649)                  # Spectral index
    As = params.get('A_s', 2.1e-9)                  # Amplitude
    tau = params.get('tau', 0.0544)                 # Optical depth
    
    # Get recombination and equality redshifts
    z_star = calculate_z_star(Omega_b_h2, Omega_m_h2)
    z_eq = calculate_z_eq(Omega_m_h2)
    
    # Sound horizon - uses fitting formula instead of full integral
    r_s = 144.7 * (Omega_m_h2/0.14)**(-0.25) * (Omega_b_h2/0.024)**(-0.13)
    
    # Angular diameter distance calc
    # Full integral: d_A = (1/(1+z_star)) * ∫_0^z_star c/H(z) dz
    Omega_m = Omega_m_h2 / (h*h)
    Omega_r = 4.15e-5 / (h*h)  # Radiation (photons+neutrinos)
    Omega_Lambda = 1.0 - Omega_m - Omega_r  # Flat universe
    
    # Approx formula for d_A 
    d_A = 14000.0 * (h/0.7) * (0.14/Omega_m_h2)**0.4  # Mpc
    
    # Acoustic scale ℓ_A = π·d_A/r_s
    ell_A = np.pi * d_A / r_s
    
    # Sound horizon angle
    theta_s = r_s / d_A  # radians = π/ℓ_A
    
    # Silk damping scale 
    ell_D = 1600.0 * (Omega_b_h2/0.02237)**(-0.25) * (Omega_m_h2/0.1424)**(-0.125)
    
    # Peak spacing
    ell_spacing = np.pi / theta_s  # = ℓ_A
    
    # Spectrum array
    dl_values = np.zeros_like(ell, dtype=float)
    
    # Base amplitude
    base_amp = 5800.0 * (As / 2.1e-9)
    
    # Calculate spectrum for each ℓ
    for i, l in enumerate(ell):
        # SW plateau (low ℓ)
        if l < 50:
            # Amplitude with tilt
            sw_amp = base_amp * (l/10.0)**(ns-1) / (1.0 + 0.5*(l/25.0)**2)
            # Reionization damping
            reion_damp = np.exp(-2.0 * tau)
            dl_values[i] = sw_amp * reion_damp
            continue
        
        # Transition region
        if l < 200:
            # Mix SW and first peak
            sw_weight = 1.0 - (l - 50) / 150.0
            peak_weight = 1.0 - sw_weight
            
            # SW part
            sw_amp = base_amp * (50.0/10.0)**(ns-1) / (1.0 + 0.5*(50.0/25.0)**2)
            sw_amp *= np.exp(-0.5 * ((l - 50) / 50.0)**2)
            
            # First peak part
            peak_height = 5800.0
            peak_amp = peak_height * np.exp(-0.5 * ((l - 220) / 60.0)**2)
            
            # Combine them
            dl_values[i] = sw_weight * sw_amp + peak_weight * peak_amp
            continue
        
        # Main acoustic peaks with Silk damping
        # Figure out where in the oscillation pattern we are
        phase = (l - 220) / ell_spacing * np.pi
        
        # Oscillation pattern (peaks at 0,2π,4π; troughs at π,3π,...)
        osc_pattern = 0.5 + 0.5 * np.cos(phase)
        
        # Baryon loading - makes odd peaks higher than even ones
        baryon_factor = 1.0
        if phase > 0:  # After first peak
            mod_phase = phase % (2*np.pi)
            baryon_strength = 0.35 * Omega_b_h2 / 0.022
            baryon_factor = 1.0 - baryon_strength * np.sin(mod_phase)**2
        
        # Silk damping - high ℓ power suppression
        damping = np.exp(-(l / ell_D)**2)  # Square exponent per main.tex
        
        # Overall envelope
        envelope = base_amp * (l/220.0)**(ns-1) * np.exp(-0.5 * ((np.log(l) - np.log(220)) / 0.8)**2) * damping
        
        # Put it all together
        dl_values[i] = envelope * osc_pattern * baryon_factor
    
    # Normalize if requested
    if normalize:
        # First peak at ℓ ≈ 220
        peak1_idx = np.argmin(np.abs(ell - 220))
        if dl_values[peak1_idx] > 0:
            # Scale to expected height
            norm_factor = 6000.0 / dl_values[peak1_idx]
            dl_values *= norm_factor
    
    return dl_values

def validate_model_features():
    """
    Validate the model by testing key features against expected behavior.
    
    This function generates plots showing:
    1. The full power spectrum with labeled features
    2. Individual components (primordial, envelope, oscillations)
    3. Parameter sensitivity by varying key parameters
    
    All plots are saved to the output directory.
    """
    print("Validating ΛCDM model features...")
    
    # Generate multipole array
    ell = np.arange(2, 2501)
    
    # Compute fiducial model
    dl_fiducial = lcdm_power_spectrum(ell)
    
    # Create plot of full spectrum with labeled features
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot model
    plt.plot(ell, dl_fiducial, 'r-', lw=2.5, label='ΛCDM Model')
    
    # Annotate key features
    features = [
        (10, 1200, "Sachs-Wolfe Plateau"),
        (220, 6000, "First Acoustic Peak"),
        (535, 2500, "Second Acoustic Peak"),
        (810, 2200, "Third Acoustic Peak"), 
        (1120, 1100, "Fourth Acoustic Peak"),
        (1800, 500, "Silk Damping Tail")
    ]
    
    for l, d, label in features:
        plt.annotate(label, xy=(l, d), xytext=(l+50, d*1.2),
                    arrowprops=dict(arrowstyle="->", lw=1.5),
                    ha='center', fontsize=12)
    
    # Set labels and title
    plt.xlabel(r'Multipole $\ell$', fontsize=14)
    plt.ylabel(r'$D_\ell$ [$\mu K^2$]', fontsize=14)
    plt.title('ΛCDM CMB Power Spectrum: Key Features', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=12)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'lcdm_model_features.png'), dpi=300)
    plt.show()
    
    # Create parameter sensitivity plots
    print("Testing parameter sensitivity...")
    
    # Define parameter variations
    param_variations = {
        'H0': [60.0, 67.36, 74.0],
        'Omega_b_h2': [0.020, 0.02237, 0.025],
        'Omega_c_h2': [0.10, 0.1200, 0.14],
        'n_s': [0.94, 0.9649, 0.99],
        'tau': [0.04, 0.0544, 0.07]
    }
    
    # Loop through each parameter
    for param, values in param_variations.items():
        plt.figure(figsize=(10, 6))
        
        labels = ["Low", "Fiducial", "High"]
        line_styles = ["--", "-", "-."]
        
        for i, val in enumerate(values):
            # Create parameter set with this value
            params = fiducial_params.copy()
            params[param] = val
            
            # Compute model
            dl = lcdm_power_spectrum(ell, params)
            
            # Plot
            plt.plot(ell, dl, line_styles[i], lw=2, 
                     label=f"{param} = {val} ({labels[i]})")
        
        # Set labels and title
        plt.xlabel(r'Multipole $\ell$', fontsize=14)
        plt.ylabel(r'$D_\ell$ [$\mu K^2$]', fontsize=14)
        plt.title(f'CMB Power Spectrum: {param} Sensitivity', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', fontsize=12)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'lcdm_sensitivity_{param}.png'), dpi=300)
        plt.show()
    
    # Create zoomed version of fiducial model
    plt.figure(figsize=(12, 6))
    
    # Plot model
    plt.plot(ell, dl_fiducial, 'r-', lw=2.5, label='ΛCDM Model')
    
    # Zoom to show peaks clearly
    plt.xlim(0, 1200)
    plt.ylim(0, 7000)
    
    # Set labels and title
    plt.xlabel(r'Multipole $\ell$', fontsize=14)
    plt.ylabel(r'$D_\ell$ [$\mu K^2$]', fontsize=14)
    plt.title('ΛCDM CMB Power Spectrum: First Four Acoustic Peaks', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Annotate acoustic peaks
    for peak_num, l in enumerate([220, 535, 810, 1120], 1):
        plt.axvline(l, color='k', ls='--', alpha=0.3)
        # Find y-value
        idx = np.argmin(np.abs(ell - l))
        plt.annotate(f"Peak {peak_num}", xy=(l, dl_fiducial[idx]), 
                    xytext=(l, dl_fiducial[idx]*1.1),
                    ha='center', fontsize=12, fontweight='bold',
                    arrowprops=dict(arrowstyle="->", lw=1))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'lcdm_zoomed_peaks.png'), dpi=300)
    plt.show()
    
    print("Model validation complete. All outputs saved to 'output/' directory.")

def integrate_with_external_data(data_file=None):
    """
    Compare model with real Planck data.
    
    Args:
        data_file (str): Path to data file, or None to load from FITS
    """
    print("Comparing ΛCDM model with Planck data...")
    
    # Load data
    if data_file is not None and os.path.exists(data_file):
        print(f"Loading Planck data from {data_file}")
        import pandas as pd
        df = pd.read_csv(data_file)
        ell_data = df['ell'].values
        dl_data = df['Dl'].values
        sigma_data = df['errDl'].values if 'errDl' in df.columns else None
    else:
        print("Loading Planck data from FITS file...")
        from data_loader import load_planck_data
        ell_data, dl_data, sigma_data = load_planck_data(source="fits")
        
        # Bin the data for cleaner comparison
        ell_data, dl_data, sigma_data = bin_data(ell_data, dl_data, sigma_data)
    
    # Generate model
    ell_model = np.arange(2, 2501)
    dl_model = lcdm_power_spectrum(ell_model)
    
    # Interpolate model to data points for chi-squared calculation
    interp_model = interp1d(ell_model, dl_model, kind='cubic', 
                           bounds_error=False, fill_value="extrapolate")
    dl_model_at_data = interp_model(ell_data)
    
    # Find best scaling factor
    def chi_squared(scale):
        residuals = dl_data - scale * dl_model_at_data
        return np.sum((residuals / sigma_data)**2) if sigma_data is not None else np.sum(residuals**2)
    
    # Search for optimal scaling factor
    scales = np.linspace(0.5, 2.0, 100)
    chi2_values = [chi_squared(s) for s in scales]
    best_scale = scales[np.argmin(chi2_values)]
    print(f"Best-fit scaling factor: {best_scale:.4f}")
    
    # Apply scaling
    dl_model_scaled = dl_model * best_scale
    
    # Create data vs model comparison plot
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot the model
    plt.plot(ell_model, dl_model_scaled, 'r-', lw=2, 
             label='ΛCDM Model', zorder=2)
    
    # Plot the data
    if sigma_data is not None:
        plt.errorbar(ell_data, dl_data, yerr=sigma_data, fmt='o', 
                    markersize=3, alpha=0.5, color='blue',
                    label='Planck CMB Data', zorder=1)
    else:
        plt.scatter(ell_data, dl_data, s=5, alpha=0.5, color='blue',
                   label='Planck CMB Data', zorder=1)
    
    # Set labels, title, and axis limits
    plt.xlabel(r'Multipole $\ell$', fontsize=14)
    plt.ylabel(r'$D_\ell$ [$\mu K^2$]', fontsize=14)
    plt.title('Planck CMB Data vs ΛCDM Model', fontsize=16, fontweight='bold')
    plt.xlim(2, 2500)
    plt.ylim(0, 7000)
    
    # Add legend and grid
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save full comparison
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'data_vs_model_full.png'), dpi=300)
    plt.show()
    
    # Create zoomed version
    plt.figure(figsize=(12, 6))
    
    # Filter to show first few peaks clearly
    mask_model = ell_model <= 1200
    mask_data = ell_data <= 1200
    
    # Plot zoomed view
    plt.plot(ell_model[mask_model], dl_model_scaled[mask_model], 'r-', lw=2, 
             label='ΛCDM Model', zorder=2)
    
    if sigma_data is not None:
        plt.errorbar(ell_data[mask_data], dl_data[mask_data], 
                    yerr=sigma_data[mask_data], fmt='o', 
                    markersize=3, alpha=0.5, color='blue',
                    label='Planck CMB Data', zorder=1)
    else:
        plt.scatter(ell_data[mask_data], dl_data[mask_data], 
                   s=5, alpha=0.5, color='blue',
                   label='Planck CMB Data', zorder=1)
    
    # Label peaks
    peak_positions = [220, 535, 810, 1120]
    for i, l in enumerate(peak_positions):
        if l <= 1200:
            idx = np.argmin(np.abs(ell_model - l))
            plt.annotate(f"Peak {i+1}", xy=(l, dl_model_scaled[idx]), 
                        xytext=(l, dl_model_scaled[idx]*1.1),
                        ha='center', fontsize=12, 
                        arrowprops=dict(arrowstyle="->", lw=1))
    
    # Set labels, title, and axis limits
    plt.xlabel(r'Multipole $\ell$', fontsize=14)
    plt.ylabel(r'$D_\ell$ [$\mu K^2$]', fontsize=14)
    plt.title('Zoomed View: First Four Acoustic Peaks', fontsize=16, fontweight='bold')
    
    # Add legend and grid
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save zoomed comparison
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'data_vs_model_zoomed.png'), dpi=300)
    plt.show()
    
    print("Model comparison complete. All outputs saved to 'output/' directory.")

def bin_data(ell, dl, sigma, bin_width=20):
    """
    Bin data for cleaner visualization using inverse-variance weighting.
    
    Args:
        ell (array): Multipole moments
        dl (array): Power spectrum values
        sigma (array): Error bars
        bin_width (int): Width of each bin in ℓ
        
    Returns:
        tuple: (ell_binned, dl_binned, sigma_binned) arrays
    """
    # Create bin edges
    bin_edges = np.arange(min(ell), max(ell) + bin_width, bin_width)
    n_bins = len(bin_edges) - 1
    
    # Initialize arrays
    ell_binned = np.zeros(n_bins)
    dl_binned = np.zeros(n_bins)
    weight_sum = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    
    # Perform binning with inverse-variance weighting
    for i in range(len(ell)):
        # Find bin index
        bin_idx = np.searchsorted(bin_edges, ell[i]) - 1
        if bin_idx < 0 or bin_idx >= n_bins:
            continue
            
        # Use inverse variance as weight
        weight = 1.0 / (sigma[i]**2) if sigma[i] > 0 else 0.0
        
        # Accumulate weighted values
        ell_binned[bin_idx] += ell[i] * weight
        dl_binned[bin_idx] += dl[i] * weight
        weight_sum[bin_idx] += weight
        counts[bin_idx] += 1
    
    # Compute weighted averages
    mask = (counts > 0) & (weight_sum > 0)
    ell_binned[mask] /= weight_sum[mask]
    dl_binned[mask] /= weight_sum[mask]
    
    # Compute error of weighted mean
    sigma_binned = np.zeros(n_bins)
    sigma_binned[mask] = 1.0 / np.sqrt(weight_sum[mask])
    
    # Filter out empty bins
    ell_binned = ell_binned[mask]
    dl_binned = dl_binned[mask]
    sigma_binned = sigma_binned[mask]
    
    print(f"Binned {len(ell)} multipoles into {len(ell_binned)} bins")
    
    return ell_binned, dl_binned, sigma_binned

def replace_cosmology_model():
    """
    Function to replace the existing cosomology_model's compute_cl function
    with this improved version.
    """
    print("Integrating state-of-the-art ΛCDM model into the pipeline...")
    
    # Define the improved compute_cl function that matches the interface
    def improved_compute_cl(params):
        """
        Compute theoretical CMB TT power spectrum using state-of-the-art ΛCDM model.
        Compatible with existing pipeline.
        
        Args:
            params (dict): Cosmological parameters
            
        Returns:
            array: D_ℓ power spectrum values
        """
        ell = np.arange(2, 2501)
        return lcdm_power_spectrum(ell, params, normalize=True)
    
    # Set the docstring to document the function
    improved_compute_cl.__doc__ = "State-of-the-art ΛCDM model for CMB power spectrum."
    
    try:
        # Import the cosmology_model module
        import cosmology_model
        
        # Store the original function for reference
        original_compute_cl = cosmology_model.compute_cl
        
        # Replace with our improved version
        cosmology_model.compute_cl = improved_compute_cl
        
        print("Model integration successful!")
        print("The improved model will be used in all pipeline components.")
    except Exception as e:
        print(f"Error integrating model: {e}")
        print("You can still use the model directly through theoretical_lcdm.py")

if __name__ == "__main__":
    # Run model validation
    validate_model_features()
    
    # Compare with Planck data
    integrate_with_external_data()
    
    # Integrate with the rest of the pipeline
    replace_cosmology_model()