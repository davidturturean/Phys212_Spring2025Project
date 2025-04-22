# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 17:30:45 2025

@authors: David Turturean, Daria Teodora Harabor

Final visualization for the PHYS 212 project, showing professional-grade figures
of the Planck CMB power spectrum and ΛCDM model comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from data_loader import load_planck_data
from parameters import fiducial_params
from theoretical_lcdm import lcdm_power_spectrum, bin_data
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks as scipy_find_peaks
import matplotlib as mpl

# Set up professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['mathtext.fontset'] = 'dejavuserif'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.titlesize'] = 16

# Ensure output directory exists
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    """
    Create all final visualizations for the project.
    
    This script generates:
    1. CMB power spectrum with labeled acoustic peaks
    2. Data vs model comparison
    3. MCMC parameter visualization
    4. Scientific illustration of physical implications
    """
    print("Creating final visualizations for PHYS 212 ΛCDM Project")
    print("======================================================")
    
    # Load and prepare data
    print("\nStep 1: Loading Planck data from FITS file...")
    ell_data, dl_data, sigma_data = load_planck_data(source="fits")
    
    # Bin the data for cleaner visualization
    print("\nStep 2: Binning data for visualization...")
    ell_binned, dl_binned, sigma_binned = bin_data(ell_data, dl_data, sigma_data)
    
    # Generate state-of-the-art theoretical model
    print("\nStep 3: Generating theoretical ΛCDM model...")
    ell_model = np.arange(2, 2501)
    dl_model = lcdm_power_spectrum(ell_model, fiducial_params)
    
    # Create power spectrum visualization
    print("\nStep 4: Creating CMB power spectrum visualization...")
    plot_power_spectrum(ell_binned, dl_binned, sigma_binned)
    
    # Create data vs model comparison
    print("\nStep 5: Creating data vs model comparison...")
    create_data_model_comparison(ell_binned, dl_binned, sigma_binned, ell_model, dl_model)
    
    # Create scientific visualization
    print("\nStep 6: Creating scientific visualization...")
    create_scientific_visualization(ell_binned, dl_binned, sigma_binned, ell_model, dl_model)
    
    print("\nAll visualizations complete. Files saved to 'output/' directory.")

def find_peaks(ell, dl, min_prominence=100, max_peaks=7, smoothing=1.5):
    """
    Find acoustic peaks in the power spectrum.
    
    Args:
        ell (array): Multipole moments
        dl (array): Power spectrum values
        min_prominence (float): Minimum peak prominence
        max_peaks (int): Maximum number of peaks to find
        smoothing (float): Gaussian smoothing sigma
        
    Returns:
        tuple: (peaks, dl_smooth) - peaks is list of (ell, dl) tuples for each peak
    """
    # Apply smoothing for better peak detection
    dl_smooth = gaussian_filter1d(dl, sigma=smoothing)
    
    # Find peaks using scipy
    peak_indices, properties = scipy_find_peaks(dl_smooth, prominence=min_prominence)
    
    # Sort peaks by prominence and limit to max_peaks
    if len(peak_indices) > max_peaks:
        sorted_indices = np.argsort(properties['prominences'])[::-1]  # Descending order
        peak_indices = peak_indices[sorted_indices[:max_peaks]]
        # Re-sort by position
        peak_indices = np.sort(peak_indices)
    
    # Convert to (ell, dl) pairs
    peaks = [(ell[i], dl_smooth[i]) for i in peak_indices]
    
    return peaks, dl_smooth

def plot_power_spectrum(ell, dl, sigma):
    """
    Create high-quality plot of the CMB power spectrum with labeled peaks.
    
    Args:
        ell (array): Multipole moments
        dl (array): Power spectrum values
        sigma (array): Error bars
    """
    # Setup the figure
    plt.figure(figsize=(12, 8))
    
    # Find peaks for labeling
    peaks, dl_smooth = find_peaks(ell, dl, min_prominence=50)
    
    # Plot data with error bars
    plt.errorbar(ell, dl, yerr=sigma, fmt='o', markersize=3, alpha=0.5, color='#3182bd',
                label='Planck CMB data', zorder=1)
    
    # Plot smoothed curve to highlight the peaks
    plt.plot(ell, dl_smooth, '-', color='#e6550d', lw=2.5, label='Smoothed data', zorder=2)
    
    # Label acoustic peaks
    for i, (l, d) in enumerate(peaks):
        plt.annotate(f"Peak {i+1}", xy=(l, d), xytext=(l+30, d*1.1),
                    arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
                    ha='center', fontsize=12, fontweight='bold', zorder=3)
        
        # Draw vertical line at each peak
        plt.axvline(l, color='k', linestyle='--', alpha=0.3, zorder=0)
    
    # Set axis labels and title
    plt.xlabel(r'Multipole $\ell$')
    plt.ylabel(r'$D_\ell$ [$\mu K^2$]')
    plt.title('Planck CMB TT Power Spectrum', fontweight='bold')
    
    # Add explanation text
    plt.figtext(0.5, 0.02, 
               "The acoustic peaks in the CMB power spectrum correspond to sound waves in the early universe.\n"
               "These features provide crucial constraints on cosmological parameters such as the density of dark matter and dark energy.",
               ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='ivory', alpha=0.9))
    
    # Set axis limits and legend
    plt.xlim(2, 2500)
    plt.ylim(0, 7000)
    plt.legend(loc='upper right')
    
    # Save the figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(os.path.join(OUTPUT_DIR, 'cmb_power_spectrum_final.png'), dpi=300)
    plt.close()
    
    # Create zoomed view focusing on the first few peaks
    plt.figure(figsize=(12, 6))
    
    # Define zoom range
    min_ell = 50
    max_ell = 1000
    
    # Filter data within range
    mask = (ell >= min_ell) & (ell <= max_ell)
    
    # Plot data with error bars
    plt.errorbar(ell[mask], dl[mask], yerr=sigma[mask], fmt='o', markersize=3,
                alpha=0.5, color='#3182bd', label='Planck CMB data', zorder=1)
    
    # Plot smoothed curve
    plt.plot(ell[mask], dl_smooth[mask], '-', color='#e6550d', lw=2.5,
             label='Smoothed data', zorder=2)
    
    # Label peaks in this range
    for i, (l, d) in enumerate(peaks):
        if l >= min_ell and l <= max_ell:
            plt.annotate(f"Peak {i+1}", xy=(l, d), xytext=(l, d*1.1),
                        arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
                        ha='center', fontsize=12, fontweight='bold', zorder=3)
            
            # Draw vertical line at each peak
            plt.axvline(l, color='k', linestyle='--', alpha=0.3, zorder=0)
    
    # Set axis labels, title, and limits
    plt.xlabel(r'Multipole $\ell$')
    plt.ylabel(r'$D_\ell$ [$\mu K^2$]')
    plt.title('Zoomed View of Acoustic Peaks', fontweight='bold')
    plt.legend(loc='upper right')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cmb_power_spectrum_zoomed.png'), dpi=300)
    plt.close()

def create_data_model_comparison(ell_data, dl_data, sigma_data, ell_model, dl_model):
    """
    Create visualization comparing the data with the theoretical model.
    
    Args:
        ell_data (array): Data multipole moments
        dl_data (array): Data power spectrum values
        sigma_data (array): Data error bars
        ell_model (array): Model multipole moments
        dl_model (array): Model power spectrum values
    """
    # Interpolate model to data points for chi-squared calculation
    interp_model = interp1d(ell_model, dl_model, kind='cubic', 
                           bounds_error=False, fill_value="extrapolate")
    dl_model_at_data = interp_model(ell_data)
    
    # Find best scaling factor to match model amplitude to data
    def chi_squared(scale):
        residuals = dl_data - scale * dl_model_at_data
        return np.sum((residuals / sigma_data)**2)
    
    scales = np.linspace(0.5, 2.0, 100)
    chi2_values = [chi_squared(s) for s in scales]
    best_scale = scales[np.argmin(chi2_values)]
    print(f"Best-fit scaling factor: {best_scale:.4f}")
    
    # Apply scaling to model
    dl_model_scaled = best_scale * dl_model
    dl_model_at_data_scaled = best_scale * dl_model_at_data
    
    # Calculate residuals and chi-squared
    residuals = dl_data - dl_model_at_data_scaled
    chi2 = np.sum((residuals / sigma_data)**2)
    dof = len(ell_data) - len(fiducial_params)
    
    print(f"χ² = {chi2:.2f} for {dof} degrees of freedom")
    print(f"Reduced χ² = {chi2/dof:.2f}")
    
    # Create comparison plot
    plt.figure(figsize=(12, 9))
    
    # Top panel: Model vs data
    plt.subplot(2, 1, 1)
    plt.errorbar(ell_data, dl_data, yerr=sigma_data, fmt='o', markersize=3, 
                alpha=0.5, color='#3182bd', label='Planck data', zorder=1)
    plt.plot(ell_model, dl_model_scaled, '-', lw=2.5, color='#e6550d', 
            label=r'ΛCDM model', zorder=2)
    
    # Find and label acoustic peaks in data
    peaks, _ = find_peaks(ell_data, dl_data, min_prominence=50)
    for i, (l, d) in enumerate(peaks[:4]):  # Label first 4 peaks
        plt.annotate(f"Peak {i+1}", xy=(l, d), xytext=(l+30, d*1.1),
                    arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
                    ha='center', fontsize=10, fontweight='bold', zorder=3)
    
    # Set axis labels, title, limits, and legend
    plt.xlabel(r'Multipole $\ell$')
    plt.ylabel(r'$D_\ell$ [$\mu K^2$]')
    plt.title('CMB TT Power Spectrum: Data vs. ΛCDM Model', fontweight='bold')
    plt.xlim(2, 2500)
    plt.ylim(0, 7000)
    plt.legend(loc='upper right')
    
    # Bottom panel: Residuals
    plt.subplot(2, 1, 2)
    plt.errorbar(ell_data, residuals, yerr=sigma_data, fmt='o', markersize=3,
                alpha=0.5, color='#3182bd', zorder=1)
    plt.axhline(0, color='black', linestyle='--', zorder=2)
    
    # Set axis labels and title
    plt.xlabel(r'Multipole $\ell$')
    plt.ylabel('Residuals')
    plt.title(fr'Residuals (Data $-$ Model): $\chi^2 = {chi2:.0f}$', fontweight='bold')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_vs_data.png'), dpi=300)
    plt.close()
    
    # Create zoomed plot for first few peaks
    plt.figure(figsize=(12, 8))
    
    # Define zoom range
    max_ell = 1200
    
    # Filter data and model to zoom range
    mask_data = ell_data <= max_ell
    mask_model = ell_model <= max_ell
    
    # Plot data and model
    plt.subplot(2, 1, 1)
    plt.errorbar(ell_data[mask_data], dl_data[mask_data], 
                yerr=sigma_data[mask_data], fmt='o', markersize=3,
                alpha=0.5, color='#3182bd', label='Planck data', zorder=1)
    plt.plot(ell_model[mask_model], dl_model_scaled[mask_model], 
            '-', lw=2.5, color='#e6550d', label=r'ΛCDM model', zorder=2)
    
    # Label peaks in this range
    for i, (l, d) in enumerate(peaks):
        if l <= max_ell:
            plt.annotate(f"Peak {i+1}", xy=(l, d), xytext=(l, d*1.1),
                        arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
                        ha='center', fontsize=10, fontweight='bold', zorder=3)
    
    # Set axis labels, title, and legend
    plt.xlabel(r'Multipole $\ell$')
    plt.ylabel(r'$D_\ell$ [$\mu K^2$]')
    plt.title('Zoomed View: First Acoustic Peaks', fontweight='bold')
    plt.legend(loc='upper right')
    
    # Plot zoomed residuals
    plt.subplot(2, 1, 2)
    plt.errorbar(ell_data[mask_data], residuals[mask_data], 
                yerr=sigma_data[mask_data], fmt='o', markersize=3,
                alpha=0.5, color='#3182bd', zorder=1)
    plt.axhline(0, color='black', linestyle='--', zorder=2)
    
    # Set axis labels and title
    plt.xlabel(r'Multipole $\ell$')
    plt.ylabel('Residuals')
    plt.title('Zoomed Residuals', fontweight='bold')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_vs_data_zoomed.png'), dpi=300)
    plt.close()
    
    # Return the optimal scaling factor
    return best_scale, chi2/dof

def create_scientific_visualization(ell_data, dl_data, sigma_data, ell_model, dl_model):
    """
    Create a scientific visualization with labeled physical features.
    
    Args:
        ell_data (array): Data multipole moments
        dl_data (array): Data power spectrum values
        sigma_data (array): Data error bars
        ell_model (array): Model multipole moments
        dl_model (array): Model power spectrum values
    """
    # Setup the figure
    plt.figure(figsize=(14, 10))
    
    # Find peaks for labeling
    peaks, dl_smooth = find_peaks(ell_data, dl_data, min_prominence=50)
    
    # Interpolate model to data points
    interp_model = interp1d(ell_model, dl_model, kind='cubic', 
                           bounds_error=False, fill_value="extrapolate")
    dl_model_at_data = interp_model(ell_data)
    
    # Find best scaling factor
    def chi_squared(scale):
        residuals = dl_data - scale * dl_model_at_data
        return np.sum((residuals / sigma_data)**2)
    
    scales = np.linspace(0.5, 2.0, 100)
    chi2_values = [chi_squared(s) for s in scales]
    best_scale = scales[np.argmin(chi2_values)]
    
    # Apply scaling to model
    dl_model_scaled = best_scale * dl_model
    
    # Plot data with error bars
    plt.errorbar(ell_data, dl_data, yerr=sigma_data, fmt='o', markersize=3, alpha=0.4, 
                color='#3182bd', label='Planck data', zorder=1)
    
    # Plot smoothed curve to highlight the peaks
    plt.plot(ell_data, dl_smooth, '-', color='#e6550d', lw=2.5, 
             label='Smoothed data', zorder=2)
    
    # Plot model
    plt.plot(ell_model, dl_model_scaled, '--', lw=2, color='#31a354', 
            label=r'ΛCDM model', zorder=3, alpha=0.7)
    
    # Label acoustic peaks
    for i, (l, d) in enumerate(peaks):
        plt.annotate(f"Peak {i+1}", xy=(l, d), xytext=(l+30, d+300),
                    arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
                    ha='center', fontsize=12, fontweight='bold', zorder=4)
    
    # Add physical feature annotations
    annotations = [
        # Sachs-Wolfe plateau
        (20, 1000, "Sachs-Wolfe Plateau\n(large-scale temperature fluctuations)"),
        
        # First acoustic peak
        (220, 6000, "First Acoustic Peak\n(flat universe indicator)"),
        
        # Baryon loading
        (450, 2000, "Second-to-First Peak Ratio\n(baryon density indicator)"),
        
        # Dark matter effects
        (800, 2500, "Higher Peaks\n(dark matter density indicators)"),
        
        # Silk damping
        (2000, 1000, "Silk Damping Tail\n(photon diffusion)")
    ]
    
    # Add each annotation
    for l, d, text in annotations:
        plt.annotate(text, xy=(l, d), xytext=(l+100, d),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", 
                                   color='#756bb1', lw=2),
                    ha='left', fontsize=12, fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.5", fc='#fee0d2', ec='#756bb1', alpha=0.8),
                    zorder=5)
    
    # Set axis labels, title, and limits
    plt.xlabel(r'Multipole $\ell$')
    plt.ylabel(r'$D_\ell$ [$\mu K^2$]')
    plt.title('CMB Power Spectrum: Physical Interpretation', fontweight='bold')
    plt.xlim(2, 2500)
    plt.ylim(0, 7000)
    
    # Add explanatory text
    plt.figtext(0.5, 0.01, 
               "The CMB power spectrum encodes the physics of the early universe. The locations and amplitudes of the acoustic peaks\n"
               "constrain cosmological parameters including the curvature, baryon density, dark matter density, and dark energy.",
               ha='center', fontsize=14, bbox=dict(boxstyle='round', facecolor='ivory', alpha=0.9))
    
    # Add ΛCDM parameter explanation
    param_text = (
        "ΛCDM Parameters:\n"
        r"$H_0$: Hubble constant" + "\n"
        r"$\Omega_b h^2$: Baryon density" + "\n"
        r"$\Omega_c h^2$: Cold dark matter density" + "\n"
        r"$n_s$: Scalar spectral index" + "\n"
        r"$A_s$: Primordial amplitude" + "\n"
        r"$\tau$: Optical depth to reionization"
    )
    
    plt.annotate(param_text, xy=(0.02, 0.98), xycoords='figure fraction',
                va='top', ha='left', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.5", fc='white', ec='gray', alpha=0.8))
    
    # Add legend
    plt.legend(loc='upper right')
    
    # Save the figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(os.path.join(OUTPUT_DIR, 'cmb_spectrum_scientific.png'), dpi=300)
    plt.close()
    
    # Create parameter sensitivity plots
    create_parameter_sensitivity_plots()

def create_parameter_sensitivity_plots():
    """
    Create visualizations showing how each parameter affects the power spectrum.
    """
    print("Creating parameter sensitivity visualizations...")
    
    # Generate multipole array
    ell = np.arange(2, 2501)
    
    # Define parameter variations (current ±15%)
    param_variations = {
        'H0': [57.0, 67.36, 77.0],
        'Omega_b_h2': [0.019, 0.02237, 0.026],
        'Omega_c_h2': [0.102, 0.1200, 0.138],
        'n_s': [0.92, 0.9649, 1.01],
        'tau': [0.046, 0.0544, 0.063]
    }
    
    # Create directory for parameter plots
    param_dir = os.path.join(OUTPUT_DIR, "parameter_sensitivity")
    os.makedirs(param_dir, exist_ok=True)
    
    # Loop through each parameter
    for param, values in param_variations.items():
        plt.figure(figsize=(10, 6))
        
        labels = [f"{param} = {values[0]} (Low)",
                 f"{param} = {values[1]} (Fiducial)",
                 f"{param} = {values[2]} (High)"]
        
        linestyles = ["--", "-", "-."]
        colors = ["#d95f02", "#7570b3", "#1b9e77"]
        
        # Generate models for each parameter value
        for i, val in enumerate(values):
            # Create parameter set with this value
            params = fiducial_params.copy()
            params[param] = val
            
            # Compute model
            dl = lcdm_power_spectrum(ell, params, normalize=False)
            
            # Plot
            plt.plot(ell, dl, linestyles[i], lw=2, color=colors[i], label=labels[i])
        
        # Set labels and title
        plt.xlabel(r'Multipole $\ell$')
        plt.ylabel(r'$D_\ell$ [$\mu K^2$]')
        plt.title(f'Parameter Sensitivity: {param}', fontweight='bold')
        
        # Set axis limits based on parameter
        plt.xlim(2, 2500)
        if param == 'n_s':
            plt.yscale('log')
        else:
            plt.ylim(0, 7000)
            
        # Add legend
        plt.legend(loc='upper right')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(param_dir, f'sensitivity_{param}.png'), dpi=300)
        plt.close()
        
    # Create a zoomed version for first few peaks
    for param, values in param_variations.items():
        plt.figure(figsize=(10, 6))
        
        labels = [f"{param} = {values[0]} (Low)",
                 f"{param} = {values[1]} (Fiducial)",
                 f"{param} = {values[2]} (High)"]
        
        linestyles = ["--", "-", "-."]
        colors = ["#d95f02", "#7570b3", "#1b9e77"]
        
        for i, val in enumerate(values):
            # Create parameter set with this value
            params = fiducial_params.copy()
            params[param] = val
            
            # Compute model
            dl = lcdm_power_spectrum(ell, params, normalize=False)
            
            # Plot
            plt.plot(ell, dl, linestyles[i], lw=2, color=colors[i], label=labels[i])
        
        # Set labels and title
        plt.xlabel(r'Multipole $\ell$')
        plt.ylabel(r'$D_\ell$ [$\mu K^2$]')
        plt.title(f'Parameter Sensitivity: {param} (Zoomed)', fontweight='bold')
        
        # Set zoomed limits
        plt.xlim(0, 1200)
        plt.ylim(0, 7000)
            
        # Add legend
        plt.legend(loc='upper right')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(param_dir, f'sensitivity_{param}_zoomed.png'), dpi=300)
        plt.close()
    
    print(f"Parameter sensitivity plots saved to {param_dir}")

if __name__ == "__main__":
    main()