# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 17:05:23 2025

@authors: David Turturean, Daria Teodora Harabor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import healpy as hp
from scipy.interpolate import interp1d

def load_planck_data(source="fits", lmax=2500):
    """
    Load Planck CMB TT power spectrum data from FITS file.
    Always uses actual data, never simulated data.
    
    Args:
        source (str): Data source - "fits", "txt", or "csv"
        lmax (int): Maximum multipole to return
        
    Returns:
        tuple: (ell, D_ell, sigma) arrays
    """
    print(f"Loading Planck data using source: {source}")
    
    # FITS file containing Planck data
    fits_file = "COM_CMB_IQU-smica_2048_R3.00_full.fits"
    
    # CSV/TXT files might contain pre-processed data
    txt_file = "planck_tt_spectrum_from_fits.txt"
    csv_file = os.path.join("output", "cmb_power_spectrum_final.csv")
    
    # Always prioritize FITS file (original data)
    if source == "fits" or source == "auto":
        if os.path.exists(fits_file):
            return load_from_fits(fits_file, lmax)
        else:
            print(f"Warning: FITS file {fits_file} not found.")
            # Fall back to other sources if FITS not available
            if os.path.exists(csv_file):
                return load_from_csv(csv_file, lmax)
            elif os.path.exists(txt_file):
                return load_from_txt(txt_file, lmax)
            else:
                raise FileNotFoundError(f"Cannot find Planck data files. Please ensure {fits_file} exists.")
    
    # Support other formats if explicitly requested
    elif source == "txt" and os.path.exists(txt_file):
        return load_from_txt(txt_file, lmax)
    elif source == "csv" and os.path.exists(csv_file):
        return load_from_csv(csv_file, lmax)
    else:
        raise ValueError(f"Unknown or unavailable source: {source}")

def load_from_fits(fits_file, lmax=2500):
    """
    Load power spectrum directly from the Planck FITS map.
    
    Args:
        fits_file (str): Path to the FITS file
        lmax (int): Maximum multipole to return
        
    Returns:
        tuple: (ell, D_ell, sigma) arrays
    """
    print(f"Extracting power spectrum from FITS file: {fits_file}")
    
    # Read the map
    t_map = hp.read_map(fits_file, field=0)
    
    # Compute power spectrum
    cl = hp.anafast(t_map, lmax=lmax)
    
    # Convert to D_ell = ell*(ell+1)*C_ell/(2π)
    ell = np.arange(len(cl))
    dl = ell * (ell + 1) * cl / (2 * np.pi)
    
    # Calculate error bars (cosmic variance approximation)
    # For TT spectrum, delta_Cl/Cl = sqrt(2/(2l+1))
    sigma = np.zeros_like(dl)
    for l in range(2, len(ell)):
        sigma[l] = dl[l] * np.sqrt(2 / (2 * l + 1))
    
    # We're only interested in ell >= 2 (remove monopole and dipole)
    ell = ell[2:]
    dl = dl[2:]
    sigma = sigma[2:]
    
    # Scaling factor to convert to standard CMB temperature units (μK^2)
    # The map is in K_CMB units, need μK^2
    scale_factor = 1e12  # (1e6)^2 for K to μK
    dl *= scale_factor
    sigma *= scale_factor
    
    print(f"Extracted power spectrum with {len(ell)} multipoles")
    print(f"D_ell range: {np.min(dl):.2e} to {np.max(dl):.2e} μK^2")
    
    return ell, dl, sigma

def load_from_txt(txt_file, lmax=2500):
    """
    Load power spectrum from a text file.
    
    Args:
        txt_file (str): Path to the text file
        lmax (int): Maximum multipole to return
        
    Returns:
        tuple: (ell, D_ell, sigma) arrays
    """
    print(f"Loading power spectrum from text file: {txt_file}")
    
    try:
        # Load data from text file
        data = np.loadtxt(txt_file, comments='#')
        
        # Extract columns
        ell = data[:, 0].astype(int)
        dl = data[:, 1]
        sigma = data[:, 2]
        
        # Trim to lmax
        mask = ell <= lmax
        ell = ell[mask]
        dl = dl[mask]
        sigma = sigma[mask]
        
        print(f"Loaded {len(ell)} multipoles from text file")
        print(f"D_ell range: {np.min(dl):.2e} to {np.max(dl):.2e}")
        
        return ell, dl, sigma
    
    except Exception as e:
        print(f"Error loading from text file: {e}")
        raise FileNotFoundError(f"Cannot load text file {txt_file} and no fallback data available.")

def load_from_csv(csv_file, lmax=2500):
    """
    Load power spectrum from a CSV file.
    
    Args:
        csv_file (str): Path to the CSV file
        lmax (int): Maximum multipole to return
        
    Returns:
        tuple: (ell, D_ell, sigma) arrays
    """
    print(f"Loading power spectrum from CSV file: {csv_file}")
    
    try:
        # Load data from CSV file
        df = pd.read_csv(csv_file)
        
        # Extract columns
        ell = df["ell"].values.astype(int)
        dl = df["Dl"].values
        sigma = df["errDl"].values
        
        # Trim to lmax
        mask = ell <= lmax
        ell = ell[mask]
        dl = dl[mask]
        sigma = sigma[mask]
        
        print(f"Loaded {len(ell)} multipoles from CSV file")
        print(f"D_ell range: {np.min(dl):.2e} to {np.max(dl):.2e}")
        
        return ell, dl, sigma
    
    except Exception as e:
        print(f"Error loading from CSV file: {e}")
        raise FileNotFoundError(f"Cannot load CSV file {csv_file} and no fallback data available.")

# Removed simulated data function - we only use real data from the FITS file

def plot_power_spectrum(ell, dl, sigma, title="CMB TT Power Spectrum", save_path=None):
    """
    Plot the power spectrum.
    
    Args:
        ell (array): Multipole values
        dl (array): Power spectrum values
        sigma (array): Error bars
        title (str): Plot title
        save_path (str): Path to save the plot, or None to display
        
    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    
    # Plot the data points with error bars
    plt.errorbar(ell, dl, yerr=sigma, fmt='o', markersize=3, color='blue', 
                 label="Planck Data", alpha=0.7)
    
    # Plot a smooth line connecting the points
    if len(ell) >= 4:  # Need at least 4 points for cubic interpolation
        ell_smooth = np.linspace(min(ell), max(ell), 1000)
        interp = interp1d(ell, dl, kind='cubic', bounds_error=False, fill_value="extrapolate")
        dl_smooth = interp(ell_smooth)
        plt.plot(ell_smooth, dl_smooth, '-', color='red', alpha=0.5, label="Interpolation")
    
    plt.xlabel(r"Multipole $\ell$")
    plt.ylabel(r"$D_\ell$ [$\mu K^2$]")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Load data from the best available source
    ell, dl, sigma = load_planck_data()
    
    # Plot the power spectrum
    plot_power_spectrum(ell, dl, sigma)
    
    # Save the data to a CSV file for later use
    output_file = "planck_tt_spectrum_processed.csv"
    df = pd.DataFrame({"ell": ell, "Dl": dl, "errDl": sigma})
    df.to_csv(output_file, index=False)
    print(f"Saved processed spectrum to {output_file}")