# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 20:29:47 2025

@author: Daria Teodora Harabor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------
# 1. PLANCK 2018 PARAMS
# ---------------------------
H0 = 67.36  # km/s/Mpc
h = H0 / 100.0
ombh2 = 0.02237  # Ω_b h^2
omch2 = 0.1200   # Ω_c h^2
As = 2.1e-9      # Amplitude
ns = 0.9649      # Tilt
tau = 0.0544     # Optical depth


# ---------------------------
# 2. TT SPECTRUM MODEL
# ---------------------------
def primordial_power(k, As, ns, k0=0.05):
    return As * (k / k0)**(ns - 1)

def model_Dl_TT(ell, As, ns):
    """Basic TT spectrum with peaks and damping as in Section 1.1."""
    # k ≈ ℓ/r(z*)
    k = ell / 14000  # r(z*) ≈ 14000 Mpc
    
    # Power spectrum with tilt
    Pk = primordial_power(k, As, ns)
    
    # Get scales for Planck 2018 cosmology
    h = 0.6736
    Omega_b_h2 = 0.02237
    Omega_m_h2 = 0.1424
    
    # Use formulas from Section 1.1
    r_s = 144.7 * (Omega_m_h2/0.14)**(-0.25) * (Omega_b_h2/0.024)**(-0.13)  # Sound horizon
    # Parameter-dependent d_A
    d_A = 14000.0 * (h/0.7) * (0.14/Omega_m_h2)**0.4  # Mpc
    ell_A = np.pi * d_A / r_s  # Acoustic scale = π·d_A/r_s
    
    # Silk damping scale
    ell_D = 1600.0 * (Omega_b_h2/0.02237)**(-0.25) * (Omega_m_h2/0.1424)**(-0.125)
    
    # Acoustic pattern
    acoustic = np.sin(np.pi * ell / ell_A)**2
    
    # Silk damping = exp(-ℓ²/ℓ_D²)
    envelope = np.exp(-(ell / ell_D)**2)
    
    # Final spectrum
    Dl = Pk * acoustic * envelope * 1e9  # in μK^2
    return Dl

# Compute model spectrum
ell_model = np.arange(2, 2501)
Dl_model = model_Dl_TT(ell_model, As, ns)

# ---------------------------
# 3. LOAD PLANCK BINNED SPECTRUM
# ---------------------------
planck_file = "COM_PowerSpect_CMB-TT-binned_R3.01.txt"  # Must be in same folder

# Full column names in the file
columns = ["ell", "Dl", "errDl", "Cl", "errCl", "covCC", "covDD", "covDC"]

# Only attempt to load data if file exists
if os.path.exists(planck_file):
    try:
        df = pd.read_csv(planck_file,
                        sep=r'\s+',
                        comment='#',
                        names=columns,
                        encoding='latin1',
                        engine='python')
        
        # Extract relevant columns
        ell_data = df["ell"]
        Dl_data = df["Dl"]
        Dl_err = df["errDl"]
        
        # Optional: inspect values
        print(df.head())
        print(df.describe())
    except Exception as e:
        print(f"Error loading Planck data: {e}")
        # Create placeholder data
        ell_data = np.arange(30, 2500, 50)
        Dl_data = model_Dl_TT(ell_data, As, ns)  # Use model as placeholder
        Dl_err = Dl_data * 0.1  # 10% error as placeholder
else:
    print(f"Planck data file {planck_file} not found. Using model data as placeholder.")
    # Create placeholder data
    ell_data = np.arange(30, 2500, 50)
    Dl_data = model_Dl_TT(ell_data, As, ns)  # Use model as placeholder
    Dl_err = Dl_data * 0.1  # 10% error as placeholder

# ---------------------------
# 4. PLOT COMPARISON
# ---------------------------
"""
plt.figure(figsize=(10, 5))
plt.plot(ell_model, Dl_model, label="Model (ΛCDM approx)", lw=2, color='tab:blue')
plt.errorbar(ell_data, Dl_data, yerr=Dl_err, fmt='o', markersize=3,
             label="Planck 2018 Binned", alpha=0.6, capsize=2, color='tab:orange')

plt.xlabel("Multipole $\\ell$")
plt.ylabel("$D_\\ell^{TT}$ [$\\mu K^2$]")
plt.title("CMB TT Power Spectrum: Model vs Planck 2018")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""

# ---------------------------
# 5. RESIDUAL & CHI-SQUARED COMPARISON
# ---------------------------

from scipy.interpolate import interp1d

# Interpolate your model to match Planck ell values
interp_model = interp1d(ell_model, Dl_model, kind='linear', bounds_error=False, fill_value=0)
Dl_model_interp = interp_model(ell_data)

# Compute residuals and chi-squared
residuals = Dl_data - Dl_model_interp
chi2 = np.sum((residuals / Dl_err)**2)

# Plot residuals
"""
plt.figure(figsize=(10, 3))
plt.errorbar(ell_data, residuals, yerr=Dl_err, fmt='o', markersize=3,
             label="Residuals (Planck - Model)", color='purple', alpha=0.6)
plt.axhline(0, color='gray', ls='--')
plt.xlabel("Multipole $\\ell$")
plt.ylabel("Residual $D_\\ell$ [$\\mu K^2$]")
plt.title("Residuals Between Planck 2018 and Model")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
"""

# Print chi-squared summary
dof = len(ell_data) - 2  # Degrees of freedom (subtracting parameters As, ns)
print(f"\nTotal χ²: {chi2:.2f} for {dof} degrees of freedom")
print(f"Reduced χ²: {chi2/dof:.2f}")

# ---------------------------
# 4+5. OVERLAID MODEL, DATA, AND RESIDUALS
# ---------------------------

from scipy.interpolate import interp1d

# Interpolate model at Planck ell values
interp_model = interp1d(ell_model, Dl_model, kind='linear', bounds_error=False, fill_value=0)
Dl_model_interp = interp_model(ell_data)
residuals = Dl_data - Dl_model_interp
chi2 = np.sum((residuals / Dl_err)**2)
dof = len(ell_data) - 2

"""
# Make figure with 2 subplots: spectrum + residuals
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, height_ratios=[3, 1])

# Top: spectrum
ax1.plot(ell_model, Dl_model, label="Model (ΛCDM approx)", lw=2, color='tab:blue')
ax1.errorbar(ell_data, Dl_data, yerr=Dl_err, fmt='o', markersize=3,
             label="Planck 2018 Binned", alpha=0.6, capsize=2, color='tab:orange')
ax1.set_ylabel(r"$D_\ell^{TT}$ [$\mu K^2$]")
ax1.set_title("CMB TT Power Spectrum: Model vs Planck 2018")
ax1.grid(True)
ax1.legend()

# Bottom: residuals
ax2.errorbar(ell_data, residuals, yerr=Dl_err, fmt='o', markersize=3,
             label="Residuals (Planck - Model)", color='purple', alpha=0.7)
ax2.axhline(0, color='gray', ls='--')
ax2.set_xlabel("Multipole $\\ell$")
ax2.set_ylabel("Residual $D_\\ell$")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
"""

# -------------------------------------------

# Try to load the CSV file that might have been created by check.py
csv_file = "planck_binned_tt.csv"
if os.path.exists(csv_file):
    try:
        # Load the Planck binned power spectrum from CSV
        planck_df = pd.read_csv(csv_file)
        
        # Extract relevant columns
        ell_data = planck_df["ell"]
        Dl_data = planck_df["Dl"]
        Dl_err = planck_df["errDl"]
        
        print(f"Loaded Planck data from {csv_file}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        # Continue with previously defined ell_data, Dl_data, Dl_err
else:
    print(f"CSV file {csv_file} not found. Using previously defined data.")
    # Continue with previously defined ell_data, Dl_data, Dl_err


ell_model = np.arange(2, 2501)
Dl_model = model_Dl_TT(ell_model, As, ns)

# Interpolate model to Planck ell values
from scipy.interpolate import interp1d
model_interp = interp1d(ell_model, Dl_model, kind='linear', bounds_error=False, fill_value=0)
Dl_model_interp = model_interp(ell_data)

# Compute best-fit amplitude scaling factor (least-squares fit)
scaling_factor = np.sum(Dl_model_interp * Dl_data / Dl_err**2) / np.sum((Dl_model_interp / Dl_err)**2)
Dl_model_scaled = scaling_factor * Dl_model_interp

# Plot comparison
if False:  # Disable plotting during imports
    plt.figure(figsize=(10, 5))
    plt.errorbar(ell_data, Dl_data, yerr=Dl_err, fmt='o', label='Planck 2018 Binned', color='tab:orange', alpha=0.6)
    plt.plot(ell_data, Dl_model_scaled, label=f'Scaled Model (A = {scaling_factor:.2f})', color='tab:blue', lw=2)
    plt.xlabel("Multipole $\\ell$")
    plt.ylabel("$D_\\ell^{TT}$ [$\\mu K^2$]")
    plt.title("CMB TT Power Spectrum: Scaled Semi-Analytic Model vs Planck 2018")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# If this is run as a script, test the model
if __name__ == "__main__":
    print("Testing CAMB.py model...")
    
    # Plot comparison with Planck data (if available)
    plt.figure(figsize=(10, 5))
    plt.plot(ell_model, Dl_model, label="Model (ΛCDM approx)", lw=2, color='tab:blue')
    plt.errorbar(ell_data, Dl_data, yerr=Dl_err, fmt='o', markersize=3,
                 label="Planck 2018 Binned", alpha=0.6, capsize=2, color='tab:orange')
    plt.xlabel("Multipole $\\ell$")
    plt.ylabel("$D_\\ell^{TT}$ [$\\mu K^2$]")
    plt.title("CMB TT Power Spectrum: Model vs Planck 2018")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()