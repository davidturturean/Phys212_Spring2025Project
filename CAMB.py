# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 20:29:47 2025

@author: Daria Teodora Harabor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# 1. SET COSMOLOGICAL PARAMETERS (Planck 2018 best-fit)
# ---------------------------
H0 = 67.36  # Hubble parameter in km/s/Mpc
h = H0 / 100.0
ombh2 = 0.02237  # Ω_b h^2
omch2 = 0.1200   # Ω_c h^2
As = 2.1e-9      # Scalar amplitude
ns = 0.9649      # Spectral index
tau = 0.0544     # Optical depth


# ---------------------------
# 2. APPROXIMATE MODEL FOR TT SPECTRUM
# ---------------------------
def primordial_power(k, As, ns, k0=0.05):
    return As * (k / k0)**(ns - 1)

def model_Dl_TT(ell, As, ns):
    """Semi-analytic mockup TT spectrum with acoustic structure and damping."""
    k = ell / 14000  # rough ell-to-k mapping
    Pk = primordial_power(k, As, ns)

    acoustic = np.sin(np.pi * ell / 340.0)**2
    envelope = np.exp(-(ell / 1800)**2)
    Dl = Pk * acoustic * envelope * 1e9  # scale up for visibility in μK^2
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

# Load the Planck binned power spectrum from uploaded CSV
planck_df = pd.read_csv("planck_binned_tt.csv")

# Extract relevant columns
ell_data = planck_df["ell"]
Dl_data = planck_df["Dl"]
Dl_err = planck_df["errDl"]


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





