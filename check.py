# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 22:51:18 2025

@author: Daria Teodora Harabor
"""

import numpy as np
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
# 2. SEMI-ANALYTIC MODEL FOR TT SPECTRUM
# ---------------------------
def primordial_power(k, As, ns, k0=0.05):
    return As * (k / k0)**(ns - 1)

def model_Dl_TT(ell, As, ns):
    """Semi-analytic mockup TT spectrum with acoustic structure and damping."""
    k = ell / 14000  # rough ell-to-k mapping
    Pk = primordial_power(k, As, ns)

    acoustic = np.sin(np.pi * ell / 340.0)**2
    envelope = np.exp(-(ell / 1800)**2)
    
    # Apply reionization damping at low ℓ
    damping = np.exp(-2 * tau * (ell < 30))
    
    Dl = Pk * acoustic * envelope * damping * 1e9  # μK² scaling
    return Dl

# Compute model spectrum
ell_model = np.arange(2, 2501)
Dl_model = model_Dl_TT(ell_model, As, ns)

# ---------------------------
# 3. PLOT THE MODEL ALONE
# ---------------------------
plt.figure(figsize=(10, 5))
plt.plot(ell_model, Dl_model, label="Semi-Analytic Model", color='tab:blue')
plt.xlabel("Multipole $\\ell$")
plt.ylabel("$D_\\ell^{TT}$ [$\\mu K^2$]")
plt.title("CMB TT Power Spectrum — Semi-Analytic Approximation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


import pandas as pd

# File path (must match location of your .txt file)
planck_file = "COM_PowerSpect_CMB-TT-binned_R3.01.txt"

# Read ASCII Planck 2018 binned power spectrum
df = pd.read_csv(planck_file,
                 sep=r'\s+',
                 comment='#',
                 names=["ell", "Dl", "errDl", "covCC", "covDD", "covDC"],
                 encoding='latin1',
                 engine='python')

# Save to CSV for later use
df.to_csv("planck_binned_tt.csv", index=False)
print("✅ Saved as planck_binned_tt.csv")
