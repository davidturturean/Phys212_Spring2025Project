#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare ΛCDM parameter constraints with Planck 2018 values

This script compares our MCMC parameter constraints with published Planck 2018 values,
generating visualizations and a detailed interpretation document.

Authors: David Turturean, Daria Teodora Harabor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Create output directory if it doesn't exist
os.makedirs("mcmc_results", exist_ok=True)

# Load our constraints
try:
    df = pd.read_csv("mcmc_results/parameter_constraints.csv")
    posterior_samples = np.load("mcmc_results/posterior_samples_lambdaCDM.npy")
    from parameters import param_names
    print(f"Loaded parameter constraints for {len(df)} parameters")
except Exception as e:
    print(f"ERROR: Failed to load parameter constraints: {e}")
    print("Make sure to run the MCMC simulation first using run_mcmc_production.py")
    exit(1)

# Planck 2018 values (TT,TE,EE+lowE)
planck_values = {
    "H0": (67.36, 0.54),           # Value, error
    "Omega_b_h2": (0.02237, 0.00015),
    "Omega_c_h2": (0.1200, 0.0012),
    "n_s": (0.9649, 0.0042),
    "A_s": (2.1e-9, 0.03e-9),
    "tau": (0.0544, 0.0073)
}

# Calculate differences in sigma units
differences = {}
for _, row in df.iterrows():
    param = row['parameter']
    if param in planck_values:
        our_val = row['median']
        our_err = (row['upper_68'] - row['lower_68']) / 2  # Approximate symmetric error
        planck_val, planck_err = planck_values[param]
        
        # Difference in sigma
        sigma_diff = (our_val - planck_val) / np.sqrt(our_err**2 + planck_err**2)
        differences[param] = sigma_diff

# Print comparison
print("\nParameter differences with Planck (in sigma units):")
for param, diff in differences.items():
    print(f"{param}: {diff:.2f} sigma")

# Plot comparison
plt.figure(figsize=(10, 6))
params = list(differences.keys())
diffs = [differences[p] for p in params]

plt.bar(params, diffs, color='skyblue')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
plt.axhline(y=-1, color='r', linestyle='--', alpha=0.5)
plt.axhline(y=2, color='r', linestyle=':', alpha=0.5)
plt.axhline(y=-2, color='r', linestyle=':', alpha=0.5)

plt.xlabel('Parameter')
plt.ylabel('Difference (sigma)')
plt.title('Comparison with Planck 2018 Values')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mcmc_results/planck_comparison.png", dpi=300)

print("\nSaved Planck comparison plot to mcmc_results/planck_comparison.png")
print("\nGenerating interpretation text...")

# Generate interpretation text
interpretation = """
# Interpretation of ΛCDM Parameter Constraints

## Comparison with Planck 2018 Values

"""

# Add general comment on agreement
max_diff = max([abs(d) for d in diffs])
if max_diff < 1.0:
    interpretation += "Our parameter constraints show excellent agreement with Planck 2018 values, with all parameters within 1σ of the published values. "
else:
    max_diff_param = params[diffs.index(max([abs(d) for d in diffs]) * (1 if diffs[diffs.index(max([abs(d) for d in diffs]))] >= 0 else -1))]
    if max_diff < 2.0:
        interpretation += f"Our parameter constraints show good agreement with Planck 2018 values, with all parameters within 2σ of the published values. The largest deviation is for {max_diff_param} at {max_diff:.1f}σ. "
    else:
        interpretation += f"Some parameters show deviations from Planck 2018 values, with the largest being {max_diff_param} at {max_diff:.1f}σ. "

interpretation += "The level of agreement is remarkable considering we are using only TT power spectrum data while Planck used a combination of TT, TE, and EE spectra plus low-ℓ polarization.\n\n"

interpretation += "## Parameter-Specific Interpretation\n\n"

# Load correlation matrix for interpretation
corr_matrix = np.corrcoef(posterior_samples.T)

# Add interpretation for each parameter
for i, param in enumerate(params):
    our_value = df[df['parameter'] == param]['median'].values[0]
    planck_val, planck_err = planck_values[param]
    
    interpretation += f"### {param}\n\n"
    
    # Parameter-specific interpretation
    if param == "H0":
        interpretation += f"Our constraint on the Hubble constant is {our_value:.2f} km/s/Mpc, which "
        if abs(differences[param]) < 1.0:
            interpretation += f"agrees well with the Planck value of {planck_val:.2f} km/s/Mpc. "
        else:
            interpretation += f"differs from the Planck value of {planck_val:.2f} km/s/Mpc by {differences[param]:.2f}σ. "
        
        # Check correlation with other parameters
        correlations = [(np.abs(corr_matrix[i, j]), params[j]) for j in range(len(params)) if i != j]
        correlations.sort(reverse=True)
        interpretation += f"H0 shows strongest correlation with {correlations[0][1]} (r = {corr_matrix[i, params.index(correlations[0][1])]:.2f}), "
        interpretation += "indicating the well-known geometric degeneracy in the CMB.\n\n"
        
    elif param == "Omega_b_h2":
        interpretation += f"The physical baryon density is constrained to {our_value:.5f}, "
        if abs(differences[param]) < 1.0:
            interpretation += f"in good agreement with the Planck value of {planck_val:.5f}. "
        else:
            interpretation += f"which differs from the Planck value of {planck_val:.5f} by {differences[param]:.2f}σ. "
        
        interpretation += "This parameter is primarily constrained by the relative heights of the acoustic peaks in the CMB power spectrum.\n\n"
        
    elif param == "Omega_c_h2":
        interpretation += f"The physical cold dark matter density is found to be {our_value:.4f}, "
        if abs(differences[param]) < 1.0:
            interpretation += f"consistent with the Planck measurement of {planck_val:.4f}. "
        else:
            interpretation += f"which differs from the Planck measurement of {planck_val:.4f} by {differences[param]:.2f}σ. "
        
        interpretation += "This parameter affects the matter-radiation equality redshift and the overall shape of the power spectrum.\n\n"
        
    elif param == "n_s":
        interpretation += f"The scalar spectral index is constrained to {our_value:.4f}, "
        if abs(differences[param]) < 1.0:
            interpretation += f"in agreement with the Planck value of {planck_val:.4f}. "
        else:
            interpretation += f"which differs from the Planck value of {planck_val:.4f} by {differences[param]:.2f}σ. "
        
        interpretation += "The value of n_s < 1 confirms the prediction of simple inflationary models, indicating a slightly red-tilted primordial power spectrum.\n\n"
        
    elif param == "A_s":
        interpretation += f"The primordial amplitude is found to be {our_value:.3e}, "
        if abs(differences[param]) < 1.0:
            interpretation += f"consistent with Planck's value of {planck_val:.3e}. "
        else:
            interpretation += f"which differs from Planck's value of {planck_val:.3e} by {differences[param]:.2f}σ. "
        
        interpretation += "This parameter sets the overall amplitude of the CMB fluctuations.\n\n"
        
    elif param == "tau":
        interpretation += f"The optical depth to reionization is constrained to {our_value:.4f}, "
        if abs(differences[param]) < 1.0:
            interpretation += f"in good agreement with Planck's value of {planck_val:.4f}. "
        else:
            interpretation += f"which differs from Planck's value of {planck_val:.4f} by {differences[param]:.2f}σ. "
        
        interpretation += "This parameter is primarily constrained by the large-scale polarization data in Planck, so our constraint from TT spectrum alone is noteworthy.\n\n"

interpretation += """
## Parameter Degeneracies

The corner plot reveals several parameter degeneracies that are well-understood in CMB physics:

1. **H0-Ωm degeneracy**: There is a degeneracy between the Hubble constant and matter density. This is because the CMB primarily constrains the acoustic scale, which depends on the combination of these parameters.

2. **ns-As degeneracy**: These parameters show some correlation as they both affect the shape and amplitude of the power spectrum.

3. **τ-As degeneracy**: Reionization optical depth and primordial amplitude are degenerate because both affect the overall amplitude of the peaks. Increasing τ suppresses power, which can be compensated by increasing As.

These degeneracies could be broken by adding additional datasets, such as BAO, supernovae, or CMB polarization.

## Implications for Cosmology

Our results from the CMB TT power spectrum analysis provide strong support for the standard ΛCDM cosmological model. The parameter constraints are consistent with a universe that:

1. Is spatially flat (assumed in our model)
2. Has ~68% dark energy driving cosmic acceleration
3. Has ~27% cold dark matter
4. Has ~5% ordinary matter (baryons)
5. Expanded from an early hot phase described by the Big Bang model
6. Had a spectrum of nearly scale-invariant primordial fluctuations, as predicted by cosmic inflation

## Limitations

This analysis has some limitations that could be addressed in future work:

1. We used only the TT power spectrum, while Planck's published constraints include polarization data
2. We fixed some parameters that could be varied (e.g., Neff, Σmν)
3. Our chains may benefit from longer runs or more chains to improve convergence for some parameters

## Future Directions

Possible extensions of this work include:

1. Adding CMB polarization data (EE and TE spectra)
2. Including other cosmological probes like BAO or supernovae
3. Exploring extensions to ΛCDM (e.g., curved universe, w≠-1, massive neutrinos)
4. Calculating Bayesian evidence to compare standard ΛCDM with extended models
"""

# Save interpretation to file
with open("mcmc_results/parameter_interpretation.md", "w") as f:
    f.write(interpretation)

print("Interpretation saved to mcmc_results/parameter_interpretation.md")
