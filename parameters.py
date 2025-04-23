# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 14:30:15 2025

@author: David Turturean
"""

import numpy as np

# Define fiducial ΛCDM parameter values (Planck 2018 best-fit)
fiducial_params = {
    "H0": 67.36,         # Hubble constant [km/s/Mpc]
    "Omega_b_h2": 0.02237,  # Physical baryon density
    "Omega_c_h2": 0.1200,   # Physical cold dark matter density
    "n_s": 0.9649,        # Scalar spectral index
    "A_s": 2.1e-9,        # Primordial amplitude
    "tau": 0.0544,        # Optical depth to reionization
}

# Define prior distributions for each ΛCDM parameter
priors = {
    "H0": {
        "dist": "uniform",
        "min": 40.0, "max": 100.0,               # Hubble constant [km/s/Mpc]
    },
    "Omega_b_h2": {
        "dist": "gaussian",
        "mean": 0.0224, "sigma": 0.0005,         # Baryon density mean and 1σ
        "min": 0.01, "max": 0.03                 # hard bounds to reflect BBN/Planck limits
    },
    "Omega_c_h2": {
        "dist": "uniform",
        "min": 0.001, "max": 0.99,              # CDM density (broad bounds; physically >0)
    },
    "n_s": {
        "dist": "uniform",
        "min": 0.8, "max": 1.2,                 # Spectral index (flat prior in a reasonable range)
    },
    "A_s": {
        "dist": "uniform",
        "min": 5e-10, "max": 5e-9,              # Primordial amplitude
    },
    "tau": {
        "dist": "gaussian",
        "mean": 0.0544, "sigma": 0.0073,        # Optical depth prior (based on Planck)
        "min": 0.0, "max": 0.3                  # τ must be between 0 and 0.3
    }
}

# Parameter ranges for MCMC initialization (for multiple starting points)
# Using broad ranges to avoid being too close to known values
param_ranges = {
    "H0": (50.0, 90.0),         # Wide range including both Planck and local measurements
    "Omega_b_h2": (0.015, 0.030), # Wide range around BBN constraints
    "Omega_c_h2": (0.05, 0.25),   # Wide range for dark matter density
    "n_s": (0.85, 1.05),          # Wide range around scale-invariant n_s=1
    "A_s": (1.0e-9, 4.0e-9),      # Wide range for amplitude
    "tau": (0.02, 0.12)           # Wide range for reionization optical depth
}

# Get the list of parameter names for use in MCMC
param_names = list(fiducial_params.keys())