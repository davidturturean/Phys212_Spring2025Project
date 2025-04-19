# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 14:35:03 2025

@author: David Turturean
"""

import numpy as np
from scipy.interpolate import interp1d
from data import load_planck_tt_data
import sys
import os

# Ensure the current directory is in the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# First try to import from cosmology_model (enhanced model)
try:
    from cosmology_model import compute_cl as compute_cl_full
    use_enhanced_model = True
    print("Using enhanced cosmological model")
except ImportError:
    use_enhanced_model = False
    print("Enhanced model not available, falling back to simple model")

# Fall back to Daria's simpler model if enhanced model fails
from CAMB import model_Dl_TT

# Load the Planck TT data once (so it's cached for likelihood calculations)
try:
    ell_obs, D_ell_obs, sigma_obs = load_planck_tt_data()
    print(f"Loaded Planck TT data with {len(ell_obs)} multipoles")
except Exception as e:
    print(f"Warning: Could not preload Planck data: {e}")
    # Define placeholder data in case loading fails
    ell_obs = np.arange(2, 2500, 50)
    D_ell_obs = np.zeros_like(ell_obs)
    sigma_obs = np.ones_like(ell_obs)

def compute_cl(params):
    """
    Compute theoretical CMB TT power spectrum given cosmological parameters.
    Uses enhanced model if available, falls back to simple model if not.
    
    Args:
        params (dict): Dictionary with cosmological parameters
            Required keys: 'A_s', 'n_s', 'tau' (optional)
            
    Returns:
        ndarray: Array of D_ell values for ell from 2 to 2500
    """
    # Define the ell range
    ell_model = np.arange(2, 2501)
    
    # Try the enhanced model first
    if use_enhanced_model:
        try:
            D_ell_model = compute_cl_full(params)
            return D_ell_model
        except Exception as e:
            print(f"Enhanced model failed: {e}, falling back to simple model")
            # Fall back to simple model if enhanced model fails
    
    # Fall back to Daria's simple model
    # Extract relevant parameters
    A_s = params['A_s']
    n_s = params['n_s']
    
    # Call Daria's model to compute D_ell values
    D_ell_model = model_Dl_TT(ell_model, A_s, n_s)
    
    # Apply reionization effect if tau is considered in the model
    if 'tau' in params:
        # Simple dampening factor at low-ℓ due to reionization
        tau = params['tau']
        damping = np.exp(-2 * tau * (ell_model < 30))
        D_ell_model *= damping
    
    return D_ell_model

def compute_loglikelihood(params):
    """
    Compute the log-likelihood of the Planck TT data given cosmological parameters.
    Uses a chi-squared Gaussian likelihood approximation for the power spectrum.
    
    Args:
        params (dict): Dictionary with cosmological parameters
            Required keys: 'A_s', 'n_s', 'tau'
            
    Returns:
        float: Log-likelihood value
    """
    # Get theoretical D_ell for TT spectrum
    ell_model = np.arange(2, 2501)
    try:
        D_ell_theory = compute_cl(params)
    except Exception as e:
        print(f"Error in compute_cl: {e}")
        # Return a very negative log-likelihood if computation fails
        return -1e10
    
    # Interpolate theoretical spectrum to match observed multipole values
    interp_model = interp1d(ell_model, D_ell_theory, kind='linear', bounds_error=False, fill_value=0)
    D_ell_theory_interp = interp_model(ell_obs)
    
    # Compute residuals (data - model)
    residuals = D_ell_obs - D_ell_theory_interp
    
    # Separate high-ℓ and low-ℓ components
    highL_mask = (ell_obs >= 30)
    lowL_mask = (ell_obs < 30)
    
    # Compute log-likelihood for high-ℓ (Gaussian approximation)
    logL_highL = -0.5 * np.sum((residuals[highL_mask] / sigma_obs[highL_mask])**2)
    
    # Compute log-likelihood for low-ℓ (also Gaussian approximation, could be refined later)
    # For low-ℓ, we could use a more accurate likelihood, but this is a starting point
    logL_lowL = -0.5 * np.sum((residuals[lowL_mask] / sigma_obs[lowL_mask])**2)
    
    # Total log-likelihood
    logL = logL_highL + logL_lowL
    
    # Add normalization factors for completeness (optional, doesn't affect MCMC)
    n_data = len(ell_obs)
    norm = -0.5 * n_data * np.log(2 * np.pi) - np.sum(np.log(sigma_obs))
    
    return logL + norm

# Example usage (if run as main script)
if __name__ == "__main__":
    # Test using fiducial parameter values
    from parameters import fiducial_params
    
    # Compute likelihood with fiducial parameters
    logL = compute_loglikelihood(fiducial_params)
    print(f"Log-likelihood at fiducial parameters: {logL:.2f}")
    
    # Compute chi-squared for convenience
    chi2 = -2 * (logL - (len(ell_obs) * (-0.5 * np.log(2 * np.pi)) - np.sum(np.log(sigma_obs))))
    dof = len(ell_obs) - len(fiducial_params)  # Degrees of freedom
    print(f"χ² = {chi2:.2f} for {dof} degrees of freedom")
    print(f"Reduced χ² = {chi2/dof:.2f}")