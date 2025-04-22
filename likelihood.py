# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 14:35:03 2025

@author: David Turturean
"""

import numpy as np
from scipy.interpolate import interp1d
import sys
import os
import logging

# Set up logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "likelihood.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LIKELIHOOD")

# Ensure the current directory is in the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# First try to import from cosmology_model (enhanced model)
try:
    from cosmology_model import compute_cl as compute_cl_full
    use_enhanced_model = True
    logger.info("Using enhanced cosmological model")
except ImportError:
    use_enhanced_model = False
    logger.info("Enhanced model not available, falling back to simple model")

# Fall back to Daria's simpler model if enhanced model fails
from CAMB import model_Dl_TT

# Load the Planck TT data once (so it's cached for likelihood calculations)
try:
    from data_loader import load_planck_data
    ell_obs, D_ell_obs, sigma_obs = load_planck_data(source="fits")
    logger.info(f"Loaded Planck TT data with {len(ell_obs)} multipoles")
except Exception as e:
    logger.error(f"Warning: Could not preload Planck data: {e}")
    # Define placeholder data in case loading fails
    ell_obs = np.arange(2, 2500, 50)
    D_ell_obs = np.zeros_like(ell_obs)
    sigma_obs = np.ones_like(ell_obs)

def build_covariance_matrix(ell, dl, sigma_ell=None):
    """
    Build a proper covariance matrix for the CMB TT spectrum.
    
    This function creates a covariance matrix that accounts for:
    1. Measurement errors (diagonal elements)
    2. Correlations between neighboring multipoles
    3. Effects of fractional sky coverage
    
    Args:
        ell (array): Multipole moments
        dl (array): Power spectrum values
        sigma_ell (array, optional): Per-ell error bars
        
    Returns:
        array: Covariance matrix
    """
    n_ell = len(ell)
    
    # Use provided errors or default to cosmic variance + instrument noise
    if sigma_ell is None:
        # Approximate cosmic variance
        sigma_ell = dl * np.sqrt(2 / (2 * ell + 1))
    
    # Initialize covariance matrix with diagonal elements
    cov = np.diag(sigma_ell**2)
    
    # Add correlations between neighboring multipoles
    f_sky = 0.8  # Approximate fractional sky coverage
    corr_length = 10  # Correlation length in multipoles
    
    for i in range(n_ell):
        for j in range(n_ell):
            if i != j:
                l_diff = abs(ell[i] - ell[j])
                if l_diff <= corr_length:
                    # Exponential correlation function based on multipole separation
                    rho = np.exp(-0.5 * (l_diff / corr_length)**2) * (1 - f_sky)
                    cov[i, j] = rho * np.sqrt(cov[i, i] * cov[j, j])
    
    return cov

def invert_cov_matrix(cov_matrix):
    """
    Safely invert a covariance matrix using SVD with conditioning.
    
    Args:
        cov_matrix (array): Covariance matrix to invert
        
    Returns:
        array: Inverted covariance matrix
    """
    try:
        # Use singular value decomposition for stable inversion
        u, s, vh = np.linalg.svd(cov_matrix, full_matrices=False)
        
        # Condition the eigenvalues to ensure stability
        # Discard eigenvalues smaller than a threshold related to the largest eigenvalue
        rcond = 1e-10  # Relative condition number
        s_new = np.where(s > rcond * s[0], s, rcond * s[0])
        
        # Compute the pseudo-inverse
        inv_cov = np.dot(vh.T, np.dot(np.diag(1/s_new), u.T))
        
        return inv_cov
    
    except Exception as e:
        logger.error(f"Error inverting covariance matrix: {e}")
        # Fall back to diagonal-only inverse in case of error
        inv_cov = np.diag(1.0 / np.diag(cov_matrix))
        logger.warning("Using diagonal-only covariance matrix as fallback")
        return inv_cov

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
            logger.error(f"Enhanced model failed: {e}, falling back to simple model")
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

def compute_loglikelihood(params, use_full_cov=True):
    """
    Compute the log-likelihood of the Planck TT data given cosmological parameters.
    Uses a proper covariance matrix for the power spectrum.
    
    Args:
        params (dict): Dictionary with cosmological parameters
            Required keys: 'A_s', 'n_s', 'tau'
        use_full_cov (bool): Whether to use the full covariance matrix (True)
                          or just diagonal elements (False)
            
    Returns:
        float: Log-likelihood value
    """
    # Get theoretical D_ell for TT spectrum
    ell_model = np.arange(2, 2501)
    try:
        D_ell_theory = compute_cl(params)
    except Exception as e:
        logger.error(f"Error in compute_cl: {e}")
        # Return a very negative log-likelihood if computation fails
        return -1e10
    
    # Interpolate theoretical spectrum to match observed multipole values
    interp_model = interp1d(ell_model, D_ell_theory, kind='linear', bounds_error=False, fill_value=0)
    D_ell_theory_interp = interp_model(ell_obs)
    
    # Compute residuals (data - model)
    residuals = D_ell_obs - D_ell_theory_interp
    
    # Compute log-likelihood using full covariance matrix if requested
    if use_full_cov:
        try:
            # Build covariance matrix with correlations
            cov_matrix = build_covariance_matrix(ell_obs, D_ell_obs, sigma_obs)
            
            # Invert covariance matrix safely
            inv_cov = invert_cov_matrix(cov_matrix)
            
            # Compute chi-squared using full covariance
            chi2 = np.dot(residuals, np.dot(inv_cov, residuals))
            
            # Log determinant term (optional, doesn't affect MCMC sampling)
            sign, logdet = np.linalg.slogdet(cov_matrix)
            if sign <= 0:
                logger.warning("Covariance matrix has non-positive determinant, using fallback")
                # Fall back to diagonal-only likelihood
                return -0.5 * np.sum((residuals / sigma_obs)**2)
            
            # Full log-likelihood with determinant term
            n_data = len(ell_obs)
            logL = -0.5 * (chi2 + logdet + n_data * np.log(2 * np.pi))
            return logL
            
        except Exception as e:
            logger.error(f"Error computing likelihood with full covariance: {e}")
            logger.warning("Falling back to diagonal-only likelihood")
            # Fall back to diagonal-only likelihood
            return -0.5 * np.sum((residuals / sigma_obs)**2)
    else:
        # Separate high-ℓ and low-ℓ components
        highL_mask = (ell_obs >= 30)
        lowL_mask = (ell_obs < 30)
        
        # Compute log-likelihood for high-ℓ (Gaussian approximation)
        logL_highL = -0.5 * np.sum((residuals[highL_mask] / sigma_obs[highL_mask])**2)
        
        # Compute log-likelihood for low-ℓ (also Gaussian approximation)
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
    import matplotlib.pyplot as plt
    import time
    
    # Compare diagonal-only and full covariance likelihoods
    start_time = time.time()
    
    # Compute likelihood with fiducial parameters
    logL_diag = compute_loglikelihood(fiducial_params, use_full_cov=False)
    logL_full = compute_loglikelihood(fiducial_params, use_full_cov=True)
    
    end_time = time.time()
    
    print(f"Diagonal-only log-likelihood: {logL_diag:.2f}")
    print(f"Full covariance log-likelihood: {logL_full:.2f}")
    print(f"Computation time: {end_time - start_time:.3f} seconds")
    
    # Compute chi-squared for convenience
    chi2_diag = -2 * (logL_diag - (len(ell_obs) * (-0.5 * np.log(2 * np.pi)) - np.sum(np.log(sigma_obs))))
    dof = len(ell_obs) - len(fiducial_params)  # Degrees of freedom
    print(f"Diagonal χ² = {chi2_diag:.2f} for {dof} degrees of freedom")
    print(f"Diagonal reduced χ² = {chi2_diag/dof:.2f}")
    
    # Compute model at fiducial parameters
    ell_model = np.arange(2, 2501)
    dl_model = compute_cl(fiducial_params)
    
    # Visualize covariance matrix
    cov_matrix = build_covariance_matrix(ell_obs, D_ell_obs, sigma_obs)
    
    # Make sure the output directory exists
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot covariance matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(np.log10(np.abs(cov_matrix)), cmap='viridis')
    plt.colorbar(label='log10(Covariance)')
    plt.title('Log10 of CMB Power Spectrum Covariance Matrix')
    plt.xlabel('Multipole index')
    plt.ylabel('Multipole index')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "covariance_matrix.png"))
    plt.close()
    
    print(f"Saved covariance matrix visualization to {os.path.join(output_dir, 'covariance_matrix.png')}")