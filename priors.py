# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 14:36:52 2025

@author: David Turturean
"""

import numpy as np
import logging
import os
from parameters import priors
from likelihood import compute_loglikelihood

# Set up logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "priors.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PRIORS")

def log_prior(params):
    """
    Compute the log prior probability for the given parameters.
    Returns -inf if any parameter is outside its prior bounds.
    Otherwise, returns the sum of log-probabilities of each prior.
    
    Args:
        params (dict): Dictionary of parameter values
        
    Returns:
        float: Log-prior probability, or -inf if any parameter is outside bounds
    """
    logp = 0.0
    
    for name, value in params.items():
        if name not in priors:
            continue  # Ignore parameters not defined in priors
            
        pinfo = priors[name]
        
        # Check bounds first
        if ("min" in pinfo and value < pinfo["min"]) or ("max" in pinfo and value > pinfo["max"]):
            return -np.inf  # Parameter outside allowed range -> prior is zero
            
        # Calculate log-prior contribution based on distribution type
        if pinfo["dist"] == "uniform":
            # Uniform prior: constant within [min, max]
            width = pinfo["max"] - pinfo["min"]
            logp += -np.log(width)  # log of 1/width (normalization)
            
        elif pinfo["dist"] == "gaussian":
            # Gaussian prior: log (1/(sqrt(2π)σ) * exp(-0.5*((x-mu)/σ)^2))
            mu, sigma = pinfo["mean"], pinfo["sigma"]
            logp += -0.5 * ((value - mu)/sigma)**2 - np.log(sigma * np.sqrt(2*np.pi))
            
        else:
            raise ValueError(f"Unknown prior distribution type for {name}")
            
    return logp

def log_posterior(params):
    """
    Compute the log posterior for the given parameters.
    log_posterior = log_prior + log_likelihood.
    Always uses the full covariance matrix for improved accuracy.
    
    Args:
        params (dict): Dictionary of parameter values
        
    Returns:
        float: Log-posterior value, or -inf if prior is invalid
    """
    # Check prior first (more efficient)
    lp = log_prior(params)
    if not np.isfinite(lp):
        # If prior is zero (params out of bounds), return -inf immediately
        return -np.inf
        
    # Compute likelihood (more expensive) - always use full covariance matrix
    ll = compute_loglikelihood(params, use_full_cov=True)
    
    # Return the sum (log posterior)
    return lp + ll

# Example usage (if run as main script)
if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from parameters import fiducial_params
    
    # Calculate prior at fiducial values
    log_p = log_prior(fiducial_params)
    print(f"Log-prior at fiducial parameters: {log_p:.2f}")
    
    # Calculate posterior with full covariance matrix (always used now)
    log_post = log_posterior(fiducial_params)
    
    print(f"Log-posterior with full covariance: {log_post:.2f}")
    
    # Test tau's Gaussian prior
    tau_vals = np.linspace(0.01, 0.2, 20)
    test_params = fiducial_params.copy()
    
    prior_vals = []
    post_vals = []
    
    for tau in tau_vals:
        test_params["tau"] = tau
        lp = log_prior(test_params)
        lpost = log_posterior(test_params)
        
        prior_vals.append(lp)
        post_vals.append(lpost)
    
    # Convert to probabilities (for visualization)
    prior_prob = np.exp(prior_vals - np.max(prior_vals))
    prior_prob /= np.sum(prior_prob)
    
    post_prob = np.exp(post_vals - np.max(post_vals))
    post_prob /= np.sum(post_prob)
    
    # Make sure the output directory exists
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot prior and posterior distributions
    plt.figure(figsize=(10, 6))
    plt.plot(tau_vals, prior_prob, 'b-', label="Prior")
    plt.plot(tau_vals, post_prob, 'r-', label="Posterior (full covariance)")
    plt.axvline(fiducial_params["tau"], color='k', linestyle='--', label="Fiducial value")
    plt.xlabel(r"$\tau$ (optical depth)")
    plt.ylabel("Probability density")
    plt.title(r"Prior and Posterior for $\tau$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tau_prior_posterior.png"))
    plt.close()
    
    print(f"Saved prior and posterior visualization to {os.path.join(output_dir, 'tau_prior_posterior.png')}")