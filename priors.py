# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 14:36:52 2025

@author: David Turturean
"""

import numpy as np
from parameters import priors
from likelihood import compute_loglikelihood

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
        
    # Compute likelihood (more expensive)
    ll = compute_loglikelihood(params)
    
    # Return the sum (log posterior)
    return lp + ll

# Example usage (if run as main script)
if __name__ == "__main__":
    from parameters import fiducial_params
    
    # Calculate prior at fiducial values
    log_p = log_prior(fiducial_params)
    print(f"Log-prior at fiducial parameters: {log_p:.2f}")
    
    # Check tau's Gaussian prior specifically
    tau_vals = np.linspace(0.01, 0.2, 20)
    test_params = fiducial_params.copy()
    
    print("\nTesting tau's Gaussian prior:")
    print("tau\tlog_prior")
    print("-" * 20)
    for tau in tau_vals:
        test_params["tau"] = tau
        lp = log_prior(test_params)
        print(f"{tau:.3f}\t{lp:.2f}")