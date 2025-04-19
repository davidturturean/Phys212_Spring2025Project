# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 14:38:10 2025

@author: David Turturean
"""

import numpy as np
import matplotlib.pyplot as plt
from priors import log_posterior
import corner

def evaluate_posterior_grid(grid_params, fixed_params):
    """
    Compute the posterior on a grid for one or two parameters, keeping others fixed.
    
    Args:
        grid_params (dict): Keys are parameter names to vary (1 or 2 keys),
                           values are sequences (list or array) of values for those parameters.
        fixed_params (dict): Parameter values for all other parameters (those not varied).
    
    Returns:
        If one parameter is varied: returns (values, post_values)
        If two parameters are varied: returns (grid_vals_param1, grid_vals_param2, post_grid)
    """
    params_to_vary = list(grid_params.keys())
    
    if len(params_to_vary) == 1:
        # 1D grid evaluation
        param = params_to_vary[0]
        values = np.array(grid_params[param])
        logpost_values = []
        
        for val in values:
            params = fixed_params.copy()
            params[param] = val
            lp = log_posterior(params)
            logpost_values.append(lp)
            
        return values, np.array(logpost_values)
        
    elif len(params_to_vary) == 2:
        # 2D grid evaluation
        p1, p2 = params_to_vary
        vals1 = np.array(grid_params[p1])
        vals2 = np.array(grid_params[p2])
        post_grid = np.zeros((len(vals1), len(vals2)))
        
        for i, val1 in enumerate(vals1):
            for j, val2 in enumerate(vals2):
                params = fixed_params.copy()
                params[p1] = val1
                params[p2] = val2
                post_grid[i, j] = log_posterior(params)
                
        return vals1, vals2, post_grid
        
    else:
        raise ValueError("Can only handle 1D or 2D grid evaluations.")

def plot_trace(chain, param_names, save_file=None):
    """
    Plot trace (parameter value vs step) for each parameter in the MCMC chain.
    
    Args:
        chain (ndarray): MCMC chain array. Shape can be:
                        - (n_steps, n_params) for a single chain
                        - (n_chains, n_steps, n_params) for multiple chains
        param_names (list): List of parameter name strings
        save_file (str, optional): Path to save figure to. If None, displays figure instead.
    
    Returns:
        fig: The matplotlib figure object
    """
    chain = np.array(chain)
    n_params = len(param_names)
    
    # Determine if multiple chains are present (3D array)
    if chain.ndim == 3:
        n_chains, n_steps, _ = chain.shape
        fig, axes = plt.subplots(n_params, 1, figsize=(12, 2*n_params), sharex=True)
        
        if n_params == 1:
            axes = [axes]  # Make it iterable for single parameter case
            
        # Plot each parameter
        for i, name in enumerate(param_names):
            for c in range(n_chains):
                axes[i].plot(chain[c, :, i], alpha=0.7, lw=0.8, 
                           label=f"Chain {c+1}" if i == 0 else None)
            axes[i].set_ylabel(name)
            axes[i].grid(True, alpha=0.3)
        
        # Only add legend to first subplot
        if n_chains <= 10:  # Only show legend if not too many chains
            axes[0].legend(fontsize=8)
            
        axes[-1].set_xlabel("MCMC step")
        plt.suptitle("MCMC Trace Plots", fontsize=14)
        
    else:
        # Single chain
        n_steps, _ = chain.shape
        fig, axes = plt.subplots(n_params, 1, figsize=(12, 2*n_params), sharex=True)
        
        if n_params == 1:
            axes = [axes]  # Make it iterable for single parameter case
            
        for i, name in enumerate(param_names):
            axes[i].plot(chain[:, i], color="black", lw=0.8)
            axes[i].set_ylabel(name)
            axes[i].grid(True, alpha=0.3)
            
        axes[-1].set_xlabel("MCMC step")
        plt.suptitle("MCMC Trace Plot", fontsize=14)
    
    plt.tight_layout()
    
    # Save or show the figure
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        
    return fig

def plot_corner(samples, param_names, truths=None, figsize=(10, 10), save_file=None):
    """
    Create a corner plot of the posterior samples for all parameters.
    
    Args:
        samples (ndarray): Array of shape (N_samples, N_parameters) with MCMC samples after burn-in.
        param_names (list): List of parameter names for labeling axes.
        truths (dict, optional): True or reference values for parameters (to mark on plots).
        figsize (tuple, optional): Size of the figure, default (10, 10).
        save_file (str, optional): Path to save figure to. If None, displays figure instead.
        
    Returns:
        fig: The matplotlib figure object
    """
    # Prepare truth values in the same order as param_names if provided
    truth_list = None
    if truths is not None:
        truth_list = [truths.get(name, None) for name in param_names]
        
    # Create the corner plot
    fig = corner.corner(samples, labels=param_names, truths=truth_list,
                      show_titles=True, title_fmt=".3f", title_kwargs={"fontsize": 12},
                      figsize=figsize)
                      
    fig.suptitle("ΛCDM Parameter Posterior Distributions", fontsize=16, y=0.99)
    
    # Save or show the figure
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    
    return fig  # Return the figure for potential further customization

def gelman_rubin(chains):
    """
    Calculate Gelman-Rubin R-hat statistic for MCMC convergence diagnosis.
    
    Args:
        chains (ndarray): MCMC chains with shape (n_chains, n_steps, n_params)
        
    Returns:
        ndarray: R-hat values for each parameter
    """
    m, n, d = chains.shape  # m chains, n samples, d parameters
    R_hat = []
    
    for i in range(d):
        # Calculate mean of each chain
        chain_means = np.mean(chains[:, :, i], axis=1)
        # Calculate variance of each chain
        chain_vars = np.var(chains[:, :, i], axis=1, ddof=1)
        
        # Between-chain variance
        B = n * np.var(chain_means, ddof=1)
        # Within-chain variance
        W = np.mean(chain_vars)
        
        # Estimate of marginal posterior variance
        V_hat = (1 - 1/n) * W + B/n
        # R-hat statistic
        R = np.sqrt(V_hat / W)
        
        R_hat.append(R)
        
    return np.array(R_hat)

# Example usage (if run as main script)
if __name__ == "__main__":
    from parameters import fiducial_params, param_names
    import time
    
    # Example: 1D posterior grid for A_s
    print("Computing 1D posterior grid for A_s...")
    start = time.time()
    A_s_vals = np.linspace(1e-9, 3e-9, 20)
    fixed_params = fiducial_params.copy()
    vals, logpost = evaluate_posterior_grid({"A_s": A_s_vals}, fixed_params)
    end = time.time()
    print(f"Computed in {end-start:.2f} seconds")
    
    # Convert to linear probability and normalize
    prob = np.exp(logpost - np.max(logpost))
    prob /= np.sum(prob)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(vals, prob, 'o-')
    plt.axvline(fiducial_params["A_s"], color='r', linestyle='--', label="Fiducial value")
    plt.xlabel("$A_s$")
    plt.ylabel("Posterior probability (normalized)")
    plt.title("1D Posterior for $A_s$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()