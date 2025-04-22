# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 14:40:30 2025

@author: David Turturean
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import os
import logging
from priors import log_posterior
from parameters import param_names, fiducial_params, param_ranges

# Set up logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "mcmc_run.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MCMC")

def run_mcmc(log_posterior_func, initial_params, n_steps=10000, proposal_scale=None, 
             checkpoint_every=1000, chain_file=None, early_stopping=False, 
             convergence_checker=None, chain_id=None):
    """
    Run a simple Metropolis-Hastings MCMC sampler with checkpointing and logging.
    
    Args:
        log_posterior_func: Function that returns log-posterior given parameters.
        initial_params: Dict of initial parameter values (must be within prior bounds).
        n_steps: Number of MCMC steps to run.
        proposal_scale: Dict of proposal std-dev for each parameter.
        checkpoint_every: How often to save intermediate results.
        chain_file: If provided, will save checkpoints to this file.
        early_stopping: Whether to enable early stopping based on convergence.
        convergence_checker: Function to check for chain convergence.
        chain_id: ID of this chain for convergence checking.
        
    Returns:
        chain (ndarray): Array of shape (n_steps, n_params) with sampled parameter values.
        log_posterior_values (ndarray): Array of log-posterior values at each step.
    """
    # Initialize
    params = initial_params.copy()
    param_keys = list(initial_params.keys())
    
    logger.info(f"Starting MCMC run with {n_steps} steps")
    logger.info(f"Initial parameters: {initial_params}")
    
    # Try to get initial log-posterior
    current_logpost = log_posterior_func(params)
    if not np.isfinite(current_logpost):
        logger.error("Initial parameters have zero probability. Check priors.")
        raise ValueError("Initial parameters have zero probability. Check priors.")
    
    logger.info(f"Initial log-posterior: {current_logpost:.4f}")
    
    # Initialize chain and log-posterior storage
    chain = np.zeros((n_steps, len(param_keys)))
    log_posterior_values = np.zeros(n_steps)
    
    # If no proposal scales specified, set defaults (5% of initial values)
    if proposal_scale is None:
        proposal_scale = {name: 0.05 * abs(val) if val != 0 else 0.01 
                         for name, val in params.items()}
    
    logger.info(f"Proposal scales: {proposal_scale}")
    
    # MCMC loop
    accept_count = 0
    start_time = time.time()
    last_checkpoint_time = start_time
    early_stop = False
    convergence_check_every = 100  # Check for convergence every 100 steps
    
    for i in range(n_steps):
        # Propose new parameters by sampling from a Gaussian around current values
        proposal = {}
        for name, val in params.items():
            prop_val = np.random.normal(loc=val, scale=proposal_scale[name])
            proposal[name] = prop_val
        
        # Evaluate log-posterior at proposal
        prop_logpost = log_posterior_func(proposal)
        
        # Acceptance probability (log domain to avoid overflow)
        log_accept_ratio = prop_logpost - current_logpost
        
        # Accept or reject
        if np.log(np.random.rand()) < log_accept_ratio:
            # Accept the proposal
            params = proposal
            current_logpost = prop_logpost
            accept_count += 1
        
        # Store current state
        for j, param in enumerate(param_keys):
            chain[i, j] = params[param]
        log_posterior_values[i] = current_logpost
        
        # Check for convergence if early stopping is enabled
        if early_stopping and convergence_checker is not None and i > 0 and i % convergence_check_every == 0:
            # Check convergence on current chain data
            current_chain_data = chain[:i+1]
            try:
                converged, convergence_info = convergence_checker(current_chain_data, param_keys, chain_id, i)
            except Exception as e:
                logger.warning(f"Error in convergence check: {e}")
                converged, convergence_info = False, {}
            
            if converged:
                logger.info(f"Early stopping criteria met at step {i+1}/{n_steps}.")
                if 'max_r_hat' in convergence_info:
                    logger.info(f"Max R-hat value: {convergence_info['max_r_hat']:.4f}")
                
                # Save final state before stopping
                if chain_file:
                    final_file = f"{chain_file}_final_{i+1}.npz"
                    np.savez(
                        final_file,
                        chain=chain[:i+1],
                        log_posterior=log_posterior_values[:i+1],
                        accept_count=accept_count,
                        i=i,
                        elapsed_time=time.time() - start_time,
                        converged=True,
                        convergence_info=convergence_info
                    )
                    logger.info(f"Saved final state to {final_file}")
                
                early_stop = True
                break
        
        # Print and log progress occasionally
        if (i+1) % checkpoint_every == 0:
            accept_rate = accept_count / (i+1)
            elapsed_time = time.time() - start_time
            time_per_step = elapsed_time / (i+1)
            remaining_time = time_per_step * (n_steps - i - 1)
            
            log_message = (
                f"Step {i+1}/{n_steps} ({(i+1)/n_steps*100:.1f}%) - "
                f"Log-posterior: {current_logpost:.2f} - "
                f"Accept rate: {accept_rate:.2f} - "
                f"Elapsed: {elapsed_time:.1f}s - "
                f"Est. remaining: {remaining_time:.1f}s"
            )
            logger.info(log_message)
            
            # Also print current parameter values
            param_message = "Current parameters: " + ", ".join([f"{k}={params[k]:.6g}" for k in param_keys])
            logger.info(param_message)
            
            # Save checkpoint if requested
            if chain_file and (time.time() - last_checkpoint_time > 300):  # At most every 5 minutes
                checkpoint_data = {
                    'chain': chain[:i+1],
                    'log_posterior': log_posterior_values[:i+1],
                    'accept_count': accept_count,
                    'i': i,
                    'elapsed_time': elapsed_time
                }
                checkpoint_file = f"{chain_file}_checkpoint_{i+1}.npz"
                np.savez(checkpoint_file, **checkpoint_data)
                logger.info(f"Saved checkpoint to {checkpoint_file}")
                last_checkpoint_time = time.time()
    
    # Final acceptance rate and timing stats
    actual_steps = i + 1  # Account for early stopping
    final_accept_rate = accept_count / actual_steps
    total_time = time.time() - start_time
    
    if early_stop:
        logger.info(f"MCMC early stopped at step {actual_steps}/{n_steps} ({actual_steps/n_steps*100:.1f}%)")
        logger.info(f"Early stopping saved {n_steps - actual_steps} steps, approx. {(n_steps - actual_steps) * (total_time/actual_steps):.1f} seconds")
        # Trim arrays to actual size used
        chain = chain[:actual_steps]
        log_posterior_values = log_posterior_values[:actual_steps]
    else:
        logger.info(f"MCMC completed full {n_steps} steps")
    
    logger.info(f"Total runtime: {total_time:.2f} seconds")
    logger.info(f"Final acceptance rate: {final_accept_rate:.2f}")
    logger.info(f"Average time per step: {total_time/actual_steps:.4f} seconds")
    
    return chain, log_posterior_values

def run_multi_chain_mcmc(n_chains=4, n_steps=10000, burn_in_fraction=0.1, output_dir=None):
    """
    Run multiple MCMC chains in parallel from different starting points.
    
    Args:
        n_chains: Number of chains to run
        n_steps: Number of steps per chain
        burn_in_fraction: Fraction of initial samples to discard as burn-in
        output_dir: Directory to save chains and results (defaults to current directory)
        
    Returns:
        all_chains: Array of shape (n_chains, n_steps, n_params)
        trimmed_chains: Array with burn-in removed
    """
    # First ensure we have the real data loaded
    from data_loader import load_planck_data
    try:
        logger.info("Testing data loading to ensure FITS file is used...")
        ell, dl, sigma = load_planck_data(source="fits")
        logger.info(f"Successfully loaded {len(ell)} multipoles from FITS file")
    except Exception as e:
        logger.error(f"Error loading FITS data: {e}")
        logger.error("Please ensure the FITS file is available before running MCMC")
        raise
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcmc_results")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results will be saved to {output_dir}")
    
    # Initialize storage for all chains
    all_chains = np.zeros((n_chains, n_steps, len(param_names)))
    all_log_posts = np.zeros((n_chains, n_steps))
    
    logger.info(f"Running {n_chains} MCMC chains with {n_steps} steps each...")
    total_start_time = time.time()
    
    for i in range(n_chains):
        logger.info(f"Starting chain {i+1}/{n_chains}...")
        
        # Generate random initial parameters from specified ranges
        initial_params = {}
        for param in param_names:
            if param in param_ranges:
                min_val, max_val = param_ranges[param]
                # Random value within range
                initial_params[param] = min_val + np.random.rand() * (max_val - min_val)
            else:
                # Fall back to fiducial if no range specified
                initial_params[param] = fiducial_params[param]
        
        # Set proposal scales to ~5% of parameter range
        proposal_scale = {}
        for param in param_names:
            if param in param_ranges:
                min_val, max_val = param_ranges[param]
                proposal_scale[param] = 0.05 * (max_val - min_val)
            else:
                # Default for parameters without ranges
                proposal_scale[param] = 0.05 * abs(fiducial_params[param])
        
        logger.info(f"Initial parameters for chain {i+1}:")
        for param, value in initial_params.items():
            logger.info(f"  {param}: {value}")
        
        # Run the MCMC chain with checkpointing
        chain_file = os.path.join(output_dir, f"chain_{i+1}")
        checkpoint_every = max(100, min(1000, n_steps // 100))  # Adjust based on chain length
        
        chain, log_posts = run_mcmc(
            log_posterior_func=log_posterior, 
            initial_params=initial_params, 
            n_steps=n_steps, 
            proposal_scale=proposal_scale,
            checkpoint_every=checkpoint_every,
            chain_file=chain_file
        )
        
        # Store results
        all_chains[i] = chain
        all_log_posts[i] = log_posts
        
        # Save individual chain
        chain_outfile = os.path.join(output_dir, f"chain_{i+1}_full.npz")
        np.savez(
            chain_outfile, 
            chain=chain, 
            log_posterior=log_posts, 
            initial_params=initial_params,
            param_names=param_names
        )
        logger.info(f"Saved chain {i+1} to {chain_outfile}")
    
    total_time = time.time() - total_start_time
    logger.info(f"All chains completed in {total_time:.1f} seconds")
    
    # Calculate burn-in length (as fraction of chain length)
    burn_in = int(n_steps * burn_in_fraction)
    logger.info(f"Discarding first {burn_in} steps ({burn_in_fraction*100:.1f}%) as burn-in")
    
    # Create trimmed chains with burn-in removed
    trimmed_chains = all_chains[:, burn_in:, :]
    
    # Save all chains together
    all_chains_file = os.path.join(output_dir, "chains_lambdaCDM.npy")
    np.save(all_chains_file, all_chains)
    logger.info(f"Saved all chains to {all_chains_file}")
    
    return all_chains, trimmed_chains

def analyze_chains(all_chains, trimmed_chains, labels=None, output_dir=None):
    """
    Analyze MCMC chains: convergence, posterior distributions, etc.
    
    Args:
        all_chains: Complete chains (with burn-in)
        trimmed_chains: Chains with burn-in removed
        labels: Parameter labels for plotting
        output_dir: Directory to save analysis outputs
    """
    if labels is None:
        labels = param_names
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcmc_results")
    os.makedirs(output_dir, exist_ok=True)
    
    n_chains, n_steps, n_params = all_chains.shape
    logger.info(f"Analyzing {n_chains} chains with {n_steps} steps each")
    
    # Import analysis functions from analysis.py
    try:
        from analysis import plot_trace, plot_corner, gelman_rubin
    except ImportError:
        logger.error("Could not import analysis functions. Check that analysis.py exists.")
        raise
    
    # 1. Plot complete trace for all chains to check convergence
    logger.info("Plotting trace plots of all chains...")
    trace_plot_file = os.path.join(output_dir, "trace_plots.png")
    plot_trace(all_chains, labels, save_file=trace_plot_file)
    logger.info(f"Saved trace plots to {trace_plot_file}")
    
    # 2. Calculate Gelman-Rubin statistics
    logger.info("Calculating Gelman-Rubin statistics...")
    r_hat = gelman_rubin(trimmed_chains)
    
    logger.info("Gelman-Rubin R-hat values:")
    for i, name in enumerate(labels):
        converged = "Yes" if r_hat[i] < 1.1 else "No"
        logger.info(f"{name}: {r_hat[i]:.3f} (Converged: {converged})")
    
    # 3. Flatten chains for corner plot
    flat_samples = trimmed_chains.reshape((-1, n_params))
    
    # 4. Create corner plot
    logger.info("Creating corner plot...")
    corner_plot_file = os.path.join(output_dir, "corner_plot.png")
    plot_corner(flat_samples, labels, truths=fiducial_params, save_file=corner_plot_file)
    logger.info(f"Saved corner plot to {corner_plot_file}")
    
    # 5. Calculate parameter statistics
    logger.info("Parameter estimates (median and 68% credible interval):")
    
    # Create results array
    results = {
        'parameter': [],
        'median': [],
        'lower_68': [],
        'upper_68': [],
        'fiducial': [],
        'offset_pct': []
    }
    
    for i, name in enumerate(labels):
        mcmc_median = np.percentile(flat_samples[:, i], 50)
        mcmc_low = np.percentile(flat_samples[:, i], 16)
        mcmc_high = np.percentile(flat_samples[:, i], 84)
        fiducial = fiducial_params[name]
        offset_pct = (mcmc_median-fiducial)/fiducial*100 if fiducial != 0 else float('nan')
        
        logger.info(f"{name}: {mcmc_median:.6g} + {mcmc_high-mcmc_median:.6g} - {mcmc_median-mcmc_low:.6g}")
        logger.info(f"  (Fiducial: {fiducial:.6g}, Offset: {offset_pct:.1f}%)")
        
        # Store results
        results['parameter'].append(name)
        results['median'].append(mcmc_median)
        results['lower_68'].append(mcmc_low)
        results['upper_68'].append(mcmc_high)
        results['fiducial'].append(fiducial)
        results['offset_pct'].append(offset_pct)
    
    # Save results to CSV
    try:
        import pandas as pd
        results_df = pd.DataFrame(results)
        results_file = os.path.join(output_dir, "parameter_constraints.csv")
        results_df.to_csv(results_file, index=False)
        logger.info(f"Saved parameter constraints to {results_file}")
    except ImportError:
        logger.warning("pandas not available, skipping CSV export of results")
    
    # Save flat samples
    flat_samples_file = os.path.join(output_dir, "posterior_samples_lambdaCDM.npy")
    np.save(flat_samples_file, flat_samples)
    logger.info(f"Saved posterior samples to {flat_samples_file}")
    
    return flat_samples, results

if __name__ == "__main__":
    # Create output directory for MCMC results
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcmc_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure file logging for this run
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_log_file = os.path.join(output_dir, f"mcmc_run_{run_timestamp}.log")
    file_handler = logging.FileHandler(run_log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting MCMC analysis run at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Results will be saved to {output_dir}")
    
    # Run MCMC with 4 chains, 30000 steps each (for larger dataset), and 20% burn-in
    all_chains, trimmed_chains = run_multi_chain_mcmc(
        n_chains=4, 
        n_steps=30000, 
        burn_in_fraction=0.2,
        output_dir=output_dir
    )
    
    # Analyze the chains
    flat_samples, results = analyze_chains(
        all_chains, 
        trimmed_chains,
        output_dir=output_dir
    )
    
    logger.info("MCMC analysis complete")
    print(f"MCMC analysis complete. Results saved to {output_dir}")
    print(f"Log file: {run_log_file}")
    
    # Create a README file in the output directory
    readme_content = f"""# MCMC Analysis Results

Run Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Files
- chains_lambdaCDM.npy: Raw MCMC chains for all parameters
- posterior_samples_lambdaCDM.npy: Flattened posterior samples after burn-in
- parameter_constraints.csv: Summary statistics for all parameters
- trace_plots.png: Trace plots for visual convergence inspection
- corner_plot.png: Corner plot showing parameter distributions and correlations
- mcmc_run_{run_timestamp}.log: Detailed log of this MCMC run

## Run Configuration
- Number of chains: 4
- Steps per chain: 30000
- Burn-in fraction: 20%
- Data source: Planck FITS file

## Parameter Constraints
"""
    
    # Add parameter results to README
    for i, param in enumerate(results['parameter']):
        median = results['median'][i]
        lower = results['lower_68'][i]
        upper = results['upper_68'][i]
        fiducial = results['fiducial'][i]
        offset = results['offset_pct'][i]
        
        readme_content += f"- {param}: {median:.6g} + {upper-median:.6g} - {median-lower:.6g}"
        readme_content += f" (Fiducial: {fiducial:.6g}, Offset: {offset:.1f}%)\n"
    
    # Write README file
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme_content)
    
    logger.info(f"Created README.md in {output_dir}")
    print(f"Summary README created: {os.path.join(output_dir, 'README.md')}")