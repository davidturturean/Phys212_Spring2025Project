#!/usr/bin/env python
"""
Script to analyze just the final steps of annealing chains and compute Gelman-Rubin statistics
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive environments
import matplotlib.pyplot as plt
import os
import argparse
import glob

def load_chain_file(file, use_final_steps_only=False, n_steps=10800):
    """
    Load a single chain file with error handling
    Optionally, extract only the final n_steps
    """
    try:
        data = np.load(file, allow_pickle=True)
        
        # Get chain data - may be stored as 'chain' or 'samples'
        if 'chain' in data:
            chain = data['chain']
        elif 'samples' in data:
            chain = data['samples']
        else:
            raise KeyError("Neither 'chain' nor 'samples' found in archive")
        
        # Get log probability - may be stored as 'log_prob', 'log_probs', or 'log_posterior'
        if 'log_prob' in data:
            log_prob = data['log_prob']
        elif 'log_probs' in data:
            log_prob = data['log_probs']
        elif 'log_posterior' in data:
            log_prob = data['log_posterior']
        else:
            # If we can't find log probabilities, create dummy values
            print(f"Warning: No log probability data found in {file}, using dummy values")
            log_prob = np.zeros(len(chain))
        
        # Get temperatures if available
        temperatures = data.get('temperatures', None)
        if temperatures is None:
            temperatures = data.get('temperature_schedule', None)
        
        # Print available keys to help with debugging
        print(f"Available keys in {file}: {list(data.keys())}")
        
        # Extract only the final n_steps if requested
        if use_final_steps_only and len(chain) > n_steps:
            chain = chain[-n_steps:]
            if len(log_prob) > n_steps:
                log_prob = log_prob[-n_steps:]
            if temperatures is not None and len(temperatures) > n_steps:
                temperatures = temperatures[-n_steps:]
            print(f"Using only the final {n_steps} steps from {file}")
        
        result = {
            'chain': chain,
            'log_prob': log_prob,
            'temperatures': temperatures,
            'file': file
        }
        print(f"Loaded chain from {file} with shape {chain.shape}")
        return result
    except Exception as e:
        print(f"Error loading {file}: {e}")
        return None

def load_annealing_chains(directories=None, use_final_steps_only=False, n_steps=10800):
    """
    Load annealing chains from specified directories
    Optionally, extract only the final n_steps
    """
    chain_files = []
    
    if directories:
        for directory in directories:
            pattern = os.path.join(directory, "**", "*annealing*final*.npz")
            files = glob.glob(pattern, recursive=True)
            chain_files.extend(files)
    else:
        # Search for annealing chains in common directories
        patterns = [
            "./mcmc_results/**/annealing*final*.npz",
            "./mcmc_results_annealing*/**/*annealing*final*.npz",
            "./mcmc_results_merged/**/*annealing*final*.npz",
            "./mcmc_results/**/continued_annealing*final*.npz",
            "./mcmc_results_annealing*/**/continued_annealing*final*.npz",
            "./mcmc_results_merged/**/continued_annealing*final*.npz"
        ]
        
        for pattern in patterns:
            files = glob.glob(pattern, recursive=True)
            chain_files.extend(files)
    
    # Remove duplicates while preserving order
    unique_files = []
    seen = set()
    for file in chain_files:
        if file not in seen:
            seen.add(file)
            unique_files.append(file)
    chain_files = unique_files
    
    print(f"Found {len(chain_files)} annealing chain files")
    for file in chain_files:
        print(f"  - {file}")
    
    if chain_files:
        # Load chains one by one
        all_chains = []
        for file in chain_files:
            chain = load_chain_file(file, use_final_steps_only, n_steps)
            if chain is not None:
                all_chains.append(chain)
    else:
        all_chains = []
    
    return all_chains

def gelman_rubin(chains, param_names=None):
    """
    Calculate Gelman-Rubin statistic for assessing chain convergence.
    """
    if param_names is None:
        param_names = ['H0', 'Omega_b_h2', 'Omega_c_h2', 'n_s', 'A_s', 'tau']
    
    n_chains = len(chains)
    if n_chains < 2:
        print("Warning: At least 2 chains are required for the Gelman-Rubin diagnostic")
        return None, None
    
    # Get lengths and number of parameters
    chain_lengths = [len(chain) for chain in chains]
    min_length = min(chain_lengths)
    n_params = chains[0].shape[1]
    
    # Truncate chains to the same length
    chains_equal = [chain[:min_length] for chain in chains]
    
    # Initialize results
    r_hat = {}
    converged = {}
    
    # Calculate Gelman-Rubin statistic for each parameter
    for i, param in enumerate(param_names):
        # Extract parameter values from each chain
        param_chains = [chain[:, i] for chain in chains_equal]
        
        # Calculate within-chain variance W
        within_chain_var = np.mean([np.var(chain, ddof=1) for chain in param_chains])
        
        # Calculate between-chain variance B
        chain_means = [np.mean(chain) for chain in param_chains]
        overall_mean = np.mean(chain_means)
        between_chain_var = min_length * np.var(chain_means, ddof=1)
        
        # Calculate pooled variance estimate
        var_pooled = ((min_length - 1) / min_length) * within_chain_var + (1 / min_length) * between_chain_var
        
        # Calculate potential scale reduction factor
        r = np.sqrt(var_pooled / within_chain_var)
        r_hat[param] = r
        converged[param] = r < 1.1
    
    return r_hat, converged

def compute_statistics(chains, param_names=None):
    """
    Compute statistics for parameters across all chains
    """
    if param_names is None:
        param_names = ['H0', 'Omega_b_h2', 'Omega_c_h2', 'n_s', 'A_s', 'tau']
    
    stats_dict = {}
    
    # Combine all chains
    all_samples = np.vstack(chains)
    
    for i, param in enumerate(param_names):
        # Extract parameter values
        values = all_samples[:, i]
        
        # Compute median and percentiles
        median = np.median(values)
        lower = np.percentile(values, 16)
        upper = np.percentile(values, 84)
        
        stats_dict[param] = {
            'median': median,
            'lower': lower,
            'upper': upper,
            'error_minus': median - lower,
            'error_plus': upper - median
        }
        
        print(f"{param}: {median:.6g} + {upper-median:.6g} - {median-lower:.6g} (68% CI)")
    
    return stats_dict, all_samples

def plot_corner(samples, param_names, output_file):
    """
    Create corner plot for the parameters
    """
    try:
        import corner
        # Use actual corner package if available
        fig = corner.corner(
            samples,
            labels=param_names,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            title_fmt='.4g',
            use_math_text=True
        )
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
    except ImportError:
        # Simplified version using matplotlib
        print("WARNING: Using simplified corner plot implementation. Install corner package for better plots.")
        n_params = len(param_names)
        fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
        
        # If only one parameter, make axes subscriptable
        if n_params == 1:
            axes = np.array([[axes]])
        
        for i in range(n_params):
            for j in range(n_params):
                if i > j:
                    # 2D histogram for parameter pairs
                    axes[i, j].hist2d(
                        samples[:, j], samples[:, i], 
                        bins=50, cmap='Blues', density=True
                    )
                    axes[i, j].set_xlabel(param_names[j])
                    axes[i, j].set_ylabel(param_names[i])
                elif i == j:
                    # Histogram for single parameter
                    axes[i, i].hist(samples[:, i], bins=50, density=True)
                    axes[i, i].set_xlabel(param_names[i])
                    axes[i, i].set_ylabel('Density')
                else:
                    # Remove upper triangle
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Saved corner plot to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze final steps of annealing chains")
    parser.add_argument("--n-steps", type=int, default=10800, 
                        help="Number of final steps to use (default: 10800)")
    parser.add_argument("--output-dir", default="./annealing_analysis_final_steps", 
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load chains with only the final steps
    print(f"Loading chains, extracting final {args.n_steps} steps...")
    chains_data = load_annealing_chains(use_final_steps_only=True, n_steps=args.n_steps)
    
    if len(chains_data) < 2:
        print("Error: At least 2 chains are required for Gelman-Rubin diagnostic")
        return
    
    # Extract the actual chain arrays
    chains = [chain_data['chain'] for chain_data in chains_data]
    param_names = ['H0', 'Omega_b_h2', 'Omega_c_h2', 'n_s', 'A_s', 'tau']
    
    # Calculate statistics on final steps
    print(f"\nParameter statistics for final {args.n_steps} steps:")
    stats_dict, combined_samples = compute_statistics(chains, param_names)
    
    # Save parameter constraints to CSV
    import pandas as pd
    df = pd.DataFrame.from_dict(stats_dict, orient='index')
    df.to_csv(os.path.join(args.output_dir, 'final_steps_parameter_constraints.csv'))
    print(f"Saved parameter constraints to {os.path.join(args.output_dir, 'final_steps_parameter_constraints.csv')}")
    
    # Create corner plot from final steps
    plot_corner(combined_samples, param_names, os.path.join(args.output_dir, 'final_steps_corner_plot.png'))
    
    # Compare with Planck values
    fiducial_values = {
        'H0': 67.36,
        'Omega_b_h2': 0.02237,
        'Omega_c_h2': 0.1200,
        'n_s': 0.9649,
        'A_s': 2.1e-9,
        'tau': 0.0544
    }
    
    # Print comparison with fiducial values
    print("\nComparison with Planck 2018 values:")
    for param, values in stats_dict.items():
        fiducial = fiducial_values.get(param, None)
        if fiducial:
            offset_percent = 100 * (values['median'] - fiducial) / fiducial
            print(f"{param}: {values['median']:.6g} (Fiducial: {fiducial}, Offset: {offset_percent:.1f}%)")
    
    # Calculate Gelman-Rubin statistic
    print("\nCalculating Gelman-Rubin statistics on final steps...")
    r_hat, converged = gelman_rubin(chains, param_names)
    
    # Save results to file
    with open(os.path.join(args.output_dir, 'gelman_rubin_final_steps.txt'), 'w') as f:
        f.write(f"Gelman-Rubin Statistics (Final {args.n_steps} Steps Only)\n")
        f.write("=" * 80 + "\n\n")
        
        for param in param_names:
            convergence = "Yes" if converged[param] else "No"
            f.write(f"{param}: R-hat = {r_hat[param]:.6f} (Converged: {convergence})\n")
    
    # Print results
    print("\nGelman-Rubin R-hat values (Final steps only):")
    for param in param_names:
        convergence = "Yes" if converged[param] else "No"
        print(f"{param}: {r_hat[param]:.6f} (Converged: {convergence})")
    
    # Create trace plots of final steps
    fig, axes = plt.subplots(len(param_names), 1, figsize=(12, 3*len(param_names)), sharex=True)
    
    colors = plt.cm.tab10.colors
    
    for i, param in enumerate(param_names):
        ax = axes[i] if len(param_names) > 1 else axes
        
        for j, chain in enumerate(chains):
            color = colors[j % len(colors)]
            ax.plot(chain[:, i], alpha=0.7, color=color, label=f"Chain {j+1}")
        
        ax.set_ylabel(param)
        ax.grid(True, alpha=0.3)
        
        # Add R-hat value to plot
        ax.text(0.98, 0.95, f"R-hat = {r_hat[param]:.4f}", 
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Add median value to plot
        median = stats_dict[param]['median']
        ax.axhline(median, color='k', linestyle='--', alpha=0.5)
        ax.text(0.98, 0.85, f"Median = {median:.4g}", 
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.8))
    
    axes[-1].set_xlabel('Step (within final portion)')
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'final_steps_trace_plots.png'), dpi=300)
    plt.close(fig)
    
    # Create summary file
    with open(os.path.join(args.output_dir, 'final_steps_summary.md'), 'w') as f:
        f.write(f"# Analysis of Final {args.n_steps} Steps of Annealing Chains\n\n")
        
        f.write("## Parameter Constraints\n\n")
        f.write("| Parameter | Value | 68% CI Lower | 68% CI Upper | Planck 2018 | Offset (%) |\n")
        f.write("|-----------|-------|-------------|-------------|------------|------------|\n")
        
        for param, values in stats_dict.items():
            fiducial = fiducial_values.get(param, None)
            if fiducial:
                offset_percent = 100 * (values['median'] - fiducial) / fiducial
                f.write(f"| {param} | {values['median']:.6g} | -{values['error_minus']:.6g} | +{values['error_plus']:.6g} | {fiducial} | {offset_percent:.1f}% |\n")
            else:
                f.write(f"| {param} | {values['median']:.6g} | -{values['error_minus']:.6g} | +{values['error_plus']:.6g} | - | - |\n")
        
        f.write("\n## Gelman-Rubin Convergence Statistics\n\n")
        f.write("| Parameter | R-hat | Converged? |\n")
        f.write("|-----------|-------|------------|\n")
        
        for param in param_names:
            convergence = "Yes" if converged[param] else "No"
            f.write(f"| {param} | {r_hat[param]:.6f} | {convergence} |\n")
    
    print(f"\nAll results saved to {args.output_dir}")
    print(f"See {os.path.join(args.output_dir, 'final_steps_summary.md')} for a full summary")

if __name__ == "__main__":
    main()