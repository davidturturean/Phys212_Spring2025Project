#!/usr/bin/env python
"""
Script to analyze standard (non-annealing) MCMC chains and compute statistics
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive environments
import matplotlib.pyplot as plt
import os
import argparse
import glob
import pandas as pd

def load_chain_file(file):
    """
    Load a single chain file with error handling
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
        
        # Print available keys to help with debugging
        print(f"Available keys in {file}: {list(data.keys())}")
        
        result = {
            'chain': chain,
            'log_prob': log_prob,
            'file': file
        }
        print(f"Loaded chain from {file} with shape {chain.shape}")
        return result
    except Exception as e:
        print(f"Error loading {file}: {e}")
        return None

def load_standard_chains(directories=None, burn_in_fraction=0.2):
    """
    Load standard MCMC chains from specified directories
    Apply burn-in and return cleaned chains
    """
    chain_files = []
    
    if directories:
        for directory in directories:
            pattern = os.path.join(directory, "**", "chain_*_full.npz")
            files = glob.glob(pattern, recursive=True)
            chain_files.extend(files)
    else:
        # Search for standard chains in common directories
        patterns = [
            "./mcmc_results/chain_*_full.npz",
            "./mcmc_results_merged/**/merged_standard_chain.npy",
            "./mcmc_results_final/**/chain_*_full.npz"
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
    
    print(f"Found {len(chain_files)} standard chain files")
    for file in chain_files:
        print(f"  - {file}")
    
    if chain_files:
        # Load chains one by one
        all_chains = []
        for file in chain_files:
            chain_data = load_chain_file(file)
            if chain_data is not None:
                # Apply burn-in
                burn_in = int(burn_in_fraction * len(chain_data['chain']))
                chain_data['chain'] = chain_data['chain'][burn_in:]
                chain_data['log_prob'] = chain_data['log_prob'][burn_in:] if len(chain_data['log_prob']) > burn_in else chain_data['log_prob']
                
                all_chains.append(chain_data)
    else:
        all_chains = []
    
    return all_chains

def compute_statistics(chains, param_names=None):
    """
    Compute statistics for parameters across all chains
    """
    if param_names is None:
        param_names = ['H0', 'Omega_b_h2', 'Omega_c_h2', 'n_s', 'A_s', 'tau']
    
    stats_dict = {}
    
    # Combine all chains
    all_samples = np.vstack([chain_data['chain'] for chain_data in chains])
    
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

def plot_trace(chains, param_names, output_file):
    """
    Create trace plots for all parameters
    """
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 1, figsize=(12, 3*n_params), sharex=True)
    
    colors = plt.cm.tab10.colors
    
    for i, param in enumerate(param_names):
        ax = axes[i] if n_params > 1 else axes
        
        for j, chain in enumerate(chains):
            color = colors[j % len(colors)]
            ax.plot(chain[:, i], alpha=0.7, color=color, label=f"Chain {j+1}")
        
        ax.set_ylabel(param)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Step')
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Saved trace plots to {output_file}")

def compare_with_annealing(standard_stats, annealing_stats, param_names, output_dir):
    """
    Compare standard MCMC results with annealing MCMC results
    """
    if annealing_stats is None:
        print("No annealing results available for comparison.")
        return
    
    # Create comparison table
    comparison_data = []
    
    for param in param_names:
        std_median = standard_stats[param]['median']
        std_err_p = standard_stats[param]['error_plus']
        std_err_m = standard_stats[param]['error_minus']
        
        ann_median = annealing_stats[param]['median']
        ann_err_p = annealing_stats[param]['error_plus']
        ann_err_m = annealing_stats[param]['error_minus']
        
        # Calculate difference in sigma units
        diff = abs(std_median - ann_median)
        combined_sigma = np.sqrt(std_err_p * std_err_m + ann_err_p * ann_err_m)
        sigma_diff = diff / combined_sigma if combined_sigma > 0 else 0
        
        comparison_data.append({
            'Parameter': param,
            'Standard': f"{std_median:.6g} + {std_err_p:.6g} - {std_err_m:.6g}",
            'Annealing': f"{ann_median:.6g} + {ann_err_p:.6g} - {ann_err_m:.6g}",
            'Diff': f"{diff:.6g}",
            'Sigma': f"{sigma_diff:.2f}σ"
        })
    
    # Save as CSV
    df = pd.DataFrame(comparison_data)
    df.to_csv(os.path.join(output_dir, 'standard_vs_annealing.csv'), index=False)
    
    # Create comparison plots
    fig, axes = plt.subplots(len(param_names), 1, figsize=(10, 3*len(param_names)))
    
    for i, param in enumerate(param_names):
        ax = axes[i] if len(param_names) > 1 else axes
        
        # Standard MCMC
        std_median = standard_stats[param]['median']
        std_err_p = standard_stats[param]['error_plus']
        std_err_m = standard_stats[param]['error_minus']
        
        # Annealing MCMC
        ann_median = annealing_stats[param]['median']
        ann_err_p = annealing_stats[param]['error_plus']
        ann_err_m = annealing_stats[param]['error_minus']
        
        # Plot values
        ax.errorbar(0, std_median, yerr=[[std_err_m], [std_err_p]], 
                   fmt='o', capsize=5, color='blue', label='Standard MCMC')
        ax.errorbar(1, ann_median, yerr=[[ann_err_m], [ann_err_p]], 
                   fmt='o', capsize=5, color='red', label='Annealing MCMC')
        
        ax.set_ylabel(param)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Standard', 'Annealing'])
        ax.grid(True, alpha=0.3)
    
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'standard_vs_annealing_comparison.png'), dpi=300)
    plt.close(fig)
    
    # Create markdown report
    with open(os.path.join(output_dir, 'standard_vs_annealing_comparison.md'), 'w') as f:
        f.write("# Comparison of Standard vs. Annealing MCMC Results\n\n")
        
        f.write("## Parameter Constraints\n\n")
        f.write("| Parameter | Standard MCMC | Annealing MCMC | Difference | Significance |\n")
        f.write("|-----------|--------------|----------------|------------|-------------|\n")
        
        for row in comparison_data:
            f.write(f"| {row['Parameter']} | {row['Standard']} | {row['Annealing']} | {row['Diff']} | {row['Sigma']} |\n")
        
        f.write("\n## Interpretation\n\n")
        f.write("This comparison shows how the standard MCMC and temperature annealing MCMC methods differ in their parameter estimates. ")
        f.write("A difference of less than 1σ is statistically consistent, while larger differences may indicate ")
        f.write("that one method is exploring the parameter space more effectively than the other.\n\n")
        
        f.write("### Advantages of Annealing MCMC\n\n")
        f.write("- Better at escaping local minima\n")
        f.write("- More thorough exploration of parameter space\n")
        f.write("- Less sensitive to initial conditions\n")
        f.write("- Usually provides more robust uncertainty estimates\n\n")
        
        f.write("### Advantages of Standard MCMC\n\n")
        f.write("- Computationally more efficient\n")
        f.write("- Simpler implementation\n")
        f.write("- Direct sampling from the posterior\n")
        f.write("- No temperature parameter to tune\n")
    
    print(f"Comparison saved to {output_dir}/standard_vs_annealing_comparison.md")

def load_annealing_results(annealing_dir):
    """
    Load existing annealing results for comparison
    """
    try:
        # Try to load from final_steps_parameter_constraints.csv first
        constraints_file = os.path.join(annealing_dir, 'final_steps_parameter_constraints.csv')
        if os.path.exists(constraints_file):
            df = pd.read_csv(constraints_file, index_col=0)
            
            stats_dict = {}
            for param, row in df.iterrows():
                stats_dict[param] = {
                    'median': row['median'],
                    'lower': row['lower'],
                    'upper': row['upper'],
                    'error_minus': row['error_minus'],
                    'error_plus': row['error_plus']
                }
            
            print(f"Loaded annealing results from {constraints_file}")
            return stats_dict
        
        # Try alternative file formats
        constraints_file = os.path.join(annealing_dir, 'annealing_parameter_constraints.csv')
        if os.path.exists(constraints_file):
            df = pd.read_csv(constraints_file, index_col=0)
            
            stats_dict = {}
            for param, row in df.iterrows():
                stats_dict[param] = {
                    'median': row['median'],
                    'lower': row['lower'],
                    'upper': row['upper'],
                    'error_minus': row['error_minus'],
                    'error_plus': row['error_plus']
                }
            
            print(f"Loaded annealing results from {constraints_file}")
            return stats_dict
        
        print("No suitable annealing results found for comparison")
        return None
    
    except Exception as e:
        print(f"Error loading annealing results: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Analyze standard MCMC chains")
    parser.add_argument("--burn-in", type=float, default=0.2, 
                      help="Burn-in fraction to discard (default: 0.2)")
    parser.add_argument("--output-dir", default="./standard_chains_analysis", 
                      help="Output directory for results")
    parser.add_argument("--annealing-dir", default="./annealing_analysis_final_steps",
                      help="Directory with annealing results for comparison")
    
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load standard chains
    print(f"Loading standard MCMC chains with burn-in {args.burn_in}...")
    chains_data = load_standard_chains(burn_in_fraction=args.burn_in)
    
    if not chains_data:
        print("Error: No standard chains found!")
        return
    
    # Extract chain arrays for analysis
    chains = [chain_data['chain'] for chain_data in chains_data]
    param_names = ['H0', 'Omega_b_h2', 'Omega_c_h2', 'n_s', 'A_s', 'tau']
    
    # Calculate statistics
    print("\nParameter statistics for standard chains:")
    stats_dict, combined_samples = compute_statistics(chains_data, param_names)
    
    # Save parameter constraints to CSV
    df = pd.DataFrame.from_dict(stats_dict, orient='index')
    df.to_csv(os.path.join(args.output_dir, 'standard_parameter_constraints.csv'))
    print(f"Saved parameter constraints to {os.path.join(args.output_dir, 'standard_parameter_constraints.csv')}")
    
    # Create corner plot
    plot_corner(combined_samples, param_names, os.path.join(args.output_dir, 'standard_corner_plot.png'))
    
    # Create trace plots
    plot_trace(chains, param_names, os.path.join(args.output_dir, 'standard_trace_plots.png'))
    
    # Calculate Gelman-Rubin statistic if multiple chains
    if len(chains) >= 2:
        print("\nCalculating Gelman-Rubin statistics for standard chains...")
        r_hat, converged = gelman_rubin(chains, param_names)
        
        # Save to file
        with open(os.path.join(args.output_dir, 'standard_gelman_rubin.txt'), 'w') as f:
            f.write("Gelman-Rubin Statistics for Standard MCMC Chains\n")
            f.write("=" * 80 + "\n\n")
            
            for param in param_names:
                convergence = "Yes" if converged[param] else "No"
                f.write(f"{param}: R-hat = {r_hat[param]:.6f} (Converged: {convergence})\n")
        
        # Print results
        print("\nGelman-Rubin R-hat values for standard chains:")
        for param in param_names:
            convergence = "Yes" if converged[param] else "No"
            print(f"{param}: {r_hat[param]:.6f} (Converged: {convergence})")
    else:
        print("\nNot enough chains for Gelman-Rubin diagnostic")
    
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
    
    # Load annealing results for comparison
    annealing_stats = load_annealing_results(args.annealing_dir)
    
    # Compare with annealing results
    if annealing_stats:
        compare_with_annealing(stats_dict, annealing_stats, param_names, args.output_dir)
    
    # Create summary file
    with open(os.path.join(args.output_dir, 'standard_chains_summary.md'), 'w') as f:
        f.write("# Analysis of Standard MCMC Chains\n\n")
        
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
        
        # Add Gelman-Rubin results if available
        if len(chains) >= 2:
            f.write("\n## Gelman-Rubin Convergence Statistics\n\n")
            f.write("| Parameter | R-hat | Converged? |\n")
            f.write("|-----------|-------|------------|\n")
            
            for param in param_names:
                convergence = "Yes" if converged[param] else "No"
                f.write(f"| {param} | {r_hat[param]:.6f} | {convergence} |\n")
        
        f.write("\n## Analysis Details\n\n")
        f.write(f"- Number of chains analyzed: {len(chains)}\n")
        f.write(f"- Total number of samples: {len(combined_samples)}\n")
        f.write(f"- Burn-in fraction discarded: {args.burn_in * 100}%\n")
        f.write(f"- Chain files analyzed: {len(chains_data)}\n")
    
    print(f"\nAll results saved to {args.output_dir}")
    print(f"See {os.path.join(args.output_dir, 'standard_chains_summary.md')} for a full summary")

if __name__ == "__main__":
    main()