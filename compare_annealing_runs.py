import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive environments
import matplotlib.pyplot as plt
import os
import argparse
import glob
from analyze_annealing_chains import load_annealing_chains, compute_statistics, plot_corner
import numpy as np
from scipy import stats

def gelman_rubin(chains, param_names=None):
    """
    Calculate Gelman-Rubin statistic for assessing chain convergence.
    
    Parameters:
    -----------
    chains : list of numpy arrays
        MCMC chains to analyze, each of shape (n_steps, n_params)
    param_names : list of str
        Names of parameters
    
    Returns:
    --------
    r_hat : dict
        Gelman-Rubin statistic for each parameter
    converged : dict
        Boolean indicating whether each parameter has converged (R < 1.1)
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

def compare_annealing_runs(run_dirs, labels=None, burn_in_fraction=0.2, output_dir='./annealing_comparison'):
    """
    Compare statistics from multiple annealing runs
    
    Parameters:
    -----------
    run_dirs : list of str
        Directories containing the annealing chains
    labels : list of str
        Labels for each run
    burn_in_fraction : float
        Fraction of samples to discard as burn-in
    output_dir : str
        Directory to save results
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(run_dirs))]
    
    param_names = ['H0', 'Omega_b_h2', 'Omega_c_h2', 'n_s', 'A_s', 'tau']
    
    # Process each run directory
    all_stats = []
    all_samples = []
    all_chains = []
    
    for i, directory in enumerate(run_dirs):
        print(f"\nProcessing {labels[i]} in {directory}")
        chains = load_annealing_chains([directory])
        
        if not chains:
            print(f"No chains found in {directory}, skipping")
            continue
        
        stats, samples = compute_statistics(chains, burn_in_fraction, param_names)
        all_stats.append(stats)
        all_samples.append(samples)
        
        # Store raw chains for Gelman-Rubin calculation
        for chain_data in chains:
            all_chains.append(chain_data['chain'][int(burn_in_fraction * len(chain_data['chain'])):])
        
        # Create individual corner plot
        output_file = os.path.join(output_dir, f'corner_plot_{labels[i].replace(" ", "_")}.png')
        plot_corner(samples, param_names, output_file)
        
    # Calculate Gelman-Rubin statistic across all chains from all runs
    if len(all_chains) >= 2:
        print("\nCalculating Gelman-Rubin statistics across all runs...")
        r_hat, converged = gelman_rubin(all_chains, param_names)
        
        with open(os.path.join(output_dir, 'gelman_rubin_stats.txt'), 'w') as f:
            f.write("Gelman-Rubin Statistics Across All Runs\n")
            f.write("=====================================\n\n")
            
            for param in param_names:
                convergence = "Yes" if converged[param] else "No"
                f.write(f"{param}: R-hat = {r_hat[param]:.4f} (Converged: {convergence})\n")
        
        print("Gelman-Rubin R-hat values:")
        for param in param_names:
            convergence = "Yes" if converged[param] else "No"
            print(f"{param}: {r_hat[param]:.4f} (Converged: {convergence})")
    else:
        print("\nNot enough chains for Gelman-Rubin diagnostic across runs")
    
    # Check if we have any stats to compare
    if not all_stats:
        print("No stats to compare. Skipping comparison plots.")
        return [], []
    
    # Single run case
    if len(all_stats) == 1:
        print("Only one run found. Creating parameter summary instead of comparison.")
        # Create summary table
        with open(os.path.join(output_dir, 'parameter_summary.txt'), 'w') as f:
            f.write("Parameter Summary\n")
            f.write("=" * 80 + "\n\n")
            
            for param in param_names:
                param_stats = all_stats[0][param]
                f.write(f"{param}: {param_stats['median']:.6g} + {param_stats['error_plus']:.6g} - {param_stats['error_minus']:.6g}\n")
                
                if param in fiducial_values:
                    f.write(f"  Fiducial: {fiducial_values[param]}\n")
                f.write("\n")
        
        return all_stats, all_samples
    
    # Compare parameter constraints
    fig, axes = plt.subplots(len(param_names), 1, figsize=(10, 3*len(param_names)))
    
    fiducial_values = {
        'H0': 67.36,
        'Omega_b_h2': 0.02237,
        'Omega_c_h2': 0.1200,
        'n_s': 0.9649,
        'A_s': 2.1e-9,
        'tau': 0.0544
    }
    
    colors = plt.cm.tab10.colors
    
    for i, param in enumerate(param_names):
        ax = axes[i] if len(param_names) > 1 else axes
        
        # Set x ticks before plotting
        x_positions = list(range(len(all_stats)))
        ax.set_xticks(x_positions)
        
        for j, stats in enumerate(all_stats):
            param_stats = stats[param]
            median = param_stats['median']
            lower = param_stats['median'] - param_stats['error_minus']
            upper = param_stats['median'] + param_stats['error_plus']
            
            ax.errorbar(
                j, median, yerr=[[median-lower], [upper-median]],
                fmt='o', capsize=5, color=colors[j % len(colors)],
                label=labels[j]
            )
        
        # Add fiducial value
        if param in fiducial_values:
            ax.axhline(fiducial_values[param], color='k', linestyle='--', alpha=0.7, label='Fiducial')
        
        ax.set_ylabel(param)
        # Now set the labels after setting ticks
        ax.set_xticklabels(labels, rotation=45)
        ax.grid(True, alpha=0.3)
    
    # Add legend to the first subplot
    if len(param_names) > 1:
        axes[0].legend(loc='upper right')
    else:
        axes.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_comparison.png'), dpi=300)
    plt.close()
    
    # Create comparison table
    print("\nParameter Comparison Table:")
    print("-" * 80)
    print(f"{'Parameter':<15} " + " ".join([f"{label:<20}" for label in labels]))
    print("-" * 80)
    
    for param in param_names:
        values_str = []
        for stats in all_stats:
            param_stats = stats[param]
            value_str = f"{param_stats['median']:.6g} ± {param_stats['error_plus']:.6g}"
            values_str.append(value_str)
        
        print(f"{param:<15} " + " ".join([f"{value:<20}" for value in values_str]))
    
    print("-" * 80)
    
    # Save comparison summary
    with open(os.path.join(output_dir, 'comparison_summary.txt'), 'w') as f:
        f.write("Parameter Comparison Summary\n")
        f.write("=" * 80 + "\n\n")
        
        for param in param_names:
            f.write(f"{param}:\n")
            for i, stats in enumerate(all_stats):
                param_stats = stats[param]
                f.write(f"  {labels[i]}: {param_stats['median']:.6g} + {param_stats['error_plus']:.6g} - {param_stats['error_minus']:.6g}\n")
            
            if param in fiducial_values:
                f.write(f"  Fiducial: {fiducial_values[param]}\n")
            
            f.write("\n")
    
    print(f"Comparison results saved to {output_dir}")
    
    return all_stats, all_samples

def find_annealing_run_dirs():
    """
    Automatically find directories that contain annealing runs
    """
    potential_dirs = [
        "./mcmc_results",
        "./mcmc_results_annealing*",
        "./mcmc_results_merged"
    ]
    
    run_dirs = []
    
    for pattern in potential_dirs:
        dirs = glob.glob(pattern)
        for d in dirs:
            if os.path.isdir(d):
                # Check if directory contains annealing chains
                pattern = os.path.join(d, "**", "*annealing*final.npz")
                files = glob.glob(pattern, recursive=True)
                if files:
                    run_dirs.append(d)
    
    return run_dirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare multiple annealing MCMC runs")
    parser.add_argument("--directories", nargs="+", help="Directories to compare")
    parser.add_argument("--labels", nargs="+", help="Labels for each directory")
    parser.add_argument("--burn-in", type=float, default=0.2, help="Burn-in fraction (default: 0.2)")
    parser.add_argument("--output-dir", default="./annealing_comparison", help="Output directory for comparison results")
    parser.add_argument("--auto-find", action="store_true", help="Automatically find annealing run directories")
    
    args = parser.parse_args()
    
    if args.auto_find:
        run_dirs = find_annealing_run_dirs()
        labels = [os.path.basename(d) for d in run_dirs]
        print(f"Found {len(run_dirs)} annealing run directories:")
        for d in run_dirs:
            print(f"  - {d}")
    else:
        run_dirs = args.directories
        labels = args.labels
    
    if not run_dirs:
        print("No annealing run directories specified or found!")
        exit(1)
    
    if labels and len(labels) != len(run_dirs):
        print("Warning: Number of labels doesn't match number of directories!")
        labels = [f"Run {i+1}" for i in range(len(run_dirs))]
    
    compare_annealing_runs(
        run_dirs=run_dirs,
        labels=labels,
        burn_in_fraction=args.burn_in,
        output_dir=args.output_dir
    )