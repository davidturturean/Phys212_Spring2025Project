import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive environments
import matplotlib.pyplot as plt
import os
import corner
import glob
import argparse
from scipy import stats
import pandas as pd
from multiprocessing import Pool
import multiprocessing

def load_chain_file(file):
    """
    Load a single chain file with error handling
    Handle different key names in the archive
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

def load_annealing_chains(directories=None):
    """
    Load all annealing chains from specified directories or search for them
    Using parallel processing for faster loading
    """
    chain_files = []
    
    if directories:
        for directory in directories:
            pattern = os.path.join(directory, "**", "*annealing*final*.npz")
            files = glob.glob(pattern, recursive=True)
            chain_files.extend(files)
            
            # Also look for checkpoint files if no final files found
            if not files:
                checkpoint_pattern = os.path.join(directory, "**", "*annealing*checkpoint*.npz")
                checkpoint_files = glob.glob(checkpoint_pattern, recursive=True)
                if checkpoint_files:
                    # Sort by step number to get the latest
                    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else 0, reverse=True)
                    # Take the latest checkpoint for each chain
                    seen_chains = set()
                    latest_checkpoints = []
                    for f in checkpoint_files:
                        # Extract chain number
                        chain_parts = os.path.basename(f).split('_')
                        chain_num = None
                        for part in chain_parts:
                            if part.isdigit():
                                chain_num = part
                                break
                        
                        if chain_num and chain_num not in seen_chains:
                            seen_chains.add(chain_num)
                            latest_checkpoints.append(f)
                            print(f"Using latest checkpoint for chain {chain_num}: {f}")
                    
                    chain_files.extend(latest_checkpoints)
    else:
        # Search for annealing chains in common directories with expanded patterns
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
        
        # If no final files found, look for checkpoints
        if not chain_files:
            checkpoint_patterns = [
                "./mcmc_results/**/annealing*checkpoint*.npz",
                "./mcmc_results_annealing*/**/*annealing*checkpoint*.npz",
                "./mcmc_results_merged/**/*annealing*checkpoint*.npz"
            ]
            
            checkpoint_files = []
            for pattern in checkpoint_patterns:
                files = glob.glob(pattern, recursive=True)
                checkpoint_files.extend(files)
            
            if checkpoint_files:
                # Group by chain number and take the latest for each
                checkpoints_by_chain = {}
                for f in checkpoint_files:
                    # Extract chain number and step
                    basename = os.path.basename(f)
                    parts = basename.split('_')
                    
                    chain_num = None
                    step_num = 0
                    
                    # Find chain number and step number
                    for i, part in enumerate(parts):
                        if part.isdigit() and i < len(parts) - 1 and parts[i-1] in ['chain', 'annealing']:
                            chain_num = part
                        elif part.isdigit() and i == len(parts) - 1:
                            step_num = int(part.split('.')[0])
                    
                    if chain_num:
                        if chain_num not in checkpoints_by_chain or step_num > checkpoints_by_chain[chain_num][1]:
                            checkpoints_by_chain[chain_num] = (f, step_num)
                
                # Add latest checkpoint for each chain
                for chain_num, (checkpoint_file, step) in checkpoints_by_chain.items():
                    chain_files.append(checkpoint_file)
                    print(f"Using latest checkpoint for chain {chain_num}: {checkpoint_file} (step {step})")
    
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
    
    # Use parallel processing to load chains
    num_cores = min(multiprocessing.cpu_count(), 8)  # Limit to 8 cores to prevent overload
    print(f"Using {num_cores} cores for parallel loading")
    
    if chain_files:
        # Load chains one by one for better error handling
        all_chains = []
        for file in chain_files:
            chain = load_chain_file(file)
            if chain is not None:
                all_chains.append(chain)
    else:
        all_chains = []
    
    return all_chains

def compute_statistics(chains, burn_in_fraction=0.2, param_names=None):
    """
    Compute statistics for parameters across all chains
    """
    if param_names is None:
        param_names = ['H0', 'Omega_b_h2', 'Omega_c_h2', 'n_s', 'A_s', 'tau']
    
    stats_dict = {}
    
    # Combine all chains after burn-in
    all_samples = []
    
    for chain_data in chains:
        chain = chain_data['chain']
        burn_in = int(burn_in_fraction * len(chain))
        samples = chain[burn_in:]
        all_samples.append(samples)
    
    combined_samples = np.vstack(all_samples)
    
    for i, param in enumerate(param_names):
        # Extract parameter values
        values = combined_samples[:, i]
        
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
    
    return stats_dict, combined_samples

def plot_corner(samples, param_names=None, output_file='annealing_corner_plot.png'):
    """
    Create corner plot for the parameters
    """
    if param_names is None:
        param_names = ['H0', 'Omega_b_h2', 'Omega_c_h2', 'n_s', 'A_s', 'tau']
    
    # Create corner plot
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
    print(f"Saved corner plot to {output_file}")

def plot_trace(chains, param_names=None, output_file='annealing_trace_plots.png'):
    """
    Create trace plots for all parameters
    """
    if param_names is None:
        param_names = ['H0', 'Omega_b_h2', 'Omega_c_h2', 'n_s', 'A_s', 'tau']
    
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 1, figsize=(12, 3*n_params), sharex=True)
    
    colors = plt.cm.tab10.colors
    
    for i, param in enumerate(param_names):
        ax = axes[i] if n_params > 1 else axes
        
        for j, chain_data in enumerate(chains):
            chain = chain_data['chain']
            color = colors[j % len(colors)]
            ax.plot(chain[:, i], alpha=0.7, color=color, label=f"Chain {j+1}")
        
        ax.set_ylabel(param)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Step')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Saved trace plots to {output_file}")

def plot_temperatures(chains, output_file='annealing_temperature_schedule.png'):
    """
    Plot temperature schedules for annealing chains
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10.colors
    
    for i, chain_data in enumerate(chains):
        temps = chain_data.get('temperatures')
        if temps is not None:
            color = colors[i % len(colors)]
            ax.plot(temps, color=color, label=f"Chain {i+1}")
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Temperature')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Saved temperature schedule to {output_file}")

def export_to_csv(stats_dict, output_file='annealing_parameter_constraints.csv'):
    """
    Export parameter constraints to CSV
    """
    df = pd.DataFrame.from_dict(stats_dict, orient='index')
    df.to_csv(output_file)
    print(f"Saved parameter constraints to {output_file}")

def analyze_all_annealing_chains(directories=None, burn_in_fraction=0.2, output_dir='./annealing_analysis'):
    """
    Analyze all annealing chains and generate plots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load all annealing chains
    chains = load_annealing_chains(directories)
    
    if not chains:
        print("No annealing chains found!")
        return
    
    # Compute statistics
    param_names = ['H0', 'Omega_b_h2', 'Omega_c_h2', 'n_s', 'A_s', 'tau']
    stats_dict, combined_samples = compute_statistics(chains, burn_in_fraction, param_names)
    
    # Create plots
    plot_corner(combined_samples, param_names, os.path.join(output_dir, 'annealing_corner_plot.png'))
    plot_trace(chains, param_names, os.path.join(output_dir, 'annealing_trace_plots.png'))
    plot_temperatures(chains, os.path.join(output_dir, 'annealing_temperature_schedule.png'))
    
    # Export results
    export_to_csv(stats_dict, os.path.join(output_dir, 'annealing_parameter_constraints.csv'))
    
    # Save samples for later use
    np.save(os.path.join(output_dir, 'annealing_posterior_samples.npy'), combined_samples)
    
    # Print summary
    print("\nParameter Constraints Summary:")
    fiducial_values = {
        'H0': 67.36,
        'Omega_b_h2': 0.02237,
        'Omega_c_h2': 0.1200,
        'n_s': 0.9649,
        'A_s': 2.1e-9,
        'tau': 0.0544
    }
    
    for param, values in stats_dict.items():
        fiducial = fiducial_values.get(param, None)
        if fiducial:
            offset_percent = 100 * (values['median'] - fiducial) / fiducial
            print(f"{param}: {values['median']:.6g} + {values['error_plus']:.6g} - {values['error_minus']:.6g} (68% CI)")
            print(f"  (Fiducial: {fiducial}, Offset: {offset_percent:.1f}%)")
        else:
            print(f"{param}: {values['median']:.6g} + {values['error_plus']:.6g} - {values['error_minus']:.6g} (68% CI)")
    
    return stats_dict, combined_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze annealing MCMC chains")
    parser.add_argument("--directories", nargs="+", help="Directories to search for annealing chains")
    parser.add_argument("--burn-in", type=float, default=0.2, help="Burn-in fraction (default: 0.2)")
    parser.add_argument("--output-dir", default="./annealing_analysis", help="Output directory for plots and data")
    parser.add_argument("--num-processes", type=int, default=None, 
                      help="Number of processes to use for parallel processing (default: use all available cores)")
    
    args = parser.parse_args()
    
    # Set number of processes for pool if specified
    if args.num_processes:
        print(f"Using {args.num_processes} cores as specified")
        multiprocessing.set_start_method('spawn', force=True)
    
    analyze_all_annealing_chains(
        directories=args.directories,
        burn_in_fraction=args.burn_in,
        output_dir=args.output_dir
    )