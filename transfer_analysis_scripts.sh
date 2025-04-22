#!/bin/bash
# Script to transfer combined analysis scripts to MIT Engaging cluster

# Define your username and path
USERNAME="davidct"  # Replace with your actual username if different
REMOTE_DIR="/nfs/home2/davidct/FinalProject"  # Replace if different

# Files to transfer
FILES=(
  "analyze_combined_results.py"
  "run_combined_analysis.sh"
)

# Transfer files
echo "Transferring files to MIT Engaging cluster..."
for file in "${FILES[@]}"; do
  echo "Transferring $file..."
  scp "$file" "$USERNAME@engaging-login.mit.edu:$REMOTE_DIR/"
done

echo "Files transferred successfully!"
echo ""
echo "Next steps:"
echo "1. Log in to the cluster: ssh $USERNAME@engaging-login.mit.edu"
echo "2. Navigate to your project directory: cd $REMOTE_DIR"
echo "3. Make sure the script is executable: chmod +x run_combined_analysis.sh"
echo "4. Submit the job: sbatch run_combined_analysis.sh"
echo ""
echo "FIXED: The script now properly handles chains with different shapes"
echo "and properly normalizes them for analysis."
echo ""
echo "This script will analyze ONLY your original MCMC chains from:"
echo "- mcmc_results (with all the checkpoint files)"
echo "- mcmc_results_final_20250422_080408"
echo ""
echo "The analysis will:"
echo "- Process checkpoint files from mcmc_results"
echo "- Handle both standard and annealing chains"
echo "- Properly process chains with different shapes"
echo "- Generate comprehensive parameter constraints"