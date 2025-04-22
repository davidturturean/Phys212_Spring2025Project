#!/bin/bash
# Script to transfer fixed restart continue scripts to MIT Engaging cluster

# Define your username and path
USERNAME="davidct"  # Replace with your actual username if different
REMOTE_DIR="/nfs/home2/davidct/FinalProject"  # Replace if different

# Files to transfer
FILES=(
  "restart_continue_chains.py"
  "run_restart_continue_mki_fixed.sh"
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
echo "3. Make sure the script is executable: chmod +x run_restart_continue_mki_fixed.sh"
echo "4. Submit the job: sbatch run_restart_continue_mki_fixed.sh"
echo ""
echo "This script will continue both your standard and annealing MCMC runs"
echo "from where they left off, using runtime patches to ensure proper"
echo "FITS data loading and NumPy compatibility."