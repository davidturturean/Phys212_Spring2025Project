#!/bin/bash
# Script to upload the cleaned project to GitHub

# Configuration
REPO_URL="https://github.com/davidturturean/Phys212_Spring2025Project.git"
OLD_BRANCH="old_Apr22"
MAIN_BRANCH="main"

echo "Starting GitHub upload process..."

# Initialize Git if not already initialized
if [ ! -d ".git" ]; then
  echo "Initializing git repository..."
  git init
  git remote add origin $REPO_URL
else
  echo "Git repository already initialized."
  # Make sure we have the latest from remote
  git fetch origin
fi

# Create and checkout to old_Apr22 branch if not exists
if ! git show-ref --verify --quiet refs/heads/$OLD_BRANCH; then
  echo "Creating $OLD_BRANCH branch..."
  git checkout -b $OLD_BRANCH
else
  echo "Switching to $OLD_BRANCH branch..."
  git checkout $OLD_BRANCH
fi

# Push current main to old_Apr22 branch
echo "Pushing current main to $OLD_BRANCH branch..."
git push origin main:$OLD_BRANCH -f

# Switch back to main branch
echo "Switching to $MAIN_BRANCH branch..."
git checkout -B $MAIN_BRANCH

# Add all files
echo "Adding files to git..."
git add .

# Commit the changes
echo "Committing changes..."
git commit -m "Cleaned and organized ΛCDM Cosmological Parameter Inference codebase

- Removed redundant files and scripts
- Organized core functionality
- Improved documentation
- Optimized project structure
- Removed large data files from repository"

# Push to main branch
echo "Pushing to $MAIN_BRANCH branch..."
git push -f origin $MAIN_BRANCH

echo "Upload completed successfully!"
echo ""
echo "The previous main branch has been moved to $OLD_BRANCH"
echo "The cleaned codebase is now on the $MAIN_BRANCH branch"