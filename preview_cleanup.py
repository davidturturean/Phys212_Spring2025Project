#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preview what files would remain after cleanup.
This script doesn't actually delete anything.
"""

import os

# Define the project root directory
project_dir = os.path.dirname(os.path.abspath(__file__))

# Lists of directories and files to be removed
dirs_to_remove = [
    "archived",
    "archived_code",
    "cleanup_temp",
    "__pycache__",
    "mcmc_test_results",
    "output/extra",
]

files_to_remove = [
    "cleanup.py",
    "cleanup_final.py",
    "check.py",
    "mcmc_test.py",
    "readheader.py",
    "MCMC.py",
    "theoretical_model.py",
    "data.py",
    "run_improved.py",
    ".DS_Store",
    "output/.DS_Store",
]

# Collect all files and directories
all_dirs = []
all_files = []

for dirpath, dirnames, filenames in os.walk(project_dir):
    # Skip the venv directory
    if "venv" in dirpath:
        continue
    
    rel_dirpath = os.path.relpath(dirpath, project_dir)
    if rel_dirpath == ".":
        rel_dirpath = ""
    
    # Add directories
    for dirname in dirnames:
        if dirname != "venv" and "venv" not in dirname:
            rel_dirname = os.path.join(rel_dirpath, dirname)
            all_dirs.append(rel_dirname)
    
    # Add files
    for filename in filenames:
        rel_filename = os.path.join(rel_dirpath, filename)
        all_files.append(rel_filename)

# Find directories and files that would remain
remaining_dirs = []
for directory in sorted(all_dirs):
    should_remove = False
    for rm_dir in dirs_to_remove:
        if directory == rm_dir or directory.startswith(rm_dir + os.sep):
            should_remove = True
            break
    
    if not should_remove:
        remaining_dirs.append(directory)

remaining_files = []
for file in sorted(all_files):
    should_remove = False
    for rm_dir in dirs_to_remove:
        if file.startswith(rm_dir + os.sep):
            should_remove = True
            break
    
    for rm_file in files_to_remove:
        if file == rm_file:
            should_remove = True
            break
    
    if not should_remove:
        remaining_files.append(file)

# Print the results
print("DIRECTORIES THAT WOULD REMAIN AFTER CLEANUP:")
for d in remaining_dirs:
    print(f"  {d}")

print("\nFILES THAT WOULD REMAIN AFTER CLEANUP:")
for f in remaining_files:
    print(f"  {f}")

print("\nNote: This is just a preview. No files were actually deleted.")