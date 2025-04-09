#!/bin/bash
#SBATCH --job-name=hello_world
#SBATCH --open-mode=append
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH -c 1
#SBATCH --gres=gpu:1  # Request 1 GPU (remove if you don't need a GPU)

# Load the Python module
module purge
module load python/intel/3.8.6

# Create and activate a virtual environment (optional)
# python -m venv myenv
# source myenv/bin/activate

# Install packages if needed
pip install torch

# Run the Python script
python hello_world.py

