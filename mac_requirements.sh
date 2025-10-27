#!/bin/bash
# create and activate environment
conda create --name rl_test python=3.11 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rl_test

# install dependencies
pip install gymnasium tqdm distracting-control opencv-python