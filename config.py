"""
Configuration file for the project.

This file contains all global configurations used in the project,
including paths, hyperparameters, and environment-specific options.
"""

import os

# ========================
# General
# ========================
SEED = 42  # Sets the seed for reproducible results

# ========================
# Data
# ========================
SPEC_RES = 800
ACC_TRAINING = 300
N_POINTS_TRAINING = 1215

# ========================
# Paths
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(BASE_DIR, "models"))
LOG_DIR = os.getenv("LOG_DIR", os.path.join(BASE_DIR, "logs"))

# ========================
# Hyperparameters
# ========================