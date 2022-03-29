
"""
Main file of the project
Experiment will be executed in this file
"""

import warnings

from src.experiment_definitions import phi_experiment_flip_transform, phi_experiment_no_transform
warnings.filterwarnings('ignore')

BATCH_SIZE = 2500
LR = 0.01
EPOCH_COUNT = 150
SEED = 42

def main():
    """Main function. Point of execution"""
    # Execution of the experiment for PHI Model with no augmentation transformations
    phi_experiment_no_transform(EPOCH_COUNT, BATCH_SIZE, SEED, LR)

    # Execution of the experiment for PHI Model with flip augmentation transformations
    phi_experiment_flip_transform(EPOCH_COUNT, BATCH_SIZE, SEED, LR)

if __name__ == "__main__":
    main()
