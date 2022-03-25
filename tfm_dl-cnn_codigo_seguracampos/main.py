
"""
Main file of the project
Experiment will be executed in this file
"""

import warnings

from src.experiment_definitions import phi_experiment_no_transform
warnings.filterwarnings('ignore')

BATCH_SIZE = 2500
LR = 0.01
EPOCH_COUNT = 150
DATA_FOLDER = 'data/interim/experiment'
SEED = 42

def main():
    """Main function. Point of execution"""
    phi_experiment_no_transform(EPOCH_COUNT, BATCH_SIZE, SEED, LR, DATA_FOLDER)

if __name__ == "__main__":
    main()
