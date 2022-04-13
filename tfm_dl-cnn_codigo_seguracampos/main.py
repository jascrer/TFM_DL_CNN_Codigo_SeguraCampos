
"""
Main file of the project
Experiment will be executed in this file
"""

import warnings
# import torch
import src.experiment_definitions as experiments

warnings.filterwarnings('ignore')

PHI_BATCH_SIZE = 2500
ALEX_BATCH_SIZE = 250
PAPER_LR = 0.01
GDEXPLOIT_LR = 0.001
EPOCH_COUNT = 150
SEED = 42

def main():
    """Main function. Point of execution"""
    # Execution of the experiment for PHI Model with no augmentation transformations
    experiments.phi_model_no_transform(EPOCH_COUNT, PHI_BATCH_SIZE, SEED, PAPER_LR)

    # Execution of the experiment for PHI Model with flip augmentation transformations
    experiments.phi_model_flip_transform(EPOCH_COUNT, PHI_BATCH_SIZE, SEED, PAPER_LR)

    # Empty CUDA memory
    #torch.cuda.empty_cache()

    # Execution of the experiment for AlexNet Model with no augmentation transformations
    #experiments.alexnet_model_no_transform(EPOCH_COUNT, ALEX_BATCH_SIZE, SEED, GDEXPLOIT_LR)

    # Empty CUDA memory
    #torch.cuda.empty_cache()

    # Execution of the experiment for AlexNet Model with flip augmentation transformations
    # experiments.alexnet_model_flip_transform(EPOCH_COUNT, ALEX_BATCH_SIZE, SEED, GDEXPLOIT_LR)

if __name__ == "__main__":
    main()
