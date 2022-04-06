
"""
Main file of the project
Experiment will be executed in this file
"""

import warnings

import torch

from src.experiment_definitions import alexnet_model_flip_transform, alexnet_model_no_transform # , phi_model_flip_transform, phi_model_no_transform
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
    # phi_model_no_transform(EPOCH_COUNT, PHI_BATCH_SIZE, SEED, LR)

    # Execution of the experiment for PHI Model with flip augmentation transformations
    # phi_model_flip_transform(EPOCH_COUNT, PHI_BATCH_SIZE, SEED, LR)

    # Empty CUDA memory
    torch.cuda.empty_cache()

    # Execution of the experiment for AlexNet Model with no augmentation transformations
    alexnet_model_no_transform(EPOCH_COUNT, ALEX_BATCH_SIZE, SEED, GDEXPLOIT_LR)

    # Empty CUDA memory
    torch.cuda.empty_cache()

    # Execution of the experiment for AlexNet Model with flip augmentation transformations
    alexnet_model_flip_transform(EPOCH_COUNT, ALEX_BATCH_SIZE, SEED, GDEXPLOIT_LR)

if __name__ == "__main__":
    main()
