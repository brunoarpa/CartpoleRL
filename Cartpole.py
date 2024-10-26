import math
import random
import gymnasium
import torch
import time

# Make the cartpole environment from gymnasium  :)))
env = gymnasium.make("CartPole-v1")
# Use GPU for better processing
device = torch.device(
    "mps" if torch.backends.mps.is_available() else # Apple 
    "cuda" if torch.cuda.is_available() else    
    "cpu"
)

