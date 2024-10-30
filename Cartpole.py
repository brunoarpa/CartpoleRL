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

# Create Neural Network
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions): 
        super(DQN, self).__init__() #Initializes parent class nn.Module
        self.layer1 = nn.Linear(n_observations, 128) # Each layer (inputs to layer, outputs of layer)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x): # Called with either one element to determine next action, or a batch during optimization
        x = F.relu(self.layer1(x)) # Going through each layer
        x = F.relu(self.layer2(x))
        return self.layer3(x) # Returns tensor([[left0exp,right0exp]...])
