import math
import random
import gymnasium as gym
from collections import namedtuple, deque
from itertools import count
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt

# Make the cartpole environment from gymnasium  :)))
env = gym.make("CartPole-v1")

# Set up matplotlib (graph for plotting in graph of episode, duration, results)
is_ipython = 'inline' in matplotlib.get_backend() 
if is_ipython:
    from IPython import display
plt.ion()

# Use GPU for better processing
device = torch.device(
    "mps" if torch.backends.mps.is_available() else # Apple 
    "cuda" if torch.cuda.is_available() else    
    "cpu"
)

# Create Neural Network :o
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

# To respresent a single transition that maps state-action pair to next_state-reward result
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward')) 

# To temporarily store and manage a dataset of transitions that the agent can sample from during training
class ReplayMemory():

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity) # Creates double ended list with max capacity so oldest removed when reached

    def push(self, *args):
        self.memory.append(Transition(*args)) # Save a transition *args means it accepts a variable number of arguments

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size) # Returns a random sample of length batch_size of transitions from the memory

    def __len__(self):
        return len(self.memory) # Returns number of transitions in memory
