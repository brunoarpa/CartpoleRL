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

BATCH_SIZE = 128 # BATCH_SIZE is the number of transitions sampled from the ReplayMemory()
GAMMA = 0.99 # GAMMA is the discount factor to calculate future rewards value
EPS_START = 0.9 # EPS_START is the starting value of epsilon for the decay (exploration)
EPS_END = 0.05 # EPS_END is the minimum value of epsilon so no more decay (explotation)
EPS_DECAY = 1000 # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005 # TAU is the update rate of the target_net
LR = 1e-4 # LR is the learning rate of the ``AdamW`` optimizer in this case 1*10^-4

n_actions = env.action_space.n # The number of possible actions the agent can take
state, info = env.reset() # State from the reseted environment
n_observations = len(state) # The number of features in the state space
policy_net = DQN(n_observations, n_actions).to(device) # The network that is used to select actions during training
target_net = DQN(n_observations, n_actions).to(device) # A copy of the policy network that is updated less frequently to provide stable Q-value targets.
target_net.load_state_dict(policy_net.state_dict()) # Put policy_net's parameters(weights and biases) into target_net as a copy

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True) #AdamW variant, prevents overfitting, amsgrad = True is more stable gradient descent
memory = ReplayMemory(10000) # Double ended queue memory created with 10000 of maxlength

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random() # Random float from 0 to 1
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY) # Probability of choosing a random action decaying exponentially
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad(): # Disabling gradient calculation for inference, sure of not calling Tensor.backward() better memory computation
            return policy_net(state).max(1).indices.view(1, 1) # returns maximum Q-value for state, .indices re-format the tensor
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long) # agent explores by selecting a random action

episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1) # Creates plotter with identifier 1
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf() #clear currect figure
        plt.title('Training...')

    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy()) # durations green
    
    if len(durations_t) >= 100: 
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1) # Take 100 episode averages and plot them too
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy()) # average duration red 

    plt.pause(0.001)  # pause a bit so that plots are updated

    if is_ipython:
        if not show_result:
            display.display(plt.gcf()) # Displays current figure
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf()) # Displays final figure

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE) # Get 128 (BATCH_SIZE) random transition samples
    batch = Transition(*zip(*transitions)) # Turns batch into each attribute of Transitions as a batch of tensors

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool) # If next_state non final
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]) # Concatenates to turn list of tensors into only one tensor

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch) # Gets actions which would've been taken for each batch state

    next_state_values = torch.zeros(BATCH_SIZE, device=device) # Creates a batch of zeros
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values # Best actions in non_final_next_states

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch # Expected best actions with state Q(s,a) values
    
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()# Clears all gradients torch has stored
    loss.backward() # Backpropagation

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100) # Max gradient value of 100 to stabilize gradient descent
    optimizer.step() # Updates parameters from network

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    state, info = env.reset() # Initialize the environment and get its state
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) # Turn state into tensor
    for t in count():
        action = select_action(state) 
        observation, reward, terminated, truncated, _ = env.step(action.item()) # Execute action
        reward = torch.tensor([reward], device=device) # Turn reward into tensor
        done = terminated or truncated

        if terminated:
            next_state = None # Final state
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0) # Not final state

        memory.push(state, action, next_state, reward) # Store the transition in memory

        state = next_state # Move to the next state

        optimize_model() # Perform one step of the optimization (on the policy network)

        target_net_state_dict = target_net.state_dict() # Gets weights and biases of target network
        policy_net_state_dict = policy_net.state_dict() # Gets weights and biases of policy network
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU) # Iterate over each parameter of network
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
env.close()
