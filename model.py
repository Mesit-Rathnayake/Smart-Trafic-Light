import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque

# ==========================================
# 1. THE NEURAL NETWORK ARCHITECTURE
# ==========================================
class DQN(nn.Module):
    def __init__(self, input_dim=37, output_dim=4):
        """
        Input: 37 dimensions (Vehicle Count, Queue, Wait Time for 12 lanes + 1 Current Phase)
        Output: 4 dimensions (Q-values for N-S Green, E-W Green, etc.)
        """
        super(DQN, self).__init__()
        
        # First Hidden Layer: Maps 37 inputs to 128 neurons
        self.fc1 = nn.Linear(input_dim, 128)
        
        # Second Hidden Layer: Maps 128 neurons to 64 neurons
        self.fc2 = nn.Linear(128, 64)
        
        # Output Layer: Maps 64 neurons to 4 possible actions
        self.out = nn.Linear(64, output_dim)

    def forward(self, x):
        """
        Defines how the data flows through the network using ReLU activation.
        """
        # Pass through layer 1 with ReLU activation
        x = F.relu(self.fc1(x))
        
        # Pass through layer 2 with ReLU activation
        x = F.relu(self.fc2(x))
        
        # Output layer (No activation function here because we want raw Q-values)
        return self.out(x)


# ==========================================
# 2. EXPERIENCE REPLAY BUFFER
# ==========================================
class ReplayMemory:
    def __init__(self, capacity=10000):
        """
        Initializes a memory buffer with a maximum capacity.
        When full, it automatically pushes out the oldest memories.
        """
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Saves a single step/experience to memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Pulls a random batch of experiences to train the network.
        Random sampling prevents the network from overfitting to chronological data.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Returns the current size of the memory buffer."""
        return len(self.memory)