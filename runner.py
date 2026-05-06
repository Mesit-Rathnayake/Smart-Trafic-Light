import os
import sys
import traci
import torch
import torch.optim as optim
import torch.nn as nn
import random
import math

# Import Member 2's Brain!
from model import DQN, ReplayMemory

# ==========================================
# 1. ENVIRONMENT SETUP
# ==========================================
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# ==========================================
# 2. RL HYPERPARAMETERS (Member 3's Playground)
# ==========================================
BATCH_SIZE = 32      # How many memories to learn from at once
GAMMA = 0.99         # Discount factor (values future rewards)
EPS_START = 1.0      # Start with 100% random actions (Exploration)
EPS_END = 0.05       # End with 5% random actions
EPS_DECAY = 1000     # How fast to decay the randomness
TARGET_UPDATE = 10   # How often to update the target network

# Initialize the Neural Network and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(37, 4).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
memory = ReplayMemory(10000)

steps_done = 0

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def get_state():
    """Extracts the 37-dimension state."""
    incoming_lanes = [
        "B2C_0", "B2C_1", "B2C_2", "T2C_0", "T2C_1", "T2C_2", 
        "L2C_0", "L2C_1", "L2C_2", "R2C_0", "R2C_1", "R2C_2"
    ]
    state = []
    for lane in incoming_lanes:
        state.extend([
            traci.lane.getLastStepVehicleNumber(lane),
            traci.lane.getLastStepHaltingNumber(lane),  # The correct SUMO function for queues!
            traci.lane.getWaitingTime(lane)
        ])
    state.append(traci.trafficlight.getPhase("center"))
    return torch.tensor([state], dtype=torch.float32, device=device)

def get_reward():
    """Calculates the negative reward based on wait times and queues."""
    incoming_lanes = [
        "B2C_0", "B2C_1", "B2C_2", "T2C_0", "T2C_1", "T2C_2", 
        "L2C_0", "L2C_1", "L2C_2", "R2C_0", "R2C_1", "R2C_2"
    ]
    total_wait = sum([traci.lane.getWaitingTime(lane) for lane in incoming_lanes])
    total_queue = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in incoming_lanes])
    return torch.tensor([- (total_wait + total_queue)], dtype=torch.float32, device=device)

def select_action(state):
    """Epsilon-greedy action selection."""
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold:
        # EXPLOIT: Use the Brain
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # EXPLORE: Random Phase (0, 1, 2, or 3)
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)

def optimize_model():
    """Trains the Neural Network using past memories."""
    if len(memory) < BATCH_SIZE:
        return # Not enough memories yet!

    # Pull a random batch of memories
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

    # Convert to PyTorch Tensors
    state_batch = torch.cat(batch_state)
    action_batch = torch.cat(batch_action)
    reward_batch = torch.cat(batch_reward)
    next_state_batch = torch.cat(batch_next_state)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states
    next_state_values = policy_net(next_state_batch).max(1)[0].detach()
    
    # Compute the expected Q values (Bellman Equation)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Calculate Loss (Huber Loss is stable for DQN)
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model (Backpropagation)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ==========================================
# 4. MAIN TRAINING LOOP
# ==========================================
def run():
    # Start SUMO and immediately press play (--start)
    traci.start(["sumo-gui", "-c", "simulation.sumocfg", "--start"])
    
    # Take ONE initial step to wake up the simulation
    traci.simulationStep()
    
    try:
        state = get_state()
        step = 0
        MAX_STEPS = 3600  # Train for 1 simulated hour
        
        while step < MAX_STEPS:
            # 1. Ask AI for the next light phase
            action = select_action(state)
            
            # 2. Change the traffic light (Action is 0, 1, 2, or 3)
            traci.trafficlight.setPhase("center", action.item())
            
            # 3. Let the simulation run for 10 steps to see the effect
            for _ in range(10): 
                traci.simulationStep()
                step += 1
                
            # 4. Observe the new state and calculate the reward
            next_state = get_state()
            reward = get_reward()
            
            # 5. Save this experience to Memory (False means episode is not done)
            memory.push(state, action, reward, next_state, False)
            
            # 6. Move to the next state
            state = next_state
            
            # 7. Train the Brain!
            optimize_model()

        # Save the trained model at the end
        torch.save(policy_net.state_dict(), "trained_brain.pth")
        print("\n🎉 Training Complete. Brain saved as trained_brain.pth")

    except Exception as e:
        # If Python crashes, catch it and print it loudly!
        print("\n" + "="*50)
        print(f"🚨 PYTHON ERROR CAUGHT: {e}")
        print("="*50 + "\n")
        
    finally:
        # Ensure SUMO closes cleanly even if Python crashes
        traci.close()

if __name__ == "__main__":
    run()