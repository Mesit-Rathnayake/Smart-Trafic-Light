import os
import sys
import traci
import torch
import torch.optim as optim
import torch.nn as nn
import random
import math

# We use the EXACT same brain architecture!
from model import DQN, ReplayMemory

# ==========================================
# ENVIRONMENT SETUP
# ==========================================
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NEW AGENT B: 28 Inputs, 3 Actions
policy_net = DQN(28, 3).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
memory = ReplayMemory(10000)

steps_done = 0

# ==========================================
# T-JUNCTION HELPER FUNCTIONS
# ==========================================
def get_state():
    # Only 9 incoming lanes now (No Top/T2C lanes)
    incoming_lanes = [
        "B2C_0", "B2C_1", "B2C_2", 
        "L2C_0", "L2C_1", "L2C_2", 
        "R2C_0", "R2C_1", "R2C_2"
    ]
    state = []
    for lane in incoming_lanes:
        state.extend([
            traci.lane.getLastStepVehicleNumber(lane),
            traci.lane.getLastStepHaltingNumber(lane),
            traci.lane.getWaitingTime(lane)
        ])
    state.append(traci.trafficlight.getPhase("center"))
    return torch.tensor([state], dtype=torch.float32, device=device)

def get_reward():
    incoming_lanes = [
        "B2C_0", "B2C_1", "B2C_2", 
        "L2C_0", "L2C_1", "L2C_2", 
        "R2C_0", "R2C_1", "R2C_2"
    ]
    total_wait = sum([traci.lane.getWaitingTime(lane) for lane in incoming_lanes])
    total_queue = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in incoming_lanes])
    return torch.tensor([- (total_wait + total_queue)], dtype=torch.float32, device=device)

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # T-Junctions only use phase 0, 1, or 2 (3 actions)
        return torch.tensor([[random.randrange(3)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE: return
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

    state_batch = torch.cat(batch_state)
    action_batch = torch.cat(batch_action)
    reward_batch = torch.cat(batch_reward)
    next_state_batch = torch.cat(batch_next_state)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = policy_net(next_state_batch).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ==========================================
# MAIN LOOP
# ==========================================
def run():
    # Use the specific T-Junction config
    traci.start(["sumo-gui", "-c", "t_simulation.sumocfg", "--start"])
    traci.simulationStep()
    
    try:
        state = get_state()
        step = 0
        MAX_STEPS = 3600 
        
        while step < MAX_STEPS:
            action = select_action(state)
            
            # Action is 0, 1, or 2 for the T-Junction
            traci.trafficlight.setPhase("center", action.item())
            
            for _ in range(10): 
                traci.simulationStep()
                step += 1
                
            next_state = get_state()
            reward = get_reward()
            memory.push(state, action, reward, next_state, False)
            state = next_state
            optimize_model()

        # Save Agent B's brain separately!
        torch.save(policy_net.state_dict(), "t_trained_brain.pth")
        print("\n🎉 Agent B (T-Junction) Training Complete. Brain saved as t_trained_brain.pth")

    except Exception as e:
        print(f"🚨 ERROR: {e}")
    finally:
        traci.close()

if __name__ == "__main__":
    run()