import os
import sys
import traci
import torch
import torch.optim as optim
import torch.nn as nn
import random
import math
import csv # Added for logging

from model import DQN, ReplayMemory

# ==========================================
# ENVIRONMENT & HYPERPARAMETERS
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
policy_net = DQN(37, 4).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
memory = ReplayMemory(10000)

steps_done = 0

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_state():
    incoming_lanes = ["B2C_0", "B2C_1", "B2C_2", "T2C_0", "T2C_1", "T2C_2", "L2C_0", "L2C_1", "L2C_2", "R2C_0", "R2C_1", "R2C_2"]
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
    incoming_lanes = ["B2C_0", "B2C_1", "B2C_2", "T2C_0", "T2C_1", "T2C_2", "L2C_0", "L2C_1", "L2C_2", "R2C_0", "R2C_1", "R2C_2"]
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
            return policy_net(state).max(1)[1].view(1, 1), eps_threshold
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long), eps_threshold

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return 0.0 # Return 0 loss if not training yet

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
    
    return loss.item() # Return the loss metric

# ==========================================
# MAIN LOOP WITH LOGGING
# ==========================================
def run():
    traci.start(["sumo", "-c", "simulation.sumocfg", "--start"]) # Using 'sumo' instead of 'sumo-gui' for faster headless training
    traci.simulationStep()
    
    # Open a CSV file to log our PhD metrics
    with open('training_log.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "Epsilon", "Reward", "NN_Loss"]) # Headers
        
        try:
            state = get_state()
            step = 0
            MAX_STEPS = 3600 
            
            while step < MAX_STEPS:
                action, current_eps = select_action(state)
                traci.trafficlight.setPhase("center", action.item())
                
                for _ in range(10): 
                    traci.simulationStep()
                    step += 1
                    
                next_state = get_state()
                reward = get_reward()
                memory.push(state, action, reward, next_state, False)
                state = next_state
                
                loss_value = optimize_model()
                
                # Log the data to our CSV every action step
                writer.writerow([step, current_eps, reward.item(), loss_value])

            torch.save(policy_net.state_dict(), "trained_brain_advanced.pth")
            print("\n🎉 Advanced Training Complete. Data logged to training_log.csv")

        except Exception as e:
            print(f"🚨 ERROR: {e}")
        finally:
            traci.close()

if __name__ == "__main__":
    run()