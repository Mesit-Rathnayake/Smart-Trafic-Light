import os
import sys
import traci
import torch
import torch.optim as optim
import torch.nn as nn
import random
import math

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
EPS_DECAY = 5000   
NUM_EPOCHS = 30    
TARGET_UPDATE = 5  # NEW: How often we update the Target Network (in epochs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AGENT B: 28 Inputs, 3 Actions
# NEW: Create TWO networks. One for learning, one for stable targets.
policy_net = DQN(28, 3).to(device)
target_net = DQN(28, 3).to(device)
target_net.load_state_dict(policy_net.state_dict()) # Clone the weights
target_net.eval() # Freeze the target network

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
memory = ReplayMemory(10000)

steps_done = 0

# ==========================================
# T-JUNCTION HELPER FUNCTIONS
# ==========================================
def get_state():
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
        return torch.tensor([[random.randrange(3)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE: return
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

    state_batch = torch.cat(batch_state)
    action_batch = torch.cat(batch_action)
    reward_batch = torch.cat(batch_reward)
    next_state_batch = torch.cat(batch_next_state)

    # Q-values expected by our current policy network
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # NEW: Q-values of next state calculated by our STABLE target network
    next_state_values = target_net(next_state_batch).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ==========================================
# MAIN MULTI-EPOCH LOOP
# ==========================================
def run():
    global steps_done
    sumo_cmd = ["sumo", "-c", "t_simulation.sumocfg", "--no-step-log", "true", "-W"]
    
    # NEW: Track the best reward so we can checkpoint the model
    best_reward = -float('inf')
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n🚀 Starting Epoch {epoch + 1}/{NUM_EPOCHS}...")
        traci.start(sumo_cmd)
        traci.simulationStep()
        
        try:
            state = get_state()
            step = 0
            MAX_STEPS = 3600 
            epoch_reward = 0
            
            while step < MAX_STEPS:
                action = select_action(state)
                traci.trafficlight.setPhase("center", action.item())
                
                for _ in range(10): 
                    traci.simulationStep()
                    step += 1
                    
                next_state = get_state()
                reward = get_reward()
                epoch_reward += reward.item()
                
                memory.push(state, action, reward, next_state, False)
                state = next_state
                optimize_model()

                if traci.simulation.getMinExpectedNumber() <= 0: break

            # NEW: Checkpoint logic - save the best model dynamically!
            if epoch_reward > best_reward:
                best_reward = epoch_reward
                torch.save(policy_net.state_dict(), "t_trained_brain_multi_best.pth")
                print(f"✅ Epoch {epoch + 1} finished | Total Reward: {epoch_reward:,.0f} 🌟 (NEW BEST!)")
            else:
                print(f"✅ Epoch {epoch + 1} finished | Total Reward: {epoch_reward:,.0f}")

        except Exception as e:
            print(f"🚨 ERROR in Epoch {epoch + 1}: {e}")
        finally:
            traci.close()
            
        # NEW: Update the target network periodically to stabilize learning
        if epoch % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    # Save the final Brain as a fallback
    torch.save(policy_net.state_dict(), "t_trained_brain_multi_final.pth")
    print(f"\n🎉 Multi-Epoch Training Complete.")
    print(f"🏆 Your most optimal AI was saved as 't_trained_brain_multi_best.pth'!")

if __name__ == "__main__":
    run()