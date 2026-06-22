import os
import sys
import traci
import torch
import torch.optim as optim
import torch.nn as nn
import random
import math
import csv

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
TARGET_UPDATE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4-WAY CROSS JUNCTION: 37 Inputs, 4 Actions
policy_net = DQN(37, 4).to(device)
target_net = DQN(37, 4).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
memory = ReplayMemory(10000)

steps_done = 0

# ==========================================
# 4-WAY HELPER FUNCTIONS
# ==========================================


def get_state():
    # All 12 lanes for the 4-way intersection
    incoming_lanes = ["B2C_0", "B2C_1", "B2C_2", "T2C_0", "T2C_1",
                      "T2C_2", "L2C_0", "L2C_1", "L2C_2", "R2C_0", "R2C_1", "R2C_2"]
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
    incoming_lanes = ["B2C_0", "B2C_1", "B2C_2", "T2C_0", "T2C_1",
                      "T2C_2", "L2C_0", "L2C_1", "L2C_2", "R2C_0", "R2C_1", "R2C_2"]
    total_wait = sum([traci.lane.getWaitingTime(lane)
                     for lane in incoming_lanes])
    total_queue = sum([traci.lane.getLastStepHaltingNumber(lane)
                      for lane in incoming_lanes])
    return torch.tensor([- (total_wait + total_queue)], dtype=torch.float32, device=device)


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1), eps_threshold
    else:
        # 4 possible actions (phases 0, 1, 2, 3)
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long), eps_threshold


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return 0.0
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(
        *transitions)

    state_batch = torch.cat(batch_state)
    action_batch = torch.cat(batch_action)
    reward_batch = torch.cat(batch_reward)
    next_state_batch = torch.cat(batch_next_state)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = target_net(next_state_batch).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# ==========================================
# MAIN MULTI-EPOCH LOOP
# ==========================================


def run():
    global steps_done

    # Robust Path Bypass: Dynamically target the 'sumo' binary
    sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
    sumo_cmd = [sumo_binary, "-c", "simulation.sumocfg",
                "--no-step-log", "true", "-W"]

    best_reward = -float('inf')

    # Open training log inside the root workspace
    with open('training_log_offpeak.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # Standard CSV Headers
        writer.writerow(["Step", "Epsilon", "Reward", "NN_Loss"])

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
                    action, current_eps = select_action(state)
                    traci.trafficlight.setPhase("center", action.item())

                    for _ in range(10):
                        traci.simulationStep()
                        step += 1

                    next_state = get_state()
                    reward = get_reward()
                    epoch_reward += reward.item()

                    memory.push(state, action, reward, next_state, False)
                    state = next_state

                    loss_val = optimize_model()

                    # Output step-by-step metric logs to the CSV file
                    writer.writerow(
                        [steps_done, current_eps, reward.item(), loss_val])

                    if traci.simulation.getMinExpectedNumber() <= 0:
                        break

                if epoch_reward > best_reward:
                    best_reward = epoch_reward
                    # Save checkpoint for the optimal offpeak weight matrix
                    torch.save(policy_net.state_dict(),
                               "trained_brain_offpeak_best.pth")
                    print(
                        f"✅ Epoch {epoch + 1} finished | Total Reward: {epoch_reward:,.0f} 🌟 (NEW BEST!)")
                else:
                    print(
                        f"✅ Epoch {epoch + 1} finished | Total Reward: {epoch_reward:,.0f}")

            except Exception as e:
                print(f"🚨 ERROR in Epoch {epoch + 1}: {e}")
            finally:
                traci.close()

            if epoch % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

    print(f"\n🎉 Off-Peak Training Complete.")
    print(f"🏆 Your most optimal AI was saved as 'trained_brain_offpeak_best.pth'!")
    print("📊 Training metrics successfully logged to 'training_log_offpeak.csv'!")


if __name__ == "__main__":
    run()
