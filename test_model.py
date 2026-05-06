import os
import sys
import traci
import torch
from model import DQN

# ==========================================
# 1. ENVIRONMENT SETUP
# ==========================================
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. LOAD THE TRAINED BRAIN
# ==========================================
# Rebuild the network architecture
policy_net = DQN(37, 4).to(device)

# Load the saved weights into the network
policy_net.load_state_dict(torch.load("trained_brain.pth", weights_only=True))

# Set the network to "Evaluation Mode" (locks the weights, no more learning)
policy_net.eval()

# ==========================================
# 3. HELPER FUNCTION
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
            traci.lane.getLastStepHaltingNumber(lane), 
            traci.lane.getWaitingTime(lane)
        ])
    state.append(traci.trafficlight.getPhase("center"))
    return torch.tensor([state], dtype=torch.float32, device=device)

# ==========================================
# 4. MAIN TESTING LOOP
# ==========================================
def run_test():
    # Start SUMO automatically
    traci.start(["sumo-gui", "-c", "simulation.sumocfg", "--start"])
    traci.simulationStep()
    
    total_waiting_time = 0
    step = 0
    
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            state = get_state()
            
            # THE AI MAKES A DECISION (No random guessing allowed!)
            with torch.no_grad():
                action = policy_net(state).max(1)[1].view(1, 1)
            
            # Change the traffic light
            traci.trafficlight.setPhase("center", action.item())
            
            # Let traffic flow for 10 steps
            for _ in range(10): 
                traci.simulationStep()
                
                # Track metrics for your final report
                for lane in ["B2C_0", "B2C_1", "B2C_2", "T2C_0", "T2C_1", "T2C_2", "L2C_0", "L2C_1", "L2C_2", "R2C_0", "R2C_1", "R2C_2"]:
                    total_waiting_time += traci.lane.getWaitingTime(lane)
                
                step += 1
                if traci.simulation.getMinExpectedNumber() <= 0: break

        print(f"\n✅ Test Complete!")
        print(f"Total Accumulated Waiting Time: {total_waiting_time} seconds")
        
    except Exception as e:
        print(f"\n🚨 ERROR: {e}")
    finally:
        traci.close()

if __name__ == "__main__":
    run_test()