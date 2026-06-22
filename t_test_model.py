import os
import sys
import traci
import torch
from model import DQN

# ==========================================
# 1. ENVIRONMENT SETUP
# ==========================================
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. LOAD THE TRAINED BRAIN (AGENT B)
# ==========================================
policy_net = DQN(28, 3).to(device)

# WE ARE LOADING THE BEST MULTI-EPOCH BRAIN HERE!
policy_net.load_state_dict(torch.load("t_trained_brain_multi_best.pth", weights_only=True))
policy_net.eval()

# ==========================================
# 3. HELPER FUNCTION
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

# ==========================================
# 4. MAIN TESTING LOOP
# ==========================================
def run_test():
    traci.start(["sumo-gui", "-c", "t_simulation.sumocfg", "--start"])
    traci.simulationStep()
    
    total_waiting_time = 0
    
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            state = get_state()
            
            with torch.no_grad():
                action = policy_net(state).max(1)[1].view(1, 1)
            
            traci.trafficlight.setPhase("center", action.item())
            
            for _ in range(10): 
                traci.simulationStep()
                
                for lane in ["B2C_0", "B2C_1", "B2C_2", "L2C_0", "L2C_1", "L2C_2", "R2C_0", "R2C_1", "R2C_2"]:
                    total_waiting_time += traci.lane.getWaitingTime(lane)
                
                if traci.simulation.getMinExpectedNumber() <= 0: break

        print(f"\n✅ T-Junction Test Complete!")
        print(f"Total Accumulated Waiting Time: {total_waiting_time} seconds")
        
    except Exception as e:
        print(f"\n🚨 ERROR: {e}")
    finally:
        traci.close()

if __name__ == "__main__":
    run_test()