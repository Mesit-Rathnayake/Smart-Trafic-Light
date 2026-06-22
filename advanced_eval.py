import os
import sys
import traci
import torch
from model import DQN

# ==========================================
# SETUP
# ==========================================
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(37, 4).to(device)

# Load the brain (ensure the filename matches your trained model)
policy_net.load_state_dict(torch.load("trained_brain_advanced.pth", weights_only=True))
policy_net.eval()

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

# ==========================================
# ADVANCED METRICS LOOP
# ==========================================
def run_test():
    traci.start(["sumo-gui", "-c", "simulation.sumocfg", "--start"])
    traci.simulationStep()
    
    # PhD Level Trackers
    total_waiting_time = 0
    total_co2_emissions = 0.0 # in mg
    max_queue_length = 0
    total_throughput = 0
    
    incoming_lanes = ["B2C_0", "B2C_1", "B2C_2", "T2C_0", "T2C_1", "T2C_2", "L2C_0", "L2C_1", "L2C_2", "R2C_0", "R2C_1", "R2C_2"]

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            state = get_state()
            
            with torch.no_grad():
                action = policy_net(state).max(1)[1].view(1, 1)
            traci.trafficlight.setPhase("center", action.item())
            
            for _ in range(10): 
                traci.simulationStep()
                
                # 1. Track Throughput (Cars that reached their destination this step)
                total_throughput += traci.simulation.getArrivedNumber()
                
                current_step_queue = 0
                for lane in incoming_lanes:
                    # 2. Track Waiting Time
                    total_waiting_time += traci.lane.getWaitingTime(lane)
                    # 3. Track CO2 Emissions (SUMO outputs mg/s)
                    total_co2_emissions += traci.lane.getCO2Emission(lane)
                    # 4. Track Queues
                    current_step_queue += traci.lane.getLastStepHaltingNumber(lane)
                
                if current_step_queue > max_queue_length:
                    max_queue_length = current_step_queue

                if traci.simulation.getMinExpectedNumber() <= 0: break

        # Convert CO2 from mg to kg for readability
        co2_kg = total_co2_emissions / 1000000.0

        print(f"\n" + "="*50)
        print(f"🎓 PHD-LEVEL EVALUATION REPORT")
        print(f"="*50)
        print(f"🚦 Total Accumulated Wait Time: {total_waiting_time:,.2f} seconds")
        print(f"🚗 Intersection Throughput:     {total_throughput} vehicles cleared")
        print(f"🛑 Maximum Queue Recorded:      {max_queue_length} vehicles at once")
        print(f"🌍 Total CO2 Emissions:         {co2_kg:,.2f} kg of CO2")
        print(f"="*50 + "\n")
        
    except Exception as e:
        print(f"\n🚨 ERROR: {e}")
    finally:
        traci.close()

if __name__ == "__main__":
    run_test()