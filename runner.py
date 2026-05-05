import os
import sys
import traci

# 1. Environment Setup: Ensure SUMO_HOME is correctly linked to your D: drive
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME' as per setup_instructions.txt")

def get_state():
    """
    Retrieves the 37-dimension state vector defined in Section 4.3.
    """
    # Organized list based on Edge IDs: South, North, West, East
    incoming_lanes = [
        "B2C_0", "B2C_1", "B2C_2", # South
        "T2C_0", "T2C_1", "T2C_2", # North
        "L2C_0", "L2C_1", "L2C_2", # West
        "R2C_0", "R2C_1", "R2C_2"  # East
    ]
    
    state = []
    for lane in incoming_lanes:
        # Extracting metrics for each lane
        veh_count = traci.lane.getLastStepVehicleNumber(lane)
        queue_len = traci.lane.getLastStepHaltingNumber(lane)  # Number of halting vehicles
        wait_time = traci.lane.getWaitingTime(lane)
        
        state.extend([veh_count, queue_len, wait_time])
    
    # 37th variable: Current traffic signal phase
    current_phase = traci.trafficlight.getPhase("center")
    state.append(current_phase)
    
    return state

def get_reward():
    """
    Calculates the reward to minimize congestion as per Section 4.5.
    """
    incoming_lanes = ["B2C_0", "B2C_1", "B2C_2", "T2C_0", "T2C_1", "T2C_2", 
                      "L2C_0", "L2C_1", "L2C_2", "R2C_0", "R2C_1", "R2C_2"]
    
    total_wait = 0
    total_queue = 0
    
    for lane in incoming_lanes:
        total_wait += traci.lane.getWaitingTime(lane)
        total_queue += traci.lane.getLastStepHaltingNumber(lane)

    # Negative reward for high waiting time and long queues
    return -(total_wait + total_queue)

def run():
    # Start the simulation with GUI
    try:
        traci.start(["sumo-gui", "-c", "simulation.sumocfg"])
        
        step = 0
        max_steps = 3600  # Match simulation end time in config
        
        while step < max_steps and traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            
            # Observe and print the state/reward every 10 steps
            if step % 10 == 0:
                current_state = get_state()
                current_reward = get_reward()
                
                # Print feedback to verify data extraction is working
                print(f"Step: {step} | Reward: {current_reward:.2f} | State Size: {len(current_state)}")
                
            step += 1
        
        print(f"Simulation completed successfully after {step} steps")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        try:
            traci.close()
        except:
            pass

if __name__ == "__main__":
    run()