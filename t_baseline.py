import os
import sys
import traci

# ==========================================
# 1. ENVIRONMENT SETUP
# ==========================================
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# ==========================================
# 2. MAIN BASELINE LOOP
# ==========================================
def run_baseline():
    # Use the T-Junction configuration
    traci.start(["sumo-gui", "-c", "t_simulation.sumocfg", "--start"])
    traci.simulationStep()
    
    total_waiting_time = 0
    
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            # No AI here. Just let SUMO run its default fixed-timer lights.
            traci.simulationStep()
            
            # Track wait times for the 9 lanes
            for lane in ["B2C_0", "B2C_1", "B2C_2", "L2C_0", "L2C_1", "L2C_2", "R2C_0", "R2C_1", "R2C_2"]:
                total_waiting_time += traci.lane.getWaitingTime(lane)
            
            if traci.simulation.getMinExpectedNumber() <= 0: break

        print(f"\n✅ T-Junction Baseline Test Complete!")
        print(f"Total Accumulated Waiting Time (Fixed-Time): {total_waiting_time} seconds")
        
    except Exception as e:
        print(f"\n🚨 ERROR: {e}")
    finally:
        traci.close()

if __name__ == "__main__":
    run_baseline()