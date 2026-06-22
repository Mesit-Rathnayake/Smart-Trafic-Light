import os
import sys
import traci

# ==========================================
# 1. ENVIRONMENT SETUP
# ==========================================
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# ==========================================
# 2. MAIN BASELINE LOOP
# ==========================================


def run_baseline():

    # Start SUMO automatically
    # Construct the exact path to sumo-gui using SUMO_HOME
    sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui')
    traci.start([sumo_binary, "-c", "simulation.sumocfg", "--start"])
    traci.simulationStep()

    total_waiting_time = 0
    step = 0

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            # We don't ask the AI for anything. We just let SUMO run its default lights.
            traci.simulationStep()

            # Track metrics for your final report
            for lane in ["B2C_0", "B2C_1", "B2C_2", "T2C_0", "T2C_1", "T2C_2", "L2C_0", "L2C_1", "L2C_2", "R2C_0", "R2C_1", "R2C_2"]:
                total_waiting_time += traci.lane.getWaitingTime(lane)

            step += 1
            if traci.simulation.getMinExpectedNumber() <= 0:
                break

        print(f"\n✅ Baseline Test Complete!")
        print(
            f"Total Accumulated Waiting Time (Fixed-Time): {total_waiting_time} seconds")

    except Exception as e:
        print(f"\n🚨 ERROR: {e}")
    finally:
        traci.close()


if __name__ == "__main__":
    run_baseline()
