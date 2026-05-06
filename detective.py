import os
import sys
import traci

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)

# Start SUMO without the GUI just to read the data
traci.start(["sumo", "-c", "simulation.sumocfg"])

print("\n=== DETECTIVE REPORT ===")
print(f"Traffic Light IDs: {traci.trafficlight.getIDList()}")

# Get all lanes, but only filter the ones ending in '2C' to check your incoming lanes
all_lanes = traci.lane.getIDList()
incoming = [lane for lane in all_lanes if "2C" in lane]
print(f"Valid Incoming Lanes: {incoming}")
print("========================\n")

traci.close()