import csv
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD THE DATA
# ==========================================
steps = []
epsilons = []
rewards = []
losses = []

try:
    with open('training_log.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            steps.append(int(row[0]))
            epsilons.append(float(row[1]))
            rewards.append(float(row[2]))
            losses.append(float(row[3]))
except FileNotFoundError:
    print("🚨 ERROR: training_log.csv not found! Did you run advanced_runner.py?")
    exit()

# Convert to numpy arrays for math operations
steps = np.array(steps)
rewards = np.array(rewards)
losses = np.array(losses)

# Helper function to smooth out the noisy Reinforcement Learning data
def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# ==========================================
# 2. CREATE THE DASHBOARD
# ==========================================
# Set global style for academic look
plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Deep Q-Network (DQN) Training Dynamics & Evaluation', fontsize=20, fontweight='bold', y=0.98)

window = 50 # Smoothing window

# --- GRAPH 1: The Learning Curve (Reward) ---
axs[0, 0].plot(steps, rewards, alpha=0.3, color='#1f77b4', label='Raw Reward')
if len(rewards) > window:
    axs[0, 0].plot(steps[window-1:], moving_average(rewards, window), color='#00008b', linewidth=2, label=f'Trend ({window}-step Avg)')
axs[0, 0].set_title('Cumulative Reward over Time (Learning Curve)', fontsize=14, fontweight='bold')
axs[0, 0].set_xlabel('Training Steps')
axs[0, 0].set_ylabel('Reward (Negative Wait Time)')
axs[0, 0].legend()

# --- GRAPH 2: Neural Network Convergence (Loss) ---
axs[0, 1].plot(steps, losses, alpha=0.3, color='#ff7f0e', label='Raw Loss')
if len(losses) > window:
    axs[0, 1].plot(steps[window-1:], moving_average(losses, window), color='#cc4400', linewidth=2, label=f'Trend ({window}-step Avg)')
axs[0, 1].set_title('Neural Network Loss (Huber Loss)', fontsize=14, fontweight='bold')
axs[0, 1].set_xlabel('Training Steps')
axs[0, 1].set_ylabel('Loss Value')
axs[0, 1].set_yscale('log') # Log scale is standard for viewing NN loss
axs[0, 1].legend()

# --- GRAPH 3: Exploration vs. Exploitation (Epsilon Decay) ---
axs[1, 0].plot(steps, epsilons, color='#2ca02c', linewidth=3)
axs[1, 0].set_title('Epsilon Decay (Exploration → Exploitation)', fontsize=14, fontweight='bold')
axs[1, 0].set_xlabel('Training Steps')
axs[1, 0].set_ylabel('Probability of Random Action')
axs[1, 0].fill_between(steps, epsilons, alpha=0.2, color='#2ca02c')

# --- GRAPH 4: Final Impact (Baseline vs AI) ---
# NOTE: These are your numbers from our previous chats!
labels = ['Fixed-Time Baseline', 'DQN Adaptive AI']
values = [2319962, 31452] 
colors = ['#d62728', '#1f77b4']

bars = axs[1, 1].bar(labels, values, color=colors)
axs[1, 1].set_title('Total Accumulated Waiting Time', fontsize=14, fontweight='bold')
axs[1, 1].set_ylabel('Seconds (Lower is Better)')

# Add commas to the numbers on top of the bars
for bar in bars:
    yval = bar.get_height()
    axs[1, 1].text(bar.get_x() + bar.get_width()/2, yval + 50000, 
            f'{int(yval):,} sec', ha='center', va='bottom', fontsize=12, fontweight='bold')

# ==========================================
# 3. SAVE AND SHOW
# ==========================================
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('phd_dashboard.png', dpi=300, bbox_inches='tight')
print("\n✅ Dashboard successfully generated and saved as 'phd_dashboard.png'!")
plt.show()