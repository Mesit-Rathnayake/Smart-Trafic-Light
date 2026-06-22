import csv
import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CONSTANTS & EVALUATION METRICS
# ==========================================
PEAK_BASELINE_WAIT = 2319962.00
PEAK_AI_WAIT = 26488.00

OFFPEAK_BASELINE_WAIT = 104991.00
OFFPEAK_AI_WAIT = 1311.00


def moving_average(data, window_size=50):
    """Smoothes out noisy Reinforcement Learning learning curves."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def generate_dashboard(log_filepath, output_filename, title_suffix, baseline_val, ai_val, theme_color_primary, theme_color_secondary):
    """Generates a complete 4-panel training and performance dashboard for a given run."""
    steps = []
    epsilons = []
    rewards = []
    losses = []

    print(f"📖 Reading training log: {log_filepath}...")
    try:
        with open(log_filepath, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                steps.append(int(row[0]))
                epsilons.append(float(row[1]))
                rewards.append(float(row[2]))
                losses.append(float(row[3]))
    except FileNotFoundError:
        print(
            f"⚠️ Warning: Could not find {log_filepath}. Skipping this dashboard.")
        return False
    except Exception as e:
        print(f"🚨 Error reading {log_filepath}: {e}")
        return False

    steps = np.array(steps)
    rewards = np.array(rewards)
    losses = np.array(losses)

    # Use a clean academic style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        # Fallback theme if seaborn style is not installed
        plt.style.use('ggplot')

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f'Deep Q-Network (DQN) Training Dynamics & Metrics - {title_suffix}', fontsize=18, fontweight='bold', y=0.98)

    window = 50  # Moving average window size

    # --- PANEL 1: Cumulative Reward Curve (Learning Progress) ---
    axs[0, 0].plot(steps, rewards, alpha=0.25,
                   color=theme_color_primary, label='Raw Step Reward')
    if len(rewards) > window:
        axs[0, 0].plot(steps[window-1:], moving_average(rewards, window),
                       color=theme_color_secondary, linewidth=2.5, label=f'Trend ({window}-step Avg)')
    axs[0, 0].set_title(
        'Cumulative Reward Function (Traffic Penalty Minimization)', fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel('Simulated Training Steps')
    axs[0, 0].set_ylabel('Reward Value (Negative Waiting Time)')
    axs[0, 0].legend(loc='lower right')

    # --- PANEL 2: Neural Network Loss (Huber Loss Convergence) ---
    # Strip out zeros to prevent log-scale rendering errors
    valid_loss_indices = losses > 0
    if np.any(valid_loss_indices):
        axs[0, 1].plot(steps[valid_loss_indices], losses[valid_loss_indices],
                       alpha=0.25, color='#ff7f0e', label='Raw Huber Loss')
        if len(losses[valid_loss_indices]) > window:
            smoothed_loss = moving_average(losses[valid_loss_indices], window)
            axs[0, 1].plot(steps[valid_loss_indices][window-1:], smoothed_loss,
                           color='#cc4400', linewidth=2.5, label=f'Trend ({window}-step Avg)')
        axs[0, 1].set_yscale('log')
    else:
        axs[0, 1].plot(steps, losses, alpha=0.25,
                       color='#ff7f0e', label='Raw Huber Loss')
    axs[0, 1].set_title(
        'Neural Network Optimization Loss (Log-Scale)', fontsize=12, fontweight='bold')
    axs[0, 1].set_xlabel('Simulated Training Steps')
    axs[0, 1].set_ylabel('Smooth L1 Loss')
    axs[0, 1].legend(loc='upper right')

    # --- PANEL 3: Exploration Policy (Epsilon Decay Curve) ---
    axs[1, 0].plot(steps, epsilons, color='#2ca02c',
                   linewidth=2.5, label='Epsilon (ε)')
    axs[1, 0].fill_between(steps, epsilons, alpha=0.15, color='#2ca02c')
    axs[1, 0].set_title(
        'Exploration vs. Exploitation Rate (Epsilon Decay)', fontsize=12, fontweight='bold')
    axs[1, 0].set_xlabel('Simulated Training Steps')
    axs[1, 0].set_ylabel('Exploration Probability (ε-Greedy)')
    axs[1, 0].legend(loc='upper right')

    # --- PANEL 4: Performance Evaluation Comparison ---
    labels = ['Fixed-Time Baseline', 'Adaptive DQN AI']
    bar_values = [baseline_val, ai_val]
    colors = ['#d62728', theme_color_secondary]

    bars = axs[1, 1].bar(labels, bar_values, color=colors, width=0.5)
    axs[1, 1].set_title(
        'Total Accumulated Waiting Time Comparison', fontsize=12, fontweight='bold')
    axs[1, 1].set_ylabel('Seconds (Lower is Better)')

    # Calculate and display percentage optimization
    optimization_pct = ((baseline_val - ai_val) / baseline_val) * 100

    # Render value labels above the bar charts
    y_limits_max = max(bar_values) * 1.15
    axs[1, 1].set_ylim(0, y_limits_max)
    for bar in bars:
        height = bar.get_height()
        axs[1, 1].text(bar.get_x() + bar.get_width()/2., height + (max(bar_values) * 0.02),
                       f'{int(height):,} s', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Draw improvement overlay text
    axs[1, 1].text(0.5, max(bar_values) * 0.8, f'Optimization: -{optimization_pct:.2f}%',
                   ha='center', va='center', color='#006400', weight='bold', fontsize=13,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='#2ca02c', boxstyle='round,pad=0.5'))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Successful! Visual dashboard saved as: '{output_filename}'")
    return True


def main():
    print("🎨 Starting Dashboard Generation Pipeline...")

    # 1. Process Peak-Traffic Data
    peak_success = generate_dashboard(
        log_filepath='training_log_peak.csv',
        output_filename='dashboard_peak.png',
        title_suffix='Peak Traffic Profile',
        baseline_val=PEAK_BASELINE_WAIT,
        ai_val=PEAK_AI_WAIT,
        theme_color_primary='#1f77b4',
        theme_color_secondary='#00008b'
    )

    # 2. Process Off-Peak Traffic Data
    offpeak_success = generate_dashboard(
        log_filepath='training_log_offpeak.csv',
        output_filename='dashboard_offpeak.png',
        title_suffix='Off-Peak Traffic Profile',
        baseline_val=OFFPEAK_BASELINE_WAIT,
        ai_val=OFFPEAK_AI_WAIT,
        theme_color_primary='#17becf',
        theme_color_secondary='#008080'
    )

    if not peak_success and not offpeak_success:
        print("\n🚨 ERROR: No training logs were found! Please run runner_multi_peak.py or runner_multi_offpeak.py first.")
    else:
        print("\n🎉 All available dashboards have been generated! Ready for your project report.")


if __name__ == "__main__":
    main()
