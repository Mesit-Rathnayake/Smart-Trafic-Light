import matplotlib.pyplot as plt

# Your data
labels = ['Fixed-Time Baseline', 'AI (DQN) Controller']
times = [2319962, 30482]

# Create the bar chart
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(labels, times)

# Add text labels above the bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 50000, 
            f'{int(yval):,} sec', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Formatting the chart
ax.set_ylabel('Total Accumulated Waiting Time (Seconds)', fontsize=12)
ax.set_title('Performance Comparison: Peak Traffic Scenario', fontsize=14, fontweight='bold')
ax.set_ylim(0, 2600000) # Give room for the top label
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Save the image
plt.savefig('results_chart.png', dpi=300, bbox_inches='tight')
print("Chart saved as results_chart.png")