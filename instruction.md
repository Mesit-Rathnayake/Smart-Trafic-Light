# 🚦 Intelligent Traffic Light Control using Reinforcement Learning

This repository contains a mini-project for **EC6301 - Artificial Intelligence** at the **University of Ruhuna**. The system uses **Q-Learning/Deep Q-Networks (DQN)** to optimize traffic signal timings in a simulated urban environment.

---

## 🛠️ Environment Setup

Follow these instructions to ensure your development environment matches the project requirements.

### 1. SUMO Installation (Secondary Drive Setup)
If your `C:` drive is full, you can install SUMO on another drive (e.g., `D:`).

1. **Download:** Get the latest installer from the [Official SUMO Website](https://sumo.dlr.de/docs/Downloads.php).
2. **Install:** Run the `.msi` and set the path to `D:\Eclipse\Sumo` (or your preferred directory).
3. **Set Environment Variables (Critical):**
   * Open **Edit the system environment variables**.
   * Click **Environment Variables**.
   * Under **System Variables**, click **New**:
     * **Variable Name:** `SUMO_HOME`
     * **Variable Value:** `D:\Eclipse\Sumo`
   * Find the **Path** variable, click **Edit**, then **New**, and add:
     * `D:\Eclipse\Sumo\bin`

### 2. Python Dependencies
Ensure you have Python 3.x installed, then run:
```bash
pip install traci sumolib torch numpy matplotlib