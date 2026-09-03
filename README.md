# **A2C-2: Autonomous Lunar Landing** 

# **DSCD 614: Reinforcement Learning — Group Project A2C-2** 

University of Ghana | Examiner: Dr. Michael Agbo Tettey Soli 

Submission Date: 4 September 2026 

# **Overview** 

This repository trains an **Advantage Actor-Critic (A2C)** agent to land a simulated spacecraft on the lunar surface using the Gymnasium LunarLander-v3 environment. The agent is evaluated against a hard-coded heuristic controller and standard statistical benchmarks across a strict 3- seed protocol. 

- **Algorithm:** A2C (PyTorch custom implementation) 

- **Environment:** LunarLander-v3 (Discrete action space, 8-D continuous observation space) 

- **Seeds:** 42, 101, 2024 

- **Compute budget:** 300 episodes per seed 

# **Repository Structure** 

GROUP-4_A2C-2-Lunar-Lander/ 

├── docs/ 

│ ├── Analysis_Report.docx 

│ ├── a2c_evaluation.mp4 

│ ├── comparison_bar.png 

│ ├── formulation.docx 

│ └── learning_curve.png 

├── logs/ 

│ ├── a2c_multiseed_results.csv    # Combined raw per-episode training logs 

│ ├── a2c_training_results.csv     # Individual run logs 

│ └── baseline_results.csv         # Baseline comparison metrics 

├── src/ 

│ ├── a2c_lunar_lander.pth         # Saved model weights for demonstration 

│ ├── evaluate.py                  # Deterministic 30-episode evaluation script 

│ ├── plot_results.py              # Generates presentation-ready figures 

│ └── train_a2c.py                 # Multi-seed training loop and entry point 

├── README.md 

├── REPORT.md 

└── requirements.txt 

# **Quickstart — Reproduce Everything in One Command** 

Bash 

git clone https://github.com/AgbekoEmmanuel/GROUP-4_A2C-2-Lunar-Lander.git cd GROUP-4_A2C-2-Lunar-Lander 

pip install -r requirements.txt python src/train_a2c.py 

This will: 

1. Execute the multi-seed training loop across seeds 42, 101, and 2024 (300 episodes each). 

2. Export the consolidated metrics to logs/a2c_multiseed_results.csv. 

3. Save the final actor-critic weights to src/a2c_lunar_lander.pth. 

4. Generate the performance learning curve inside docs/learning_curve.png. 

# **Installation** 

Bash 

pip install -r requirements.txt 

# **requirements.txt** 

gymnasium[box2d]>=0.29.0 torch>=2.2.0 swig>=4.1.0 pyvirtualdisplay>=3.0 

imageio>=2.30.0 imageio-ffmpeg>=0.4.8 numpy>=1.24.0 pandas>=2.0.0 matplotlib>=3.7.0 scipy>=1.10.0 Python 3.10+ required. 

# **Evaluation & Baseline Comparison** 

- **Protocol & Held-Out Data:** Evaluated over 90 total episodes (30 episodes per seed across seeds 42, 101, and 2024) serving as a strictly held-out test period. Both the heuristic baseline and the A2C agent were tested under identical conditions. 

- **Deterministic Evaluation:** Exploration was strictly disabled during evaluation by bypassing the categorical distribution sampler and using torch.argmax to select the highest-probability action at every step. 

- **Metric Summary:** Over the 90 evaluation episodes, the baseline heuristic achieved a mean reward of **243.03** ( _±_ 92.24), while the trained A2C agent achieved a mean reward of **-216.06** ( _±_ 129.27). 

- **Variation Analysis:** The observed performance difference strictly exceeds the variation across seeds, indicating a statistically significant gap due to early training limits (300 episodes). 

# **Hyperparameters** 

|**Hyperparameter**|**Value**|**Description**|
|---|---|---|
|**Learning Rate**|1_×_10<br>-3|Optimizer step size|
|**Discount Factor (**_γ_**)**|0.99|Reward horizon weighting|
|**Hidden Layers**|[128,128]|Two fully connected layers|
|**Activation**|ReLU|Hidden layer non-linearity|



|**Function**|||
|---|---|---|
|**Optimizer**|Adam|First-order gradient optimization|
|**Episodes**|300|Training duration per seed|
|**Evaluated Seeds**|42, 101,<br>2024|Ensured reproducibility protocol|



# **MDP Formulation** 

- 8 

- **State (** _s_ **):** ℝ vector containing horizontal/vertical coordinates, linear velocities, angle, angular velocity, and two left/right ground-contact touch sensors. 

- **Action (** _A_ **):** Discrete space of 4 options — 

   - {0:do nothing _,_ 1:fire left orientation engine _,_ 2:fire main engine _,_ 3:fire right orientation engine } . 

- **Reward Function:** Dense shaping based on distance to landing pad, velocity penalties, tilt penalties, plus a terminal reward of +100 or -100 for safe landing or crashing. 

- **Markov Property:** Fully holds as the kinematic state vector provides complete visibility into system dynamics without historical frame-stacking. 

# **Group Members** 

|**Name**|**Student ID**|**Contribution**|
|---|---|---|
|Emmanuel Agbeko|22424224|Network architecture design, training pipelines,|
|||and multi-seed logging scripts|
|Adams Diiwu Amadu|22424232|MDP formulation, environment structuring,|
|||scoping, and business requirement mapping|
|Dominic Adjin|22424159|Evaluation metrics framework, data|



visualization, and report compilation 

# **Submission Links** 

See `Submission_Links.txt` for: 

- GitHub repo URL and final commit SHA 

- YouTube presentation URL 

