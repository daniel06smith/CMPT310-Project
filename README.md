# PPO/DQN RL AI Driving Project 
Group Project for CMPT 310 (Fall 2025)

Members: Daniel, Diar, Amir, Jim
---
### What is this project?
This project uses DQN and PPO to train a car to drive around an unseen racing track in the CarRacing-v3 enviroment in Gymnasium.

Installation
---
To replicate this project on your own (windows) device, follow the below steps:
---
1. Download [Anaconda](https://www.anaconda.com/download) (required)
2. Run the following lines in Anaconda Prompt:
```bash
# Clone the repo
git clone https://github.com/daniel06smith/cmpt310-Project
cd cmpt310-Project

# Install Option A: 
conda create -f environment.yml
conda activate gymenv

# Install Option B:
conda create -n gymenv python=3.11 -y
conda activate gymenv
conda install -c conda-forge gymnasium gymnasium-box2d swig stable-baselines3

# Optional: to use Jupyter Notebooks
conda install -n gymenv ipykernel --update-deps --force-reinstall
```

Setting up VSCode
---
After creating `train_dqn.py`, press `ctrl + P` and select ">Python: Select Interpreter". Select the `gymenv` environment we created.

![Select the `gymenv` environment](image.png)

Next, open the terminal (`ctrl + ~`) - if your default terminal is `Windows Powershell`, then press `ctrl + P` and select ">Terminal: Select Default Profile". Select `Command Prompt`. Delete your powershell terminal, and open a new terminal.

-> Our `gymenv` environment will not launch in Windows Powershell