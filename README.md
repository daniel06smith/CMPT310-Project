# PPO/DQN RL AI Driving Project 
Group Project for CMPT 310 (Fall 2025)

Members: Daniel, Diar, Amir, Jim
---
### What is this project?
This project uses DQN and PPO to train a car to drive around an unseen racing track in the CarDriving-V3 enviroment in Gymnasium.

Installation
---
To replicate this project on your own (windows) device, follow the below steps:
---
1. Download [Anaconda](https://www.anaconda.com/download)
2. Download the [C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools)
* a) Launch the installer, and checkmark "Desktop Development with C++"
* b) Press "Install"
* c) (You may close the build tools window now)
3. Run the following lines in Anaconda Prompt:
```python
# Create virtual environment
conda create -n gymenv
# Activate the virtual environment
conda activate gymenv
# Install Python 3.11
conda install python=3.11
# Install SWIG + Gymnasium[bod2d]
conda install swig
conda install gymnasium[box2d]
```

Setting up VSCode
---
After creating `train_dqn.py`, press `ctrl + P` and select ">Python: Select Interpreter". Select the `gymenv` environment we created.

![Select the `gymenv` environment](image.png)

Next, open the terminal (`ctrl + ~`) - if your default terminal is `Windows Powershell`, then press `ctrl + P` and select ">Terminal: Select Default Profile". Select `Command Prompt`. Delete your powershell terminal, and open a new terminal.

-> Our `gymenv` environment will not launch in Windows Powershell

