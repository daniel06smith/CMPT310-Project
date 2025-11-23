# CMPT 310 Project
## Car Racing RL with DQN
By: Amir Matianiu, Daniel Smith, Jim Chen, Diar Shakimov

### Overview
---
Custom racing environment using pygame and wraping with Gymnasium API. Trains the agent using DQN. Plots are kept in the runs folder.

### Requirements
---
```bash
pip install numpy gymnasium pygame torch torchvision matplotlib 
```

### Track selection/creation
Using the provided shapes, you can make custom tracks here: https://docs.google.com/presentation/d/1ciblqaFEaYMehfQGdd6yzgFPScmP8V11G4hfrsuJBuQ/edit?usp=sharing (just copy and paste the shapes in your program)

### Training
---
```bash
python agent.py [hyperparameter.yml] --train

# Example
python agent.py racer1 --train
```

### Test
---
```bash
python agent.py racer1
```
