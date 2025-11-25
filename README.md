# CMPT 310 Project
## Car Racing RL with DQN
By: Amir Matianiu, Daniel Smith, Jim Chen, Diar Shakimov

A Pygame racing environment with Gymnasium API for training RL agents using LIDAR sensors.

## Features

- **Gymnasium-compatible API** - Standard RL interface for easy integration
- **Headless mode support** - Fast training without rendering (1000s of steps/second)
- **LIDAR-based observation** - 5 distance sensors for environment perception
- **Checkpoint system** - Track progress through colored checkpoints
- **Multiple tracks** - Support for different track layouts, make yours [HERE](https://docs.google.com/presentation/d/1ciblqaFEaYMehfQGdd6yzgFPScmP8V11G4hfrsuJBuQ/edit?slide=id.p#slide=id.p)
- **Collision detection** - Rotated hitbox-based collision system

## Installation

```bash
pip install pygame gymnasium numpy torch matplotlib pyyaml
```

## Quick Start

### Training
```bash
python agent.py your_hyperparameters --train
```

### Testing/Visualization
```bash
python agent.py your_hyperparameters
```

### Basic Usage
Checkout the [env_demo](env_demo.py)

## Env Specifications

### Action Space: Discrete(6)

| Action | Description |
|--------|-------------|
| 0 | Steer left |
| 1 | Steer right |
| 2 | Accelerate |
| 3 | Steer left + Accelerate |
| 4 | Steer right + Accelerate |
| 5 | Do nothing |

### Observation Space: Box(5,)

5 normalized LIDAR distances [0.0-1.0] at angles: -60°, -30°, 0°, +30°, +60°
- 0.0 = far from obstacle
- 1.0 = obstacle adjacent

### Rewards

| Event | Reward | Notes |
|-------|--------|-------|
| Speed | +0.1 × speed | Max +0.8/step |
| Stationary | -0.1 | Speed < 0.5 |
| Collision | -1.0 | Episode ends |
| Checkpoint | +10.0 | 5 per lap |
| Complete Lap | +50.0 | All checkpoints |

### Car Dynamics

- Max Speed: 8.0 | Acceleration: 0.25 | Friction: 5% | Turn: 4°/action

## Files

```
racer.py                 # Environment
agent.py                 # DQN trainer
hyperparameters.yml      # Config
tracks/track*.png        # Track images
runs/                    # Training outputs
```