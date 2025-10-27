# CMPT 310 Project
## Car Racing RL with DQN/PPO
By: Daniel Smith, Amir, Diar Shakimov, Jim Chen
---
### Requirements
1. Python 3.13.0
2. (Optional) UV for package handling ( `pip install uv` )

(uv is just a better package handler for Python, works faster)
---
### Installation
```bash
# setup + activate virtual environment
python -m venv .venv    
.venv\Scripts\Activate      # windows
source venv/bin/activate    # macOS/linux
# install required packages
pip install -r requirements.txt
```
optionally (using uv):
```bash
# setup + activate virtual environment
uv venv
.venv\Scripts\Activate
# install required packages
uv pip install -r requirements.txt
```
---
### Playing
```bash
python main.py
```
---

### Controls
* [W] to accelerate
* [S] to brake
* [A/D] to steer left/right