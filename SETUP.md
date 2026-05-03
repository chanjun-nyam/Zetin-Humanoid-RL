# Setup Manual


## Requirements
- [Isaac Sim 5.1.0](https://github.com/isaac-sim)
- [Isaac Lab 2.3.2](https://github.com/isaac-sim/IsaacLab)
- [uv](https://github.com/astral-sh/uv) (we assume using uv for python package manager)


## Python environment
```Bash
# modify to directory where your isaaclab is installed
ISAACLAB_PTH="/home/chanjun/workspace/NVIDIA-Omniverse/IsaacLab"

# create python venv
uv sync

# install isaaclab packages
source ./.venv/bin/activate
"${ISAACLAB_PTH}/isaaclab.sh" --install
deactivate

# reinforcement learning environment which we implemented
uv pip install -e ./source/Bipedal-Lab

# reinforcement learning library
uv pip install git+https://github.com/chanjun-nyam/Simple-RL.git

# NVIDIA-GPU monitoring tool
uv pip install nvitop

# setup isaacsim packages (this must be run for every new terminal)
source ./load_isaacsim_package.sh
```

```
uv pip install mkdocs mkdocs-material "mkdocstrings[python]"
```


## Visual Studio Code workspace
```Bash
TODO
```
