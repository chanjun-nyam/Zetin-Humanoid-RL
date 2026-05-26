from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils import configclass

from dataclasses import MISSING
from typing import List

from bipedal_lab.base.managers import (
    ActionManagerCfg,
    ObservationManagerCfg,
    RewardManagerCfg,
    RobotDataManagerCfg,
    TerminationManagerCfg,
)



@configclass
class BipedalEnvCfg(DirectRLEnvCfg):
    """Configuration class for `BipedalEnv`.
    """

    # config terms derived from `DirectRLEnv`
    # only required fields are rewritten

    sim: SimulationCfg = MISSING

    decimation: int = MISSING

    episode_length_s: float = MISSING

    scene: InteractiveSceneCfg = MISSING

    action_space: int = MISSING

    observation_space: int = MISSING

    # config terms derived from `BipedalEnv`

    action_scale: float | List[float] = MISSING

    robot_cfg: SceneEntityCfg = MISSING

    rdm_cfg: RobotDataManagerCfg = MISSING

    act_cfg: ActionManagerCfg = MISSING

    obs_cfg: ObservationManagerCfg = MISSING

    rwd_cfg: RewardManagerCfg = MISSING

    ter_cfg: TerminationManagerCfg = MISSING
