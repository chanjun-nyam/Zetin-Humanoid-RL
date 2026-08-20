from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils import configclass

from dataclasses import MISSING
from typing import Tuple, List, Dict, Callable

import torch as th

from .managers import (
    ActionManagerCfg,
    CommandManagerCfg,
    RandomizeManagerCfg,
    RewardManagerCfg,
    RobotDataManagerCfg,
    TerminationManagerCfg,
)



@configclass
class BipedalEnvCfg(DirectRLEnvCfg):
    """Configuration class for `BipedalEnv`.
    """

    ar_robot: SceneEntityCfg = MISSING
    """`SceneEntityCfg` of `Articulation`.
    """

    @configclass
    class SubTerrainCfg:
        prop: float = MISSING
        check: Callable[[th.Tensor], th.Tensor] = MISSING

    sub_terrains: Dict[str, SubTerrainCfg] = MISSING
    """Sub-terrain configurations.
    """

    vel_err_sma_window: int = MISSING
    """Window size for velocity error sma buffer.
    """

    foll_boundary: float = MISSING
    """When norm of xy-velocity error is less than the product of norm of xy-command and `foll_boundary`, it is defined as **following** state.
    """

    foll_hyst: Tuple[float, float] = MISSING
    """Hysteresis range of following rate.
    """

    max_stride: float = MISSING
    """Maximum stride distance which defines feasible input command.
    """

    rdm_cfg: RobotDataManagerCfg = MISSING
    """Configuration for `RobotDataManager`.
    """

    act_cfg: ActionManagerCfg = MISSING
    """Configuration for `ActionManager`.
    """

    cmd_cfg: CommandManagerCfg = MISSING
    """Configuration for `CommandManager`.
    """

    rnd_cfg: RandomizeManagerCfg = MISSING
    """Configuration for `RandomizeManager`.
    """

    rwd_cfg: RewardManagerCfg = MISSING
    """Configuration for `RewardManager`.
    """

    ter_cfg: TerminationManagerCfg = MISSING
    """Configuration for `TerminationManager`.
    """

    obs_q_names: List[str] = MISSING
