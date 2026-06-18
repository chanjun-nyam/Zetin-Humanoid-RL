from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils import configclass

from dataclasses import MISSING
from typing import List, Tuple, Dict, Callable

import torch as th

from bipedal_lab.base.managers import (
    ActionManagerCfg,
    CommandManagerCfg,
    ObservationManagerCfg,
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

    rdm_cfg: RobotDataManagerCfg = MISSING
    """Configuration for `RobotDataManager`.
    """

    act_cfg: ActionManagerCfg = MISSING
    """Configuration for `ActionManager`.
    """

    cmd_cfg: CommandManagerCfg = MISSING
    """Configuration for `CommandManager`.
    """

    obs_cfg: ObservationManagerCfg = MISSING
    """Configuration for `ObservationManager`.
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
