from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.envs import DirectRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from dataclasses import MISSING
import math

import torch as th

from .robot_data_manager import RobotDataManager



@configclass
class TerminationManagerCfg:
    """Configuration class for `TerminationManager`.
    """

    co_termin: SceneEntityCfg = MISSING
    """`SceneEntityCfg` of `ContactSensor`.

    Requires:
        `body_names`: Bodies which trigger termination when contact occurs.
    """

    max_tilt_angle: float = MISSING
    """Terminate when tilting angle[degree] of robot's root is greater than this value.
    """

    max_episode_length: int = MISSING
    """Truncate when episode length reaches this value.
    """



class TerminationManager:
    """Manager class which handles termination and truncation of environment.
    """


    def __init__(self, cfg: TerminationManagerCfg, env: DirectRLEnv, rdm: RobotDataManager):
        """Initialize the manager.

        Args:
            cfg (TerminationManagerCfg): Configuration instance for the manager.
            env (DirectRLEnv): Environment instance.
            rdm (RobotDataManager): `RobotDataManager` instance.
        """
        self.cfg = cfg
        self.env = env
        self.rdm = rdm

        self.max_tilt_cos = math.cos(math.radians(self.cfg.max_tilt_angle))
        self.cfg.co_termin.resolve(self.env.scene)

        dummy_done = th.zeros((env.num_envs,), dtype=th.bool, device=env.device)

        # source buffers for defensive copying
        self._terminated = th.zeros_like(dummy_done)
        self._truncated = th.zeros_like(dummy_done)

        self._info = {}

        # TODO: episode length buff uniform randomization


    def update(self):
        """Update the manager.
        """
        robot_tilt_cos = th.sum(th.mul(
            self.rdm.gravity_dir_b,
            self.rdm.GRAVITY_DIR_W,
        ), dim=-1)

        # termination
        self._terminated.copy_(
            (robot_tilt_cos < self.max_tilt_cos) |
            (self.rdm.is_cont[:,self.cfg.co_termin.body_ids].any(dim=-1))
        )

        # truncation
        self._truncated.copy_(self.env.episode_length_buf >= self.cfg.max_episode_length)

        # info dict
        self._info = {
            'terminated': self._terminated.sum().item(),
            'truncated': self._truncated.sum().item(),
        }


    @property
    def terminated(self):
        """Terminated tensor. Shape is (n_env,).
        """
        return self._terminated.clone()


    @property
    def truncated(self):
        """Truncated tensor. Shape is (n_env,).
        """
        return self._truncated.clone()


    @property
    def info(self):
        """Info dictionary.
        """
        return self._info
