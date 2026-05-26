from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.envs import DirectRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from dataclasses import MISSING
import math

import torch as th



@configclass
class TerminationManagerCfg:
    """Configuration class for `TerminationManager`.
    """

    ar_robot: SceneEntityCfg = MISSING
    """`SceneEntityCfg` of `Articulation`.
    """

    co_termin: SceneEntityCfg = MISSING
    """`SceneEntityCfg` of `ContactSensor`.

    Requires:
        `body_names`: Bodies which trigger termination when contact normal force exceeds the `max_normal_force`.
    """

    max_tilt_angle: float = MISSING
    """Terminate when tilting angle[degree] of robot's root is greater than this value.
    """

    max_normal_force: float = MISSING
    """Terminate when any body in `sensor_cfg.body_names` experience normal force greater than this value.
    """

    normal_force_history_length: int = MISSING
    """History length which normal force is computed across.
    """

    max_episode_length: int = MISSING
    """Truncate when episode length reaches this value.
    """



class TerminationManager:
    """Manager class which handles termination and truncation of environment.
    """


    def __init__(self, cfg: TerminationManagerCfg, env: DirectRLEnv):
        """Initialize the manager.

        Args:
            cfg (TerminationManagerCfg): Configuration instance for the manager.
            env (DirectRLEnv): Environment instance.
        """
        self.cfg = cfg
        self.env = env

        self.max_tilt_cos = math.cos(math.radians(self.cfg.max_tilt_angle))
        self.max_normal_force_sq = self.cfg.max_normal_force ** 2

        self.robot: Articulation = self.env.scene[self.cfg.ar_robot.name]
        self.cont_snsr: ContactSensor = self.env.scene[self.cfg.co_termin.name]
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
            self.robot.data.projected_gravity_b,
            self.robot.data.GRAVITY_VEC_W,
        ), dim=-1)

        nforce_hist = self.cont_snsr.data.net_forces_w_history[:,:,self.cfg.co_termin.body_ids,:]
        nforce = nforce_hist[:,:self.cfg.normal_force_history_length,:,:].mean(dim=1)
        nforce_sq = nforce.square().sum(dim=-1).amax(dim=-1)

        # termination
        self._terminated.copy_(
            (robot_tilt_cos < self.max_tilt_cos) |
            (nforce_sq > self.max_normal_force_sq)
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
