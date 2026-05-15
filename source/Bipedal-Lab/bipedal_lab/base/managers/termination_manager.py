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

    robot_cfg: SceneEntityCfg = MISSING
    """`SceneEntityCfg` for `Articulation`.

    Actually only name of it is necessary.
    """

    sensor_cfg: SceneEntityCfg = MISSING
    """`SceneEntityCfg` for `ContactSensor`.
    """

    max_tilt_angle: float = MISSING
    """Termination when robot tilting angle is grather than this value. [degree]
    """

    max_episode_length: int = MISSING
    """Truncation when episode length reach this value.
    """

    max_normal_force: float = MISSING
    """Termination when any body in `sensor_cfg.body_names` experiences normalforce grather than this value.
    """

    normal_force_history_length: int = MISSING
    """History length for final normal force computation.
    """



class TerminationManager:
    """Manager class which handles termination and truncation.
    """
    
    terminated: th.Tensor
    """Termination tensor with shape (n_env,).
    """

    truncated: th.Tensor
    """Truncation tensor with shape (n_env,).
    """

    info: dict
    """Dictionary containing informations for monitoring.
    """
    
    
    def __init__(self, cfg: TerminationManagerCfg, env: DirectRLEnv):
        """Initialization.

        Args:
            cfg (TerminationManagerCfg): configuration
            env (DirectRLEnv): rl environment
        """
        self.cfg = cfg
        self.env = env

        self.max_tilt_cos = math.cos(math.radians(self.cfg.max_tilt_angle))
        self.max_normal_force_sq = self.cfg.max_normal_force ** 2

        self.robot: Articulation = self.env.scene[self.cfg.robot_cfg.name]
        self.sensor: ContactSensor = self.env.scene[self.cfg.sensor_cfg.name]
        self.cfg.sensor_cfg.resolve(self.env.scene)

        # access to these tensor before any update call is undefined behavior
        self.terminated = None
        self.truncated = None


    def update(self):
        """Update.
        """
        robot_tilt_cos = th.sum(th.mul(
            self.robot.data.projected_gravity_b,
            self.robot.data.GRAVITY_VEC_W,
        ), dim=-1)

        nforce_hist = self.sensor.data.net_forces_w_history[:,:,self.cfg.sensor_cfg.body_ids,:]
        nforce = nforce_hist[:,:self.cfg.normal_force_history_length,:,:].mean(dim=1)
        nforce_sq = nforce.square().sum(dim=-1).amax(dim=-1)

        # termination
        self.terminated = th.logical_or(
            robot_tilt_cos < self.max_tilt_cos,
            nforce_sq > self.max_normal_force_sq,
        )

        # truncation
        self.truncated = self.env.episode_length_buf >= self.cfg.max_episode_length

        # info dict
        self.info = {
            'terminated': self.terminated.sum().item(),
            'truncated': self.truncated.sum().item(),
        }
