from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from collections.abc import Sequence
import torch as th

from bipedal_lab.base.utils import SMABuffer
from bipedal_lab.base.math_utils import (
    twist_swing_decomposition,
    quat_apply_inv,
)



@configclass
class ArticulationDataManagerCfg:
    """Configuration class for `ArticulationDataManager`.
    """

    asset_cfg: SceneEntityCfg

    n_window: int



class ArticulationDataManager:
    """Manager class which process articulation data and track some useful quantities.
    """


    def __init__(self, cfg: ArticulationDataManagerCfg, env: DirectMARLEnv):
        """Initialization.

        Args:
            cfg (ArticulationDataManagerCfg): configuration
            env (DirectMARLEnv): environment
        """
        self.cfg = cfg
        self.env = env

        self.asset_cfg = self.cfg.asset_cfg
        self.asset: Articulation = self.env.scene[self.asset_cfg.name]

        self._root_quat_w: th.Tensor
        self._gravity_dir_b: th.Tensor
        self._root_quat_w_twist: th.Tensor
        self._root_quat_w_swing: th.Tensor
        self._inst_root_linvel_t: th.Tensor
        self._compute()

        # sma buffers
        self._qvel = SMABuffer.init_like(self.asset.data.joint_vel, self.cfg.n_window)
        self._qtau = SMABuffer.init_like(self.asset.data.applied_torque, self.cfg.n_window)
        self._root_linvel_t = SMABuffer.init_like(self.asset.data.root_lin_vel_w, self.cfg.n_window)
        self._root_angvel_b = SMABuffer.init_like(self.asset.data.root_ang_vel_b, self.cfg.n_window)

        # TODO: consider applying rotation coning compensation for _root_angvel_b


    def _compute(self):
        self._root_quat_w = self.asset.data.root_quat_w
        self._gravity_dir_b = self.asset.data.projected_gravity_b

        # compute linear velocity respect to root's twist quaternion frame
        self._root_quat_w_twist, self._root_quat_w_swing = twist_swing_decomposition(self._root_quat_w)
        self._inst_root_linvel_t = quat_apply_inv(self._root_quat_w_twist, self.asset.data.root_lin_vel_w)
    

    def update(self):
        """Update.
        """
        self._compute()

        # update sma buffers
        self._qvel.update(self.asset.data.joint_vel)
        self._qtau.update(self.asset.data.applied_torque)
        self._root_linvel_t.update(self._inst_root_linvel_t)
        self._root_angvel_b.update(self.asset.data.root_ang_vel_b)


    def reset(self, env_ids: Sequence[int]):
        """Reset.

        Args:
            env_ids (Sequence[int]): Sequence of environment indices to reset.
        """
        self._compute()

        # reset sma buffers
        self._qvel.reset(env_ids, self.asset.data.joint_vel[env_ids,:])
        self._qtau.reset(env_ids, self.asset.data.applied_torque[env_ids,:])
        self._root_linvel_t.reset(env_ids, self._inst_root_linvel_t[env_ids,:])
        self._root_angvel_b.reset(env_ids, self.asset.data.root_ang_vel_b[env_ids,:])
    

    @property
    def root_quat_w(self):
        """Identical to `isaaclab.assets.ArticulationData.root_quat_w`
        """
        return self._root_quat_w
    

    @property
    def gravity_dir_b(self):
        """Identical to `isaaclab.assets.ArticulationData.projected_gravity_b`.
        """
        return self._gravity_dir_b
    

    @property
    def root_linvel_t(self):
        """SMA of linear velocity which is computed respect to root twist quaternion frame.
        """
        return self._root_linvel_t.sma
    

    @property
    def root_angvel_b(self):
        """SMA of `isaaclab.assets.ArticulationData.root_ang_vel_b`.
        """
        return self._root_angvel_b.sma


    @property
    def n_qdim(self):
        """Identical to `isaaclab.assets.Articulation.num_joints`.
        """
        return self.asset.num_joints
    

    @property
    def qpos(self):
        """Identical to `isaaclab.assets.ArticulationData.joint_pos`.
        """
        return self.asset.data.joint_pos
    

    @property
    def qvel(self):
        """SMA of `isaaclab.assets.ArticulationData.joint_vel`.
        """
        return self._qvel.sma
    

    @property
    def qtau(self):
        """SMA of `isaaclab.assets.ArticulationData.applied_torque`.
        """
        return self._qtau.sma
    

    @property
    def qpos_default(self):
        """Identical to `isaaclab.assets.ArticulationData.default_joint_pos`
        """
        return self.asset.data.default_joint_pos
    

    @property
    def qvel_default(self):
        """Identical to `isaaclab.assets.ArticulationData.default_joint_vel`
        """
        return self.asset.data.default_joint_vel
    

    @property
    def qpos_limit(self):
        """Identical to `isaaclab.assets.ArticulationData.joint_pos_limits`.
        """
        return self.asset.data.joint_pos_limits
