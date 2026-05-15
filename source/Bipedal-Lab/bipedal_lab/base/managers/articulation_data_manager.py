from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from collections.abc import Sequence
from dataclasses import MISSING

import torch as th

from bipedal_lab.base.utils import SMABuffer
from bipedal_lab.base.math_utils import (
    quat_apply,
    quat_mul,
    quat_conj,
    quat_twist,
)



@configclass
class ArticulationDataManagerCfg:
    """Configuration class for `ArticulationDataManager`.
    """

    asset_cfg: SceneEntityCfg = MISSING
    """`SceneEntityCfg` for `Articulation`.

    Actually only name of it is necessary.
    """

    n_window: int = MISSING
    """Size of the SMA buffer window.
    """



class ArticulationDataManager:
    """Manager class which process articulation data and track some useful quantities.
    """


    def __init__(self, cfg: ArticulationDataManagerCfg, env: DirectRLEnv):
        """Initialization.

        Args:
            cfg (ArticulationDataManagerCfg): configuration
            env (DirectRLEnv): environment
        """
        self.cfg = cfg
        self.env = env

        self.asset: Articulation = self.env.scene[self.cfg.asset_cfg.name]

        self.VEC3_Z = th.zeros_like(self.asset.data.root_pos_w)
        self.VEC3_Z[:,2] = 1.0

        self._root_quat_w: th.Tensor
        self._gravity_dir_b: th.Tensor
        self._twist_quat: th.Tensor
        self._swing_quat: th.Tensor
        self._inst_twist_linvel: th.Tensor
        self._compute()

        # sma buffers
        self._qvel = SMABuffer.init_like(self.asset.data.joint_vel, self.cfg.n_window)
        self._qtau = SMABuffer.init_like(self.asset.data.applied_torque, self.cfg.n_window)
        self._root_linvel_t = SMABuffer.init_like(self.asset.data.root_lin_vel_w, self.cfg.n_window)
        self._root_angvel_b = SMABuffer.init_like(self.asset.data.root_ang_vel_b, self.cfg.n_window)

        # TODO: consider applying rotation coning compensation for _root_angvel_b

        print('articulation data manager ||||||||||||||||||||||||||')
        print(self.asset.joint_names)


    def _compute(self):
        self._root_quat_w = self.asset.data.root_quat_w
        self._gravity_dir_b = self.asset.data.projected_gravity_b

        # twist-swing decomposition
        self._twist_quat = quat_twist(self._root_quat_w, self.VEC3_Z)
        self._swing_quat = quat_mul(quat_conj(self._twist_quat), self._root_quat_w)
        
        # instantanious linear velocity in twist quaternion frame
        self._inst_root_linvel_t = quat_apply(quat_conj(self._twist_quat), self.asset.data.root_lin_vel_w)

        # TODO: angular velocity of twist quaternion
    

    def update(self):
        """Update.

        Note:
            This function should be called at the simulation cycle, not the policy cycle.
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
    def twist_quat(self):
        """Twist quaternion of `root_quat_w`.
        """
        return self._twist_quat
    

    @property
    def swing_quat(self):
        """Swing quaternion of `root_quat_w`.

        Note:
            Decomposition order is twist-swing.
        """
        return self._swing_quat
    

    @property
    def gravity_dir_b(self):
        """Identical to `isaaclab.assets.ArticulationData.projected_gravity_b`.
        """
        return self._gravity_dir_b
    

    @property
    def root_linvel_t(self):
        """SMA of root's linear velocity in twist quaternion frame.
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
