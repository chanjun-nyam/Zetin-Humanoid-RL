from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.envs import DirectRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from collections.abc import Sequence
from dataclasses import MISSING

import torch as th

from bipedal_lab.base.utils import SMABuffer
from bipedal_lab.base.math_utils import (
    quat_apply,
    quat_conj,
    vec_sq_norm,
)



@configclass
class RobotDataManagerCfg:
    """Configuration class for `RobotDataManager`.
    """

    ar_robot: SceneEntityCfg = MISSING
    """`SceneEntityCfg` of `Articulation`.
    """

    co_robot: SceneEntityCfg = MISSING
    """`SceneEntityCfg` of `ContactSensor`.
    """



class RobotDataManager:
    """Manager class which track some useful quantities for robot respect to policy loop.

    Note:
        Tensors provided by this manager don't change their id.
        Which means, identical tensor will only change the value after the update call.
        Therefore, clone the tensor when you need to track it over across the update calls.
    """


    def __init__(self, cfg: RobotDataManagerCfg, env: DirectRLEnv):
        """Initialize the manager.

        Args:
            cfg (ArticulationDataManagerCfg): Configuration instance for the manager.
            env (DirectRLEnv): Environment instance.
        
        Raises:
            ValueError: When history length of contact sensor is not equal with decimation of environment.
        """
        self.cfg = cfg
        self.env = env

        self.robot: Articulation = self.env.scene[self.cfg.ar_robot.name]
        self.cont_snsr: ContactSensor = self.env.scene[self.cfg.co_robot.name]

        if self.cont_snsr.cfg.history_length != self.env.cfg.decimation:
            raise ValueError('We restrict history length of contact sensor to be decimation of environment.')

        self._n_env = env.num_envs
        self._n_body = self.robot.num_bodies
        self._n_cbody = self.cont_snsr.num_bodies
        self._n_qdim = self.robot.num_joints

        self._body_names = self.robot.body_names
        self._cbody_names = self.cont_snsr.body_names
        self._q_names = self.robot.joint_names

        decim = self.env.cfg.decimation

        def _zeros(*args, dtype=th.float32, device=env.device):
            return th.zeros(size=args, dtype=dtype, device=device)

        # source tensors/buffers for defensive copying
        # ----- contact -----
        self._cont_force_w = _zeros(self.n_env, self.n_cbody, 3)

        self._is_cont = _zeros(self.n_env, self.n_cbody, dtype=th.bool)
        self._is_air = _zeros(self.n_env, self.n_cbody, dtype=th.bool)
        self._first_cont = _zeros(self.n_env, self.n_cbody, dtype=th.bool)
        self._first_air = _zeros(self.n_env, self.n_cbody, dtype=th.bool)

        if self.cont_snsr.cfg.track_air_time:
            self._cont_time = _zeros(self.n_env, self.n_cbody)
            self._air_time = _zeros(self.n_env, self.n_cbody)
            self._last_cont_time = _zeros(self.n_env, self.n_cbody)
            self._last_air_time = _zeros(self.n_env, self.n_cbody)
            self._cont_period = _zeros(self.n_env, self.n_cbody)
            self._air_period = _zeros(self.n_env, self.n_cbody)
        else:
            self._cont_time = None
            self._air_time = None
            self._last_cont_time = None
            self._last_air_time = None
            self._cont_period = None
            self._air_period = None

        # ----- root -----
        self._root_pos_w = _zeros(self.n_env, 3)
        self._root_quat_w = _zeros(self.n_env, 4)
        self._root_linvel_w = SMABuffer.init_like(_zeros(self.n_env, 3), (1,), decim)
        self._root_angvel_w = SMABuffer.init_like(_zeros(self.n_env, 3), (1,), decim)
        self._root_linvel_b = _zeros(self.n_env, 3)
        self._root_angvel_b = _zeros(self.n_env, 3)

        # ----- body -----
        self._body_pos_w = _zeros(self.n_env, self.n_body, 3)
        self._body_quat_w = _zeros(self.n_env, self.n_body, 4)
        self._body_linvel_w = SMABuffer.init_like(_zeros(self.n_env, self.n_body, 3), (1, 2), decim)
        self._body_angvel_w = SMABuffer.init_like(_zeros(self.n_env, self.n_body, 3), (1, 2), decim)
        self._body_linvel_b = _zeros(self.n_env, self.n_body, 3)
        self._body_angvel_b = _zeros(self.n_env, self.n_body, 3)

        # ----- gravity -----
        self._GRAVITY_DIR_W = _zeros(self.n_env, 3)
        self._gravity_dir_b = _zeros(self.n_env, 3)

        # ----- q(generalized coordinates) -----
        self._qpos = _zeros(self.n_env, self.n_qdim)
        self._qvel = SMABuffer.init_like(_zeros(self.n_env, self.n_qdim), (1,), decim)
        self._qtau = SMABuffer.init_like(_zeros(self.n_env, self.n_qdim), (1,), decim)
        self._qpos_default = _zeros(self.n_env, self.n_qdim)
        self._qvel_default = _zeros(self.n_env, self.n_qdim)

        # initialize tensors and buffers
        self.ALL_INDICES = th.arange(self.n_env, dtype=th.int64, device=env.device)
        self.reset(self.ALL_INDICES)


    def _update_co_source_tensors(self):
        self._cont_force_w.copy_(self.cont_snsr.data.net_forces_w_history.mean(dim=1))

        last_is_cont = self._is_cont.clone()
        last_is_air = self._is_air.clone()

        self._is_cont.copy_(vec_sq_norm(self._cont_force_w) > self.cont_snsr.cfg.force_threshold ** 2)
        self._is_air.copy_(self._is_cont.logical_not())
        self._first_cont.copy_(th.logical_and(self._is_cont, last_is_air))
        self._first_air.copy_(th.logical_and(self._is_air, last_is_cont))

        if self.cont_snsr.cfg.track_air_time:
            self._cont_period.copy_(th.where(
                self._first_cont,
                self._air_time + self._last_cont_time,
                self._cont_period,
            ))
            self._air_period.copy_(th.where(
                self._first_air,
                self._cont_time + self._last_air_time,
                self._air_period,
            ))
            self._last_cont_time.copy_(th.where(
                self._first_air,
                self._cont_time,
                self._last_cont_time,
            ))
            self._last_air_time.copy_(th.where(
                self._first_cont,
                self._air_time,
                self._last_air_time,
            ))
            self._cont_time.copy_(th.where(
                self._is_cont,
                self._cont_time + self.env.step_dt,
                0.0,
            ))
            self._air_time.copy_(th.where(
                self._is_air,
                self._air_time + self.env.step_dt,
                0.0,
            ))


    def _update_ar_source_tensors(self):
        # ----- root -----
        self._root_pos_w.copy_(self.robot.data.root_pos_w)
        self._root_quat_w.copy_(self.robot.data.root_quat_w)
        self._root_linvel_b.copy_(quat_apply(quat_conj(self._root_quat_w), self._root_linvel_w.sma))
        self._root_angvel_b.copy_(quat_apply(quat_conj(self._root_quat_w), self._root_angvel_w.sma))

        # ----- body -----
        self._body_pos_w.copy_(self.robot.data.body_pos_w)
        self._body_quat_w.copy_(self.robot.data.body_quat_w)
        self._body_linvel_b.copy_(quat_apply(quat_conj(self._body_quat_w), self._body_linvel_w.sma))
        self._body_angvel_b.copy_(quat_apply(quat_conj(self._body_quat_w), self._body_angvel_w.sma))

        # ----- gravity -----
        self._GRAVITY_DIR_W.copy_(self.robot.data.GRAVITY_VEC_W)
        self._gravity_dir_b.copy_(quat_apply(quat_conj(self._root_quat_w), self._GRAVITY_DIR_W))

        # ----- q(generalized coordinates) -----
        self._qpos.copy_(self.robot.data.joint_pos)
        self._qpos_default.copy_(self.robot.data.default_joint_pos)
        self._qvel_default.copy_(self.robot.data.default_joint_vel)


    def _update_ar_source_buffers(self):
        # ----- root -----
        self._root_linvel_w.update(self.robot.data.root_lin_vel_w)
        self._root_angvel_w.update(self.robot.data.root_ang_vel_w)

        # ----- body -----
        self._body_linvel_w.update(self.robot.data.body_lin_vel_w)
        self._body_angvel_w.update(self.robot.data.body_ang_vel_w)

        # ----- q(generalized coordinates) -----
        self._qvel.update(self.robot.data.joint_vel)
        self._qtau.update(self.robot.data.applied_torque)


    def update(self, last: bool):
        """Update the manager.

        Args:
            last (bool): Whether this `update` call is the last one of local sim loop.

        Note: This needs to be called at the simulation cycle, not the policy cycle.
        """
        self._update_ar_source_buffers()

        if last:
            self._update_co_source_tensors()
            self._update_ar_source_tensors()


    def reset(self, env_ids: Sequence[int]):
        """Reset the manager.

        Args:
            env_ids (Sequence[int]): Environment indices to reset.
        """
        # ----- contact -----
        self._cont_force_w[env_ids] = 0.0
        self._is_cont[env_ids] = False
        self._is_air[env_ids] = False
        self._first_cont[env_ids] = False
        self._first_air[env_ids] = False

        if self.cont_snsr.cfg.track_air_time:
            self._cont_time[env_ids] = 0.0
            self._air_time[env_ids] = 0.0
            self._last_cont_time[env_ids] = 0.0
            self._last_air_time[env_ids] = 0.0
            self._cont_period[env_ids] = 0.0
            self._air_period[env_ids] = 0.0

        # reset source buffers
        # ----- root -----
        self._root_linvel_w.reset(env_ids, self.robot.data.root_lin_vel_w[env_ids])
        self._root_angvel_w.reset(env_ids, self.robot.data.root_ang_vel_w[env_ids])

        # ----- body -----
        self._body_linvel_w.reset(env_ids, self.robot.data.body_lin_vel_w[env_ids])
        self._body_angvel_w.reset(env_ids, self.robot.data.body_ang_vel_w[env_ids])

        # ----- q(generalized coordinates) -----
        self._qvel.reset(env_ids, self.robot.data.joint_vel[env_ids])
        self._qtau.reset(env_ids, self.robot.data.applied_torque[env_ids])

        # reset source tensors
        self._update_ar_source_tensors()


    # ----- numbers -----
    @property
    def n_env(self):
        """Number of environments."""
        return self._n_env

    @property
    def n_body(self):
        """Number of bodies in articulation."""
        return self._n_body

    @property
    def n_cbody(self):
        """Number of bodies in contact sensor."""
        return self._n_cbody
    
    @property
    def n_qdim(self):
        """Number of joints in articulation"""
        return self._n_qdim

    # ----- names -----
    @property
    def body_names(self):
        """Names of bodies in articulation."""
        return self._body_names

    @property
    def cbody_names(self):
        """Names of bodies in contact sensor."""
        return self._cbody_names

    @property
    def q_names(self):
        """Names of joints in articulation."""
        return self._q_names

    # ----- contact -----
    @property
    def cont_force_w(self):
        """Contact normal force in world frame. Shape is (n_env, n_cbody, 3)."""
        return self._cont_force_w

    @property
    def is_cont(self):
        """Whether the body is in contact. Shape is (n_env, n_cbody)."""
        return self._is_cont

    @property
    def is_air(self):
        """Whether the body is not in contact. Shape is (n_env, n_cbody)."""
        return self._is_air

    @property
    def first_cont(self):
        """Whether the body just made contact. Shape is (n_env, n_cbody)."""
        return self._first_cont

    @property
    def first_air(self):
        """Whether the body just made not contact. Shape is (n_env, n_cbody)."""
        return self._first_air

    @property
    def cont_time(self):
        """Current contact time. Shape is (n_env, n_cbody)."""
        return self._cont_time

    @property
    def air_time(self):
        """Current air time. Shape is (n_env, n_cbody)."""
        return self._air_time

    @property
    def last_cont_time(self):
        """Last contact time. Shape is (n_env, n_cbody)."""
        return self._last_cont_time

    @property
    def last_air_time(self):
        """Last air time. Shape is (n_env, n_cbody)."""
        return self._last_air_time

    @property
    def cont_period(self):
        """Contact period. Shape is (n_env, n_cbody)."""
        return self._cont_period

    @property
    def air_period(self):
        """Air period. Shape is (n_env, n_cbody)."""
        return self._air_period

    # ----- root -----
    @property
    def root_pos_w(self):
        """Position of root in world frame. Shape is (n_env, 3)."""
        return self._root_pos_w

    @property
    def root_quat_w(self):
        """Rotation of root in world frame. Shape is (n_env, 4)."""
        return self._root_quat_w

    @property
    def root_linvel_w(self):
        """Linear velocity of root in world frame. Shape is (n_env, 3)."""
        return self._root_linvel_w.sma

    @property
    def root_angvel_w(self):
        """Angular velocity of root in world frame. Shape is (n_env, 3)."""
        return self._root_angvel_w.sma

    @property
    def root_linvel_b(self):
        """Linear velocity of root in base(root) frame. Shape is (n_env, 3)."""
        return self._root_linvel_b

    @property
    def root_angvel_b(self):
        """Angular velocity of root in base(root) frame. Shape is (n_env, 3)."""
        return self._root_angvel_b

    # ----- body -----
    @property
    def body_pos_w(self):
        """Position of body in world frame. Shape is (n_env, n_body, 3)."""
        return self._body_pos_w

    @property
    def body_quat_w(self):
        """Rotation of body in world frame. Shape is (n_env, n_body, 4)."""
        return self._body_quat_w

    @property
    def body_linvel_w(self):
        """Linear velocity of body in world frame. Shape is (n_env, n_body, 3)."""
        return self._body_linvel_w.sma

    @property
    def body_angvel_w(self):
        """Angular velocity of body in world frame. Shape is (n_env, n_body, 3)."""
        return self._body_angvel_w.sma

    @property
    def body_linvel_b(self):
        """Linear velocity of body in base(body) frame. Shape is (n_env, n_body, 3)."""
        return self._body_linvel_b

    @property
    def body_angvel_b(self):
        """Angular velocity of body in base(body) frame. Shape is (n_env, n_body, 3)."""
        return self._body_angvel_b

    # ----- gravity -----
    @property
    def GRAVITY_DIR_W(self):
        """Gravity unit vector measured in world frame. Shape is (n_env, 3)."""
        return self._GRAVITY_DIR_W

    @property
    def gravity_dir_b(self):
        """Gravity unit vector measured in base frame. Shape is (n_env, 3)."""
        return self._gravity_dir_b

    # ----- q(generalized coordinates) -----
    @property
    def qpos(self):
        """Generalized coordinates of articulation. Shape is (n_env, n_qdim)."""
        return self._qpos

    @property
    def qvel(self):
        """Generalized velocities of articulation. Shape is (n_env, n_qdim)."""
        return self._qvel.sma

    @property
    def qtau(self):
        """Generalized force/torque of articulation. Shape is (n_env, n_qdim)."""
        return self._qtau.sma

    @property
    def qpos_default(self):
        """Default value for `qpos`. Shape is (n_env, n_qdim)."""
        return self._qpos_default


    @property
    def qvel_default(self):
        """Default value for `qvel`. Shape is (n_env, n_qdim)."""
        return self._qvel_default
