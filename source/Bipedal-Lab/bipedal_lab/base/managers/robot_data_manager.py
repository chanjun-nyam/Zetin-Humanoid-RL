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
    quat_mul,
    quat_conj,
    quat_twist,
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

        # dummy tensors for tensor/buffer initialization
        dummy_vec3 = th.zeros_like(self.robot.data.root_link_pos_w)
        dummy_vec4 = th.zeros_like(self.robot.data.root_link_quat_w)
        dummy_vecq = th.zeros_like(self.robot.data.joint_pos)

        dummy_cbody_vec3 = th.zeros_like(self.cont_snsr.data.net_forces_w)
        dummy_cbody_float = th.zeros_like(self.cont_snsr.data.current_air_time)
        dummy_cbody_bool = th.zeros_like(dummy_cbody_float, dtype=th.bool)

        # source tensors for defensive copying
        self._TWIST_AXIS = th.zeros_like(dummy_vec3)

        self._root_quat_w = th.zeros_like(dummy_vec4)
        self._twist_quat = th.zeros_like(dummy_vec4)
        self._swing_quat = th.zeros_like(dummy_vec4)

        self._gravity_dir_b = th.zeros_like(dummy_vec3)

        self._qpos = th.zeros_like(dummy_vecq)
        self._qpos_default = th.zeros_like(dummy_vecq)
        self._qvel_default = th.zeros_like(dummy_vecq)

        self._cont_force_w = th.zeros_like(dummy_cbody_vec3)

        self._is_cont = th.zeros_like(dummy_cbody_bool)
        self._is_air = th.zeros_like(dummy_cbody_bool)
        self._first_cont = th.zeros_like(dummy_cbody_bool)
        self._first_air = th.zeros_like(dummy_cbody_bool)

        if self.cont_snsr.cfg.track_air_time:
            self._cont_time = th.zeros_like(dummy_cbody_float)
            self._air_time = th.zeros_like(dummy_cbody_float)
            self._last_cont_time = th.zeros_like(dummy_cbody_float)
            self._last_air_time = th.zeros_like(dummy_cbody_float)
            self._cont_period = th.zeros_like(dummy_cbody_float)
            self._air_period = th.zeros_like(dummy_cbody_float)
        else:
            self._cont_time = None
            self._air_time = None
            self._last_cont_time = None
            self._last_air_time = None
            self._cont_period = None
            self._air_period = None

        # source buffers for defensive copying
        decimation = self.env.cfg.decimation

        self._root_linvel_b = SMABuffer.init_like(dummy_vec3, decimation)
        self._root_angvel_b = SMABuffer.init_like(dummy_vec3, decimation)
        self._root_linvel_t = SMABuffer.init_like(dummy_vec3, decimation)
        self._root_angvel_t = SMABuffer.init_like(dummy_vec3, decimation)

        self._qvel = SMABuffer.init_like(dummy_vecq, decimation)
        self._qtau = SMABuffer.init_like(dummy_vecq, decimation)

        # initialize tensors and buffers
        self.ALL_INDICES = th.arange(dummy_vec3.shape[0], dtype=th.int64, device=dummy_vec3.device)
        self.reset(self.ALL_INDICES)

        # TODO: implement angvel decomposition for twist-swing decomposed frame
        # TODO: consider applying rotation coning compensation for root_angvel_b


    def _update_ar_source_tensors(self):
        self._TWIST_AXIS[:,:] = 0.0
        self._TWIST_AXIS[:,2] = 1.0

        self._root_quat_w.copy_(self.robot.data.root_quat_w)
        self._twist_quat.copy_(quat_twist(self._root_quat_w, self._TWIST_AXIS))
        self._swing_quat.copy_(quat_mul(quat_conj(self._twist_quat), self._root_quat_w))
        
        self._gravity_dir_b.copy_(self.robot.data.projected_gravity_b)

        self._qpos.copy_(self.robot.data.joint_pos)
        self._qpos_default.copy_(self.robot.data.default_joint_pos)
        self._qvel_default.copy_(self.robot.data.default_joint_vel)


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


    def update(self, last: bool):
        """Update the manager.

        Args:
            last (bool): Whether this `update` call is the last one of local sim loop.

        Note: This needs to be called at the simulation cycle, not the policy cycle.
        """
        self._update_ar_source_tensors()
        if last:
            self._update_co_source_tensors()

        # update source buffers
        root_linvel_t = quat_apply(quat_conj(self._twist_quat), self.robot.data.root_lin_vel_w)
        root_angvel_t = self.robot.data.root_ang_vel_b

        self._root_linvel_b.update(self.robot.data.root_lin_vel_b)
        self._root_angvel_b.update(self.robot.data.root_ang_vel_b)
        self._root_linvel_t.update(root_linvel_t)
        self._root_angvel_t.update(root_angvel_t)

        self._qvel.update(self.robot.data.joint_vel)
        self._qtau.update(self.robot.data.applied_torque)


    def reset(self, env_ids: Sequence[int]):
        """Reset.

        Args:
            env_ids (Sequence[int]): Environment indices to reset.
        """
        self._update_ar_source_tensors()

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

        # update source buffers
        root_linvel_t = quat_apply(quat_conj(self._twist_quat), self.robot.data.root_lin_vel_w)
        root_angvel_t = self.robot.data.root_ang_vel_b

        self._root_linvel_b.reset(env_ids, self.robot.data.root_lin_vel_b[env_ids])
        self._root_angvel_b.reset(env_ids, self.robot.data.root_ang_vel_b[env_ids])
        self._root_linvel_t.reset(env_ids, root_linvel_t[env_ids])
        self._root_angvel_t.reset(env_ids, root_angvel_t[env_ids])

        self._qvel.reset(env_ids, self.robot.data.joint_vel[env_ids])
        self._qtau.reset(env_ids, self.robot.data.applied_torque[env_ids])


    @property
    def n_qdim(self):
        """DoF of articulation.
        """
        return self.robot.num_joints
    
    
    @property
    def TWIST_AXIS(self):
        """Twist axis of twist-swing decomposition. Shape is (n_env, 3).

        It is constant vector of (0, 0, 1).
        """
        return self._TWIST_AXIS
    

    @property
    def root_quat_w(self):
        """Rotation of root in world frame. Shape is (n_env, 4).
        """
        return self._root_quat_w
    

    @property
    def twist_quat(self):
        """Twist component of `root_quat_w`. Shape is (n_env, 4).
        """
        return self._twist_quat
    

    @property
    def swing_quat(self):
        """Swing component of `root_quat_w`. Shape is (n_env, 4).

        Note:
            Swing component is decomposed after the twist; $q = q_t q_s$.
        """
        return self._swing_quat
    

    @property
    def gravity_dir_b(self):
        """Gravity unit vector measured in base frame. Shape is (n_env, 3).
        """
        return self._gravity_dir_b
    

    @property
    def root_linvel_b(self):
        """Linear velocity measured in base frame. Shape is (n_env, 3).
        """
        return self._root_linvel_b.sma
    

    @property
    def root_angvel_b(self):
        """Angular velocity measured in base frame. Shape is (n_env, 3).
        """
        return self._root_angvel_b.sma


    @property
    def root_linvel_t(self):
        """Linear velocity measured in twist frame. Shape is (n_env, 3).
        """
        return self._root_linvel_t.sma
    

    @property
    def root_angvel_t(self):
        """Angular velocity measured in twist frame. Shape is (n_env, 3).
        """
        return self._root_angvel_t.sma


    @property
    def qpos(self):
        """Generalized coordinates of articulation. Shape is (n_env, n_jnt).
        """
        return self._qpos
    

    @property
    def qvel(self):
        """Generalized velocities of articulation. Shape is (n_env, n_jnt).
        """
        return self._qvel.sma
    

    @property
    def qtau(self):
        """Generalized force/torque of articulation. Shape is (n_env, n_jnt).
        """
        return self._qtau.sma
    

    @property
    def qpos_default(self):
        """Default value for `qpos`. Shape is (n_env, n_jnt).
        """
        return self._qpos_default
    

    @property
    def qvel_default(self):
        """Default value for `qvel`. Shape is (n_env, n_jnt).
        """
        return self._qvel_default
    

    @property
    def cont_force_w(self):
        """Contact normal force in world frame. Shape is (n_env, n_body, 3).
        """
        return self._cont_force_w
    

    @property
    def is_cont(self):
        """Whether the body is in contact. Shape is (n_env, n_body).
        """
        return self._is_cont
    

    @property
    def is_air(self):
        """Whether the body is not in contact. Shape is (n_env, n_body).
        """
        return self._is_air
    

    @property
    def first_cont(self):
        """Whether the body just made contact. Shape is (n_env, n_body).
        """
        return self._first_cont
    

    @property
    def first_air(self):
        """Whether the body just made not contact. Shape is (n_env, n_body).
        """
        return self._first_air
    

    @property
    def cont_time(self):
        """Current contact time. Shape is (n_env, n_body).
        """
        return self._cont_time
    

    @property
    def air_time(self):
        """Current air time. Shape is (n_env, n_body).
        """
        return self._air_time
    

    @property
    def last_cont_time(self):
        """Last contact time. Shape is (n_env, n_body).
        """
        return self._last_cont_time
    

    @property
    def last_air_time(self):
        """Last air time. Shape is (n_env, n_body).
        """
        return self._last_air_time
    

    @property
    def cont_period(self):
        """Contact period. Shape is (n_env, n_body).
        """
        return self._cont_period
    

    @property
    def air_period(self):
        """Air period. Shape is (n_env, n_body).
        """
        return self._air_period
