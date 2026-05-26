from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, VecEnvObs, VecEnvStepReturn

import torch as th

from typing import Tuple
from collections.abc import Sequence

from bipedal_lab.base.env_cfg import BipedalEnvCfg
from bipedal_lab.base.utils import direct_rl_env_extended_step
from bipedal_lab.base.managers import (
    ActionManager,
    ObservationManager,
    RewardManager,
    RobotDataManager,
    TerminationManager,
)



class BipedalEnv(DirectRLEnv):
    """Base class for bipedal environment.
    """


    def __init__(self, cfg: BipedalEnvCfg, render_mode: str | None = None):
        """Initialization.

        Args:
            cfg (BipedalEnvCfg): configuration
            render_mode (str | None, optional): The render mode for the environment. Defaults to None.

        Raises:
            ValueError:
                If action space or observation space not matches to vector space.
                Only vector space is supported.
        """
        super().__init__(cfg, render_mode)
        self.cfg: BipedalEnvCfg
        
        # action space and observation space validity
        if not isinstance(self.cfg.action_space, int):
            raise ValueError('Only vector action space is supported.')
        if not isinstance(self.cfg.observation_space, int):
            raise ValueError('Only vector observation space is supported.')
        
        # dimension values
        self.n_env = self.num_envs
        self.n_act = self.cfg.action_space
        self.n_obs = self.cfg.observation_space
        
        # initialize managers
        self.rdm = RobotDataManager(self.cfg.rdm_cfg, self)
        self.act_mgr = ActionManager(self.cfg.act_cfg, self)
        self.obs_mgr = ObservationManager(self.cfg.obs_cfg, self.rdm)
        self.rwd_mgr = RewardManager(self.cfg.rwd_cfg, self, self.rdm)
        self.ter_mgr = TerminationManager(self.cfg.ter_cfg, self)

        # TODO: temporary implementation of command
        self.command = th.rand(self.n_env, 3, dtype=th.float32, device=self.device) * 2 - 1

        # action scale tensor
        self.act_scale = th.tensor(self.cfg.action_scale, dtype=th.float32, device=self.device)

        # access to step_info before any call of step if undefined behavior
        self.step_info = None
    
    
    def _setup_scene(self):
        self.robot: Articulation = self.scene[self.cfg.robot_cfg.name]


    def _pre_physics_step(self, action: th.Tensor):
        # update action manager
        self.act_mgr.update_action(action)

        # update command manager
        # command resample TODO: temporary implementation
        resample_ids = (self.episode_length_buf % 300 == 0).nonzero(as_tuple=False).squeeze(-1)
        self.command[resample_ids,:] = th.rand(resample_ids.shape[0], 3, dtype=th.float32, device=self.device) * 2 - 1

        # clear step info dictionary
        self.step_info = {}

        self.physics_step_cnt_ = 0


    def _apply_action(self):
        # update delayed action
        self.act_mgr.update()

        # compute setpoint
        setpoint = self.rdm.qpos_default + self.act_mgr.act_delayed * self.act_scale

        # set setpoint of pd-controller for joints
        self.robot.set_joint_position_target(setpoint)


    def _post_apply_action(self):
        self.rdm.update(self.physics_step_cnt_ == 3)
        self.physics_step_cnt_ += 1


    def _get_dones(self) -> Tuple[th.Tensor, th.Tensor]:
        # update termination manager
        self.ter_mgr.update()
        # update done info
        self.step_info['done'] = self.ter_mgr.info
        return (
            self.ter_mgr.terminated,
            self.ter_mgr.truncated,
        )
    

    def _get_rewards(self) -> th.Tensor:
        # update reward manager
        self.rwd_mgr.update(self.command, self.act_mgr.act_diff(o=1), self.act_mgr.act_diff(o=2), self.ter_mgr.terminated)
        # update reward info
        self.step_info['reward'] = self.rwd_mgr.info
        
        return self.rwd_mgr.reward
    
    
    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        
        # reset root and joint state
        def_root_state = self.robot.data.default_root_state[env_ids,:]
        def_root_state[:,:3] += self.scene.env_origins[env_ids]

        def_qpos = self.robot.data.default_joint_pos[env_ids,:]
        def_qvel = self.robot.data.default_joint_vel[env_ids,:]

        self.robot.write_root_state_to_sim(def_root_state, env_ids)
        self.robot.write_joint_state_to_sim(def_qpos, def_qvel, None, env_ids)

        # reset managers
        self.rdm.reset(env_ids)
        self.act_mgr.reset(env_ids)
        self.obs_mgr.reset(env_ids)


    def _get_observations(self) -> VecEnvObs:
        # update observation manager
        self.obs_mgr.update(self.act_mgr.act, self.command)

        # update extras (identical to step)
        self.extras = self.step_info

        return {'policy': self.obs_mgr.obs_hist.view(self.n_env, self.n_obs).clone().detach()}
    

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass


    def step(self, action: th.Tensor) -> VecEnvStepReturn:
        """Advance the environment for one step.

        Args:
            action (th.Tensor): Action tensor with shape (n_env, n_act).

        Returns:
            VecEnvStepReturn: Tuple of observation, reward, termination, truncation, info.
        """
        return direct_rl_env_extended_step(self, action)
