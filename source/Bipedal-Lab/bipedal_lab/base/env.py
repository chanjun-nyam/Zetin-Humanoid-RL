from isaaclab.envs import DirectRLEnv, VecEnvObs, VecEnvStepReturn
from isaaclab.assets import Articulation
from isaaclab.terrains import TerrainImporter

import torch as th

from typing import Tuple
from collections.abc import Sequence

from bipedal_lab.base.env_cfg import BipedalEnvCfg
from bipedal_lab.base.utils import (
    direct_rl_env_extended_step,
    distribute_list,
    SMABuffer,
)
from bipedal_lab.base.managers import (
    ActionManager,
    CommandManager,
    ObservationManager,
    RandomizeManager,
    RewardManager,
    RobotDataManager,
    TerminationManager,
)
from bipedal_lab.base.math_utils import (
    vec_sq_norm,
    quat_twist,
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

        self.ALL_INDICES = th.arange(self.n_env, dtype=th.int64, device=self.device) # (n_env,)

        self.TWIST_AXIS = th.tensor([0, 0, 1], dtype=th.float32, device=self.device).repeat(self.n_env, 1) # (n_env, 3)

        # initialize managers
        self.rdm = RobotDataManager(cfg=self.cfg.rdm_cfg, env=self)
        self.act_mgr = ActionManager(cfg=self.cfg.act_cfg, env=self)
        self.cmd_mgr = CommandManager(cfg=self.cfg.cmd_cfg, env=self)
        self.obs_mgr = ObservationManager(cfg=self.cfg.obs_cfg, rdm=self.rdm)
        self.rnd_mgr = RandomizeManager(cfg=self.cfg.rnd_cfg, env=self)
        self.rwd_mgr = RewardManager(cfg=self.cfg.rwd_cfg, env=self, rdm=self.rdm, cmd_mgr=self.cmd_mgr)
        self.ter_mgr = TerminationManager(cfg=self.cfg.ter_cfg, env=self, rdm=self.rdm)

        # curriculum init
        self._curriculum_init()

        # access to step_info before any call of step if undefined behavior
        self.step_info = None


    def _curriculum_init(self):
        # curriculum is enabled when terrain generator is used (and so terrain_origins is not None)
        self.curriculum_enabled = self.terrain.terrain_origins is not None

        # when curriculum is not enabled, we just randomly allocate env indices to each cell
        if not self.curriculum_enabled:
            self.cmd_mgr.init(home_cell_idx=th.randint_like(self.ALL_INDICES, self.cmd_mgr.n_cell))
            return

        # ---------- allocation code start ----------

        # when curriculum is enabled, env indices are allocated to each sub-terrain proportional to its configuration
        # next, in each sub-terrain, allocated env indices are re-allocated to each available cells uniformly

        # sub-terrain configurations of terrain generator's one and self.cfg's one
        gen_sub_terrains = self.terrain.cfg.terrain_generator.sub_terrains
        my_sub_terrains = self.cfg.sub_terrains

        # isaaclab's terrain generator was intended to uniformly allocate the env indices to each sub-terrain
        # and control the proportion of each sub-terrain numbers instead
        # however we are gonna use proportions defined self.cfg's sub-terrain configuration
        # to control the number of env indices allocated to each sub-terrain
        # now, as we don't need proportions to control the number of each sub-terrain,
        # we will assume proportions of terrain generator's one is fixed to 1.0 just for simplicity
        if any([s.proportion != 1.0 for s in gen_sub_terrains.values()]):
            raise ValueError('Other than 1.0 for SubTerrainBaseCfg.proportion is not allowed.')
        # check whether sub-terrain configuration of generator's one and self.cfg's one is maching
        if list(gen_sub_terrains.keys()) != list(my_sub_terrains.keys()):
            raise ValueError('Keys in sub-terrain configuration of generator\'s one and self.cfg\'s one is not match.')

        # just for convenient, we are gonna handle sub-terrain configuration
        # with it's key-list and value-list together instead of one dictionary
        self.sub_terrain_keys = list(gen_sub_terrains.keys())
        gen_sub_terrains = list(gen_sub_terrains.values())
        my_sub_terrains = list(my_sub_terrains.values())

        # the mapping we are building: every env gets exactly one sub-terrain and one cell
        env2terrain = [None] * self.n_env
        env2cell = [None] * self.n_env

        # level 1: allocate all_ids to each sub-terrain proportional to self.cfg's sub-terrain configuration
        all_ids = list(range(self.n_env))
        ids_per_terrain = distribute_list(all_ids, [s.prop for s in my_sub_terrains])

        for terrain_idx, (alloced_ids, my_sub_terrain) in enumerate(zip(ids_per_terrain, my_sub_terrains)):
            # fill the env2terrain mapping
            for alloced_id in alloced_ids:
                env2terrain[alloced_id] = terrain_idx

            # generate available cell ids for corresponding sub-terrain
            usable_cell_ids = my_sub_terrain.check(self.cmd_mgr._cell_rng).nonzero().squeeze(-1).tolist()
            # level 2: allocate alloced_ids to each available cell uniformly
            ids_per_cell = distribute_list(alloced_ids, [1.0] * len(usable_cell_ids))

            # fill the env2cell mapping
            for cell_idx, alloced2_ids in enumerate(ids_per_cell):
                for alloced2_id in alloced2_ids:
                    env2cell[alloced2_id] = usable_cell_ids[cell_idx]

        env2terrain = th.tensor(env2terrain, dtype=th.int64, device=self.device)
        env2cell = th.tensor(env2cell, dtype=th.int64, device=self.device)

        # commit the mapping to the buffers
        self.terrain.terrain_types.copy_(env2terrain)
        self.terrain.env_origins.copy_(
            self.terrain.terrain_origins[self.terrain.terrain_levels, env2terrain])
        self.cmd_mgr.init(home_cell_idx=env2cell)
        # ---------- allocation code end ----------

        # curriculum related buffers
        self.is_foll_cnt = th.zeros_like(self.ALL_INDICES) # (n_env,)
        self.move_updown = th.zeros_like(self.ALL_INDICES) # (n_env,)
        self.vel_err_buff = SMABuffer.init_like(self.cmd_mgr.cmd, (1,), self.cfg.vel_err_sma_window)


    def _setup_scene(self):
        self.robot: Articulation = self.scene[self.cfg.ar_robot.name]
        self.terrain: TerrainImporter = self.scene.terrain


    def _pre_physics_step(self, action: th.Tensor):
        # update action manager
        self.act_mgr.update_action(action=action)

        # update command manager
        twist_quat = quat_twist(self.rdm.root_quat_w, self.TWIST_AXIS)
        heading = th.atan2(*twist_quat[...,[3,0]].unbind(dim=-1)) * 2.0 # (n_env,)
        self.cmd_mgr.update(heading=heading.unsqueeze(-1))

        # update randomize manager
        self.rnd_mgr.update()

        # clear step info dictionary
        self.step_info = {}

        self.physics_step_cnt = 0


    def _apply_action(self):
        # update delayed action
        self.act_mgr.update()

        # compute setpoint
        setpoint = self.rdm.qpos_default + self.act_mgr.act_delayed * self.act_mgr.q_scale

        # set setpoint of pd-controller for joints
        self.robot.set_joint_position_target(setpoint)


    def _post_apply_action(self):
        self.rdm.update(last=self.physics_step_cnt == self.cfg.decimation - 1)
        self.physics_step_cnt += 1


    def _get_dones(self) -> Tuple[th.Tensor, th.Tensor]:
        # update termination manager
        self.ter_mgr.update()
        # update done info
        self.step_info['done'] = self.ter_mgr.info
        return (
            self.ter_mgr.terminated,
            self.ter_mgr.truncated,
        )


    def _curriculum_update(self):
        # exit when curriculum is not enabled
        if not self.curriculum_enabled:
            return

        # tensor of commanded velocity and real robot's velocity
        cmd = self.cmd_mgr.cmd # (n_env, n_cmd)
        vel = th.cat([
            self.rdm.root_linvel_b[:,0:2],
            self.rdm.root_angvel_b[:,2:3]], dim=-1) # (n_env, 3)

        # update velocity error sma-buffer
        self.vel_err_buff.update(cmd - vel)
        sma_err = self.vel_err_buff.sma

        # compute is_following
        is_foll = (vec_sq_norm(sma_err) <
                   vec_sq_norm(cmd) * (self.cfg.foll_boundary ** 2)) # (n_env,)

        # check phase-0 states
        phase0_start = (self.cmd_mgr.phase == 0) & self.cmd_mgr.phase_changed # (n_env,)
        phase0_end = (self.cmd_mgr.phase == 1) & self.cmd_mgr.phase_changed # (n_env,)
        is_phase0 = self.cmd_mgr.phase == 0 # (n_env,)

        # when phase-0 start
        self.is_foll_cnt[phase0_start] = 0

        # during phase-0
        self.is_foll_cnt[is_phase0 & is_foll] += 1

        # when phase-0 end
        foll_rate = self.is_foll_cnt / self.cfg.cmd_cfg.phase_len[0]
        self.move_updown[phase0_end & (foll_rate > self.cfg.foll_hyst[1])] = 1
        self.move_updown[phase0_end & (foll_rate < self.cfg.foll_hyst[0])] = -1

        # compute mean terrain_level for each terrain_type
        n_row, n_col = self.terrain.terrain_origins.shape[:2]
        mean_level = th.zeros(size=(n_col,), dtype=th.float32, device=self.device)
        mean_level.scatter_reduce_(
            dim=0,
            index=self.terrain.terrain_types,
            src=self.terrain.terrain_levels.to(th.float32),
            reduce='mean',
            include_self=True,
        )
        # update curriculum info
        self.step_info['curriculum'] = {
            'following_rate': is_foll.to(th.float32).mean().item(),
            'terrain_levels': {
                self.sub_terrain_keys[idx]: val for idx, val in enumerate(mean_level.tolist())
            },
        }


    def _get_rewards(self) -> th.Tensor:
        # update reward manager
        self.rwd_mgr.update(
            d_action=self.act_mgr.act_diff(o=1),
            d2_action=self.act_mgr.act_diff(o=2),
            terminated=self.ter_mgr.terminated,
        )
        # update reward info
        self.step_info['reward'] = self.rwd_mgr.info

        # curriculum update
        self._curriculum_update()

        return self.rwd_mgr.reward


    def _curriculum_reset(self, env_ids: Sequence[int]):
        # exit when curriculum is not enabled
        if not self.curriculum_enabled:
            return

        # adjust curriculum level based on move_updown buffer
        _move_updown = self.move_updown[env_ids]
        self.terrain.update_env_origins(
            env_ids=env_ids,
            move_up=_move_updown == 1,
            move_down=_move_updown == -1,
        )
        self.move_updown[env_ids] = 0

        # reset velocity error sma-buffer
        self.vel_err_buff.reset(env_ids)


    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

        # when all environment's episode is aligned, training curve is affected and showes periodical jitter
        # to prevent this, we randomly distribute the environment's episode offset
        if len(env_ids) == self.n_env:
            self.episode_length_buf.copy_(th.randint_like(self.ALL_INDICES, self.max_episode_length))

        # curriculum reset
        self._curriculum_reset(env_ids)

        # reset root and joint state
        init_root_state = self.robot.data.default_root_state[env_ids,:]
        init_root_state[:,0:3] += self.scene.env_origins[env_ids]

        self.robot.write_root_state_to_sim(
            root_state=init_root_state,
            env_ids=env_ids,
        )
        self.robot.write_joint_state_to_sim(
            position=self.rdm.qpos_default[env_ids,:],
            velocity=self.rdm.qvel_default[env_ids,:],
            joint_ids=None,
            env_ids=env_ids,
        )

        # reset managers
        self.rdm.reset(env_ids)
        self.act_mgr.reset(env_ids)
        self.obs_mgr.reset(env_ids)
        self.rwd_mgr.reset(env_ids)


    def _get_observations(self) -> VecEnvObs:
        # update observation manager
        self.obs_mgr.update(action=self.act_mgr.act)

        tmp_cmd = th.zeros_like(self.cmd_mgr.cmd,)
        tmp_cmd[:,0] = 1.0
        obs_tensor = th.cat([
            self.obs_mgr.obs_hist.view(self.n_env, -1),
            # self.cmd_mgr.cmd,
            tmp_cmd,
            self.obs_mgr.obs_priv,
        ], dim=-1)
        obs_tensor[:,-3:] = 0.0 # TODO

        # update extras
        self.extras = self.step_info

        # print(
        #     f'issac '
        #     f'quat: {" | ".join([f"{x:6.3f}" for x in self.rdm.root_quat_w.squeeze(0)])}     '
        #     f'linv: {" | ".join([f"{x:6.3f}" for x in self.rdm.root_linvel_b.squeeze(0)])}     '
        #     f'angv: {" | ".join([f"{x:6.3f}" for x in self.rdm.root_angvel_b.squeeze(0)])}     '
        #     f'qpos: {" | ".join([f"{x:6.3f}" for x in self.rdm.qpos.squeeze(0)])}     '
        #     f'qvel: {" | ".join([f"{x:6.3f}" for x in self.rdm.qvel.squeeze(0)])}     '
        #     f'qtrg: {" | ".join([f"{x:6.3f}" for x in self.robot._data.joint_pos_target.squeeze(0)])}'
        # )

        return {'policy': obs_tensor}


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
