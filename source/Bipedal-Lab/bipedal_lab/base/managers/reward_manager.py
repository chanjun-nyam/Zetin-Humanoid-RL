from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from collections.abc import Sequence
from dataclasses import MISSING
from typing import Tuple, List

import torch as th

from .robot_data_manager import RobotDataManager
from .command_manager import CommandManager
from bipedal_lab.base.math_utils import (
    vec_sq_norm,
    vec_norm,
)
from bipedal_lab.base.utils import SMABuffer



@configclass
class RewardManagerCfg:
    """Configuration class for `RewardManager`.
    """

    ar_foot: SceneEntityCfg = MISSING
    """`SceneEntityCfg` of `Articulation`

    Requires:
        `body_names`: Bodies of foot.
        `preserve_order`: `True`
    """

    co_body: SceneEntityCfg = MISSING
    """`SceneEntityCfg` of `ContactSensor`

    Requires:
        `body_names`: Bodies to penalty the contact.
    """

    co_foot: SceneEntityCfg = MISSING
    """`SceneEntityCfg` of `ContactSensor`

    Requires:
        `body_names`: Bodies of foot.
        `preserve_order`: `True`
    """

    vel_err_sma_window: int = MISSING

    track_lin_err_scale: float = MISSING

    track_ang_err_scale: float = MISSING

    q_names: List[str] = MISSING

    qpos_limit: List[Tuple[float, float]] = MISSING
    
    qtau_limit: List[Tuple[float, float]] = MISSING

    foot_stance_z: float = MISSING

    foot_clear_z: float = MISSING

    foot_min_air_ratio: float = MISSING

    foot_min_period: float = MISSING

    reward_clip: Tuple[float, float] = MISSING

    min_mean_reward: float = MISSING

    # ----- tracking -----
    k_track_lin: float = MISSING

    k_track_ang: float = MISSING

    # ----- motion penalty -----
    k_pen_lin: float = MISSING

    k_pen_ang: float = MISSING

    k_upright: float = MISSING

    # ----- dof penalty -----
    k_mec_energy: float = MISSING

    k_the_energy: float = MISSING

    k_d_action: float = MISSING

    k_d2_action: float = MISSING

    k_qpos_limit: float = MISSING

    k_qtau_limit: float = MISSING

    # ----- foot -----
    k_foot_clear: float = MISSING

    k_foot_ratio: float = MISSING

    k_foot_period: float = MISSING

    k_foot_slip: float = MISSING

    # ----- extras -----
    k_contact: float = MISSING

    k_termin: float = MISSING



class RewardManager:
    """Manager class which computes reward.
    """


    def __init__(self, cfg: RewardManagerCfg, env: DirectRLEnv, rdm: RobotDataManager, cmd_mgr: CommandManager):
        """Initialize the manager.

        Args:
            cfg (RewardManagerCfg): Configuration instance for the manager.
            env (DirectRLEnv): Environment instance.
            rdm (RobotDataManager): `RobotDataManager` instance.
            cmd_mgr (CommandManager): `CommandManager` instance.

        Raises:
            ValueError: If more than one of `Articulation` or `ContactSensor` scene entity is given.
            ValueError: When `n_cmd` of `CommandManager` is not `3`; when command is not typical velocity tracking command.
        """
        self.cfg = cfg
        self.env = env
        self.rdm = rdm
        self.cmd_mgr = cmd_mgr
        self.cfg_dict = cfg.to_dict()

        if not (cmd_mgr.n_cmd == 3):
            raise ValueError('Only velocity tracking command with form of (x_linvel, y_linvel, z_angvel) is supported')

        self.cfg.ar_foot.resolve(env.scene)
        self.cfg.co_body.resolve(env.scene)
        self.cfg.co_foot.resolve(env.scene)
        self.ar_foot_ids = cfg.ar_foot.body_ids
        self.co_body_ids = cfg.co_body.body_ids
        self.co_foot_ids = cfg.co_foot.body_ids

        # velocity sma-buffer
        self.vel_buff = SMABuffer.init_like(rdm.root_linvel_b, (1,), cfg.vel_err_sma_window)

        # joint limit tensors
        idx_map = [cfg.q_names.index(name) for name in self.rdm.q_names]

        self.qpos_limit = th.tensor(
            [cfg.qpos_limit[idx_map[i]] for i in range(len(idx_map))],
            dtype=th.float32, device=env.device,
        ) # (n_qdim, 2)

        self.qtau_limit = th.tensor(
            [cfg.qtau_limit[idx_map[i]] for i in range(len(idx_map))],
            dtype=th.float32, device=env.device,
        ) # (n_qdim, 2)

        # source buffers for defensive copying
        self._reward = None
        self._info = {}

        # TODO: periodical reward (tracking, penalty both)
        # TODO: body frame -> tilt frame
        # TODO: tracking reward, root vel vs foot vel


    def update(self, d_action: th.Tensor, d2_action: th.Tensor, terminated: th.Tensor):
        """Update the manager.

        Args:
            d_action (th.Tensor): 1st-order action difference tensor. Shape is (n_env, n_act).
            d2_action (th.Tensor): 2nd-order action difference tensor. Shape is (n_env, n_act).
            terminated (th.Tensor): Terminated tensor. Shape is (n_env,).
        """
        cmd = self.cmd_mgr.cmd # (n_env, n_cmd)
        vel = th.cat([
            self.rdm.root_linvel_b[:,0:2],
            self.rdm.root_angvel_b[:,2:3]], dim=-1) # (n_env, 3)

        # update velocity error sma-buffer
        self.vel_buff.update(vel)
        vel_sma = self.vel_buff.sma
        err = cmd - vel
        err_sma = cmd - vel_sma

        # squared tracking error
        lin_err_sq = th.minimum(vec_sq_norm(err[:,0:2]), vec_sq_norm(err_sma[:,0:2]))
        ang_err_sq = th.minimum(vec_sq_norm(err[:,2:3]), vec_sq_norm(err_sma[:,2:3]))
        # TODO, penalty lin/ang also sma??

        # joint pos/torque limit
        qpos_violate = self.rdm.qpos.unsqueeze(-1) - self.qpos_limit # (n_env, n_qdim, 2)
        qpos_violate[:,:,0].clip_(max=0.0)
        qpos_violate[:,:,1].clip_(min=0.0)

        qtau_violate = self.rdm.qtau.unsqueeze(-1) - self.qtau_limit # (n_env, n_qdim, 2)
        qtau_violate[:,:,0].clip_(max=0.0)
        qtau_violate[:,:,1].clip_(min=0.0)

        # foot related
        foot_pos_w = self.rdm.body_pos_w[:,self.ar_foot_ids,:] # (n_env, n_foot, 3)
        foot_vel_w = self.rdm.body_linvel_w[:,self.ar_foot_ids,:] # (n_env, n_foot, 3)

        foot_stance_target_z = self.rdm.root_pos_w[:,2] + self.cfg.foot_stance_z # (n_env,)
        foot_swing_target_z = foot_stance_target_z + self.cfg.foot_clear_z # (n_env,)

        foot_stance_gap = (foot_pos_w[:,:,2] - foot_stance_target_z.unsqueeze(1)).clip(min=0.0) # (n_env, n_foot)
        foot_swing_gap = (foot_swing_target_z.unsqueeze(1) - foot_pos_w[:,:,2]).clip(min=0.0) # (n_env, n_foot)

        foot_cont = self.rdm.is_cont[:,self.co_foot_ids] # (n_env, n_foot)
        foot_gap = th.where(foot_cont, foot_stance_gap, foot_swing_gap) # (n_env, n_foot)

        foot_first_cont = self.rdm.first_cont[:,self.co_foot_ids] # (n_env, n_foot)

        foot_last_air_time = self.rdm.last_air_time[:,self.co_foot_ids] # (n_env, n_foot)
        foot_cont_period = self.rdm.cont_period[:,self.co_foot_ids] # (n_env, n_foot)

        foot_air_clip = th.minimum(foot_last_air_time, self.cfg.foot_min_air_ratio * foot_cont_period) # (n_env, n_foot)
        foot_period_clip = th.minimum((foot_cont_period / self.cfg.foot_min_period) * foot_cont_period, foot_cont_period) # (n_env, n_foot)

        # compute reward terms
        reward_terms = {
            # ----- tracking -----
            'track_lin': th.exp(-self.cfg.track_lin_err_scale * lin_err_sq),
            'track_ang': th.exp(-self.cfg.track_ang_err_scale * ang_err_sq),
            # ----- motion penalty -----
            'pen_lin': vec_sq_norm(self.rdm.root_linvel_b[:,2:3]),
            'pen_ang': vec_sq_norm(self.rdm.root_angvel_b[:,0:2]),
            'upright': vec_norm(self.rdm.gravity_dir_b[:,0:2]),
            # ----- dof penalty -----
            'mec_energy': (self.rdm.qtau * self.rdm.qvel).abs().sum(dim=-1),
            'the_energy': vec_sq_norm(self.rdm.qtau),
            'd_action': vec_sq_norm(d_action),
            'd2_action': vec_sq_norm(d2_action),
            'qpos_limit': qpos_violate.abs().sum(dim=(1,2)),
            'qtau_limit': qtau_violate.abs().sum(dim=(1,2)),
            # ----- foot -----
            'foot_clear': foot_gap.sum(dim=-1),
            'foot_ratio': (foot_first_cont * foot_air_clip).sum(dim=-1) / self.env.step_dt,
            'foot_period': (foot_first_cont * foot_period_clip).sum(dim=-1) / self.env.step_dt,
            # reward scale of `foot_ratio` and `foot_period`
            # k * (first_cont) * (air/period_clip) * n_foot * (1/dt)
            # k * (1/period_step) * (air/period_clip) * n_foot * (1/dt)
            # k * (1/period_step) * (period_time * ratio_clip) * n_foot * (1/dt)
            # k * (1/period_step) * (period_step * dt * ratio_clip) * n_foot * (1/dt)
            # k * ratio_clip * n_foot
            'foot_slip': (foot_cont * vec_norm(foot_vel_w)).sum(dim=-1),
            # ----- extras -----
            'contact': self.rdm.is_cont[:,self.co_body_ids].to(th.float32).sum(dim=-1),
            'termin': terminated.to(th.float32, copy=True),
        }
        for k, v in reward_terms.items():
            v.mul_(self.cfg_dict['k_' + k])

        # compute unbiased reward
        if self._reward is None: # lazy initialization
            self._reward = th.zeros_like(reward_terms['track_lin'])
        self._reward.copy_(sum(reward_terms.values()))
        self._reward.clip_(*self.cfg.reward_clip)
        
        # compute biased reward
        min_mean_bias = max(self.cfg.min_mean_reward - self._reward.mean().item(), 0.0)
        self._reward.add_(min_mean_bias)

        # info dict
        self._info = {k: v.mean().item() for k, v in reward_terms.items()}
        self._info.update({
            'min_mean_bias': min_mean_bias,
            'metrics': {
                'lin_err': lin_err_sq.sqrt().mean().item(),
                'ang_err': ang_err_sq.sqrt().mean().item(),
                'foot_air_ratio': (foot_last_air_time / foot_cont_period.clip(min=1e-6)).mean().item(),
                'foot_contact_period': foot_cont_period.mean().item(),
            },
        })


    def reset(self, env_ids: Sequence[int]):
        """Reset the manager.

        Args:
            env_ids (Sequence[int]): Environment indices to reset.
        """
        self.vel_buff.reset(env_ids)


    @property
    def reward(self):
        """Reward tensor. Shape is (n_env,).
        """
        return self._reward.clone()
    

    @property
    def info(self):
        """Info dictionary.
        """
        return self._info
