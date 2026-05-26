from isaaclab.utils import configclass

from typing import List, Tuple
from dataclasses import MISSING

import torch as th



@configclass
class CommandManagerCfg:
    """Configuration class for `CommandManager`.
    """

    cmd_xy_rng: List[Tuple[float, float]] = MISSING

    cmd_z_rng: Tuple[float, float] = MISSING

    n_xy_div: List[int, int] = MISSING

    min_cmd_norm: float = MISSING

    heading_kp: float = MISSING

    eval_steps: int = MISSING

    play_steps: int = MISSING



class CommandManager:
    """Manager class implementing command part of square Grid Terrain Curriculum.
    """


    def __init__(self, cfg: CommandManagerCfg):
        """Initialize the manager.

        Args:
            cfg (CommandManagerCfg): Configuration instance for the manager.
        """
        self.cfg = cfg

        if not (
            self.cfg.ar_robot.name == self.cfg.ar_foot.name and
            self.cfg.co_body.name == self.cfg.co_foot.name):
            raise ValueError('Only one `Articulation` and `ContactSensor` are allowed.')

        self.robot: Articulation = env.scene[self.cfg.ar_robot.name]
        self.cont_snsr: ContactSensor = env.scene[self.cfg.co_body.name]
        self.cfg.ar_foot.resolve(env.scene)
        self.cfg.co_body.resolve(env.scene)
        self.cfg.co_foot.resolve(env.scene)
        self.ar_foot_ids = self.cfg.ar_foot.body_ids
        self.co_body_ids = self.cfg.co_body.body_ids
        self.co_foot_ids = self.cfg.co_foot.body_ids

        # joint limit tensors
        idx_map = [self.cfg.q_names.index(name) for name in self.robot.data.joint_names]
        
        self.qpos_limit = th.tensor(
            [self.cfg.qpos_limit[idx_map[i]] for i in range(len(idx_map))],
            dtype=th.float32, device=self.env.device,
        ) # (n_jnt, 2)

        self.qtau_limit = th.tensor(
            [self.cfg.qtau_limit[idx_map[i]] for i in range(len(idx_map))],
            dtype=th.float32, device=self.env.device,
        ) # (n_jnt, 2)

        # source buffers for defensive copying
        self._reward = None
        self._info = {}

        # TODO: periodical reward (tracking, penalty both)
        # TODO: body frame -> tilt frame
        # TODO: tracking reward, root vel vs foot vel


    def update(self, command: th.Tensor, d_action: th.Tensor, d2_action: th.Tensor, terminated: th.Tensor):
        """Update the manager.

        Args:
            command (th.Tensor): Command tensor. Shape is (n_env, n_cmd).
            d_action (th.Tensor): 1st-order action difference tensor. Shape is (n_env, n_act).
            d2_action (th.Tensor): 2nd-order action difference tensor. Shape is (n_env, n_act).
            terminated (th.Tensor): Terminated tensor. Shape is (n_env,).
        """
        # squared tracking error
        track_lin_err_sq = vec_sq_norm(command[:,0:2] - self.rdm.root_linvel_b[:,0:2])
        track_ang_err_sq = vec_sq_norm(command[:,2:3] - self.rdm.root_angvel_b[:,2:3])

        # joint pos/torque limit
        qpos_violate = self.rdm.qpos.unsqueeze(-1) - self.qpos_limit # (n_env, n_jnt, 2)
        qpos_violate[:,:,0].clip_(max=0.0)
        qpos_violate[:,:,1].clip_(min=0.0)

        qtau_violate = self.rdm.qtau.unsqueeze(-1) - self.qtau_limit # (n_env, n_jnt, 2)
        qtau_violate[:,:,0].clip_(max=0.0)
        qtau_violate[:,:,1].clip_(min=0.0)

        # foot related
        foot_pos_w = self.robot.data.body_pos_w[:,self.ar_foot_ids,:] # (n_env, n_foot, 3)

        foot_stance_target_z = self.robot.data.root_pos_w[:,2] + self.cfg.foot_stance_z # (n_env,)
        foot_swing_target_z = foot_stance_target_z + self.cfg.foot_clear_z # (n_env,)

        foot_stance_gap = (foot_pos_w[:,:,2] - foot_stance_target_z.unsqueeze(1)).clip(min=0.0) # (n_env, n_foot)
        foot_swing_gap = (foot_swing_target_z.unsqueeze(1) - foot_pos_w[:,:,2]).clip(min=0.0) # (n_env, n_foot)
        # print(foot_pos_w[:,:,2] - foot_stance_target_z, foot_stance_gap, foot_swing_gap) # TODO

        foot_gap = th.where(
            self.rdm.is_cont[:,self.co_foot_ids],
            foot_stance_gap,
            foot_swing_gap,
        ) # (n_env, n_foot)

        foot_first_cont = self.rdm.first_cont[:,self.co_foot_ids] # (n_env, n_foot)

        foot_last_air_time = self.rdm.last_air_time[:,self.co_foot_ids] # (n_env, n_foot)
        foot_cont_period = self.rdm.cont_period[:,self.co_foot_ids] # (n_env, n_foot)

        foot_air_left = th.minimum(foot_last_air_time, self.cfg.foot_min_air_ratio * foot_cont_period) # (n_env, n_foot)
        foot_period_left = th.minimum((foot_cont_period / self.cfg.foot_min_period) * foot_cont_period, foot_cont_period) # (n_env, n_foot)

        # compute reward terms
        reward_terms = {
            # tracking
            'track_lin': th.exp(-self.cfg.track_lin_err_scale * track_lin_err_sq),
            'track_ang': th.exp(-self.cfg.track_ang_err_scale * track_ang_err_sq),
            # motion penalty
            'pen_lin': vec_sq_norm(self.rdm.root_linvel_b[:,2:3]),
            'pen_ang': vec_sq_norm(self.rdm.root_angvel_b[:,0:2]),
            'upright': vec_norm(self.rdm.gravity_dir_b[:,0:2]),
            # dof penalty
            'mec_energy': (self.rdm.qtau * self.rdm.qvel).abs().sum(dim=-1),
            'the_energy': vec_sq_norm(self.rdm.qtau),
            'd_action': vec_sq_norm(d_action),
            'd2_action': vec_sq_norm(d2_action),
            'qpos_limit': qpos_violate.abs().sum(dim=(1,2)),
            'qtau_limit': qtau_violate.abs().sum(dim=(1,2)),
            # extras
            'contact': self.rdm.is_cont[:,self.co_body_ids].to(th.float32).sum(dim=-1),
            'foot_clear': foot_gap.sum(dim=-1),
            'foot_ratio': (foot_first_cont * foot_air_left).sum(dim=-1) / self.env.step_dt,
            'foot_period': (foot_first_cont * foot_period_left).sum(dim=-1) / self.env.step_dt,
            'termin': terminated.to(th.float32, copy=True),
        }
        for k, v in reward_terms.items():
            v.mul_(self.cfg_dict['k_' + k])

        # compute unbiased reward
        if self._reward is None: # lazy initialization
            self._reward = th.zeros_like(reward_terms['track_lin'])
        self._reward.copy_(sum(reward_terms.values()))
        
        # compute biased reward
        min_mean_bias = max(self.cfg.min_mean_reward - self._reward.mean().item(), 0.0)
        self._reward.add_(min_mean_bias)

        # info dict
        self._info = {k: v.mean().item() for k, v in reward_terms.items()}
        self._info.update({
            'min_mean_bias': min_mean_bias,
            'metrics': {
                'foot_air_ratio': (foot_last_air_time / foot_cont_period.clip(min=1e-6)).mean().item(),
                'foot_contact_period': foot_cont_period.mean().item(),
            },
        })


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





# command module 1
# command module 2
# ...
# command module n

# -> command manager


# grid terrain curriculum

# a e^-x^2 + (1-a) e^-y^2
# = e^{lna-x^2} + e^{ln(1-a)-y^2}



# grid 는 각각 evaluation value 와 uncertainty 를 가짐
# -> UCB 에서 아이디어를 따와서 search/assign



# difficulty




# P = e^diff + value / n

# stream 별로 제한


# force/torque disturbance 도 curriculum 으로


# 매 episode 가 resample 단위




# - command space: 타구 <- voxell로 나누어서 sampling 구현, margine
# - 평지: sample in command space
# - 계단, stair: heading command
# - xy 에 대해서만 gridding
# - following rate
# - (terrain, level, command xy)


# - uc
# - lc


# - xy sampling -> z sampling
# - polar sapce grid

