from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from dataclasses import MISSING

import torch as th

from .articulation_data_manager import ArticulationDataManager
from .action_manager import ActionManager
from bipedal_lab.base.math_utils import (
    vec_sq_norm,
    vec_norm,
)



@configclass
class RewardManagerCfg:
    """Configuration class for `RewardManager`.
    """

    robot_cfg: SceneEntityCfg = MISSING
    """`SceneEntityCfg` for `Articulation`.

    Note:
        `robot_cfg.body_names` must be the list of bodies which we want to avoid the collision.
    """

    sensor_cfg: SceneEntityCfg = MISSING
    """`SceneEntityCfg` for `ContactSensor`.

    Note:
        `sensor_cfg.body_names` must be the list of foot bodies.
    """

    track_lin_err_scale: float = MISSING

    track_ang_err_scale: float = MISSING

    torque_limit: float = MISSING

    min_mean_reward: float = MISSING

    normal_history_length: int = MISSING

    body_contact_norm: float = MISSING

    foot_contact_norm: float = MISSING

    foot_stance_z: float = MISSING

    foot_swing_z: float = MISSING

    k_track_lin: float = MISSING

    k_track_ang: float = MISSING

    k_pen_lin: float = MISSING

    k_pen_ang: float = MISSING

    k_upright: float = MISSING

    k_mec_energy: float = MISSING

    k_the_energy: float = MISSING

    k_d_action: float = MISSING

    k_d2_action: float = MISSING

    k_qtau_limit: float = MISSING

    k_contact: float = MISSING

    k_foot_clear: float = MISSING



class RewardManager:
    """Manager class which computes reward.
    """

    reward: th.Tensor
    """Reward tensor with shape (n_env,).
    """

    info: dict
    """Dictionary containing informations for monitoring.
    """


    def __init__(self, cfg: RewardManagerCfg, adm: ArticulationDataManager, act_mgr: ActionManager):
        """Initialization.

        Args:
            cfg (RewardManagerCfg): configuration
            adm (ArticulationDataManager): `ArticulationDataManager`
            act_mgr (ActionManager): `ActionManager`
        """
        self.cfg = cfg
        self.adm = adm
        self.act_mgr = act_mgr
        self.cfg_dict: dict = self.cfg.to_dict()

        self.robot: Articulation = self.adm.env.scene[self.cfg.robot_cfg.name]
        self.sensor: ContactSensor = self.adm.env.scene[self.cfg.sensor_cfg.name]
        self.cfg.robot_cfg.resolve(self.adm.env.scene)
        self.cfg.sensor_cfg.resolve(self.adm.env.scene)

        # access to reward and info before any update call is undefined behavior
        self.reward = None
        self.info = None

        # TODO: periodical reward (tracking, penalty both)


        from bipedal_lab.utils.tensor_debugger import TensorDebugger
        self.tensor_dbgr = TensorDebugger(rng=(-10,10))


    def update(self, command: th.Tensor):
        """Update reward and info.

        Args:
            command (th.Tensor): Command tensor in form of (vx_cmd, vy_cmd, wz_cmd).
        """
        # squared tracking error
        track_lin_err_sq = vec_sq_norm(command[:,:2] - self.adm.root_linvel_t[:,:2])
        track_ang_err_sq = vec_sq_norm(command[:,2:] - self.adm.root_angvel_b[:,2:])

        # overused torque
        qtau_overuse = (self.adm.qtau.abs() - self.cfg.torque_limit).clip(min=0.0)

        # body contact
        _normal_hist = self.sensor.data.net_forces_w_history[:,:,self.cfg.robot_cfg.body_ids,:]
        _normal = _normal_hist[:,:self.cfg.normal_history_length,:,:].mean(dim=1)
        body_contact = vec_sq_norm(_normal) > self.cfg.body_contact_norm ** 2

        # foot contact
        _normal_hist = self.sensor.data.net_forces_w_history[:,:,self.cfg.sensor_cfg.body_ids,:]
        _normal = _normal_hist[:,:self.cfg.normal_history_length,:,:].mean(dim=1)
        foot_contact = vec_sq_norm(_normal) > self.cfg.foot_contact_norm ** 2

        # foot pos
        foot_pos_w = self.robot.data.body_pos_w[:,self.cfg.sensor_cfg.body_ids,:]

        foot_stance_target_z = self.robot.data.root_pos_w[:,2] + self.cfg.foot_stance_z
        foot_swing_target_z = self.robot.data.root_pos_w[:,2] + self.cfg.foot_swing_z

        foot_stance_gap = (foot_pos_w[:,:,2] - foot_stance_target_z.unsqueeze(1)).clip(min=0.0)
        foot_swing_gap = (foot_swing_target_z.unsqueeze(1) - foot_pos_w[:,:,2]).clip(min=0.0)

        # TODO: `root_angvel_b` -> twist angvel
        # TODO: tracking reward, root vel vs foot vel
        
        reward_terms = {
            # tracking
            'track_lin': th.exp(-self.cfg.track_lin_err_scale * track_lin_err_sq),
            'track_ang': th.exp(-self.cfg.track_ang_err_scale * track_ang_err_sq),
            # motion penalty
            # 'pen_lin': vec_sq_norm(self.adm.root_linvel_t[:,2:]),
            # 'pen_ang': vec_sq_norm(self.adm.root_angvel_b[:,:2]),
            # 'pen_lin': vec_sq_norm(self.adm.asset.data.root_lin_vel_b[:,2:]),
            'pen_ang': vec_sq_norm(self.adm.asset.data.root_ang_vel_b[:,:2]),
            'upright': vec_sq_norm(self.adm.gravity_dir_b[:,:2]), # TODO
            # dof penalty
            'mec_energy': (self.adm.qtau * self.adm.qvel).abs().sum(dim=-1),
            'the_energy': vec_sq_norm(self.adm.qtau),
            'd_action': vec_sq_norm(self.act_mgr.d_act),
            'd2_action': vec_sq_norm(self.act_mgr.d2_act),
            'qtau_limit': qtau_overuse.sum(dim=-1),
            # extras
            'contact': body_contact.sum(dim=-1,dtype=th.float32),
            'foot_clear': th.where(foot_contact, foot_stance_gap, foot_swing_gap).square().sum(dim=-1),
        }
        for k, v in reward_terms.items():
            v.mul_(self.cfg_dict['k_' + k])
        # print(
        #     reward_terms['pen_lin'].min().item(),
        #     reward_terms['pen_lin'].max().item(),
        #     reward_terms['pen_lin'].mean().item(),
        #     sep='\t',
        # )
        # compute unbiased reward
        self.reward = sum(reward_terms.values())

        min_mean_bias = max(self.cfg.min_mean_reward - self.reward.mean().item(), 0.0)

        self.reward.add_(min_mean_bias)
        
        # info dict
        self.info = {k: v.mean().item() for k, v in reward_terms.items()}
        self.info['min_mean_bias'] = min_mean_bias


        if not self.tensor_dbgr.is_safe(self.reward):
            print('not safe tensor element fount!!!!!!!!!!!!!!!!!!!!!!!!!!!!! (reward)')
            print(
                self.reward.isnan().any().item(),
                self.reward.isinf().any().item(),
                self.reward.min().item(),
                self.reward.max().item(),
            )
            print(
                self.tensor_dbgr.is_safe(reward_terms['track_lin']),
                self.tensor_dbgr.is_safe(reward_terms['track_ang']),
                # self.tensor_dbgr.is_safe(reward_terms['pen_lin']),
                self.tensor_dbgr.is_safe(reward_terms['pen_ang']),
                self.tensor_dbgr.is_safe(reward_terms['upright']),
                self.tensor_dbgr.is_safe(reward_terms['mec_energy']),
                self.tensor_dbgr.is_safe(reward_terms['the_energy']),
                self.tensor_dbgr.is_safe(reward_terms['d_action']),
                self.tensor_dbgr.is_safe(reward_terms['d2_action']),
                self.tensor_dbgr.is_safe(reward_terms['qtau_limit']),
                self.tensor_dbgr.is_safe(reward_terms['contact']),
                self.tensor_dbgr.is_safe(reward_terms['foot_clear']),
            )
            self.tensor_dbgr.autofill(self.reward, val=0.0)
