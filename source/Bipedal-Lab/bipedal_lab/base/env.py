from isaaclab.envs import DirectRLEnv, VecEnvObs

import torch as th

from typing import Tuple
from collections.abc import Sequence

from bipedal_lab.base.env_cfg import BipedalEnvCfg
from bipedal_lab.base.utils import direct_rl_env_extended_step


class BipedalEnv(DirectRLEnv):
    def __init__(self, cfg: BipedalEnvCfg, render_mode: str | None = None):
        super().__init__(cfg, render_mode)
    
    def _setup_scene(self):
        pass
    
    def _reset_idx(self, env_ids: Sequence[int]):
        return super()._reset_idx(env_ids)
    
    def _pre_physics_step(self, actions: th.Tensor):
        pass
    
    def _apply_action(self):
        pass
    
    def _post_apply_action(self):
        pass
    
    def _get_observations(self) -> VecEnvObs:
        # observation manager
        # command manager
        pass
    
    def _get_states(self) -> VecEnvObs | None:
        pass
    
    def _get_rewards(self) -> th.Tensor:
        pass
    
    def _get_dones(self) -> Tuple[th.Tensor, th.Tensor]:
        pass
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        pass
    
    def step(self, action):
        return direct_rl_env_extended_step(self, action)
