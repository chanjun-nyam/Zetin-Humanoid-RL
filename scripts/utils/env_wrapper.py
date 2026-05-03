from typing import Tuple

import torch as th
import gymnasium as gym
from simple_rl.env import BaseEnv
from isaaclab.envs import ManagerBasedRLEnv


class IsaacEnvWrapper(BaseEnv):

    def __init__(self, isaac_env: ManagerBasedRLEnv, reward_mean_min: float | None = None, reward_scale: float = 1.0, info_scale: float = 1.0):
        super().__init__()

        self.isaac_env = isaac_env
        self.reward_mean_min = reward_mean_min
        self.reward_scale = reward_scale
        self.info_scale = info_scale

        self.n_env = isaac_env.num_envs
        self.device = th.device(isaac_env.sim.device)

        # check observation space validity
        obs_keys = list(isaac_env.observation_space.keys())
        if len(obs_keys) != 1 or not isinstance(isaac_env.observation_space[obs_keys[0]], gym.spaces.Box):
            raise NotImplementedError()
        
        self._obs_key = obs_keys[0]
        self.n_obs = isaac_env.observation_space[self._obs_key].shape[-1]
        self.n_action = isaac_env.action_space.shape[-1]

    def reset(self) -> Tuple[th.Tensor, dict]:
        obs, info = self.isaac_env.reset()

        obs = obs[self._obs_key]
        info = self._process_info(info)

        return obs, info

    def step(self, action:th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, dict]:
        obs, rwd, ter, tru, info = self.isaac_env.step(action)

        obs = obs[self._obs_key]
        rwd = rwd.view(self.n_env, 1) * self.reward_scale
        ter = ter.view(self.n_env, 1)
        tru = tru.view(self.n_env, 1)
        info = self._process_info(info)

        if self.reward_mean_min is not None:
            added = max(self.reward_mean_min - rwd.mean().item(), 0.0)
            rwd += added
            
            info['IsaacEnvWrapper/added'] = added

        return obs, rwd, ter, tru, info

    def close(self):
        self.isaac_env.close()

    def _process_info(self, info):
        if isinstance(info, dict):
            return {k: self._process_info(v) for k, v in info.items()}
        else:
            return float(info) * self.info_scale
