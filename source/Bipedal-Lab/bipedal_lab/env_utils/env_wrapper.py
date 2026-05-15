from isaaclab.envs import ManagerBasedRLEnv, DirectRLEnv

from typing import Tuple
import gymnasium as gym
import torch as th

from simple_rl.env import BaseEnv



class IsaacEnvWrapper(BaseEnv):
    """Wrap isaaclab's `ManagerBasedEnv` or `DirectRLEnv` to simple_rl compatible environment.
    """


    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, reward_scale: float = 1.0):
        """Initialize the environment.

        Args:
            env (ManagerBasedEnv | DirectRLEnv): Environment implemented with `ManagerBasedEnv` or `DirectRLEnv`.
            reward_scale (float, optional): Additional scaler for reward. Defaults to 1.0.
        """
        super().__init__()

        self.env = env
        self.rwd_scale = reward_scale

        self.n_env = env.num_envs
        self.device = th.device(env.device)

        # TODO: check space validity
        self.n_obs = self.env.single_observation_space['policy'].shape[-1]
        self.n_action = self.env.single_action_space.shape[-1]


    def reset(self) -> Tuple[th.Tensor, dict]:
        """Reset the environment.

        Returns:
            Tuple[th.Tensor, dict]: observation tensor and info dictionary respectively
        """
        obs, info = self.env.reset()
        return obs['policy'], info


    def step(self, action:th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, dict]:
        """Step the environment.

        Args:
            action (th.Tensor): action tensor

        Returns:
            Tuple of observation, reward, termination, truncation, info.
        """
        obs, rwd, ter, tru, info = self.env.step(action)

        rwd = rwd.view(self.n_env, 1) * self.rwd_scale
        ter = ter.view(self.n_env, 1)
        tru = tru.view(self.n_env, 1)

        return obs['policy'], rwd, ter, tru, info


    def close(self):
        """Close the environment.
        """
        self.env.close()
