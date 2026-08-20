from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from collections.abc import Sequence
from dataclasses import MISSING
from typing import List

import gymnasium.spaces as spaces
import torch as th



@configclass
class ActionManagerCfg:
    """Configuration class for `ActionManager`.
    """

    min_delayed_steps: int = MISSING
    """Minimum step count of action being delayed."""

    max_delayed_steps: int = MISSING
    """Maximum step count of action being delayed.

    Note:
        It must be less or equal than `decimation` of the environment.
    """

    ar_robot: SceneEntityCfg = MISSING
    """`SceneEntityCfg` of `Articulation`
    """

    q_names: List[str] = MISSING

    act_scale: List[float] = MISSING



class ActionManager:
    """Manager class which tracks some useful quantities related to action.
    """


    def __init__(self, cfg: ActionManagerCfg, env: DirectRLEnv):
        """Initialize the manager.

        Args:
            cfg (ActionManagerCfg): Configuration instance for the manager.
            env (DirectRLEnv): Environment instance.
        Raises:
            ValueError: When `single_action_space` of environment is not a single vector space.
        """
        self.cfg = cfg
        self.env = env

        if not (
            isinstance(env.single_action_space, spaces.Box) and
            len(env.single_action_space.shape) == 1
        ):
            raise ValueError('Only single vector action space is supported.')

        self.n_env = env.num_envs
        self.n_act = env.single_action_space.shape[-1]

        self.robot: Articulation = env.scene[cfg.ar_robot.name]

        # source tensors for defensive copying
        # action difference tensor
        self.ACT_DIFF_ORD = 2 # we don't need higher order
        self._act_diff = [
            th.zeros(
                size=(self.n_env, self.ACT_DIFF_ORD + 1 - i, self.n_act),
                dtype=th.float32,
                device=env.device,
            ) for i in range(self.ACT_DIFF_ORD + 1)
            # self._act_diff[i][:,j,:]: j-th previous i-th order difference action tensor
        ]

        # delayed action tensor
        self._act_delayed = th.zeros(
            size=(self.n_env, self.n_act),
            dtype=th.float32,
            device=env.device,
        )

        # initialize delay table
        self.delay_table = th.randint_like(
            input=self._act_delayed[:,0],
            low=self.cfg.min_delayed_steps,
            high=self.cfg.max_delayed_steps+1,
            dtype=th.int64,
        )
        self.since_update_action = 0

        # action scale tensor
        self._act_scale = th.tensor(
            cfg.act_scale,
            dtype=th.float32, device=env.device) # (n_act,)

        # q-idx mapping
        q_names, ref_q_names = cfg.q_names, self.robot.joint_names
        self.to_q_ref = [q_names.index(x) for x in ref_q_names if x in q_names]
        self.from_q_ref = [ref_q_names.index(x) for x in q_names]


    def update_action(self, action: th.Tensor):
        """Update new action.

        Args:
            action (th.Tensor): Action tensor. Shape is (n_env, n_act).
        """
        # update difference action tensor table
        for i in range(self.ACT_DIFF_ORD + 1):
            self._act_diff[i].copy_(self._act_diff[i].roll(shifts=1, dims=1))
            self._act_diff[i][:,0,:] = (
                self._act_diff[i-1][:,0,:] - self._act_diff[i-1][:,1,:]
                if i >= 1 else
                action
            )
        # reset counter
        self.since_update_action = 0


    def update(self):
        """Update delayed action tensor.

        Note:
            This function should be called at the simulation cycle, not the policy cycle.
        """
        is_current = self.delay_table <= self.since_update_action
        self._act_delayed.copy_(th.where(
            condition=is_current.unsqueeze(-1),
            input=self._act_diff[0][:,0,:], # current action
            other=self._act_diff[0][:,1,:], # previous action
        ))
        self.since_update_action += 1


    def reset(self, env_ids: Sequence[int]):
        """Reset the manager.

        Args:
            env_ids (Sequence[int]): Environment indices to reset.
        """
        # reset action difference tensors
        for i in range(self.ACT_DIFF_ORD + 1):
            self._act_diff[i][env_ids,:,:] = 0
        # reset delayed action tensor
        self._act_delayed[env_ids,:] = 0
        # reset delay table
        self.delay_table[env_ids] = th.randint_like(
            input=self._act_delayed[env_ids,0],
            low=self.cfg.min_delayed_steps,
            high=self.cfg.max_delayed_steps+1,
            dtype=th.int64,
        )


    @property
    def act(self):
        """Action tensor. Shape is (n_env, n_act).
        """
        return self._act_diff[0][:,0,:].clone()


    @property
    def act_delayed(self):
        """Delayed action tensor. Shape is (n_env, n_act).
        """
        return self._act_delayed.clone()


    def act_diff(self, o: int, t: int = 0):
        """`t`-th previous `o`-th order action difference tensor. Shape is (n_env, n_act).

        Args:
            o (int): Order of action difference.
            t (int, optional): Number of previous steps of action difference. Defaults to 0.

        Note that `o` and `t` must satifies the following condition:

        - 0 <= `o` <= `ACT_DIFF_ORD` = 2
        - 0 <= `t` <= `ACT_DIFF_ORD` = 2
        - `o` + `t` <= `ACT_DIFF_ORD` = 2
        """
        return self._act_diff[o][:,t,:].clone()


    @property
    def act_scale(self):
        """action scale tensor. Shape is (n_act,).
        """
        return self._act_scale.clone()
