from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass

from collections.abc import Sequence
from dataclasses import MISSING

import torch as th



@configclass
class ActionManagerCfg:
    """Configuration class for `ActionManager`.
    """

    min_delayed_steps: int = MISSING
    """Minimum step count of being action delayed."""

    max_delayed_steps: int = MISSING
    """Maximum step count of being action delayed.

    Note:
        It must be less or equal than `decimation` of environment.
    """



class ActionManager:
    """Manager class which tracks some useful quantities related to action.
    """

    act: th.Tensor
    """Action tensor with shape (n_env, n_act).
    """

    d_act: th.Tensor
    """First forward difference of action tensor with shape (n_env, n_act).
    """

    d2_act: th.Tensor
    """Second forward difference of action tensor with shape (n_env, n_act).
    """

    act_delayed: th.Tensor
    """Delayed action tensor with shape (n_env, n_act).
    """

    
    def __init__(self, cfg: ActionManagerCfg, env: DirectRLEnv):
        """Initialize manager.
        """
        self.cfg = cfg
        self.env = env

        self.n_env = env.num_envs
        self.n_act = env.action_space.shape[-1]
        
        dummy_act = th.zeros(
            size=(self.n_env, self.n_act),
            dtype=th.float32,
            device=env.device
        )

        # initialize action tensors
        self.act = dummy_act.clone()
        self.d_act = dummy_act.clone()
        self.d2_act = dummy_act.clone()
        self.prev_act = dummy_act.clone()
        self.prev_d_act = dummy_act.clone()
        self.act_delayed = dummy_act.clone()
        
        # initialize delay table
        self.delay_table = th.randint_like(
            input=self.act[:,0],
            low=self.cfg.min_delayed_steps,
            high=self.cfg.max_delayed_steps+1,
            dtype=th.int64,
        )
        self.since_update_action = None

        from bipedal_lab.utils.tensor_debugger import TensorDebugger
        self.tensor_dbgr = TensorDebugger(rng=(-10,10))
    

    def update_action(self, action: th.Tensor):
        """Update action.

        Args:
            action (th.Tensor): Action tensor with shape (n_env, n_act).
        """
        if not self.tensor_dbgr.is_safe(action):
            print('Anot safe tensor element fount!!!!!!!!!!!!!!!!!!!!!!!!!!!!! (action)')
            action_safe = self.tensor_dbgr._to_safe(action, self.tensor_dbgr.cond)
            print(action_safe.logical_not().sum(dim=0))
            print(action.abs().max(dim=0))
            print(action.mean(dim=0))
            print(action.std(dim=0))
            # self.tensor_dbgr.autofill(action, val=0.0)
        self.prev_act = self.act
        self.act = action
        
        self.prev_d_act = self.d_act
        self.d_act = self.act - self.prev_act

        self.d2_act = self.d_act - self.prev_d_act

        # reset counter
        self.since_update_action = 0
    

    def update(self):
        """Update `act_delayed` variable.

        Note:
            This function should be called at the simulation cycle, not the policy cycle.
        """
        self.act_delayed = th.where(
            condition=self.delay_table.unsqueeze(-1)<=self.since_update_action,
            input=self.act,
            other=self.prev_act,
        )
        self.since_update_action += 1
    

    def reset(self, env_ids: Sequence[int]):
        """Reset for given environments.

        Args:
            env_ids (Sequence[int]): Environment indices to reset.
        """
        self.act[env_ids,:] = 0
        self.d_act[env_ids,:] = 0
        self.d2_act[env_ids,:] = 0
        self.prev_act[env_ids,:] = 0
        self.prev_d_act[env_ids,:] = 0
        self.act_delayed[env_ids,:] = 0

        self.delay_table[env_ids] = th.randint_like(
            input=self.act[env_ids,0],
            low=self.cfg.min_delayed_steps,
            high=self.cfg.max_delayed_steps+1,
            dtype=th.int64,
        )
