from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass

from collections.abc import Sequence
from typing import Tuple, Dict, Callable
from dataclasses import MISSING

import torch as th



class RewardTermBase:
    def init(self, mgr, shared: dict):
        pass

    def update(self) -> Tuple[th.Tensor, dict]:
        pass

    def reset(self, env_ids: Sequence[int]):
        pass



@configclass
class RewardManagerCfg:
    """Configuration class for `RewardManager`.
    """

    clip_rng: Tuple[float, float] = MISSING

    bonus_threshold: float = MISSING

    init_shared_buff: Callable = MISSING

    update_shared_buff: Callable = MISSING

    terms: Dict[str, RewardTermBase] = MISSING



class RewardManager:
    """Manager class which computes reward.
    """


    def __init__(self, cfg: RewardManagerCfg, env: DirectRLEnv):
        """Initialize the manager.

        Args:
            cfg (RewardManagerCfg): Configuration instance for the manager.
            env (DirectRLEnv): Environment instance.
        """
        self.cfg = cfg
        self.env = env

        # initialize shared buffer
        self.shared = {}
        cfg.init_shared_buff(self, self.shared)

        # initialize reward terms
        self.terms = cfg.terms
        for term in self.terms.values():
            term.init(self, self.shared)

        # initialize source buffers
        self._reward = th.zeros(size=(env.num_envs,), dtype=th.float32, device=env.device)
        self._info = {}


    def update(self):
        """Update the manager.
        """

        # initialize source buffers
        self._reward.zero_()
        self._info = {'metrics': {}}

        # update shared buffer
        self.cfg.update_shared_buff(self, self.shared)

        # compute reward terms
        for name, term in self.terms.items():
            rwd, mtr = term.update()
            self._reward.add_(rwd)
            self._info[name] = float(rwd.mean().item())
            if len(mtr) > 0:
                self._info['metrics'][name] = mtr

        if len(self._info['metrics']) == 0:
            del self._info['metrics']

        # apply reward clipping
        self._reward.clip_(*self.cfg.clip_rng)

        # add bonus
        bonus = max(self.cfg.bonus_threshold - float(self._reward.mean().item()), 0.0)
        self._reward.add_(bonus)
        self._info['bonus'] = bonus


    def reset(self, env_ids: Sequence[int]):
        """Reset the manager.

        Args:
            env_ids (Sequence[int]): Environment indices to reset.
        """
        for term in self.terms.values():
            term.reset(env_ids)


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
