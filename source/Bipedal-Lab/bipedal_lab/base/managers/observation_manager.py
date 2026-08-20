from isaaclab.utils import configclass

from collections.abc import Sequence
from dataclasses import MISSING

import torch as th

from .robot_data_manager import RobotDataManager
from bipedal_lab.utils.buffer import HistoryBuffer



@configclass
class ObservationManagerCfg:
    """Configuration class for `ObservationManager`.
    """

    n_history: int = MISSING
    """Length of history.
    """

    obs_scale: list = MISSING
    """Scaler for each observation terms.

    It requires exactly 6 elements respectively for `root_angvel_b`, `gravity_dir_b`, `qpos`, `qvel`, `action`, `root_linvel_b`.
    """



class ObservationManager:
    """Manager class which computes observation.

    Note:
        Observation tensors from this manager doesn't include command tensor.
        Command tensor needs to be handled from external.
    """


    def __init__(self, cfg: ObservationManagerCfg, rdm: RobotDataManager):
        """Initialize the manager.

        Args:
            cfg (ObservationManagerCfg): Configuration instance for the manager.
            rdm (RobotDataManager): `RobotDataManager` instance.
        """
        self.cfg = cfg
        self.rdm = rdm

        # lazy initialize for observation tensors/buffer
        self._init_done = False


    def _lazy_init(self, action: th.Tensor):
        self._init_done = True

        # dummy observation tensors
        dummy_obs_t = th.cat([
            self.rdm.root_angvel_b,
            self.rdm.gravity_dir_b,
            self.rdm.qpos,
            self.rdm.qvel,
            action,
        ], dim=-1)
        dummy_obs_priv = th.cat([
            self.rdm.root_linvel_b,
        ], dim=-1)

        # initialize observation tensors
        self._obs_t = th.zeros_like(dummy_obs_t)
        self._obs_priv = th.zeros_like(dummy_obs_priv)

        # initialization observation buffer
        self._obs_hist = HistoryBuffer.init_like(dummy_obs_t, (1,), self.cfg.n_history)


    def update(self, action: th.Tensor):
        """Update the manager.

        Args:
            action (th.Tensor): Action tensor.
        """
        if not self._init_done:
            self._lazy_init(action)

        # observation tensors
        self._obs_t.copy_(
            th.cat([
                (self.rdm.root_angvel_b) * self.cfg.obs_scale[0],
                (self.rdm.gravity_dir_b) * self.cfg.obs_scale[1],
                (self.rdm.qpos - self.rdm.qpos_default) * self.cfg.obs_scale[2],
                (self.rdm.qvel - self.rdm.qvel_default) * self.cfg.obs_scale[3],
                (action) * self.cfg.obs_scale[4],
            ], dim=-1)
        )
        # TODO: change rdm.root_angvel_b to twist angvel and swing angvel

        self._obs_priv.copy_(
            th.cat([
                self.rdm.root_linvel_b * self.cfg.obs_scale[5],
            ], dim=-1)
        )

        # observation buffer
        self._obs_hist.update(self._obs_t)

        # TODO: noist added observation


    def reset(self, env_ids: Sequence[int]):
        """Reset the manager.

        Args:
            env_ids (Sequence[int]): Environment indices to reset.
        """
        if not self._init_done:
            return
        self._obs_t[env_ids,:] = 0
        self._obs_priv[env_ids,:] = 0
        self._obs_hist.reset(env_ids)


    @property
    def obs_t(self):
        """Observation tensor. Shape is (n_env, n_obs_t).
        """
        return self._obs_t.clone()


    @property
    def obs_priv(self):
        """Privileged observation tensor. Shape is (n_env, n_priv).
        """
        return self._obs_priv.clone()


    @property
    def obs_hist(self):
        """Historical observation tensor. Shape is (n_env, n_history, n_obs_t).
        """
        return self._obs_hist.buff.transpose(0, 1).contiguous()
