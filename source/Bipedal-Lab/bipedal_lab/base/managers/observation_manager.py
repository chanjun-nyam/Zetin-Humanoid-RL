from isaaclab.utils import configclass

from collections.abc import Sequence
from dataclasses import MISSING

import torch as th

from bipedal_lab.base.utils import HistoryBuffer
from .articulation_data_manager import ArticulationDataManager



@configclass
class ObservationManagerCfg:
    """Configuration class for `ObservationManager`.
    """

    n_history: int = MISSING
    """Length of history.
    """

    n_act: int = MISSING
    """Total number of elements in action vector.
    """

    n_cmd: int = MISSING
    """Total number of elements in command vector.
    """

    obs_scale: list = MISSING
    """Scaler for each observation terms.

    It requires exactly 7 elements respectively for `root_angvel_b`, `gravity_dir_b`, `qpos`, `qvel`, `action`, `command`, `root_linvel_t`.
    """



class ObservationManager:
    """Manager class which computes observation.

    As all observation tensors computed by this manager doesn't contain command term, it must be handled externally.
    """

    n_obs: int
    """Total number of elements in observation (for single time step).
    """

    n_priv: int
    """Total number of elements in privileged observation.
    """

    n_history: int
    """Length of history.
    """

    n_act: int
    """Total number of elements in action vector.
    """

    n_cmd: int
    """Total number of elements in action vector.
    """

    obs_t: th.Tensor
    """Current shep observation tensor with shape of (n_env, n_obs).
    """

    obs_hist: th.Tensor
    """Observation history tensor with shape of (n_env, n_history, n_obs).
    """

    obsb_hist_n: th.Tensor
    """Noise added observation history tensor with shape of (n_env, n_history, n_obs).
    """

    obs_priv: th.Tensor
    """Privileged observation tensor with shape of (n_env, n_priv).
    """


    def __init__(self, cfg: ObservationManagerCfg, adm: ArticulationDataManager):
        """Initialize observation manager.

        Args:
            cfg (ObservationManagerCfg): configuration
            adm (ArticulationDataManager): articulation data manager
        """
        self.cfg = cfg
        self.adm = adm

        self.n_act = self.cfg.n_act
        self.n_cmd = self.cfg.n_cmd
        self.n_obs = (
            self.adm.root_angvel_b.shape[-1] +
            self.adm.gravity_dir_b.shape[-1] +
            self.adm.qpos.shape[-1] +
            self.adm.qvel.shape[-1] +
            self.n_act +
            self.n_cmd
        )
        self.n_priv = (
            self.adm.root_linvel_t.shape[-1]
        )
        self.n_history = self.cfg.n_history

        # initialize history buffer
        self.obs_hist_buff = HistoryBuffer(
            n_env=self.adm.env.num_envs,
            n_history=self.n_history,
            n_dim=self.n_obs,
            device=self.adm.env.device,
            dtype=self.adm.qpos.dtype,
        )

        # access to observation tensor before any update call is undefined behavior
        self.obs_t = None
        self.obs_hist = None
        self.obs_hist_n = None
        self.obs_priv = None

        from bipedal_lab.utils.tensor_debugger import TensorDebugger
        self.tensor_dbgr = TensorDebugger(rng=(-10,10))


    def update(self, action: th.Tensor, command: th.Tensor):
        """Compute observation tensors.

        Args:
            action (th.Tensor): action applied on previous step
            command (th.Tensor): command tensor
        """
        # observation (single timestep)
        self.obs_t = th.cat([
            (self.adm.root_angvel_b) * self.cfg.obs_scale[0],
            (self.adm.gravity_dir_b) * self.cfg.obs_scale[1],
            (self.adm.qpos - self.adm.qpos_default) * self.cfg.obs_scale[2],
            (self.adm.qvel - self.adm.qvel_default) * self.cfg.obs_scale[3],
            (action) * self.cfg.obs_scale[4],
            (command) * self.cfg.obs_scale[5],
        ], dim=-1)
        # TODO: change adm.root_angvel_b to twist angvel and swing angvel
        if not self.tensor_dbgr.is_safe(self.obs_t):
            print('not safe tensor element fount!!!!!!!!!!!!!!!!!!!!!!!!!!!!! (observation)')
            print(
                self.tensor_dbgr.is_safe(self.adm.root_angvel_b * self.cfg.obs_scale[0]),
                self.tensor_dbgr.is_safe(self.adm.gravity_dir_b * self.cfg.obs_scale[1]),
                self.tensor_dbgr.is_safe((self.adm.qpos - self.adm.qpos_default) * self.cfg.obs_scale[2]),
                self.tensor_dbgr.is_safe((self.adm.qvel - self.adm.qvel_default) * self.cfg.obs_scale[3]),
                self.tensor_dbgr.is_safe(action * self.cfg.obs_scale[4]),
                self.tensor_dbgr.is_safe(command * self.cfg.obs_scale[5]),
            )
            print(
                self.obs_t.isnan().any().item(),
                self.obs_t.isinf().any().item(),
                self.obs_t.min().item(),
                self.obs_t.max().item(),
            )
            self.tensor_dbgr.autofill(self.obs_t, val=0.0)
        # update history buffer
        self.obs_hist_buff.update(self.obs_t)

        # observation history
        self.obs_hist = self.obs_hist_buff.buffer.clone().detach()

        # observation history with noise added
        self.obs_hist_n = None
        # TODO: add observation noise implementation

        # privileged observation
        self.obs_priv = th.cat([
            self.adm.root_linvel_t * self.cfg.obs_scale[6]
        ], dim=-1)
    

    def reset(self, env_ids: Sequence[int]):
        """Reset for given environment indices.

        This function only reset the history buffer.
        Therefore observation tensors `obs_t`, `obs_hist`, `obs_hist_n`, `obs_priv` will not be changed.

        Args:
            env_ids (Sequence[int]): Environment indices for reset.
        """
        self.obs_hist_buff.reset(env_ids)
