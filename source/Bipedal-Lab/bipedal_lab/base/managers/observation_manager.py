from isaaclab.utils import configclass

from collections.abc import Sequence
import torch as th

from bipedal_lab.base.utils import HistoryBuffer
from .articulation_data_manager import ArticulationDataManager



@configclass
class ObservationManagerCfg:
    """Configuration class for `ObservationManager`.
    """

    n_history: int

    n_act: int



class ObservationManager:
    """Manager class which computes observation.

    As all observation tensors computed by this manager dont't contain command terms, it must be handled externally.

    Note:
        Considering the correct implementation of reinforcement learning algorithm, the following quantities are not necessary actually.

        - observation(state) of reset environment.
    """

    n_obs: int
    """Total number of elements in observation (for single time step).
    """

    n_priv: int
    """Total number of elements in privileged observation.
    """

    n_history: int
    """Lenght of history.
    """

    n_act: int
    """Total number of elements in action.
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

        self.n_obs = (
            self.adm.root_angvel_b.shape[-1] +
            self.adm.gravity_dir_b.shape[-1] +
            self.adm.qpos.shape[-1] +
            self.adm.qvel.shape[-1] +
            self.n_act
        )
        self.n_priv = (
            self.adm.root_linvel_t.shape[-1]
        )
        self.n_history = self.cfg.n_history
        self.n_act = self.cfg.n_act

        # initialize history buffer
        self.obs_hist_buff = HistoryBuffer(
            n_env=self.adm.env.num_envs,
            n_history=self.n_history,
            n_dim=self.n_obs,
            device=self.adm.env.device,
            dtype=self.adm.qpos.dtype,
        )

        # initialize observation tensors
        self.obs_t = th.zeros(size=(self.n_env, self.n_obs), device=self.device)
        self.obs_hist = self.obs_hist_buff.buffer
        self.obs_hist_n = None
        self.obs_priv


    def update(self, prev_action: th.Tensor):
        """Compute observation tensors.

        Args:
            prev_action (th.Tensor): action applied on previous step
        """
        # observation (single timestep)
        self.obs_t = th.cat([
            self.adm.root_angvel_b,
            self.adm.gravity_dir_b,
            self.adm.qpos - self.adm.qpos_default,
            self.adm.qvel - self.adm.qvel_default,
            prev_action,
        ], dim=-1)

        # update history buffer
        self.obs_hist_buff.update(self.obs_t)

        # observation history
        self.obs_hist = self.obs_hist_buff.buffer.clone().detach()

        # observation history with noise added
        self.obs_hist_n = None
        # TODO: add observation noise implementation

        # privileged observation
        self.obs_priv = th.cat([
            self.adm.root_linvel_t
        ], dim=-1)
    

    def reset(self, env_ids: Sequence[int]):
        """Reset for given environment indices.

        This function only reset the history buffer.
        Therefore observation tensors `obs_t`, `obs_hist`, `obs_hist_n`, `obs_priv` will not be changed.

        Args:
            env_ids (Sequence[int]): Environment indices for reset.
        """
        self.obs_hist_buff.reset(env_ids)
