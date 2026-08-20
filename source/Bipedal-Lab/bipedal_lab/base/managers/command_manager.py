from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass

from itertools import product, accumulate
from typing import List, Tuple
from dataclasses import MISSING

import torch as th



@configclass
class CommandManagerCfg:
    """Configuration class for `CommandManager`.
    """

    cmd_rng: List[Tuple[float, float]] = MISSING
    """Value ranges which elements in command vector can have; `cmd_rng[k][0] <= cmd[k] <= cmd_rng[k][1]`.
    """

    cmd_div: List[int] = MISSING
    """Division numbers of each command dimension - k-th dimension is divided into `cmd_div[k]` sections.
    """

    zero_dims: List[int] = MISSING
    """Dimensions where zero-cmd is applied.
    """

    zero_ratio: List[float] = MISSING
    """Ratio of zero-cmd numbers when sampling is on k-phase (k > 0).
    """

    phase_len: List[int] = MISSING
    """Length of each phase - `phase_len[k]` is length of k-phase.

    Note:
        0-phase is home cell phase and k-phase (k > 0) is random cell phase.
        zero command is only sampled on k-phase (k > 0).
    """

    heading_dims: List[int] = MISSING
    """Dimensions where heading command is used.
    """

    heading_rng: List[Tuple[float, float]] = MISSING
    """Value ranges which elements in heading vector can have; `heading_rng[k][0] <= heading[k] <= heading_rng[k][1]`.
    """

    heading_kp: List[float] = MISSING
    """Proportional gain for heading command. Must have same length as heading_dims.
    """



class CommandManager:
    """Manager class implementing command of hyperrectangle Grid Terrain Curriculum.

    Note:
        For dimensions listed in `heading_dims`, the command is determined by p-control on `heading_err` and clipped to the global command range rather than the per-cell range.
        This assumes those dimensions are not subdivided by the grid (i.e., `cmd_div[d] == 1` for each `d` in `heading_dims`); otherwise the cell partition has no effect on those dimensions and may cause confusion.
    """


    def __init__(self, cfg: CommandManagerCfg, env: DirectRLEnv):
        """Initialize the manager.

        Args:
            cfg (CommandManagerCfg): Configuration instance for the manager.
            env (DirectRLEnv): Environment instance.
        """
        self.cfg = cfg
        self.env = env

        # compute `_cell_rng` tensor
        cmd_rng = th.tensor(cfg.cmd_rng, dtype=th.float32, device='cpu')
        cmd_div = th.tensor(cfg.cmd_div, dtype=th.int64, device='cpu')

        cell_rng_list = []

        for cell_coord_list in product(*[range(n_div) for n_div in cfg.cmd_div]):
            cell_coord = th.tensor(cell_coord_list, dtype=th.int64, device='cpu')

            low = cmd_rng[:,0] + (cmd_rng[:,1] - cmd_rng[:,0]) * (cell_coord / cmd_div)
            high = cmd_rng[:,0] + (cmd_rng[:,1] - cmd_rng[:,0]) * ((cell_coord + 1) / cmd_div)
            rng = th.stack((low, high), dim=-1) # (n_cmd, 2)

            cell_rng_list.append(rng.tolist())

        self._cmd_rng = cmd_rng.to(device=env.device)
        # (n_cmd, 2)
        self._cell_rng = th.tensor(cell_rng_list, dtype=th.float32, device=env.device)
        # (n_cell, n_cmd, 2)
        self._heading_rng = th.tensor(cfg.heading_rng, dtype=th.float32, device=env.device)
        # (n_heading, 2)
        self._zero_ratio = th.tensor(cfg.zero_ratio, dtype=th.float32, device=env.device)
        # (n_phase,)


        self.n_cmd = len(cfg.cmd_rng)
        self.n_cell = len(cell_rng_list)
        self.n_phase = len(cfg.phase_len)
        self.n_heading = len(cfg.heading_dims)
        self.cycle_len = sum(cfg.phase_len)

        self.phase_cumsum = [0] + list(accumulate(cfg.phase_len))

        self.heading_dims = th.tensor(cfg.heading_dims, dtype=th.int64, device=env.device)
        self.heading_kp = th.tensor(cfg.heading_kp, dtype=th.float32, device=env.device)


    def init(self, home_cell_idx: th.Tensor):
        """Initialize the manager.

        Args:
            home_cell_idx (th.Tensor): Shape is (n_env,). Data type is integer and range of element is [0, n_cell).
        """
        # cell index related
        self._home_cell_idx = home_cell_idx.clone().detach() # (n_env,)
        self._cell_idx = th.zeros_like(home_cell_idx) # (n_env,)

        # phase related
        self._phase = th.zeros_like(home_cell_idx) # (n_env,)
        self._phase_changed = th.zeros_like(home_cell_idx, dtype=th.bool) # (n_env,)

        # heading related
        self._heading_target = th.zeros(home_cell_idx.shape + (self.n_heading,),
                                        dtype=th.float32, device=self.env.device) # (n_env, n_heading)

        # cmd related
        self._cmd = th.zeros(home_cell_idx.shape + (self.n_cmd,),
                             dtype=th.float32, device=self.env.device) # (n_env, n_cmd)

        self._is_zero = th.zeros_like(home_cell_idx, dtype=th.bool) # (n_env,)
        self._zero_dim_mask = th.tensor(
            [d in self.cfg.zero_dims for d in range(self.n_cmd)],
            dtype=th.bool, device=self.env.device,
        )

        # reset all environments
        self._phase[:] = -1
        self.update(heading=None)


    def _sample_command(self, env_mask: th.Tensor):
        to_home_mask = env_mask & (self._phase == 0) # (n_env,)
        to_random_mask = env_mask & (self._phase != 0) # (n_env,)

        random_cell_idx = th.randint_like(self._cell_idx, self.n_cell)
        self._cell_idx.copy_(self._cell_idx.where(~to_home_mask, self._home_cell_idx))
        self._cell_idx.copy_(self._cell_idx.where(~to_random_mask, random_cell_idx))

        keep_mask = ~env_mask.unsqueeze(-1) # (n_env, 1)

        # sample command in cell range
        l, h = self._cell_rng[self._cell_idx].unbind(-1) # (n_env, n_cmd)
        random_cmd = l + (h - l) * th.rand_like(l) # (n_env, n_cmd)
        self._cmd.copy_(self._cmd.where(keep_mask, random_cmd))

        # sample target heading in heading range
        l, h = self._heading_rng.unbind(-1) # (n_heading,)
        random_heading = l + (h - l) * th.rand_like(self._heading_target) # (n_env, n_heading)
        self._heading_target.copy_(self._heading_target.where(keep_mask, random_heading))

        # sample zero-cmd
        self._is_zero.copy_(th.where(
            condition=env_mask,
            input=th.rand_like(self._is_zero, dtype=th.float32) < self._zero_ratio[self._phase],
            other=self._is_zero,
        ))


    def _update_heading_command(self, heading: th.Tensor | None):
        # assume heading is equal to heading_target when heading is not given
        if heading is None:
            heading = self._heading_target.clone()

        # compute heading error
        heading_err = self._heading_target - heading # (n_env, n_heading)

        # low <= heading_err + k * (high - low) < high
        # (low - heading_err) / (high - low) <= k < (high - heading_err) / (high - low) <= k + 1
        # k = ceil((low - heading_err) / (high - low))

        low, high = self._heading_rng[:,0], self._heading_rng[:,1]
        heading_err += ((low - heading_err) / (high - low)).ceil() * (high - low)

        # apply heading command
        self._cmd[:,self.heading_dims] = th.clip(
            input=heading_err * self.heading_kp,
            min=self._cmd_rng[self.heading_dims,0],
            max=self._cmd_rng[self.heading_dims,1],
        )


    def _update_zero_cmd(self):
        self._cmd[self._is_zero.unsqueeze(1) & self._zero_dim_mask] = 0.0


    def update(self, heading: th.Tensor | None):
        """Update the manager.

        Args:
            heading (th.Tensor | None): Heading tensor. Shape is (n_env, n_heading).
        """
        # compute new phase
        cycle_pos = self.env.episode_length_buf
        new_phase = th.zeros_like(self._phase)
        for k in range(self.n_phase):
            s1, s2 = self.phase_cumsum[k], self.phase_cumsum[k + 1]
            new_phase[(s1 <= cycle_pos) & (cycle_pos < s2)] = k

        # raise error when cycle-pos is larger then cycle-length
        if (self.cycle_len <= cycle_pos).any().item():
            raise ValueError('cycle_pos is out of range')

        # sample command when phase is changed or env is reset
        env_mask = (self._phase != new_phase) | (cycle_pos == 0)

        self._phase.copy_(new_phase)
        self._phase_changed.copy_(env_mask)

        # sample cell index
        self._sample_command(env_mask)
        self._update_heading_command(heading)
        self._update_zero_cmd()


    @property
    def phase(self):
        """Phase tensor. Shape is (n_env,).
        """
        return self._phase.clone()


    @property
    def phase_changed(self):
        """Tensor which is true only when `phase` is changed. Shape is (n_env,).
        """
        return self._phase_changed.clone()


    @property
    def cmd(self):
        """Command tensor. Shape is (n_env, n_cmd).
        """
        return self._cmd.clone()


    @property
    def is_zero(self):
        """Tensor represents whether command is zero-cmd. Shape is (n_env,).
        """
        return self._is_zero.clone()
