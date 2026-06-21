from __future__ import annotations

from isaaclab.envs import DirectRLEnv, VecEnvStepReturn

from collections.abc import Sequence
from itertools import accumulate
from typing import List, Tuple
import math

import torch as th



class HistoryBuffer:
    """History tensor buffer.
    """


    def __init__(
            self,
            shape: Tuple[int],
            zip_dims: Tuple[int],
            n_history: int,
            device: th.device,
            dtype: th.dtype,
            value: th.Tensor = 0,
        ):
        """Initialize the buffer.

        Args:
            shape (Tuple[int]): Shape of data tensor.
            zip_dims(Tuple[int]): Dimensions to zip together.
            n_history (int): Length of history.
            device (th.device): Tensor device.
            dtype (th.dtype): Tensor data type.
            value (th.Tensor, optional): Initial value for buffer. It must broadcastable to `buffer`. Defaults to 0.
        """
        self.shape = shape
        self.zip_dims = zip_dims
        self.zip_shape = tuple([s if d not in zip_dims else 1 for d, s in enumerate(shape)])
        self.n_history = n_history
        self.device = device
        self.dtype = dtype

        self._buff = th.zeros(
            size=(n_history, *shape),
            device=device,
            dtype=dtype,
        )

        # reset buffer
        self.ALL_INDICES = th.ones(size=self.zip_shape, dtype=th.bool, device=device)
        self.reset(self.ALL_INDICES, value)


    @classmethod
    def init_like(
        cls,
        data: th.Tensor,
        zip_dims: Tuple[int],
        n_history: int,
        device: th.device = None,
        dtype: th.dtype = None,
        value: th.Tensor = 0,
    ) -> HistoryBuffer:
        """Initialize the buffer with reference tensor.

        Args:
            data (th.Tensor): Reference tensor. Shape is (n_env, n_dim).
            zip_dims(Tuple[int]): Dimensions to zip together.
            n_history (int): Length of history.
            device (th.device, optional): Tensor device. Defaults to None.
            dtype (th.dtype, optional): Tensor data type. Defaults to None.
            value (th.Tensor, optional): Initial value for `buffer`. It must broadcastable to `buffer`. Defaults to 0.

        Returns:
            HistoryBuffer: Instance of initialized `HistoryBuffer`.
        """
        return HistoryBuffer(
            shape=tuple(data.shape),
            zip_dims=zip_dims,
            n_history=n_history,
            device=data.device if device is None else device,
            dtype=data.dtype if dtype is None else dtype,
            value=value,
        )


    def update(self, data: th.Tensor):
        """Update the buffer with new data.

        Args:
            data (th.Tensor): Data tensor.
        """
        self._buff.copy_(self._buff.roll(shifts=1, dims=0))
        self._buff[0,...] = data


    def reset(self, indices: th.Tensor | Tuple[Sequence[int]] | Sequence[int], value: th.Tensor = 0):
        """Reset the buffer.

        Args:
            indices (th.Tensor | Tuple[Sequence[int]] | Sequence[int]): Indices to reset. It can be either boolen mask tensor or tuple of indices or just indices when length of tuple is 1.
            value (th.Tensor, optional): Reset value for buffer.
        """
        if isinstance(indices, th.Tensor) and indices.dtype != th.bool:
            indices = (indices,)

        if isinstance(indices, th.Tensor):
            indices = indices.view(*self.zip_shape)
            self._buff.copy_(th.where(indices, value, self._buff))

        elif isinstance(indices, tuple):
            indices = tuple([i if d not in self.zip_dims else slice(None) for d, i in enumerate(indices)])
            self._buff[:,*indices] = value

        else:
            raise ValueError(f'Not supported type for `indices`: {type(indices)}')


    @property
    def buff(self):
        """Main buffer. Shape is (n_history, *shape).
        """
        return self._buff



class SMABuffer:
    """Simple moving average buffer implemented on pytorch tensor.
    """


    def __init__(
            self,
            shape: Tuple[int],
            zip_dims: Tuple[int],
            n_window: int,
            device: th.device,
            dtype: th.dtype,
            value: th.Tensor = 0,
        ):
        """Initialize the buffer.

        Args:
            shape (Tuple[int]): Shape of data tensor.
            zip_dims(Tuple[int]): Dimensions to zip together.
            n_window (int): Size of window.
            device (th.device): Device of data tensor.
            dtype (th.dtype): Data type of data tensor.
            value (th.Tensor, optional): Initial value for `sma`. It must broadcastable to `buffer`. Defaults to 0.
        """
        self.shape = shape
        self.zip_dims = zip_dims
        self.zip_shape = tuple([s if d not in zip_dims else 1 for d, s in enumerate(shape)])
        self.n_window = n_window
        self.device = device
        self.dtype = dtype

        self._buff = th.zeros(
            size=(n_window, *shape),
            device=device,
            dtype=dtype,
        )
        self._buff_len = th.zeros(
            size=self.zip_shape,
            device=device,
            dtype=th.int64,
        )
        self._sma = th.zeros(
            size=shape,
            device=device,
            dtype=dtype,
        )
        self.ptr = 0
        # index where new data must be push in (where old data must be pop out)

        self.ALL_INDICES = th.ones(size=self.zip_shape, dtype=th.bool, device=device)
        self.reset(self.ALL_INDICES, value)


    @classmethod
    def init_like(
        cls,
        data: th.Tensor,
        zip_dims: Tuple[int],
        n_window: int,
        device: th.device = None,
        dtype: th.dtype = None,
        value: th.Tensor = 0,
    ) -> SMABuffer:
        """Initialize the buffer with reference tensor.

        Args:
            data (th.Tensor): Reference tensor. Shape is (n_env, n_dim).
            zip_dims(Tuple[int]): Dimensions to zip together.
            n_window (int): Size of window.
            device (th.device, optional): Tensor device. Defaults to None.
            dtype (th.dtype, optional): Tensor data type. Defaults to None.
            value (th.Tensor, optional): Initial value for `sma`. It must broadcastable to `buffer`. Defaults to 0.

        Returns:
            SMABuffer: Instance of initialized `SMABuffer`.
        """
        return SMABuffer(
            shape=tuple(data.shape),
            zip_dims=zip_dims,
            n_window=n_window,
            device=data.device if device is None else device,
            dtype=data.dtype if dtype is None else dtype,
            value=value,
        )


    def _update_sma(self):
        self._sma.copy_(self._buff.sum(dim=0) / self._buff_len.clip(min=1))


    def update(self, data: th.Tensor):
        """Update the buffer with new data.

        Args:
            data (th.Tensor): Data tensor.
        """
        self._buff[self.ptr,...] = data
        self._buff_len.add_(1).clip_(max=self.n_window)
        self.ptr = (self.ptr + 1) % self.n_window

        self._update_sma()


    def reset(self, indices: th.Tensor | Tuple[Sequence[int]] | Sequence[int], value: th.Tensor = 0):
        """Reset the buffer.

        Args:
            indices (th.Tensor | Tuple[Sequence[int]] | Sequence[int]): Indices to reset. It can be either boolen mask tensor or tuple of indices or just indices when length of tuple is 1.
            value (th.Tensor, optional): Reset value for buffer.
        """
        if isinstance(indices, th.Tensor) and indices.dtype != th.bool:
            indices = (indices,)

        if isinstance(indices, th.Tensor):
            indices = indices.view(*self.zip_shape)

            self._buff_len.masked_fill_(indices, 0)
            self._buff.masked_fill_(indices, 0.0)
            ptr_slot = self._buff[self.ptr, ...]
            ptr_slot.copy_(th.where(indices, value, ptr_slot))

        elif isinstance(indices, tuple):
            indices = tuple([i if d not in self.zip_dims else slice(None) for d, i in enumerate(indices)])

            self._buff_len[indices] = 0
            self._buff[:,*indices] = 0.0
            self._buff[self.ptr,...][indices] = value

        else:
            raise ValueError(f'Not supported type for `indices`: {type(indices)}')

        self._update_sma()


    @property
    def sma(self):
        """SMA value.
        """
        return self._sma



def direct_rl_env_extended_step(self: DirectRLEnv, action: th.Tensor) -> VecEnvStepReturn:
    """Step function for `BipedalEnv`.

    This function is almost identical with `isaaclab.envs.DirectRLEnv.step`, but this includes calling `_post_apply_action` method after every action application on simulator.
    """
    action = action.to(self.device)
    # add action noise
    if self.cfg.action_noise_model:
        action = self._action_noise_model(action)

    # process actions
    self._pre_physics_step(action)

    # check if we need to do rendering within the physics loop
    # note: checked here once to avoid multiple checks within the loop
    is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

    # perform physics stepping
    for _ in range(self.cfg.decimation):
        self._sim_step_counter += 1
        # set actions into buffers
        self._apply_action()
        # set actions into simulator
        self.scene.write_data_to_sim()
        # simulate
        self.sim.step(render=False)
        # render between steps only if the GUI or an RTX sensor needs it
        # note: we assume the render interval to be the shortest accepted rendering interval.
        #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
        if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
            self.sim.render()
        # update buffers at sim dt
        self.scene.update(dt=self.physics_dt)

        self._post_apply_action()

    # post-step:
    # -- update env counters (used for curriculum generation)
    self.episode_length_buf += 1  # step in current episode (per env)
    self.common_step_counter += 1  # total step (common for all envs)

    self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
    self.reset_buf = self.reset_terminated | self.reset_time_outs
    self.reward_buf = self._get_rewards()

    # -- reset envs that terminated/timed-out and log the episode information
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
        self._reset_idx(reset_env_ids)
        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()

    # post-step: step interval event
    if self.cfg.events:
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

    # update observations
    self.obs_buf = self._get_observations()

    # add observation noise
    # note: we apply no noise to the state space (since it is used for critic networks)
    if self.cfg.observation_noise_model:
        self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

    # return observations, rewards, resets and extras
    return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras



def distribute_number(n: int, p_list: List[float], eps: float = 1e-8) -> List[int]:
    """Distribute int number `n` proportional to float numbers in `p_list`.

    Args:
        n (int): Int number.
        p_list (List[float]): Float numbers.
        eps (float, optional): Epsilon. Defaults to 1e-8.

    Returns:
        List[int]: Distributed int numbers.
    """
    p_sum = sum(p_list)
    p_list = [p / p_sum for p in p_list]

    k_list = [math.floor(max(n * p - eps, 0.0)) for p in p_list]
    n_left = n - sum(k_list)

    remains = [(n * p - k, -idx) for idx, (p, k) in enumerate(zip(p_list, k_list))]
    remains.sort(reverse=True)

    for i in range(n_left):
        k_list[-remains[i][1]] += 1

    return k_list



def distribute_list(x: List, p_list: List[float], eps: float = 1e-8) -> List[List]:
    """Distribute list `x` proportional to float numbers in `p_list`.

    Args:
        n (int): List.
        p_list (List[float]): Float numbers.
        eps (float, optional): Epsilon. Defaults to 1e-8.

    Returns:
        List[List]: Distributed list. Order is preserved.
    """
    len_dist = distribute_number(len(x), p_list, eps)
    len_dist_acc = [0] + list(accumulate(len_dist))

    return [x[l:r] for l, r in zip(len_dist_acc[:-1], len_dist_acc[1:])]
