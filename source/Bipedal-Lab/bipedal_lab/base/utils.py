from __future__ import annotations

from isaaclab.envs import DirectRLEnv, VecEnvStepReturn

from collections.abc import Sequence

import torch as th



class HistoryBuffer:
    """History tensor buffer.
    """


    def __init__(
            self,
            n_env: int,
            n_history: int,
            n_dim: int,
            device: th.device,
            dtype: th.dtype,
            value: th.Tensor = 0,
        ):
        """Initialize the buffer.

        Args:
            n_env (int): Number of vectorized environment dimension.
            n_history (int): Length of history.
            n_dim (int): Number of vector dimension.
            device (th.device): Tensor device.
            dtype (th.dtype): Tensor data type.
            value (th.Tensor, optional): Initial value for buffer. It must broadcastable to `buffer`. Defaults to 0.
        """
        self.n_env = n_env
        self.n_history = n_history
        self.n_dim = n_dim
        self.device = device
        self.dtype = dtype

        self._buffer = th.zeros(
            size=(n_env, n_history, n_dim),
            device=device,
            dtype=dtype,
        )

        # reset buffer
        self.ALL_INDICES = th.arange(self.n_env, dtype=th.int64, device=device)
        self.reset(self.ALL_INDICES, value)


    @classmethod
    def init_like(
        cls,
        data: th.Tensor,
        n_history: int,
        device: th.device = None,
        dtype: th.dtype = None,
        value: th.Tensor = 0,
    ) -> HistoryBuffer:
        """Initialize the buffer with reference tensor.

        Args:
            data (th.Tensor): Reference tensor. Shape is (n_env, n_dim).
            n_history (int): Length of history.
            device (th.device, optional): Tensor device. Defaults to None.
            dtype (th.dtype, optional): Tensor data type. Defaults to None.
            value (th.Tensor, optional): Initial value for `buffer`. It must broadcastable to `buffer`. Defaults to 0.
        
        Returns:
            HistoryBuffer: Instance of initialized `HistoryBuffer`.
        """
        return HistoryBuffer(
            n_env=data.shape[0],
            n_history=n_history,
            n_dim=data.shape[1],
            device=data.device if device is None else device,
            dtype=data.dtype if dtype is None else dtype,
            value=value,
        )


    def update(self, data: th.Tensor):
        """Update the buffer with new data.

        Args:
            data (th.Tensor): New data. Shape is (n_env, n_dim).
        """
        self._buffer.copy_(self._buffer.roll(shifts=1, dims=1))
        self._buffer[:,0,:] = data
    

    def reset(self, env_ids: Sequence[int], value: th.Tensor = 0):
        """Reset the buffer.

        Args:
            env_ids (Sequence[int]): Environment indices to reset.
            value (th.Tensor, optional): Reset value for buffer. It must broadcastable to (len(env_ids), n_history, n_dim). Defaults to 0.
        """
        self._buffer[env_ids,:,:] = value
    

    @property
    def buffer(self):
        """Main buffer. Shape is (n_env, n_history, n_dim)
        """
        return self._buffer



class SMABuffer:
    """Simple moving average tensor buffer.
    """


    def __init__(
            self,
            n_env: int,
            n_window: int,
            n_dim: int,
            device: th.device,
            dtype: th.dtype,
            value: th.Tensor = 0,
        ):
        """Initialize the buffer.

        Args:
            n_env (int): Number of vectorized environment dimension.
            n_window (int): Size of window.
            n_dim (int): Number of vector dimension.
            device (th.device): Tensor device.
            dtype (th.dtype): Tensor data type.
            value (th.Tensor, optional): Initial value for `sma`. It must broadcastable to `buffer`. Defaults to 0.
        """
        self.n_env = n_env
        self.n_window = n_window
        self.n_dim = n_dim
        self.device = device
        self.dtype = dtype

        self._buffer = th.zeros(
            size=(n_env, n_window, n_dim),
            device=device,
            dtype=dtype,
        )
        self._buffer_len = th.zeros(
            size=(n_env,),
            device=device,
            dtype=th.int64,
        )
        self._sma = th.zeros(
            size=(n_env, n_dim),
            device=device,
            dtype=dtype,
        )
        self.ptr = 0
        # index where new data must be push in (where old data must be pop out)

        self.ALL_INDICES = th.arange(self.n_env, dtype=th.int64, device=device)
        self.reset(self.ALL_INDICES, value)

    
    @classmethod
    def init_like(
        cls,
        data: th.Tensor,
        n_window: int,
        device: th.device = None,
        dtype: th.dtype = None,
        value: th.Tensor = 0,
    ) -> SMABuffer:
        """Initialize the buffer with reference tensor.

        Args:
            data (th.Tensor): Reference tensor. Shape is (n_env, n_dim).
            n_window (int): Size of window.
            device (th.device, optional): Tensor device. Defaults to None.
            dtype (th.dtype, optional): Tensor data type. Defaults to None.
            value (th.Tensor, optional): Initial value for `sma`. It must broadcastable to `buffer`. Defaults to 0.
        
        Returns:
            SMABuffer: Instance of initialized `SMABuffer`.
        """
        return SMABuffer(
            n_env=data.shape[0],
            n_window=n_window,
            n_dim=data.shape[1],
            device=data.device if device is None else device,
            dtype=data.dtype if dtype is None else dtype,
            value=value,
        )
    

    def _update_sma(self):
        self._sma.copy_(self._buffer.sum(dim=1) / self._buffer_len.unsqueeze(-1).clip(min=1))
    

    def update(self, data: th.Tensor):
        """Update the buffer with new data.

        Args:
            data (th.Tensor): New data. Shape is (n_env, n_dim).
        """
        # update buffer
        self._buffer[:,self.ptr,:] = data
        self._buffer_len.add_(1).clip_(max=self.n_window)
        self.ptr = (self.ptr + 1) % self.n_window

        self._update_sma()


    def reset(self, env_ids: Sequence[int], value: th.Tensor = 0):
        """Reset the buffer.

        Args:
            env_ids (Sequence[int]): Environment indices to reset.
            value (th.Tensor, optional): Reset value for buffer. It must broadcastable to (len(env_ids), n_dim). Defaults to 0.
        """
        # reset buffer
        self._buffer[env_ids,:,:] = 0.0
        self._buffer[env_ids,self.ptr,:] = value
        self._buffer_len[env_ids] = 0

        self._update_sma()
    

    @property
    def sma(self):
        """SMA value. Shape is (n_env, n_dim).
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
