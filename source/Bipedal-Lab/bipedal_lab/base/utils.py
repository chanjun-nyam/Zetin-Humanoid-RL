from isaaclab.envs import DirectRLEnv, VecEnvStepReturn

from collections.abc import Sequence
from __future__ import annotations

import torch as th



class HistoryBuffer:
    """History tensor buffer.
    
    Shape of buffer gonna be (n_env, n_history, n_dim).
    """

    buffer: th.Tensor
    """buffer tensor with shape of (n_env, n_history, n_dim)
    """


    def __init__(self, n_env: int, n_history: int, n_dim: int, device: th.device, dtype: th.dtype = th.float32):
        """Initialize the buffer filled with zeros.

        Args:
            n_env (int): number of environment
            n_history (int): length of history
            n_dim (int): dimension of vector
            device (th.device): device of tensor
            dtype (th.dtype): data type of tensor
        """
        self.n_env = n_env
        self.n_history = n_history
        self.n_dim = n_dim
        self.device = device
        self.dtype = dtype

        self.buffer = th.zeros(
            size=(n_env, n_history, n_dim),
            device=device,
            dtype=dtype,
        )


    @classmethod
    def init_like(cls, data: th.Tensor, n_history: int) -> HistoryBuffer:
        """Initialize the buffer with reference tensor.

        Args:
            data (th.Tensor):
                Reference tensor for initialization.
                It must have shape of (n_env, n_dim).
            n_history (int): length of history
        
        Returns:
            HistoryBuffer: initialized history buffer
        """
        return HistoryBuffer(
            n_env=data.shape[0],
            n_history=n_history,
            n_dim=data.shape[1],
            device=data.device,
            dtype=data.dtype,
        )
    

    def update(self, data: th.Tensor):
        """Push data to buffer.

        Args:
            data (th.Tensor):
                Tensor to push in.
                It must have shape of (n_env, n_dim).
        """
        self.buffer = self.buffer.roll(shifts=1, dims=1)
        self.buffer[:,0,:] = data
    

    def reset(self, env_ids: Sequence[int], value: th.Tensor = 0):
        """Reset buffer indices of env_ids with given value.

        Args:
            env_ids (Sequence[int]): Sequence of environment indices to reset.
            value (th.Tensor):
                Value for reset indices.
                It must have shape of (len(env_ids), n_dim) or just scalar.
                Defaults to 0.
        """
        if not isinstance(value, th.Tensor):
            value = th.tensor(value, device=self.device, dtype=self.dtype)
        value = value.expand(len(env_ids), self.n_dim)
        
        self.buffer[env_ids,:,:] = value.unsqueeze(1)



class SMABuffer:
    """Simple moving average tensor buffer.

    To prevent floating point error accumulation, it recalculate sum for every n_window updates.
    """

    buffer: th.Tensor
    """Ring buffer tensor with shape of (n_env, n_window, n_dim).
    """

    sma: th.Tensor
    """SMA value tensor with shape of (n_env, n_dim).
    """


    def __init__(self, n_env: int, n_window: int, n_dim: int, device: th.device, dtype: th.dtype = th.float32):
        """Initialize the buffer filled with zeros.

        Args:
            n_env (int): number of environment
            n_window (int): length of window
            n_dim (int): dimension of vector
            device (th.device): device of tensor
            dtype (th.dtype): data type of tensor
        """
        self.n_env = n_env
        self.n_window = n_window
        self.n_dim = n_dim
        self.device = device
        self.dtype = dtype

        self.buffer = th.zeros(
            size=(n_env, n_window, n_dim),
            device=device,
            dtype=dtype,
        )
        self.sma = th.zeros(
            size=(n_env, n_dim),
            device=device,
            dtype=dtype,
        )
        self.ptr = 0
        # indicate index where new data must be push in (where old data must be pop out)

        self.sma2 = th.zeros_like(self.sma)

    
    @classmethod
    def init_like(cls, data: th.Tensor, n_window: int) -> SMABuffer:
        """Initialize the buffer with reference tensor.

        Args:
            data (th.Tensor):
                Reference tensor for initialization.
                It must have shape of (n_env, n_dim).
            n_window (int): length of window
        
        Returns:
            SMABuffer: initialized sma buffer
        """
        return SMABuffer(
            n_env=data.shape[0],
            n_window=n_window,
            n_dim=data.shape[1],
            device=data.device,
            dtype=data.dtype,
        )
    

    def update(self, data: th.Tensor):
        """Update sma and buffer

        Args:
            data (th.Tensor):
                Tensor to update the sma.
                It must have shape of (n_env, n_dim).
        """
        self.sma += (data - self.buffer[:,self.ptr,:]) / self.n_window
        self.buffer[:,self.ptr,:] = data
        self.ptr = (self.ptr + 1) % self.n_window

        self.sma2 += data / self.n_window
        if self.ptr == 0:
            self.sma[:,:] = self.sma2[:,:]
            self.sma2[:,:] = 0


    def reset(self, env_ids: Sequence[int], value: th.Tensor = 0):
        """Reset buffer indices of env_ids with given value.

        Args:
            env_ids (Sequence[int]): Sequence of environment indices to reset.
            value (th.Tensor):
                Value for reseted indices.
                It must have shape of (len(env_ids), n_dim) or just scalar.
                Defaults to 0.
        """
        if not isinstance(value, th.Tensor):
            value = th.tensor(value, device=self.device, dtype=self.dtype)
        value = value.expand(len(env_ids), self.n_dim)

        self.buffer[env_ids,:,:] = value.unsqueeze(1)
        self.sma[env_ids,:] = value
        self.sma2[env_ids,:] = value * (self.ptr / self.n_window)



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
