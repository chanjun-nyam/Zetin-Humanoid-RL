from isaaclab.envs import DirectRLEnv, VecEnvStepReturn

from itertools import accumulate
from typing import List
import math

import torch as th



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
