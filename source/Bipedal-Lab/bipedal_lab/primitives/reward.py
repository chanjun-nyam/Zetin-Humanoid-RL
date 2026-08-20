from collections.abc import Sequence, Mapping
from typing import Tuple

import torch as th

from bipedal_lab.base.managers import (
    RobotDataManager,
    RewardManager,
    RewardTermBase,
)
from bipedal_lab.utils.buffer import SMABuffer
from bipedal_lab.utils.math import vec_norm_pow, vec_norm



class Val(RewardTermBase):
    def __init__(self, w: float, val: str | float, mask: str = None):
        self.w = w
        self.val = val
        self.mask = mask

    def init(self, mgr: RewardManager, shared: Mapping):
        self.mgr = mgr
        self.shared = shared

    def update(self) -> Tuple[th.Tensor, dict]:
        env = self.mgr.env
        val: th.Tensor = (
            self.shared[self.val]
            if isinstance(self.val, str) else
            th.full(size=(env.num_envs,), fill_value=self.val, dtype=th.float32, device=env.device)
        )
        mask: th.Tensor = (
            self.shared[self.mask]
            if self.mask is not None else th.ones_like(val, dtype=th.bool)
        )
        return self.w * val.to(th.float32) * mask, {}



class Sum(RewardTermBase):
    def __init__(self, w: float, val: str, mask: str = None):
        self.w = w
        self.val = val
        self.mask = mask

    def init(self, mgr: RewardManager, shared: Mapping):
        self.mgr = mgr
        self.shared = shared

    def update(self) -> Tuple[th.Tensor, dict]:
        val: th.Tensor = self.shared[self.val]
        mask: th.Tensor = (
            self.shared[self.mask]
            if self.mask is not None else th.ones_like(val[:,0], dtype=th.bool)
        )
        return self.w * val.sum(dim=1, dtype=th.float32) * mask, {}



class VecNormPow(RewardTermBase):
    def __init__(self, w: float, p: float, val: str, mask: str = None):
        self.w = w
        self.p = p
        self.val = val
        self.mask = mask

    def init(self, mgr: RewardManager, shared: Mapping):
        self.mgr = mgr
        self.shared = shared

    def update(self) -> Tuple[th.Tensor, dict]:
        val: th.Tensor = self.shared[self.val]
        mask: th.Tensor = (
            self.shared[self.mask]
            if self.mask is not None else th.ones_like(val[:,0], dtype=th.bool)
        )
        return self.w * vec_norm_pow(val, p=self.p) * mask, {}



class VecNorm(RewardTermBase):
    def __init__(self, w: float, p: float, val: str, mask: str = None):
        self.w = w
        self.p = p
        self.val = val
        self.mask = mask

    def init(self, mgr: RewardManager, shared: Mapping):
        self.mgr = mgr
        self.shared = shared

    def update(self) -> Tuple[th.Tensor, dict]:
        val: th.Tensor = self.shared[self.val]
        mask: th.Tensor = (
            self.shared[self.mask]
            if self.mask is not None else th.ones_like(val[:,0], dtype=th.bool)
        )
        return self.w * vec_norm(val, p=self.p) * mask, {}



class Track(RewardTermBase):
    def __init__(self, w: float, s: float, val: str, cmd: str, n_window: int):
        self.w = w
        self.s = s
        self.val = val
        self.cmd = cmd
        self.n_window = n_window

        # lazy initialization
        self.buff = None

    def init(self, mgr: RewardManager, shared: Mapping):
        self.mgr = mgr
        self.shared = shared

    def update(self) -> Tuple[th.Tensor, dict]:
        val: th.Tensor = self.shared[self.val]
        cmd: th.Tensor = self.shared[self.cmd]

        if self.buff is None:
            self.buff = SMABuffer.init_like(val, (1,), self.n_window)

        self.buff.update(val)

        err = cmd - val
        err_sma = cmd - self.buff.sma
        err_sq = th.minimum(vec_norm_pow(err, p=2.0), vec_norm_pow(err_sma, p=2.0))

        return (
            self.w * th.exp(-self.s * err_sq),
            {'err': float(err_sq.sqrt().mean().item())},
        )

    def reset(self, env_ids: Sequence[int]):
        if self.buff is not None:
            self.buff.reset(env_ids)



class FootClear(RewardTermBase):
    def __init__(
            self,
            w: float,
            p: float,
            stance_z: float,
            clear_z: float,
            rdm: str = 'rdm',
            ar_ids: str = 'ar_foot_ids',
            co_ids: str = 'co_foot_ids',
            mask: str = None,
        ):
        self.w = w
        self.p = p
        self.stance_z = stance_z
        self.clear_z = clear_z
        self.rdm = rdm
        self.ar_ids = ar_ids
        self.co_ids = co_ids
        self.mask = mask

    def init(self, mgr: RewardManager, shared: Mapping):
        self.mgr = mgr
        self.shared = shared

    def update(self) -> Tuple[th.Tensor, dict]:
        rdm: RobotDataManager = self.shared[self.rdm]
        ar_ids: Sequence[int] = self.shared[self.ar_ids]
        co_ids: Sequence[int] = self.shared[self.co_ids]
        mask: th.Tensor = (
            self.shared[self.mask]
            if self.mask is not None else th.ones_like(rdm.ALL_INDICES, dtype=th.bool)
        )

        pos_w = rdm.body_pos_w[:,ar_ids,:] # (n_env, n_foot, 3)

        stance_target_z = rdm.root_pos_w[:,2:3] + self.stance_z # (n_env, 1)
        swing_target_z = stance_target_z + self.clear_z # (n_env, 1)

        stance_gap = (pos_w[:,:,2] - stance_target_z).clip(min=0.0) # (n_env, n_foot)
        swing_gap = (swing_target_z - pos_w[:,:,2]).clip(min=0.0) # (n_env, n_foot)

        is_cont = rdm.is_cont[:,co_ids] # (n_env, n_foot)
        gap = th.where(is_cont, stance_gap, swing_gap) # (n_env, n_foot)

        return self.w * vec_norm_pow(gap, p=self.p) * mask, {}



class FootSlip(RewardTermBase):
    def __init__(
            self,
            w: float,
            rdm: str = 'rdm',
            ar_ids: str = 'ar_foot_ids',
            co_ids: str = 'co_foot_ids',
            mask: str = None,
        ):
        self.w = w
        self.rdm = rdm
        self.ar_ids = ar_ids
        self.co_ids = co_ids
        self.mask = mask

    def init(self, mgr: RewardManager, shared: Mapping):
        self.mgr = mgr
        self.shared = shared

    def update(self) -> Tuple[th.Tensor, dict]:
        rdm: RobotDataManager = self.shared[self.rdm]
        ar_ids: Sequence[int] = self.shared[self.ar_ids]
        co_ids: Sequence[int] = self.shared[self.co_ids]
        mask: th.Tensor = (
            self.shared[self.mask]
            if self.mask is not None else th.ones_like(rdm.ALL_INDICES, dtype=th.bool)
        )

        vel_w = rdm.body_linvel_w[:,ar_ids,:] # (n_env, n_foot, 3)
        is_cont = rdm.is_cont[:,co_ids] # (n_env, n_foot)

        return self.w * (is_cont * vec_norm(vel_w, p=2.0)).sum(dim=1) * mask, {}



class FootFreqInvariant(RewardTermBase):
    def __init__(
            self,
            w: float,
            p: float,
            ratio_cont: str,
            ratio_air: str,
            rdm: str = 'rdm',
            co_ids: str = 'co_foot_ids',
            mask: str = None,
        ):
        self.w = w
        self.p = p
        self.ratio_cont = ratio_cont
        self.ratio_air = ratio_air
        self.rdm = rdm
        self.co_ids = co_ids
        self.mask= mask

    def init(self, mgr: RewardManager, shared: Mapping):
        self.mgr = mgr
        self.shared = shared

    def update(self) -> Tuple[th.Tensor, dict]:
        rdm: RobotDataManager = self.shared[self.rdm]
        co_ids: Sequence[int] = self.shared[self.co_ids]
        ratio_cont: th.Tensor = self.shared[self.ratio_cont]
        ratio_air: th.Tensor = self.shared[self.ratio_air]
        mask: th.Tensor = (
            self.shared[self.mask]
            if self.mask is not None else th.ones_like(ratio_cont[:,0], dtype=th.bool)
        )

        first_cont = rdm.first_cont[:,co_ids] # (n_env, n_foot)
        first_air = rdm.first_air[:,co_ids] # (n_env, n_foot)
        cont_period = rdm.cont_period[:,co_ids] # (n_env, n_foot)
        air_period = rdm.air_period[:,co_ids] # (n_env, n_foot)

        ratio_cont = ratio_cont.clip(max=1.0) # (n_env, n_foot)
        ratio_air = ratio_air.clip(max=1.0) # (n_env, n_foot)

        if self.p == float('inf'):
            ratio_cont.add_(1e-6).floor_()
            ratio_air.add_(1e-6).floor_()
        else:
            ratio_cont.pow_(self.p)
            ratio_air.pow_(self.p)

        return (
            self.w * 0.5 * (
                first_cont * cont_period * (ratio_cont - 1.0) +
                first_air * air_period * (ratio_air - 1.0)
            ).sum(dim=1) * (1.0 / rdm.env.step_dt) * mask,
            {'ratio': 0.5 * float((ratio_cont + ratio_air).mean().item())},
        )



class Gait(RewardTermBase):
    def __init__(
            self,
            w: float,
            k: float,
            s_frc: float,
            s_spd: float,
            n_sample: int,
            rdm: str = 'rdm',
            ar_ids: str = 'ar_foot_ids',
            co_ids: str = 'co_foot_ids',
            theta: str = 'gait_theta',
            ratio: str = 'gait_ratio',
            mask: str = None
        ):
        self.w = w
        self.k = k
        self.s_frc = s_frc
        self.s_spd = s_spd
        self.n_sample = n_sample
        self.rdm = rdm
        self.ar_ids = ar_ids
        self.co_ids = co_ids
        self.theta = theta
        self.ratio = ratio
        self.mask = mask

    def init(self, mgr: RewardManager, shared: Mapping):
        self.mgr = mgr
        self.shared = shared
        self._init_cdf()

    def _init_cdf(self):
        log_prob = th.distributions.VonMises(loc=0.0, concentration=self.k).log_prob

        device = self.mgr.env.device
        x_sample = th.linspace(-th.pi, th.pi, self.n_sample+1, dtype=th.float32, device=device)
        y_sample = th.exp(log_prob(x_sample))

        self._cdf_table = th.cumulative_trapezoid(y_sample, x_sample)
        self._cdf_table.div_(self._cdf_table[-1].clone())

    def _get_cdf(self, x: th.Tensor):
        # x = 2pi(idx/n_sample-0.5)
        # (x/2pi+0.5)n_sample = idx
        idx = ((x / (2*th.pi) + 0.5) * self.n_sample).floor()
        idx = idx.to(th.int64).clip(min=0, max=self.n_sample-1)
        return self._cdf_table[idx]

    def update(self) -> Tuple[th.Tensor, dict]:
        rdm: RobotDataManager = self.shared[self.rdm]
        ar_ids: Sequence[int] = self.shared[self.ar_ids]
        co_ids: Sequence[int] = self.shared[self.co_ids]
        theta: th.Tensor = self.shared[self.theta] # (n_env, n_foot)
        ratio: th.Tensor = self.shared[self.ratio] # (n_env,)
        mask: th.Tensor = (
            self.shared[self.mask]
            if self.mask is not None else th.ones_like(theta[:,0], dtype=th.bool)
        ) # (n_env,)

        theta = th.remainder(theta, 2.0 * th.pi) # (n_env, n_foot)
        d = 2.0 * th.pi * ratio.unsqueeze(1) # (n_env, 1)

        # E[I(x)]
        # = int I(x)P(I(x)=y) dy
        # = int_A int_B P(A<x<B) dA dB
        # = int_A int_B P(A<x) * (1-P(B<x)) dA dB
        # = int_A P(A<x) dA * int_B (1-P(B<x)) dB
        # = F(x;A) * (1-F(x;B))
        # = F(x-A;0) * (1-F(x-B;0))
        # swing interval:  (a, b) = (0, d)
        # stance interval: (a, b) = (d, 2pi)
        i_swing = self._get_cdf(theta-0) * (1.0 - self._get_cdf(theta-d)) # (n_env, n_foot)
        i_stance = self._get_cdf(theta-d) * (1.0 - self._get_cdf(theta-2*th.pi)) # (n_env, n_foot)

        # compute q_frc, q_spd
        q_frc = 1.0 - th.exp(-self.s_frc * vec_norm_pow(rdm.cont_force_w[:,co_ids,:])) # (n_env, n_foot)
        q_spd = 1.0 - th.exp(-self.s_spd * vec_norm_pow(rdm.body_linvel_w[:,ar_ids,:])) # (n_env, n_foot)

        return self.w * (i_swing * q_frc + i_stance * q_spd).sum(dim=1) * mask, {}
