from isaaclab.sim import SimulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils import configclass


SIM_DT = 1/200
POLICY_DT = 1/50
DECIMATION = 4


@configclass
class BipedalEnvCfg(DirectRLEnvCfg):
    """Configuration class for :class:`~bipedal_lab.base.env.BipedalEnv`.
    """

    sim = SimulationCfg(
        device=None,
        dt=SIM_DT,
        render_interval=DECIMATION,
        physx
    )
