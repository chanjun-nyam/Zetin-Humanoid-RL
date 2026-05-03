
from isaaclab.utils import configclass

from tron_loco.velocity.bipedal_base import BipedalBaseEnvCfg
from tron_loco.assets import SOLEFOOT_CFG


# Tron Training Environment Configuration
#

@configclass
class TronEnvCfg(BipedalBaseEnvCfg):
    
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SOLEFOOT_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')


# Tron Play Environment Configuration
#

@configclass
class TronEnvCfg_Play(TronEnvCfg):
    
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 8
        self.sim.device = 'cpu'
