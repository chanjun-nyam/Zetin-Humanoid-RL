from bipedal_lab.base import BipedalEnv
from bipedal_lab.utils import env_loader

from .tron1_s_cfg import Tron1SEnvCfg
from .tron1_p_cfg import Tron1PEnvCfg


env_loader.register(
    id='Tron1-S',
    env_cls=BipedalEnv,
    cfg_cls=Tron1SEnvCfg,
    default_kwargs={},
)


env_loader.register(
    id='Tron1-P',
    env_cls=BipedalEnv,
    cfg_cls=Tron1PEnvCfg,
    default_kwargs={},
)
