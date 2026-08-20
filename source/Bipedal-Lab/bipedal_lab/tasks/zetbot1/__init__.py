from bipedal_lab.base import BipedalEnv
from bipedal_lab.utils import env_loader

from .zetbot1_s_cfg import Zetbot1SEnvCfg


env_loader.register(
    id='Zetbot1-S',
    env_cls=BipedalEnv,
    cfg_cls=Zetbot1SEnvCfg,
    default_kwargs={},
)
