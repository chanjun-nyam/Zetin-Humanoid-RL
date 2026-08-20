from bipedal_lab.base import BipedalEnv
from bipedal_lab.utils import env_loader

from .zetbot2_cfg import Zetbot2EnvCfg


env_loader.register(
    id='Zetbot2',
    env_cls=BipedalEnv,
    cfg_cls=Zetbot2EnvCfg,
    default_kwargs={},
)
