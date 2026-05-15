from bipedal_lab.base import BipedalEnv
from bipedal_lab.tasks.tron1.config import Tron1EnvCfg
from bipedal_lab.env_utils import env_loader



env_loader.register(
    id='Tron1Env',
    env_cls=BipedalEnv,
    default_kwargs={
        'cfg': Tron1EnvCfg(),
    }
)
