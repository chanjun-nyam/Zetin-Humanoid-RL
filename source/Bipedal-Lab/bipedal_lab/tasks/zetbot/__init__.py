from bipedal_lab.base import BipedalEnv
from bipedal_lab.env_utils import env_loader

from bipedal_lab.tasks.zetbot.zetbot_s_cfg import ZetbotSEnvCfg


env_loader.register(
    id='Zetbot-S',
    env_cls=BipedalEnv,
    cfg_cls=ZetbotSEnvCfg,
    default_kwargs={},
)
