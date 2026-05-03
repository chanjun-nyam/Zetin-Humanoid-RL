
from typing import Type, Tuple, Dict

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg


_env_table: Dict[str,Tuple[Type[ManagerBasedRLEnv],Type[ManagerBasedRLEnvCfg]]] = {}


def register(env_name:str, env_cls:Type[ManagerBasedRLEnv], cfg_cls:Type[ManagerBasedRLEnvCfg]) :
    _env_table[env_name] = (env_cls, cfg_cls)


def get(env_name:str) :
    return _env_table[env_name]


def make(env_name:str) :
    env_cls, cfg_cls = _env_table[env_name]
    return env_cls(cfg_cls())



from isaaclab.envs import ManagerBasedRLEnv
from tron_loco.velocity.tron_env import TronEnvCfg, TronEnvCfg_Play


register('TronEnv', ManagerBasedRLEnv, TronEnvCfg)
register('TronEnv_Play', ManagerBasedRLEnv, TronEnvCfg_Play)

