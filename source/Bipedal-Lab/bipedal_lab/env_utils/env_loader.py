from isaaclab.envs import ManagerBasedEnv, DirectRLEnv

from typing import Type, Tuple, Dict

from bipedal_lab.env_utils.env_wrapper import IsaacEnvWrapper

from simple_rl.env import BaseEnv


"""Module that register and load wrapped isaac environments.
"""


_table: Dict[str, Tuple[Type[ManagerBasedEnv] | Type[DirectRLEnv], Dict]] = {}


def register(id: str, env_cls: Type[object], default_kwargs: Dict):
    """Register new environment.

    Args:
        id (str): Id for new environment.
        env_cls (Type[object]): Class of environment.
        default_kwargs (Dict): Default kwargs when creating the environment.

    Raises:
        KeyError: When given id is already used.
    """
    if id in _table:
        raise KeyError(f'Environment id {id} is already used.')
    
    _table[id] = (env_cls, default_kwargs)


def make(id: str, reward_scale: float = 1.0, **kwargs) -> BaseEnv:
    """Make new environment.

    Args:
        id (str): Environment id which you want to make one.
        reward_scale (float, optional): Additional scaler for reward. Defaults to 1.0.
        **kwargs: This overwrite the default kwargs and used in instantiation of environment.

    Returns:
        BaseEnv: Wrapped isaac environment.
    """
    env_cls, default_kwargs = _table[id]
    env = env_cls(**(default_kwargs | kwargs))
    return IsaacEnvWrapper(env, reward_scale)
