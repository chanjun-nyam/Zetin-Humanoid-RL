from isaaclab.app import AppLauncher

app_launcher = AppLauncher(livestream=2)
simulation_app = app_launcher.app

from bipedal_lab.env_utils import env_loader
import bipedal_lab.tasks

from bipedal_lab.env_utils import IsaacEnvWrapper
from tron_loco.velocity.tron_env import TronEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

import torch as th

def main():

    # env = env_loader.make('Tron1Env')
    env = IsaacEnvWrapper(ManagerBasedRLEnv(TronEnvCfg()))

    print('###########')
    print(env.spec)
    print('###########')

    obs, info = env.reset()
    print(obs.shape)

    while simulation_app.is_running():
        obs, rwd, ter, tru, info = env.step(
            th.randn((env.n_env, env.n_action), device=env.device)
        )
        
        print(obs.shape)
        # print(obs)
        # print(rwd.shape)
        # print(ter.shape)
        # print(tru.shape)
        # print(info)

    env.close()


if __name__ == '__main__':
    main()
