from isaaclab.app import AppLauncher

app_launcher = AppLauncher(livestream=2, device='cpu')
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv

from tron_loco.velocity.tron_env import TronEnvCfg, TronEnvCfg_Play

import torch as th

import simple_rl


def main():

    env = ManagerBasedRLEnv(TronEnvCfg_Play())

    print('###########')
    print(env.action_space)
    print(env.single_action_space)
    print(env.action_space.shape)
    print(env.observation_space)
    print(env.single_observation_space)
    print(env.observation_space.shape)
    print(env.num_envs)
    print(type(env.device), env.device)
    print('###########')

    obs, info = env.reset()

    while simulation_app.is_running():
        obs, rwd, ter, tru, info = env.step(
            th.zeros(env.action_space.shape, device=env.device)
        )

        # print(obs)
        # print(rwd.shape)
        # print(ter.shape)
        # print(tru.shape)
        # print(info)

        print(rwd.mean().item())

    env.close()


if __name__ == '__main__':
    main()
