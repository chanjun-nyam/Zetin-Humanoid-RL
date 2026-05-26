from isaaclab.app import AppLauncher

app_launcher = AppLauncher(livestream=2)
simulation_app = app_launcher.app

from bipedal_lab.env_utils import env_loader
import bipedal_lab.tasks

from bipedal_lab.env_utils import IsaacEnvWrapper
from tron_loco.velocity.tron_env import TronEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

import torch as th
import bipedal_lab.tasks
from bipedal_lab.env_utils import env_loader


def main():

    # env = env_loader.make('Tron1Env')
    env = IsaacEnvWrapper(ManagerBasedRLEnv(TronEnvCfg()))

    print('###########')
    print(env.spec)
    print('###########')

    obs, info = env.reset()
    print(obs.shape)
    print('articulation num', len(env.env.scene.articulations))

    while simulation_app.is_running():

        dummy_action = th.zeros((env.n_env, env.n_action), dtype=th.float32, device=env.device)

        # action = th.zeros_like(dummy_action)
        # action = th.ones_like(dummy_action) * 0.3
        action = th.randn_like(dummy_action)

        obs, rwd, ter, tru, info = env.step(action)

        state = list(env.env.scene.articulations.values())[0].data.root_state_w
        
        # print(
        #     rwd.mean().item(),
        #     rwd.std().item(),
        #     rwd.min().item(),
        #     rwd.max().item(),
        #     sep='\t',
        # )

        # print(rwd.item())

        # print(
        #     state.mean().item(),
        #     state.std().item(),
        #     state.min().item(),
        #     state.max().item(),
        #     sep='\t',
        # )

        # print(
        #     *[state[:,k*2].item() for k in range(5)],
        #     sep='\t',
        # )

    env.close()


if __name__ == '__main__':
    main()
