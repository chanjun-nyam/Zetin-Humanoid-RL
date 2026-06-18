from typing import Literal

import typer

app = typer.Typer()


@app.command(name='play')
def main(
    env_id: str,
    terrain_type: Literal['flat', 'rough'],
    n_env: int = 64,
    device: str = 'cuda:0',
    load_path: str = None,
    deterministic: bool = True,
):
    """
    Play
    """

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True, livestream=2)
    simulation_app = app_launcher.app

    from simple_rl.runner import PPORunner
    from simple_rl.algorithms.ppo import PPO, PPOCfg
    from simple_rl.modules.modules import MlpActorCritic

    import bipedal_lab.tasks
    from bipedal_lab.env_utils import env_loader

    import torch as th


    env = env_loader.make(
        id=env_id,
        n_env=n_env,
        device=device,
        terrain_type=terrain_type,
    )

    actor_critic = MlpActorCritic(
        n_obs=env.n_obs,
        n_action=env.n_action,
        init_std=1.0,
        net_arch=[512, 256, 128],
        activ_fn=th.nn.ReLU,
    )
    ppo_cfg = PPOCfg(
        n_rollout=50,
        n_epoch=5,
        n_minibatch=4,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=1e-3,
        desired_kl=0.01,
        normalize_observation=False,
        ratio_clip_param=0.2,
        value_clip_param=0.2,
        grad_norm_clip=1.0,
        normalize_advantage=True,
        entropy_loss_coeff=0.01,
        value_loss_coeff=1.0,
    )
    ppo = PPO(env.spec, actor_critic, ppo_cfg)

    if load_path is not None:
        ppo.load(load_path)

    print(env.spec)
    print(env.env.rdm.q_names)


    import time

    current_nano = time.perf_counter_ns()
    step_cnt = 0
    obs, info = env.reset()

    while simulation_app.is_running():
        last_nano = current_nano
        current_nano = time.perf_counter_ns()
        step_cnt += 1

        # TODO
        if False:
            env.env.episode_length_buf[:] = 100
            obs[:,-6] = 1.5
            obs[:,-5] = 0.0
            obs[:,-4] = 0.0
            print(
                env.env.rdm.root_linvel_b.mean(dim=0),
                env.env.rwd_mgr.vel_buff.sma.mean(dim=0))

        act = ppo.act(obs, deterministic)
        obs, rwd, ter, tru, info = env.step(act)

        fps = (10 ** 9) / (current_nano - last_nano)

        if 'curriculum' not in info:
            info['curriculum'] = {'terrain_levels': {}}

        terrain_levels = list(info['curriculum']['terrain_levels'].values())
        # print(f'cnt: {step_cnt}, fps: {fps:.2f}, terrain-levels:', *[f'{v:.2f}' for v in terrain_levels])

        # TODO
        # env.env.episode_length_buf[:] = 100
        # print(env.env.rdm.qpos[:,[2,3,7,8]])
        # print(env.env.rdm.qpos[:,[0,1,12,13]])

        # TODO
        # print(
        #     f"mec: {info['reward']['mec_energy']:.4f}  "
        #     f"the: {info['reward']['the_energy']:.4f}  "
        #     f"d1: {info['reward']['d_action']:.4f}  "
        #     f"d2: {info['reward']['d2_action']:.4f}  "
        #     f"qpos: {info['reward']['qpos_limit']:.4f}  "
        #     f"qtau: {info['reward']['qtau_limit']:.4f}  "
        # )


    env.close()
    simulation_app.close()
