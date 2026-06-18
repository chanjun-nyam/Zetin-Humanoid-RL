from typing import Literal

import typer

app = typer.Typer()


@app.command(name='train')
def main(
    env_id: str,
    terrain_type: Literal['flat', 'rough'],
    n_env: int = 4096,
    device: str = 'cuda:0',
    run_dir: str = 'runs',
    save_path: str = 'models/model.pt',
    load_path: str = None,
):
    """
    Train
    """

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True, livestream=2)
    # app_launcher = AppLauncher(headless=True)
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
        # entropy_loss_coeff=0.01,
        entropy_loss_coeff=0.01 * (1.0 if env.n_action == 8 else 0.4), # TODO
        value_loss_coeff=1.0,
    )
    ppo = PPO(env.spec, actor_critic, ppo_cfg)

    if load_path is not None:
        ppo.load(load_path)

    print(env.spec)
    print(env.env.rdm.q_names)

    runner = PPORunner(
        env=env,
        algo=ppo,
        run_dir=run_dir,
        log_interval=2,
        checkpoint_interval=300,
    )
    runner.train(4000)
    ppo.save(save_path)


    env.close()
    simulation_app.close()
