
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(livestream=2)
simulation_app = app_launcher.app

RUN_MODE = 'train'
# RUN_MODE = 'play'
# RUN_MODE = 'test'

ENV_NAME = 'Tron1Env'
RUN_PATH = 'results/runs'
MODEL_PATH = 'results/models/model.pt'
# MODEL_PATH = '/home/chanjun/workspace/Projects/Zetin-Humanoid-RL/results/runs/2026-05-20_05.42.50/checkpoints/model_600.pt'
# STOCHASTIC = True
# STOCHASTIC = False
STOCHASTIC = RUN_MODE == 'train'

from simple_rl.runner import PPORunner
from simple_rl.algorithms.ppo import PPO, PPOCfg
from simple_rl.modules.modules import MlpActorCritic

import bipedal_lab.tasks
from bipedal_lab.env_utils import env_loader

import torch as th


def main():
    env = env_loader.make(ENV_NAME)

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

    print(env.spec)
    print(env.env.robot.data.joint_names)

    # train
    if RUN_MODE == 'train':
        runner = PPORunner(
            env=env,
            algo=ppo,
            run_dir=RUN_PATH,
            log_interval=2,
            checkpoint_interval=300,
        )
        runner.train(3000)
        ppo.save(MODEL_PATH)
    
    # test/play
    elif RUN_MODE == 'test' or RUN_MODE == 'play':
        if RUN_MODE == 'play':
            ppo.load(MODEL_PATH)

        import time
        
        obs, info = env.reset()
        
        while simulation_app.is_running():
            start_s = time.time()
            obs, rwd, ter, tru, info = env.step(
                ppo.act(obs, deterministic=(not STOCHASTIC))
            )
            end_s = time.time()

            # print(('%6.3f\t'*8) % (*env.env.rdm.qpos[0,:].tolist(),))

            # print(1 / (end_s - start_s))
        #     break

        
        # class DummyModel(th.nn.Module):
        #     def __init__(self, actor_critic):
        #         super().__init__()
        #         self.actor_critic = actor_critic
        #     def forward(self, obs):
        #         act_pdf = self.actor_critic.policy.compute(obs)
        #         return act_pdf.mean
        # dummy_input = th.randn(1, env.n_obs, device=env.device)
        
        # th.onnx.export(
        #     DummyModel(actor_critic),
        #     dummy_input,
        #     'model.onnx',
        #     export_params=True,
        #     opset_version=13,
        #     do_constant_folding=True,
        #     input_names=['input'],
        #     output_names=['output'],
        #     dynamic_axes={
        #         'input': {0: 'batch_size'},
        #         'output': {0: 'batch_size'},
        #     }
        # )
    
    env.close()


if __name__ == '__main__':
    main()
    simulation_app.close()
