from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import math

from bipedal_lab.tasks.default_cfg import DefaultEnvCfg
from bipedal_lab.tasks.tron1.robot import TRON1_PFOOT_CFG

from bipedal_lab.base.managers import (
    ActionManagerCfg,
    CommandManagerCfg,
    ObservationManagerCfg,
    RandomizeManagerCfg,
    RewardManagerCfg,
    RobotDataManagerCfg,
    TerminationManagerCfg,
)



_POST_INIT = None

_LINK = {
    'base': ['base_Link'],
    'abad': ['abad_L_Link',     'abad_R_Link'],
    'hip':  ['hip_L_Link',      'hip_R_Link'],
    'knee': ['knee_L_Link',     'knee_R_Link'],
    'foot': ['foot_L_Link',     'foot_R_Link'],
}
def _link(*args):
    return sum([_LINK[a] for a in args], start=[])

_JOINT = {
    'abad': ['abad_L_Joint',    'abad_R_Joint'],
    'hip':  ['hip_L_Joint',     'hip_R_Joint'],
    'knee': ['knee_L_Joint',    'knee_R_Joint'],
}
def _joint(*args):
    return sum([_JOINT[a] for a in args], start=[])



@configclass
class Tron1PEnvCfg(DefaultEnvCfg):

    rdm_cfg = RobotDataManagerCfg(
        ar_robot=SceneEntityCfg(name='robot'),
        co_robot=SceneEntityCfg(name='contact_sensor'),
    )

    act_cfg = ActionManagerCfg(
        min_delayed_steps=1,
        max_delayed_steps=3,

        ar_robot=SceneEntityCfg(name='robot'),
        q_names=_joint(
            'abad', # boundary sign inverted (mirrored value)
            'hip',  # boundary sign inverted
            'knee', # boundary sign inverted
        ),
        q_scale=[
            0.5, # abad_L_Joint
            0.5, # abad_R_Joint
            0.5, # hip_L_Joint
            0.5, # hip_R_Joint
            0.5, # knee_L_Joint
            0.5, # knee_R_Joint
        ],
    )

    cmd_cfg = CommandManagerCfg(
        cmd_rng=[
            (-1.5, 1.5), # x-linear velocity
            (-1.0, 1.0), # y-linear velocity
            (-1.0, 1.0), # z-angular velocity
        ],
        cmd_div=[
            12, # x-linear
            8,  # y-linear
            1,  # z-angular (heading command)
        ],

        min_cmd_norm=0.2,
        phase_len=[600, 200, 200],

        heading_dims=[2],
        heading_rng=[
            (-math.pi, math.pi), # z-angular
        ],
        heading_kp=[0.5],
    )

    obs_cfg = ObservationManagerCfg(
        n_history=10,
        obs_scale=[
            0.25,   # root_angvel_b
            1.0,    # gravity_dir_b
            1.0,    # qpos
            0.05,   # qvel
            1.0,    # action
            1.0,    # root_linvel_t
        ],
    )

    rnd_cfg = RandomizeManagerCfg(
        cof_cfgs=[
            RandomizeManagerCfg.CofCfg( # foot cof
                ar_body=SceneEntityCfg(name='robot', body_names=_link('foot')),
                static_cof_rng=(0.6, 1.0),
                kinetic_cof_rng=(0.4, 0.8),
                cor_rng=(0.0, 0.0),
                n_bucket=_POST_INIT,
            ),
        ],
        mass_cfgs=[
            RandomizeManagerCfg.MassCfg( # base mass
                ar_body=SceneEntityCfg(name='robot', body_names=_link('base')),
                add_rng=(-2.0, 2.0 + 8.0),
                add_com={
                    'x': (-0.05, 0.05),
                    'y': (-0.05, 0.05),
                    'z': (-0.02, 0.02 + 0.03),
                },
            ),
        ],
        pd_gain_cfgs=[
            RandomizeManagerCfg.PDGainCfg( # legs pd gain
                ar_joint=SceneEntityCfg(name='robot', joint_names=_joint('abad', 'hip', 'knee')),
                kp_rng=(45.0 * 0.95, 45.0 * 1.05),
                kd_rng=(1.50 * 0.95, 1.50 * 1.05),
            ),
        ],

        ar_robot=SceneEntityCfg(name='robot'),
        push_rng=[
            (-1.0, 1.0), # linvel_x
            (-1.0, 1.0), # linvel_y
            (-1.0, 1.0), # linvel_z
            (-0.5, 0.5), # angvel_x
            (-0.5, 0.5), # angvel_y
            (-0.5, 0.5), # angvel_z
        ],
        push_steps=[350, 700],
    )

    rwd_cfg = RewardManagerCfg(
        ar_foot=SceneEntityCfg(
            name='robot',
            body_names=_link('foot'),
            preserve_order=True,
        ),
        co_body=SceneEntityCfg(
            name='contact_sensor',
            body_names=_link('base', 'abad', 'hip', 'knee'),
        ),
        co_foot=SceneEntityCfg(
            name='contact_sensor',
            body_names=_link('foot'),
            preserve_order=True,
        ),

        vel_err_sma_window = 10,
        track_lin_err_scale = 4.0,
        track_ang_err_scale = 4.0,

        q_names=_joint(
            'abad', # boundary sign inverted (mirrored value)
            'hip',  # boundary sign inverted
            'knee', # boundary sign inverted
        ),
        qpos_limit=[
            (-0.2, 0.5), # abad_L_Joint
            (-0.5, 0.2), # abad_R_Joint
            (-0.6, 0.6), # hip_L_Joint
            (-0.6, 0.6), # hip_R_Joint
            (-0.3, 1.2), # knee_L_Joint
            (-1.2, 0.3), # knee_R_Joint
        ],
        qtau_limit=[
            (-30.0, 30.0), # abad_L_Joint
            (-30.0, 30.0), # abad_R_Joint
            (-30.0, 30.0), # hip_L_Joint
            (-30.0, 30.0), # hip_R_Joint
            (-30.0, 30.0), # knee_L_Joint
            (-30.0, 30.0), # knee_R_Joint
        ],

        foot_stance_z       = -0.8,
        foot_clear_z        = 0.2,
        foot_min_air_ratio  = 0.4,
        foot_min_period     = 0.4,

        reward_clip         = (-50.0, 100.0),
        min_mean_reward     = 0.0,

        # ----- tracking -----
        k_track_lin     = 1.0,
        k_track_ang     = 0.5,
        # ----- motion penalty -----
        k_pen_lin       = -2.0,
        k_pen_ang       = -0.05,
        k_upright       = -0.2,
        # ----- dof penalty -----
        k_mec_energy    = -1e-4,
        k_the_energy    = -2e-5,
        k_d_action      = -0.005,
        k_d2_action     = -0.005,
        k_qpos_limit    = -0.05,
        k_qtau_limit    = -0.001,
        # ----- foot -----
        k_foot_clear    = -0.5, # 0.5 * 0.25 * 2 = 0.25
        k_foot_ratio    = 0.5,  # 0.5 * 0.4 * 2 = 0.4
        k_foot_period   = 0.5,  # 0.5 * 1.0 * 2 = 1.0
        k_foot_slip     = -0.25,
        # ----- extras -----
        k_contact       = -0.5,
        k_termin        = -10.0,
    )

    ter_cfg = TerminationManagerCfg(
        co_termin=SceneEntityCfg(
            name='contact_sensor',
            body_names=_link('base', 'abad'),
        ),
        max_tilt_angle=180.0, # turn off tilt termination
        max_episode_length=_POST_INIT,
    )


    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = TRON1_PFOOT_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')

        n_qdim = 6
        n_cmd = len(self.cmd_cfg.cmd_rng)

        self.action_space = n_qdim
        self.observation_space = (3 * 2 + n_qdim * 3) * self.obs_cfg.n_history + n_cmd + 3

        # ----------

        self.rnd_cfg.cof_cfgs[0].n_bucket = self.n_env

        self.ter_cfg.max_episode_length = self.episode_length
