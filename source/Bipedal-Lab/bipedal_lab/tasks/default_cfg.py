from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
import isaaclab.sim as sim_utils

from dataclasses import MISSING

from bipedal_lab.base.env_cfg import BipedalEnvCfg
from bipedal_lab.base.managers import (
    ActionManagerCfg,
    ObservationManagerCfg,
    RewardManagerCfg,
    RobotDataManagerCfg,
    TerminationManagerCfg,
)



SIM_DT = 1 / 200
DECIMATION = 4
EPISODE_LENGTH = 1000
NUM_ENVS = 2 ** 12
ENV_SPACING = 4.0
DEVICE = 'cuda:0'
# NUM_ENVS, DEVICE = 1, 'cpu'


@configclass
class SceneCfg(InteractiveSceneCfg):

    terrain = TerrainImporterCfg(
        prim_path='/World/ground',
        terrain_type='plane',
        terrain_generator=None,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode='multiply',
            restitution_combine_mode='multiply',
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f'{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl',
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = MISSING

    contact_sensor = ContactSensorCfg(
        prim_path='{ENV_REGEX_NS}/Robot/.*',
        update_period=SIM_DT,
        history_length=DECIMATION,
        track_air_time=True,
    )
    
    sky_light = AssetBaseCfg(
        prim_path='/World/SkyLight',
        spawn=sim_utils.DomeLightCfg(
            intensity=800.0,
            texture_file=f'{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr',
        ),
    )



@configclass
class DefaultEnvCfg(BipedalEnvCfg):

    # config terms derived from `DirectRLEnv`

    sim = sim_utils.SimulationCfg(
        device=DEVICE,
        dt=SIM_DT,
        render_interval=DECIMATION,
    )

    decimation = DECIMATION

    episode_length_s = EPISODE_LENGTH * DECIMATION * SIM_DT

    scene = SceneCfg(NUM_ENVS, ENV_SPACING)

    action_space = MISSING

    observation_space = MISSING

    # config terms derived from `BipedalEnv`

    action_scale = 0.5

    robot_cfg = SceneEntityCfg(name='robot')

    rdm_cfg = RobotDataManagerCfg(
        ar_robot=SceneEntityCfg(name='robot'),
        co_robot=SceneEntityCfg(name='contact_sensor'),
    )

    act_cfg = ActionManagerCfg(
        # min_delayed_steps=1,
        # max_delayed_steps=DECIMATION,
        min_delayed_steps=1,
        max_delayed_steps=3,
    )

    obs_cfg = ObservationManagerCfg(
        n_history=MISSING,
        # obs_scale=[
        #     0.25,   # root_angvel_b
        #     1.0,    # gravity_dir_b
        #     1.0,    # qpos
        #     0.05,   # qvel
        #     1.0,    # action
        #     1.0,    # command
        #     1.0,    # root_linvel_t
        # ],
        obs_scale=[1.0 for _ in range(7)],
    )

    rwd_cfg = RewardManagerCfg(
        ar_robot=SceneEntityCfg(
            name='robot',
        ),
        ar_foot=SceneEntityCfg(
            name='robot',
            body_names=['ankle_(L|R)_Link'],
            preserve_order=True,
        ),
        co_body=SceneEntityCfg(
            name='contact_sensor',
            body_names=['base_Link', 'abad_(L|R)_Link', 'hip_(L|R)_Link', 'knee_(L|R)_Link'],
        ),
        co_foot=SceneEntityCfg(
            name='contact_sensor',
            body_names=['ankle_(L|R)_Link'],
            preserve_order=True,
        ),

        track_lin_err_scale = 2.0,
        track_ang_err_scale = 2.0,

        q_names=[
            'abad_L_Joint', 'abad_R_Joint', # sign inverted (mirrored value)
            'hip_L_Joint',  'hip_R_Joint',  # sign inverted
            'knee_L_Joint', 'knee_R_Joint', # sign inverted
            'ankle_L_Joint','ankle_R_Joint',# sign correct
        ],
        qpos_limit=[
            [-0.2, 0.6], # abad_L_Joint
            [-0.6, 0.2], # abad_R_Joint
            [-0.5, 0.5], # hip_L_Joint
            [-0.5, 0.5], # hip_R_Joint
            [-0.2, 0.9], # knee_L_Joint
            [-0.9, 0.2], # knee_R_Joint
            [-0.5, 0.5], # ankle_L_Joint
            [-0.5, 0.5], # ankle_R_Joint
        ],
        qtau_limit=[
            [-30.0, 30.0], # abad_L_Joint
            [-30.0, 30.0], # abad_R_Joint
            [-30.0, 30.0], # hip_L_Joint
            [-30.0, 30.0], # hip_R_Joint
            [-30.0, 30.0], # knee_L_Joint
            [-30.0, 30.0], # knee_R_Joint
            [-30.0, 30.0], # ankle_L_Joint
            [-30.0, 30.0], # ankle_R_Joint
        ],

        foot_stance_z       = -0.75,
        foot_clear_z        = 0.25,
        foot_min_air_ratio  = 0.4,
        foot_min_period     = 0.5,

        min_mean_reward     = 0.0,

        k_track_lin     = 1.0,
        k_track_ang     = 0.5,
        k_pen_lin       = -2.0,
        k_pen_ang       = -0.05,
        k_upright       = -0.2,
        # k_mec_energy    = -1e-5,
        # k_the_energy    = -1e-5,
        k_mec_energy    = -1e-4, # TODO
        k_the_energy    = -2e-5, # TODO
        k_d_action      = -0.005,
        k_d2_action     = -0.005,
        k_qpos_limit    = -0.05,
        k_qtau_limit    = -0.001,
        k_contact       = -0.5,
        # k_foot_clear    = -2.0,
        # k_foot_clear    = -0.5, # 0.5 * 0.2 * 2 = 0.2
        # k_foot_clear    = -0.3, # 0.3 * 0.2 * 2 = 0.12 #TODO
        k_foot_clear    = -0.5, # 0.3 * 0.2 * 2 = 0.12 #TODO
        # k_foot_ratio    = -0.2,
        # k_foot_period   = -0.2,
        k_foot_ratio    = 1.0 * 0.25 * 0.5, # 0.4 * 0.25 * 0.5 * 2 = 0.1
        # k_foot_period   = 0.4 * 0.25 * 2.0, # 0.4 * 0.25 * 2.0 * 2 = 0.4
        # k_foot_period   = 0.4 * 0.25 * 1.0, # 0.4 * 0.25 * 1.0 * 2 = 0.2 #TODO
        k_foot_period   = 0.4 * 0.25 * 5.0, # 0.4 * 0.25 * 5.0 * 2 = 1.0 #TODO
        k_termin        = -10.0,
    )

    ter_cfg = TerminationManagerCfg(
        ar_robot=SceneEntityCfg(name='robot'),
        co_termin=SceneEntityCfg(
            name='contact_sensor',
            body_names=['base_Link', 'abad_(L|R)_Link'],
        ),
        max_tilt_angle=180.0,
        max_episode_length=EPISODE_LENGTH,
        max_normal_force=1.0,
        normal_force_history_length=DECIMATION,
    )
