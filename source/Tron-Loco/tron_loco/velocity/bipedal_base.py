
import math
from dataclasses import MISSING

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.managers import (
    ObservationGroupCfg, ObservationTermCfg,
    TerminationTermCfg,
    RewardTermCfg,
    EventTermCfg,
    CurriculumTermCfg,
    SceneEntityCfg,
)
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


# Scene
#

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

    light = AssetBaseCfg(
        prim_path='/World/skyLight',
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f'{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr',
        ),
    )

    robot: ArticulationCfg = MISSING

    height_scanner = RayCasterCfg(
        prim_path='{ENV_REGEX_NS}/Robot/base_Link',
        mesh_prim_paths=['/World/ground'],
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.0, 1.0)),
        debug_vis=False,
    )

    contact_forces = ContactSensorCfg(
        prim_path='{ENV_REGEX_NS}/Robot/.*',
        track_air_time=True,
    )

    def __post_init__(self):
        super().__post_init__()


# Commands
#

@configclass
class CommandsCfg:

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name='robot',
        heading_command=True,
        heading_control_stiffness=0.5,
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.01,
        rel_heading_envs=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi)
        ),
    )


# Actions
#

@configclass
class ActionsCfg:

    joint_pos = mdp.JointPositionActionCfg(
        asset_name='robot',
        joint_names=['.*'],
        scale=0.5,
        use_default_offset=True,
    )


# Observations
#

@configclass
class ObservationsCfg:

    @configclass
    class ObsGroupCfg(ObservationGroupCfg):

        # base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), history_length=10)
        projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=10,
        )
        velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={'command_name': 'base_velocity'})
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=10)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), history_length=10)
        actions = ObservationTermCfg(func=mdp.last_action, history_length=10)
        height_scan = ObservationTermCfg(
            func=mdp.height_scan,
            params={'sensor_cfg': SceneEntityCfg('height_scanner')},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    obs: ObsGroupCfg = ObsGroupCfg()


# Terminations
#

@configclass
class TerminationsCfg:

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names='base_Link'), 'threshold': 1.0},
    )


# Rewards
#

@configclass
class RewardsCfg:

    track_lin = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp,
        params={
            'command_name': 'base_velocity',
            'std': 0.5,
        },
        weight=1.0
    )
    track_ang = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp,
        params={
            'command_name': 'base_velocity',
            'std': 0.5,
        },
        weight=0.5
    )

    pen_lin = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0)
    pen_ang = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
    base_height = RewardTermCfg(
        func=mdp.base_height_l2,
        params={
            'sensor_cfg': SceneEntityCfg('height_scanner'),
            'target_height': 0.75,
        },
        weight=-1.0
    )
    orientation = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-0.2)
    contact = RewardTermCfg(
        func=mdp.undesired_contacts,
        params={
            'sensor_cfg': SceneEntityCfg('contact_forces', body_names='(base|(abad|hip|knee)_(L|R))_Link'),
            'threshold': 1.0,
        },
        weight=-0.5
    )

    joint_accel = RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewardTermCfg(func=mdp.joint_torques_l2, weight=-1.0e-5)
    joint_limits = RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0)
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)

    feet_air_time = RewardTermCfg(
        func=mdp.feet_air_time,
        params={
            'sensor_cfg': SceneEntityCfg('contact_forces', body_names='ankle_.*'),
            'command_name': 'base_velocity',
            'threshold': 0.4,
        },
        weight=5.0
    )


# Events
#

@configclass
class EventsCfg:
    
    # domain randomization
    physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode='startup',
        params={
            'asset_cfg': SceneEntityCfg('robot', body_names='.*'),
            'static_friction_range': (0.4, 1.25),
            'dynamic_friction_range': (0.2, 1.0),
            'restitution_range': (0.0, 1.0),
            'num_buckets': 512,
        },
    )

    base_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode='startup',
        params={
            'asset_cfg': SceneEntityCfg('robot', body_names='base_Link'),
            'mass_distribution_params': (-5.0, 5.0),
            'operation': 'add',
        },
    )

    base_com = EventTermCfg(
        func=mdp.randomize_rigid_body_com,
        mode='startup',
        params={
            'asset_cfg': SceneEntityCfg('robot', body_names='base_Link'),
            'com_range': {'x': (-0.05, 0.05), 'y': (-0.05, 0.05), 'z': (-0.05, 0.05)},
        },
    )

    link_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode='startup',
        params={
            'asset_cfg': SceneEntityCfg('robot', body_names='.*_(L|R)_Link'),
            'mass_distribution_params': (0.8, 1.2),
            'operation': 'scale',
        }
    )

    pd_controller = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode='startup',
        params={
            'asset_cfg': SceneEntityCfg('robot', joint_names='.*'),
            'stiffness_distribution_params': (0.8, 1.2),
            'damping_distribution_params': (0.8, 1.2),
            'operation': 'scale',
        }
    )

    # reset
    reset_accel = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('robot', body_names='base_Link'),
            'force_range': (0.0, 0.0),
            'torque_range': (-0.0, 0.0),
        },
    )

    reset_root_state = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode='reset',
        params={
            'pose_range': {'x': (-0.5, 0.5), 'y': (-0.5, 0.5), 'yaw': (-math.pi, math.pi)},
            'velocity_range': {
                'x': (-0.5, 0.5),
                'y': (-0.5, 0.5),
                'z': (-0.5, 0.5),
                'roll': (-0.5, 0.5),
                'pitch': (-0.5, 0.5),
                'yaw': (-0.5, 0.5),
            },
        },
    )

    reset_joint_state = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode='reset',
        params={
            'position_range': (-0.2, 0.2),
            'velocity_range': (-0.5, 0.5),
        },
    )

    # disturbances
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode='interval',
        interval_range_s=(10.0, 15.0),
        params={
            'velocity_range': {'x': (-0.5, 0.5), 'y': (-0.5, 0.5)},
        },
    )


# Curriculums
#

@configclass
class CurriculumCfg:

    pass


# Manager-based environment
#

@configclass
class BipedalBaseEnvCfg(ManagerBasedRLEnvCfg):

    scene: SceneCfg = SceneCfg()

    commands: CommandsCfg = CommandsCfg()

    actions: ActionsCfg = ActionsCfg()

    observations: ObservationsCfg = ObservationsCfg()

    terminations: TerminationsCfg = TerminationsCfg()

    rewards: RewardsCfg = RewardsCfg()

    events: EventsCfg = EventsCfg()

    curriculum: CurriculumCfg = CurriculumCfg()


    def __post_init__(self):
        super().__post_init__()

        self.decimation = 4
        self.num_rerenders_on_reset = 0

        self.is_finite_horizon = False
        self.episode_length_s = 20.0

        self.scene.num_envs = 4096
        self.scene.env_spacing = 2.5

        self.sim.device = 'cuda:0'
        self.sim.dt = 0.005 # 200Hz
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**16

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
