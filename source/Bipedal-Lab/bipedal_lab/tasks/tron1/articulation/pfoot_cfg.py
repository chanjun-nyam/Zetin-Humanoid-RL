import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

import os

_current_dir = os.path.dirname(__file__)
_usd_path = os.path.join(_current_dir, './usd/PF_TRON1A/PF_TRON1A.usd')

TRON1_PFOOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_usd_path,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            '(abad|hip|knee|foot)_(L|R)_Joint': 0.0,
        },
        joint_vel={
            '(abad|hip|knee|foot)_(L|R)_Joint': 0.0,
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        'legs': ImplicitActuatorCfg(
            joint_names_expr=[
                '(abad|hip|knee)_(L|R)_Joint',
            ],
            effort_limit_sim=80.0,
            velocity_limit_sim=15.0,
            stiffness=45.0,
            damping=1.5,
            friction=0.0,
        ),
    },
)
