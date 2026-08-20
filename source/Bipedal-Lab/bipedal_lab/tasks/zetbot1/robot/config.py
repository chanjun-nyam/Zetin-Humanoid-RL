import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

import os



_CWD = os.path.dirname(__file__)
_ZETBOT1_SFOOT_USD_PATH = os.path.join(_CWD, './usd/SF_ZETBOT/soccerbot.usd')



ZETBOT1_SFOOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_ZETBOT1_SFOOT_USD_PATH,
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
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            # sangchae
            '(head|neck)': 0.0,
            '(sho|arm|hand)_[LR]': 0.0,
            # SF_TRON1A
            '(abad|hip|knee|ankle)_[LR]_Joint': 0.0,
        },
        joint_vel={
            # sangchae
            '(head|neck)': 0.0,
            '(sho|arm|hand)_[LR]': 0.0,
            # SF_TRON1A
            '(abad|hip|knee|ankle)_[LR]_Joint': 0.0,
        },
    ),
    actuators={
        # sangchae
        'head': ImplicitActuatorCfg(
            joint_names_expr=['(head|neck)'],
            effort_limit_sim=26.66,
            velocity_limit_sim=15.0,
            stiffness=5.0,
            damping=0.3,
            friction=0.0,
        ),
        'arms': ImplicitActuatorCfg(
            joint_names_expr=['(sho|arm|hand)_[LR]'],
            effort_limit_sim=26.66,
            velocity_limit_sim=15.0,
            stiffness=15.0,
            damping=0.5,
            friction=0.0,
        ),
        # SF_TRON1A
        'legs': ImplicitActuatorCfg(
            joint_names_expr=['(abad|hip|knee)_[LR]_Joint'],
            effort_limit_sim=80.0,
            velocity_limit_sim=15.0,
            stiffness=45.0,
            damping=1.5,
            friction=0.0,
        ),
        'ankles': ImplicitActuatorCfg(
            joint_names_expr=['(ankle)_[LR]_Joint'],
            effort_limit_sim=80.0,
            velocity_limit_sim=15.0,
            stiffness=45.0,
            damping=0.8,
            friction=0.0,
        ),
    },
)
