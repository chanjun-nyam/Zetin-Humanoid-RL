import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

import os



_CWD = os.path.dirname(__file__)
_ZETBOT2_USD_PATH = os.path.join(_CWD, './usd/zetbot2_optimized.usd')



# ---------- USD INFO: Joint ----------
_JOINT = {
    'HP': ['hip1_L', 'hip1_R'],
    'HR': ['hip2_L', 'hip2_R'],
    'HY': ['hip3_L', 'hip3_R'],
    'KP': ['knee_L', 'knee_R'],
    'A1': ['joint1_L', 'joint1_R'],
    'A2': ['joint2_L', 'joint2_R'],
    '_LINK': [
        'ankle[12]_[LR]:[0-2]', 'foot[12]_[LR]',
        # excluded from articulation
        # 'ankle[12]_[LR])01:[0-2]',
    ],
}
def _joint(*args):
    return sum([_JOINT[a] for a in args], start=[])



ZETBOT2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_ZETBOT2_USD_PATH,
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
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            'hip1_L': -0.3,
            'hip1_R': +0.3,
            'hip2_L': 0.0,
            'hip2_R': 0.0,
            'hip3_L': 0.0,
            'hip3_R': 0.0,
            'knee_L': +0.6,
            'knee_R': -0.6,
            'joint1_L': 0.0,
            'joint1_R': 0.0,
            'joint2_L': 0.0,
            'joint2_R': 0.0,
            # link
            'ankle[12]_[LR]:[0-2]': 0.0,
            'foot[12]_[LR]': 0.0,
        },
        joint_vel={
            '.*': 0.0,
        },
    ),
    actuators={
        'rs04': ImplicitActuatorCfg( # rated/peak: 35/120
            joint_names_expr=_joint('HP'),
            effort_limit_sim=100.0,
            velocity_limit_sim=15.0,
            stiffness=45.0,
            damping=1.5,
            friction=0.0,
        ),
        'rs03': ImplicitActuatorCfg( # rated/peak: 21/60
            joint_names_expr=_joint('HR', 'KP'),
            effort_limit_sim=50.0,
            velocity_limit_sim=15.0,
            stiffness=45.0,
            damping=1.5,
            friction=0.0,
        ),
        'rs06': ImplicitActuatorCfg( # rated/peak: 11/36
            joint_names_expr=_joint('A1', 'A2'),
            effort_limit_sim=30.0,
            velocity_limit_sim=15.0,
            stiffness=35.0,
            damping=0.7,
            friction=0.0,
        ),
        'rs02': ImplicitActuatorCfg( # rated/peak: 7/17
            joint_names_expr=_joint('HY'),
            effort_limit_sim=15.0,
            velocity_limit_sim=15.0,
            stiffness=45.0,
            damping=1.5,
            friction=0.0,
        ),
        '_link': ImplicitActuatorCfg(
            joint_names_expr=_joint('_LINK'),
            effort_limit_sim=0.0,
            velocity_limit_sim=15.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
        ),
    },
)
