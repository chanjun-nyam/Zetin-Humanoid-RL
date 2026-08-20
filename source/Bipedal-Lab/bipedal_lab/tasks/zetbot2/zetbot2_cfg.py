from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from typing import Tuple, List

import torch as th
import math

from bipedal_lab.tasks.default_cfg import DefaultEnvCfg
from bipedal_lab.base.managers import (
    ActionManagerCfg,
    CommandManagerCfg,
    RandomizeManagerCfg,
    RewardManagerCfg,
    RobotDataManagerCfg,
    TerminationManagerCfg,
)
import bipedal_lab.primitives.reward as rwd_prims
from .robot import ZETBOT2_CFG



# ---------- USD INFO: Link/Joint ----------
_POST_INIT = None

_LINK = {
    'base': ['base_link'],
    'thigh': [
        'pelvis_L_1', 'thigh_L_1',
        'pelvis_R_1', 'thigh_R_1',
    ],
    'calf': ['calf_L_1', 'calf_R_1'],
    'foot': ['foot_L_1', 'foot_R_1'],
}
def _link(*args):
    return sum([_LINK[a] for a in args], start=[])

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



# ---------- SHARED BUFFER: FROM REWARD MGR ----------
class _SharedBuffMgr:
    keys = [
        # ----- tracking/motion penalty -----
        'linvel_02', 'linvel_23', 'angvel_02', 'angvel_23',
        'lincmd', 'angcmd', 'gravdir_02',
        # ----- dof penalty -----
        'qacc', 'qpwr', 'qtau', 'd1_action', 'd2_action',
        'qpos_violate', 'qtau_violate', 'qpos_diff',
        # ----- foot/gait -----
        'rdm', 'ar_foot_ids', 'co_foot_ids',
        'gait_theta', 'gait_ratio', 'is_stand', 'is_walk', 'foot_cont',
        # ----- extras -----
        'pen_contact', 'terminated',
    ]


    def __init__(
            self,
            ar_foot_names: List[str],
            co_body_names: List[str],
            co_foot_names: List[str],
            q_names: List[str],
            qpos_limit: List[Tuple[float, float]],
            qtau_limit: List[Tuple[float, float]],
        ):
        self.cfg_ar_foot_names = ar_foot_names
        self.cfg_co_body_names = co_body_names
        self.cfg_co_foot_names = co_foot_names
        self.cfg_q_names = q_names
        self.cfg_qpos_limit = qpos_limit
        self.cfg_qtau_limit = qtau_limit

        self.shared: dict = None


    def _update_shared(self):
        for key in self.keys:
            if hasattr(self, key):
                self.shared[key] = getattr(self, key)


    def init(self, mgr, shared: dict):
        self.shared = shared

        # q-idx mapping
        q_names, ref_q_names = self.cfg_q_names, mgr.env.rdm.q_names
        self.to_q_ref = [q_names.index(x) for x in ref_q_names if x in q_names]
        self.from_q_ref = [ref_q_names.index(x) for x in q_names]

        # ---------- shared ----------
        self.ar_foot = SceneEntityCfg(
            name='robot',
            body_names=self.cfg_ar_foot_names,
            preserve_order=True,
        )
        self.co_body = SceneEntityCfg(
            name='contact_sensor',
            body_names=self.cfg_co_body_names,
        )
        self.co_foot = SceneEntityCfg(
            name='contact_sensor',
            body_names=self.cfg_co_foot_names,
            preserve_order=True,
        )
        self.ar_foot.resolve(mgr.env.scene)
        self.co_foot.resolve(mgr.env.scene)
        self.co_body.resolve(mgr.env.scene)

        self.qpos_limit = th.tensor(
            self.cfg_qpos_limit, dtype=th.float32, device=mgr.env.device) # (n_qdim, 2)
        self.qtau_limit = th.tensor(
            self.cfg_qtau_limit, dtype=th.float32, device=mgr.env.device) # (n_qdim, 2)

        self._update_shared()


    def update(self, mgr, shared: dict):
        rdm = mgr.env.rdm
        act_mgr = mgr.env.act_mgr
        cmd_mgr = mgr.env.cmd_mgr
        ter_mgr = mgr.env.ter_mgr

        qpos_violate = rdm.qpos[:,self.from_q_ref].unsqueeze(-1) - self.qpos_limit
        qpos_violate[:,:,0].clip_(max=0.0)
        qpos_violate[:,:,1].clip_(min=0.0)
        qpos_violate = th.sum(qpos_violate, dim=2)

        qtau_violate = rdm.qtau[:,self.from_q_ref].unsqueeze(-1) - self.qtau_limit
        qtau_violate[:,:,0].clip_(max=0.0)
        qtau_violate[:,:,1].clip_(min=0.0)
        qtau_violate = th.sum(qtau_violate, dim=2)

        # ---------- shared ----------
        # ----- tracking/motion penalty -----
        self.linvel_02 = rdm.root_linvel_b[:,0:2]
        self.linvel_23 = rdm.root_linvel_b[:,2:3]
        self.angvel_02 = rdm.root_angvel_b[:,0:2]
        self.angvel_23 = rdm.root_angvel_b[:,2:3]
        self.lincmd = cmd_mgr.cmd[:,0:2]
        self.angcmd = cmd_mgr.cmd[:,2:3]
        self.gravdir_02 = rdm.gravity_dir_b[:,0:2]
        # ----- dof penalty -----
        self.qacc = rdm.qacc[:,self.from_q_ref]
        self.qpwr = (rdm.qtau * rdm.qvel)[:,self.from_q_ref]
        self.qtau = rdm.qtau[:,self.from_q_ref]
        self.d1_action = act_mgr.act_diff(o=1)
        self.d2_action = act_mgr.act_diff(o=2)
        self.qpos_violate = qpos_violate
        self.qtau_violate = qtau_violate
        self.qpos_diff = (rdm.qpos - rdm.qpos_default)[:,self.from_q_ref]
        # ----- foot/gait -----
        self.rdm = rdm
        self.ar_foot_ids = self.ar_foot.body_ids
        self.co_foot_ids = self.co_foot.body_ids
        self.gait_theta = mgr.env.gait_theta
        self.gait_ratio = mgr.env.gait_ratio
        self.is_stand = cmd_mgr.is_zero
        self.is_walk = cmd_mgr.is_zero.logical_not()
        self.foot_cont = rdm.is_cont[:,self.co_foot.body_ids]
        # ----- extras -----
        self.pen_contact = rdm.is_cont[:,self.co_body.body_ids]
        self.terminated = ter_mgr.terminated

        self._update_shared()


_shared_buff_mgr = _SharedBuffMgr(
    ar_foot_names=_link('foot'),
    co_body_names=_link('base', 'thigh', 'calf'),
    co_foot_names=_link('foot'),
    q_names=_joint(
        'HP', 'HR', 'HY',
        'KP', 'A1', 'A2',
    ),
    qpos_limit=[
        (-1.0, 0.0), # HP-hip1_L
        (+0.0, 1.0), # HP-hip1_R
        (-0.17, 1.0), # HR-hip2_L
        (-1.0, 0.17), # HR-hip2_R
        (-0.5, 0.5), # HY-hip3_L
        (-0.5, 0.5), # HY-hip3_R
        (+0.0, 1.0), # KP-knee_L
        (-1.0, 0.0), # KP-knee_R
        (-0.8, 0.8), # A1-joint1_L
        (-0.8, 0.8), # A1-joint1_R
        (-0.8, 0.8), # A2-joint2_L
        (-0.8, 0.8), # A2-joint2_R
    ],
    qtau_limit=[
        (-35., 35.), # HP-hip1_L
        (-35., 35.), # HP-hip1_R
        (-21., 21.), # HR-hip2_L
        (-21., 21.), # HR-hip2_R
        (-7.0, 7.0), # HY-hip3_L
        (-7.0, 7.0), # HY-hip3_R
        (-21., 21.), # KP-knee_L
        (-21., 21.), # KP-knee_R
        (-11., 11.), # A1-joint1_L
        (-11., 11.), # A1-joint1_R
        (-11., 11.), # A2-joint2_L
        (-11., 11.), # A2-joint2_R
    ],
)



@configclass
class Zetbot2EnvCfg(DefaultEnvCfg):

    # ---------- ROBOT DATA ----------
    rdm_cfg = RobotDataManagerCfg(
        ar_robot=SceneEntityCfg(name='robot'),
        co_robot=SceneEntityCfg(name='contact_sensor'),
    )


    # ---------- ACTION ----------
    act_cfg = ActionManagerCfg(
        min_delayed_steps=0,
        max_delayed_steps=0,

        ar_robot=SceneEntityCfg(name='robot'),
        q_names=_joint(
            'HP', 'HR', 'HY',
            'KP', 'A1', 'A2',
        ),
        act_scale=[0.5] * 12,
    )


    # ---------- OBSERVATION ----------
    obs_q_names = _joint(
        'HP', 'HR', 'HY',
        'KP', 'A1', 'A2',
    )


    # ---------- COMMAND ----------
    cmd_cfg = CommandManagerCfg(
        cmd_rng=[
            # ----- velocity -----
            (-1.0, 1.0), # x-linear velocity [m/s]
            (-1.0, 1.0), # y-linear velocity [m/s]
            (-1.0, 1.0), # z-angular velocity [m/s]
            # ----- gait -----
            (0.8, 1.6), # gait frequency [Hz]
            (0.5, 0.5), # gait ratio
            (th.pi, th.pi), # gait offset
        ],
        cmd_div=[
            8,  # x-linear
            8,  # y-linear
            1,  # z-angular (heading command)
            1,  # gait frequency
            1,  # gait ratio
            1,  # gait offset
        ],

        zero_dims=[0, 1, 2, 3, 4, 5],
        zero_ratio=[0.0, 0.2, 0.2, 0.2, 0.2], # total: 0.2 * 0.4 = 0.08

        phase_len=[600, 100, 100, 100, 100],

        heading_dims=[2],
        heading_rng=[
            (-math.pi, math.pi), # z-angular
        ],
        heading_kp=[0.5],
    )


    # ---------- DOMAIN RANDOMIZE ----------
    rnd_cfg = RandomizeManagerCfg(
        cof_cfgs=[
            RandomizeManagerCfg.CofCfg( # foot cof
                ar_body=SceneEntityCfg(name='robot', body_names=_link('foot')),
                static_cof_rng=(0.5, 1.0),
                kinetic_cof_rng=(0.3, 0.8),
                cor_rng=(0.0, 1.0),
                n_bucket=_POST_INIT,
            ),
        ],
        mass_cfgs=[
            RandomizeManagerCfg.MassCfg( # base mass
                ar_body=SceneEntityCfg(name='robot', body_names=_link('base')),
                add_rng=(-2.0, 2.0),
                add_com={
                    'x': (-0.08, 0.08),
                    'y': (-0.05, 0.05),
                    'z': (-0.05, 0.05),
                },
            ),
        ],
        pd_gain_cfgs=[
            RandomizeManagerCfg.PDGainCfg( # legs pd gain
                ar_joint=SceneEntityCfg(name='robot', joint_names=_joint('HP', 'HR', 'HY', 'KP')),
                kp_rng=(45. * 0.8, 45. * 1.2),
                kd_rng=(1.5 * 0.8, 1.5 * 1.2),
            ),
            RandomizeManagerCfg.PDGainCfg( # ankles pd gain
                ar_joint=SceneEntityCfg(name='robot', joint_names=_joint('A1', 'A2')),
                kp_rng=(35. * 0.8, 35. * 1.2),
                kd_rng=(0.7 * 0.8, 0.7 * 1.2),
            ),
        ],

        ar_robot=SceneEntityCfg(name='robot'),
        push_rng=[
            (-1.0, 1.0), # linvel_x
            (-1.0, 1.0), # linvel_y
            (-1.0, 1.0), # linvel_z
            (-1.0, 1.0), # angvel_x
            (-1.0, 1.0), # angvel_y
            (-1.0, 1.0), # angvel_z
        ],
        push_steps=[50, 350, 800],
    )


    # ---------- REWARD ----------
    rwd_cfg = RewardManagerCfg(
        clip_rng = (-100.0, 100.0),

        bonus_threshold = 0.0,

        init_shared_buff = _shared_buff_mgr.init,

        update_shared_buff = _shared_buff_mgr.update,

        terms = {
            # ----- alive -----
            'alive': rwd_prims.Val(w=0.5, val=1.0),
            # ----- tracking -----
            'track_lin': rwd_prims.Track(w=1.0, s=4.0, val='linvel_02', cmd='lincmd', n_window=10),
            'track_ang': rwd_prims.Track(w=0.5, s=4.0, val='angvel_23', cmd='angcmd', n_window=10),
            # ----- motion penalty -----
            'pen_lin': rwd_prims.VecNormPow(w=-1.0, p=2, val='linvel_23'),
            'pen_ang': rwd_prims.VecNormPow(w=-0.02,p=2, val='angvel_02'),
            'upright': rwd_prims.VecNorm(w=-0.5, p=2, val='gravdir_02'),
            # ----- dof penalty -----
            'qacc': rwd_prims.VecNormPow(w=-1.6e-7, p=2, val='qacc'),
            'd2_action': rwd_prims.VecNormPow(w=-0.003, p=2, val='d2_action'),
            'mec_energy': rwd_prims.VecNormPow(w=-1.3e-4, p=1, val='qpwr'),
            'the_energy': rwd_prims.VecNormPow(w=-1.3e-5, p=2, val='qtau'),
            'qpos_limit': rwd_prims.VecNormPow(w=-0.1,  p=1, val='qpos_violate'),
            'qtau_limit': rwd_prims.VecNormPow(w=-0.005,p=1, val='qtau_violate'),
            'qpos': rwd_prims.VecNormPow(w=-0.03, p=1, val='qpos_diff'),
            # ----- stand -----
            'stand_cont': rwd_prims.Sum(w=0.2, val='foot_cont', mask='is_stand'),
            'stand_clear': rwd_prims.FootClear(w=-0.8, p=1, stance_z=-0.4, clear_z=0.12, mask='is_stand'),
            'stand_qpos': rwd_prims.VecNormPow(w=-0.1, p=1, val='qpos_diff', mask='is_stand'),
            # ----- gait/foot -----
            'slip': rwd_prims.FootSlip(w=-0.25),
            'clear': rwd_prims.FootClear(w=-0.8, p=1, stance_z=-0.4, clear_z=0.12),
            'gait': rwd_prims.Gait(
                w=-0.5,
                k=4,
                s_frc=25.0,
                s_spd=0.2,
                n_sample=1000,
                mask='is_walk',
            ),
            # ----- extras -----
            'contact': rwd_prims.Sum(w=-1.0, val='pen_contact'),
            'termin': rwd_prims.Val(w=-20.0, val='terminated'),
        },
    )


    # ---------- TERMINATION ----------
    ter_cfg = TerminationManagerCfg(
        co_termin=SceneEntityCfg(
            name='contact_sensor',
            body_names=_link('base', 'thigh'),
        ),
        max_tilt_angle=60.0,
        max_episode_length=_POST_INIT,
    )


    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ZETBOT2_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
        self.scene.contact_sensor.prim_path = '{ENV_REGEX_NS}/Robot/main4_7_3final_robot_only/.*'

        n_qdim = 12

        self.action_space = n_qdim
        self.observation_space = (
            (6 + n_qdim * 3) * self.n_obs_history +
            9 +
            (10 + n_qdim * 3) * 0
        )

        # ----------

        self.rnd_cfg.cof_cfgs[0].n_bucket = self.n_env

        self.ter_cfg.max_episode_length = self.episode_length
