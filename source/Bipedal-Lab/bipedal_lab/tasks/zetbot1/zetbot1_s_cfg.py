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
from .robot import ZETBOT1_SFOOT_CFG



# ---------- USD INFO: Link/Joint ----------
_POST_INIT = None

_LINK = {
    # sangchae
    'torso': ['base_link'],
    'head': ['head_1'],
    'neck': ['neck_1'],
    'sho':  ['sho_L_1', 'sho_R_1'],
    'arm':  ['arm_L_1', 'arm_R_1'],
    'hand': ['hand_L_1','hand_R_1'],
    # SF_TRON1A
    'base': ['base_Link'],
    'abad': ['abad_L_Link', 'abad_R_Link'],
    'hip':  ['hip_L_Link',  'hip_R_Link'],
    'knee': ['knee_L_Link', 'knee_R_Link'],
    'ankle':['ankle_L_Link','ankle_R_Link'],
}
def _link(*args):
    return sum([_LINK[a] for a in args], start=[])

_JOINT = {
    # sangchae
    'head': ['head'],
    'neck': ['neck'],
    'sho':  ['sho_L',   'sho_R'],
    'arm':  ['arm_L',   'arm_R'],
    'hand': ['hand_L',  'hand_R'],
    # SF_TRON1A
    'abad': ['abad_L_Joint',    'abad_R_Joint'],
    'hip':  ['hip_L_Joint',     'hip_R_Joint'],
    'knee': ['knee_L_Joint',    'knee_R_Joint'],
    'ankle':['ankle_L_Joint',   'ankle_R_Joint'],
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
    ar_foot_names=_link('ankle'),
    co_body_names=_link('base', 'abad', 'hip', 'knee'),
    co_foot_names=_link('ankle'),
    q_names=_joint(
        'head', 'neck', 'sho', 'arm', 'hand',
        'abad', 'hip', 'knee', 'ankle',
    ),
    qpos_limit=[
        # sangchae
        (-0.5, 0.5), # head
        (-0.5, 0.5), # neck
        (-0.8, 0.8), # sho_L
        (-0.8, 0.8), # sho_R
        (+0.1, 0.8), # arm_L
        (+0.1, 0.8), # arm_R
        (-0.8, 0.8), # hand_L
        (-0.8, 0.8), # hand_R
        # SF_TRON1A
        (-0.2, 0.6), # abad_L_Joint
        (-0.6, 0.2), # abad_R_Joint
        (-0.1, 0.7), # hip_L_Joint
        (-0.7, 0.1), # hip_R_Joint
        (-0.1, 1.0), # knee_L_Joint
        (-1.0, 0.1), # knee_R_Joint
        (-0.85, 0.85), # ankle_L_Joint
        (-0.85, 0.85), # ankle_R_Joint
    ],
    qtau_limit=[
        # sangchae
        (-4.0, 4.0), # head
        (-4.0, 4.0), # neck
        (-10.0, 10.0), # sho_L
        (-10.0, 10.0), # sho_R
        (-10.0, 10.0), # arm_L
        (-10.0, 10.0), # arm_R
        (-10.0, 10.0), # hand_L
        (-10.0, 10.0), # hand_R
        # SF_TRON1A
        (-30.0, 30.0), # abad_L_Joint
        (-30.0, 30.0), # abad_R_Joint
        (-30.0, 30.0), # hip_L_Joint
        (-30.0, 30.0), # hip_R_Joint
        (-30.0, 30.0), # knee_L_Joint
        (-30.0, 30.0), # knee_R_Joint
        (-15.0, 15.0), # ankle_L_Joint
        (-15.0, 15.0), # ankle_R_Joint
    ],
)



@configclass
class Zetbot1SEnvCfg(DefaultEnvCfg):

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
            # sangchae
            'head',
            'neck',
            'sho',
            'arm',
            'hand',
            # SF_TRON1A
            'abad', # boundary sign inverted (mirrored value)
            'hip',  # boundary sign inverted
            'knee', # boundary sign inverted
            'ankle',# boundary sign uninverted
        ),
        act_scale=[0.5] * 16,
        # scale * action = const.
        # reward
        # = sum action^2
        # = sum const.^2 * scale^-2
        #
        # reward_before
        # = sum_1^8 const.^2 * 4
        # ~ 4 * const.^2
        #
        # reward_after
        # = sum_1^8 const.^2 * 4 + sum_1^8 const.^2 * 1
        # ~ 4 * const.^2 + 1 * const.^2
        # = 5 * const.^2
    )


    # ---------- OBSERVATION ----------
    obs_q_names = _joint(
        'head', 'neck', 'sho', 'arm', 'hand',
        'abad', 'hip', 'knee', 'ankle',
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
                ar_body=SceneEntityCfg(name='robot', body_names=_link('ankle')),
                static_cof_rng=(0.5, 1.0),
                kinetic_cof_rng=(0.3, 0.8),
                cor_rng=(0.0, 1.0),
                n_bucket=_POST_INIT,
            ),
        ],
        mass_cfgs=[
            RandomizeManagerCfg.MassCfg( # lower body mass
                ar_body=SceneEntityCfg(name='robot', body_names=_link('base')),
                add_rng=(-2.0, 2.0),
                add_com={
                    'x': (-0.08, 0.08),
                    'y': (-0.05, 0.05),
                    'z': (-0.05, 0.05),
                },
            ),
            RandomizeManagerCfg.MassCfg( # upper body mass
                ar_body=SceneEntityCfg(name='robot', body_names=_link('torso')),
                add_rng=(-0.5, 0.5),
                add_com={
                    'x': (-0.08, 0.08),
                    'y': (-0.05, 0.05),
                    'z': (-0.05, 0.05),
                },
            ),
        ],
        pd_gain_cfgs=[
            # sangchae
            RandomizeManagerCfg.PDGainCfg( # head pd gain
                ar_joint=SceneEntityCfg(name='robot', joint_names=_joint('head', 'neck')),
                kp_rng=(5. * 0.8, 5. * 1.2),
                kd_rng=(0.2 * 0.8, 0.2 * 1.2),
            ),
            RandomizeManagerCfg.PDGainCfg( # arms pd gain
                ar_joint=SceneEntityCfg(name='robot', joint_names=_joint('sho', 'arm', 'hand')),
                kp_rng=(15. * 0.8, 15. * 1.2),
                kd_rng=(0.5 * 0.8, 0.5 * 1.2),
            ),
            # SF_TRON1A
            RandomizeManagerCfg.PDGainCfg( # legs pd gain
                ar_joint=SceneEntityCfg(name='robot', joint_names=_joint('abad', 'hip', 'knee')),
                kp_rng=(45. * 0.8, 45. * 1.2),
                kd_rng=(1.5 * 0.8, 1.5 * 1.2),
            ),
            RandomizeManagerCfg.PDGainCfg( # ankles pd gain
                ar_joint=SceneEntityCfg(name='robot', joint_names=_joint('ankle')),
                kp_rng=(45. * 0.8, 45. * 1.2),
                kd_rng=(0.8 * 0.8, 0.8 * 1.2),
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
            'pen_lin': rwd_prims.VecNormPow(w=-2.0, p=2, val='linvel_23'),
            'pen_ang': rwd_prims.VecNormPow(w=-0.01,p=2, val='angvel_02'),
            'upright': rwd_prims.VecNorm(w=-0.5, p=2, val='gravdir_02'),
            # ----- dof penalty -----
            'qacc': rwd_prims.VecNormPow(w=-1.2e-7, p=2, val='qacc'),
            'd2_action': rwd_prims.VecNormPow(w=-0.002, p=2, val='d2_action'),
            'mec_energy': rwd_prims.VecNormPow(w=-2e-4, p=1, val='qpwr'),
            'the_energy': rwd_prims.VecNormPow(w=-2e-5, p=2, val='qtau'),
            'qpos_limit': rwd_prims.VecNormPow(w=-0.1,  p=1, val='qpos_violate'),
            'qtau_limit': rwd_prims.VecNormPow(w=-0.005,p=1, val='qtau_violate'),
            'qpos': rwd_prims.VecNormPow(w=-0.03, p=1, val='qpos_diff'),
            # ----- stand -----
            'stand_cont': rwd_prims.Sum(w=0.2, val='foot_cont', mask='is_stand'),
            'stand_clear': rwd_prims.FootClear(w=-0.5, p=1, stance_z=-0.72, clear_z=0.15, mask='is_stand'),
            'stand_qpos': rwd_prims.VecNormPow(w=-0.1, p=1, val='qpos_diff', mask='is_stand'),
            # ----- gait/foot -----
            'slip': rwd_prims.FootSlip(w=-0.25),
            'clear': rwd_prims.FootClear(w=-0.5, p=1, stance_z=-0.72, clear_z=0.15),
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
            body_names=_link('base', 'abad'),
        ),
        max_tilt_angle=180.0, # turn off tilt termination
        max_episode_length=_POST_INIT,
    )


    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ZETBOT1_SFOOT_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
        # note. sangchae part in use does not includes any collider
        self.scene.contact_sensor.prim_path = '{ENV_REGEX_NS}/Robot/SF_TRON1A/.*'

        n_qdim = 8 + 8

        self.action_space = n_qdim
        self.observation_space = (
            (6 + n_qdim * 3) * self.n_obs_history +
            9 +
            (10 + n_qdim * 3) * 0
        )

        # ----------

        self.rnd_cfg.cof_cfgs[0].n_bucket = self.n_env

        self.ter_cfg.max_episode_length = self.episode_length
