from isaaclab.utils import configclass

from bipedal_lab.tasks.default_cfg import DefaultEnvCfg
from bipedal_lab.tasks.tron1.articulation import TRON1_SFOOT_CFG


@configclass
class Tron1EnvCfg(DefaultEnvCfg):

    def __post_init__(self):
        self.scene.robot = TRON1_SFOOT_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')

        self.action_space = 8
        self.observation_space = (3 + 3 + 8 + 8 + 8 + 3) * 10

        self.obs_cfg.n_history = 10

        # self.rwd_cfg.robot_cfg.body_names = ['base_Link', 'abad_(L|R)_Link',]#'(abad|hip|knee)_(L|R)_Link']
        # self.rwd_cfg.sensor_cfg.body_names = ['ankle_(L|R)_Link']

        # self.ter_cfg.sensor_cfg.body_names = ['base_Link', 'abad_(L|R)_Link']


        # self.rwd_cfg.robot_cfg.body_names = '(base|(abad|hip|knee)_(L|R))_Link'
        # self.rwd_cfg.bodypen_cfg.body_names = '(base|(abad|hip|knee)_(L|R))_Link'
        # self.rwd_cfg.sensor_cfg.body_names = 'ankle_(L|R)_Link'
        # self.ter_cfg.sensor_cfg.body_names = 'base_Link'
