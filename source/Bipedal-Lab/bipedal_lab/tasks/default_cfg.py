from isaaclab.terrains import TerrainImporterCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
import isaaclab.sim as sim_utils

from typing import Literal
from dataclasses import MISSING

import torch as th

from bipedal_lab.base.env_cfg import BipedalEnvCfg
from bipedal_lab.tasks.terrain_cfg import ROUGH_TERRAIN_CFG



_POST_INIT = None



@configclass
class SceneCfg(InteractiveSceneCfg):

    num_envs = _POST_INIT

    env_spacing = _POST_INIT

    terrain = TerrainImporterCfg(
        prim_path='/World/ground',
        terrain_type=_POST_INIT,
        terrain_generator=_POST_INIT,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode='multiply',
            restitution_combine_mode='multiply',
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
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
        update_period=_POST_INIT,
        history_length=_POST_INIT,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path='/World/SkyLight',
        spawn=sim_utils.DomeLightCfg(
            intensity=800.0,
            texture_file=f'{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr',
        ),
    )



def _get_check(t):
    def check(rng: th.Tensor):
        # input shape: (n_cell, n_cmd, 2)
        # output shape: (n_cell,)
        cent = rng.mean(dim=-1) # (n_cell, n_cmd)
        cent_x = cent[:,0] # (n_cell,)
        ans = th.zeros_like(cent_x, dtype=th.bool)
        if 's' in t:
            ans.logical_or_((0.0 <= cent_x.abs()) & (cent_x.abs() <= 1.0))
        if 'm' in t:
            ans.logical_or_((1.0 <= cent_x.abs()) & (cent_x.abs() <= 1.5))
        if 'f' in t:
            ans.logical_or_((1.5 <= cent_x.abs()) & (cent_x.abs() <= 2.0))
        return ans
    return check



@configclass
class DefaultEnvCfg(BipedalEnvCfg):

    sim_dt: float = 1 / 200

    decimation: int = 4

    episode_length: int = 1000

    n_env: int = MISSING

    device: str = MISSING

    terrain_type: Literal['flat', 'rough'] = MISSING

    # ----------

    sim = sim_utils.SimulationCfg(
        device=_POST_INIT,
        dt=_POST_INIT,
        render_interval=_POST_INIT,
        physics_material=_POST_INIT,
    )

    episode_length_s = _POST_INIT

    scene = SceneCfg()

    action_space = MISSING

    observation_space = MISSING

    # ----------

    ar_robot = SceneEntityCfg(name='robot')

    sub_terrains = {
        'stair_inv':BipedalEnvCfg.SubTerrainCfg(prop=0.4, check=_get_check('s')),
        'stair':    BipedalEnvCfg.SubTerrainCfg(prop=0.1, check=_get_check('s')),
        'wave':     BipedalEnvCfg.SubTerrainCfg(prop=0.1, check=_get_check('s')),
        'grid':     BipedalEnvCfg.SubTerrainCfg(prop=0.1, check=_get_check('s')),
        'uniform':  BipedalEnvCfg.SubTerrainCfg(prop=0.1, check=_get_check('s')),
        'slope_inv':BipedalEnvCfg.SubTerrainCfg(prop=0.1, check=_get_check('s')),
        'slope':    BipedalEnvCfg.SubTerrainCfg(prop=0.1, check=_get_check('s')),
    }

    vel_err_sma_window = 25 # 0.5s

    foll_boundary = 0.4

    foll_hyst = (0.5, 0.7) # Note. 10m / (1m/s * 12s) = 0.833

    max_stride = 1.2

    n_obs_history = 5

    # ----------


    def __post_init__(self):
        super().__post_init__()

        self.sim.device = self.device
        self.sim.dt = self.sim_dt
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**16
        self.sim.physx.gpu_collision_stack_size = 2**27

        self.episode_length_s = self.episode_length * self.decimation * self.sim_dt

        self.scene: SceneCfg
        self.scene.num_envs = self.n_env
        self.scene.env_spacing = 4.0
        self.scene.contact_sensor.update_period = self.sim_dt
        self.scene.contact_sensor.history_length = self.decimation

        is_rough = self.terrain_type == 'rough'
        self.scene.terrain.terrain_type = 'generator' if is_rough else 'plane'
        self.scene.terrain.terrain_generator = ROUGH_TERRAIN_CFG if is_rough else None
