from isaaclab.envs import DirectRLEnv
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.envs.mdp.events import (
    randomize_rigid_body_material,
    randomize_rigid_body_mass,
    randomize_rigid_body_com,
    randomize_actuator_gains,
)
from isaaclab.utils import configclass

from typing import Tuple, List, Dict, Literal
from dataclasses import MISSING



@configclass
class RandomizeManagerCfg:
    """Configuration class for `RandomizeManager`.
    """


    @configclass
    class CofCfg:
        """Configuration class for cof.
        """

        ar_body: SceneEntityCfg = MISSING
        """`SceneEntityCfg` of `Articulation`

        Requires:
            `body_names`: Bodies to configure.
        """

        static_cof_rng: Tuple[float, float] = MISSING
        """Static coefficient of friction range.
        """

        kinetic_cof_rng: Tuple[float, float] = MISSING
        """Kinetic coefficient of friction range.
        """

        cor_rng: Tuple[float, float] = MISSING
        """Coefficient of restitution range.
        """

        n_bucket: int = MISSING
        """Total number of physics meterials (defined as 3-tuple of static cof, kinetic cof, cor).
        """


    @configclass
    class MassCfg:
        """Configuration class for mass and com.
        """

        ar_body: SceneEntityCfg = MISSING
        """`SceneEntityCfg` of `Articulation`

        Requires:
            `body_names`: Bodies to configure.
        """

        add_rng: Tuple[float, float] = MISSING
        """Range of mass to add on mass.
        """

        add_com: Dict[Literal['x', 'y', 'z'], Tuple[float, float]] = MISSING
        """Range of three dimentional vector to add on com.
        """


    @configclass
    class PDGainCfg:
        """Configuration class for PD controller gains.
        """

        ar_joint: SceneEntityCfg = MISSING
        """`SceneEntityCfg` of `Articulation`

        Requires:
            `joint_names`: Joints to configure.
        """

        kp_rng: Tuple[float, float] = MISSING
        """Proportional gain range of PD controller.
        """

        kd_rng: Tuple[float, float] = MISSING
        """Derivative gain range of PD controller.
        """


    cof_cfgs: List[CofCfg] = MISSING
    """Configuration instances of `CofCfg`.
    """

    mass_cfgs: List[MassCfg] = MISSING
    """Configuration instances of `MassCfg`.
    """

    pd_gain_cfgs: List[PDGainCfg] = MISSING
    """Configuration instances of `PDGainCfg`.
    """



class RandomizeManager:
    """Manager class which applies the domain randomization.
    """


    def __init__(self, cfg: RandomizeManagerCfg, env: DirectRLEnv):
        """Initialize the manager.

        Args:
            cfg (RandomizeManagerCfg): Configuration instance for the manager.
            env (DirectRLEnv): Environment instance.
        """
        self.cfg = cfg
        self.env = env

        self._randomize_cof()
        self._randomize_mass()
        self._randomize_pd_gain()


    def _randomize_cof(self):
        for cof_cfg in self.cfg.cof_cfgs:

            cof_cfg.ar_body.resolve(self.env.scene)

            params = {
                'asset_cfg': cof_cfg.ar_body,
                'static_friction_range': cof_cfg.static_cof_rng,
                'dynamic_friction_range': cof_cfg.kinetic_cof_rng,
                'restitution_range': cof_cfg.cor_rng,
                'num_buckets': cof_cfg.n_bucket,
                'make_consistent': False,
            }
            randomize_rigid_body_material(
                cfg=EventTermCfg(params=params),
                env=self.env,
            ).__call__(self.env, None, **params)


    def _randomize_mass(self):
        for mass_cfg in self.cfg.mass_cfgs:

            mass_cfg.ar_body.resolve(self.env.scene)

            params = {
                'asset_cfg': mass_cfg.ar_body,
                'mass_distribution_params': mass_cfg.add_rng,
                'operation': 'add',
                'distribution': 'uniform',
                'recompute_inertia': True,
                'min_mass': 1e-6,
            }
            randomize_rigid_body_mass(
                cfg=EventTermCfg(params=params),
                env=self.env,
            ).__call__(self.env, None, **params)

            params = {
                'asset_cfg': mass_cfg.ar_body,
                'com_range': mass_cfg.add_com,
            }
            randomize_rigid_body_com(self.env, None, **params)


    def _randomize_pd_gain(self):
        for pd_gain_cfg in self.cfg.pd_gain_cfgs:

            pd_gain_cfg.ar_joint.resolve(self.env.scene)

            params = {
                'asset_cfg': pd_gain_cfg.ar_joint,
                'stiffness_distribution_params': pd_gain_cfg.kp_rng,
                'damping_distribution_params': pd_gain_cfg.kd_rng,
                'operation': 'abs',
                'distribution': 'uniform',
            }
            randomize_actuator_gains(
                cfg=EventTermCfg(params=params),
                env=self.env,
            ).__call__(self.env, None, **params)
