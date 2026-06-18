import isaaclab.terrains as terrain


ROUGH_TERRAIN_CFG = terrain.TerrainGeneratorCfg(
    curriculum=True,
    size=(25.0, 25.0),
    border_width=40.0, # 2m/s * 20s = 40m
    border_height=1.0,
    num_rows=10,
    num_cols=7,
    color_scheme='none',
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75, # 36.87 degree
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        'stair_inv': terrain.MeshInvertedPyramidStairsTerrainCfg(
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=2.5,
        ),
        'stair': terrain.MeshPyramidStairsTerrainCfg(
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=2.5,
        ),
        'wave': terrain.HfWaveTerrainCfg(
            amplitude_range=(0.1, 0.4),
            num_waves=5, # wave period: 20 / 5 = 4m
            border_width=2.5,
        ),
        'grid': terrain.MeshRandomGridTerrainCfg(
            grid_width=0.45,
            grid_height_range=(0.05, 0.2),
            platform_width=2.0,
        ),
        'uniform': terrain.HfRandomUniformTerrainCfg(
            noise_range=(0.02, 0.10),
            noise_step=0.02,
            border_width=2.5,
        ),
        'slope_inv': terrain.HfInvertedPyramidSlopedTerrainCfg(
            slope_range=(0.0, 0.4), # max slope angle: 2.3deg
            platform_width=2.0,
            border_width=2.5,
        ),
        'slope': terrain.HfPyramidSlopedTerrainCfg(
            slope_range=(0.0, 0.4), # max slope angle: 2.3deg
            platform_width=2.0,
            border_width=2.5,
        ),
    },
)
