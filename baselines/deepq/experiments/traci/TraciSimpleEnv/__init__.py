from gym.envs.registration import register

register(
    id='TraciSimpleEnv-v0',
    entry_point='TraciSimpleEnv.TraciSimpleEnv:TraciSimpleEnv',
)