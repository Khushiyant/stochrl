import gymnasium as gym
import shimmy  # noqa: F401  registers dm_control/* env ids


def make_flat(env_id):
    """gym.make plus dict-observation flattening (dm_control tasks via shimmy)."""
    env = gym.make(env_id)
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)
    return env
