import gymnasium as gym
import shimmy  # noqa: F401  registers dm_control/* env ids


def make_flat(env_id):
    """gym.make plus dict-observation flattening (dm_control tasks via shimmy).

    Ant is built without its 78 contact-force channels so its observation is
    positions + velocities like the other MuJoCo tasks, keeping the channel
    structure comparable across environments.
    """
    kwargs = {"include_cfrc_ext_in_observation": False} if env_id.startswith("Ant") else {}
    env = gym.make(env_id, **kwargs)
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)
    return env
