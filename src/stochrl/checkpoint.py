"""Capture and restore all environment-side randomness and simulator state,
so a run resumed from a checkpoint continues bit-identically. Covers Gymnasium
MuJoCo (qpos/qvel) and dm_control via shimmy (physics.get_state), plus the
env / action-space / observation-space RNGs and any noise-wrapper RNGs.
"""

from __future__ import annotations

import pickle


def _noise_wrappers(env):
    out = []
    w = env
    while w is not None:
        if hasattr(w, "_rng") and hasattr(w, "model"):
            out.append(w)
        w = getattr(w, "env", None)
    return out


def _sim_state(base):
    if hasattr(base, "data") and hasattr(base, "set_state"):
        return ("mujoco", (base.data.qpos.copy(), base.data.qvel.copy()))
    if hasattr(base, "physics") and hasattr(base.physics, "get_state"):
        return ("dmc", base.physics.get_state().copy())
    return ("none", None)


def snapshot_env(env) -> dict:
    base = env.unwrapped
    return {
        "sim": _sim_state(base),
        "env_rng": pickle.dumps(base.np_random),
        "act_rng": pickle.dumps(env.action_space.np_random),
        "obs_rng": pickle.dumps(env.observation_space.np_random),
        "noise": [(pickle.dumps(w._rng), pickle.dumps(w.model)) for w in _noise_wrappers(env)],
    }


def restore_env(env, snap: dict) -> None:
    base = env.unwrapped
    kind, data = snap["sim"]
    if kind == "mujoco":
        base.set_state(*data)
    elif kind == "dmc":
        base.physics.set_state(data)
        base.physics.forward()
    base.np_random = pickle.loads(snap["env_rng"])
    env.action_space._np_random = pickle.loads(snap["act_rng"])
    env.observation_space._np_random = pickle.loads(snap["obs_rng"])
    for w, (rng, model) in zip(_noise_wrappers(env), snap["noise"]):
        w._rng = pickle.loads(rng)
        w.model = pickle.loads(model)
