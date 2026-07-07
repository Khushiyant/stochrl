"""Soft Actor-Critic — CleanRL implementation, with pluggable calibrated noise.

The SAC algorithm here is CleanRL's `sac_continuous_action.py` (Huang et al., 2022,
JMLR), used as the trusted reference. The `Actor`, `SoftQNetwork`, the get_action
log-prob, the full critic/actor/temperature update block, and all hyperparameters
are COPIED VERBATIM from CleanRL (https://github.com/vwxyzjn/cleanrl). The replay
buffer is the same stable-baselines3 `ReplayBuffer` CleanRL imports.

Two deliberate deviations, both forced and documented:
  1. A single-env loop instead of `gym.vector.SyncVectorEnv([...])` with num_envs=1.
     CleanRL's vector-autoreset code (`infos["final_info"]/["final_observation"]`)
     targets gymnasium 0.29; we run gymnasium 1.3, whose vector autoreset API differs.
     A single env is exactly equivalent to num_envs=1 and side-steps that mismatch.
  2. Noise wrappers + a clean-eval + CSV logging are added around the algorithm.

Everything between the "BEGIN/END CleanRL verbatim" markers is unchanged CleanRL.

  uv run python scripts/sac_continuous_action.py --env-id HalfCheetah-v5 --noise-mode realistic --rho 0.1
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from stochrl import ActionNoise, ObservationNoise, collect_signal_stats, presets


@dataclass
class Args:
    env_id: str = "HalfCheetah-v5"
    total_timesteps: int = 1_000_000
    seed: int = 1
    device: str = "cpu"
    torch_deterministic: bool = True

    # --- noise (research knobs) --- #
    noise_mode: str = "none"  # none | uniform | uniform-calibrated | realistic | vel-flat | vel-statedep
    noise_target: str = "obs"  # obs | action | both
    rho: float = 0.1
    calib_steps: int = 10_000
    calib_seed: int = 0

    # --- evaluation & logging --- #
    eval_interval: int = 0
    eval_episodes: int = 5
    csv_path: str = ""
    torch_threads: int = 0

    # --- SAC hyperparameters (CleanRL defaults, verbatim) --- #
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5_000
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True


# ----------------------- noise / eval (project additions) ------------------- #
def build_noise(env, args, seed):
    """Wrap `env` with the requested noise model(s). Calibration uses a fixed seed
    (constant treatment across runs); obs/action noise get independent RNG streams."""
    if args.noise_mode == "none":
        return env
    obs_ss, act_ss = np.random.SeedSequence(seed).spawn(2)
    if args.noise_target in ("obs", "both"):
        stats = collect_signal_stats(gym.make(args.env_id), steps=args.calib_steps, seed=args.calib_seed)
        if args.noise_mode == "realistic":
            model = presets.realistic_sensors(stats, rho=args.rho)
        elif args.noise_mode in ("vel-flat", "vel-statedep", "pos-flat", "pos-statedep"):
            model = presets.channel_isolation(
                stats, args.rho, args.env_id, args.noise_mode, args.calib_steps, args.calib_seed)
        elif args.noise_mode.startswith("both-"):
            kinds = {"f": "flat", "s": "statedep"}
            code = args.noise_mode.split("-")[1]  # e.g. 'sf' -> vel=statedep, pos=flat
            model = presets.combined_isolation(
                stats, args.rho, args.env_id, kinds[code[0]], kinds[code[1]],
                args.calib_steps, args.calib_seed)
        else:
            model = presets.uniform_gaussian(
                stats, rho=args.rho, calibrated=(args.noise_mode == "uniform-calibrated"))
        env = ObservationNoise(env, model, seed=int(obs_ss.generate_state(1)[0]))
    if args.noise_target in ("action", "both"):
        model = presets.actuator_noise(env.action_space.shape[0], rho=args.rho)
        env = ActionNoise(env, model, seed=int(act_ss.generate_state(1)[0]))
    return env


def make_env(args, seed):
    env = gym.make(args.env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = build_noise(env, args, seed)
    return env


def evaluate(actor, env_id, episodes, device, seed):
    """Deterministic (mean-action) eval on a CLEAN env — comparable across modes."""
    eval_env = gym.make(env_id)
    returns = []
    for e in range(episodes):
        obs, _ = eval_env.reset(seed=seed + 10_000 + e)
        done, total = False, 0.0
        while not done:
            with torch.no_grad():
                _, _, mean = actor.get_action(torch.Tensor(obs).unsqueeze(0).to(device))
            obs, r, term, trunc, _ = eval_env.step(mean[0].cpu().numpy())
            total += float(r)
            done = term or trunc
        returns.append(total)
    eval_env.close()
    return float(np.mean(returns))


# ===================== BEGIN CleanRL verbatim (networks) ==================== #
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
# ====================== END CleanRL verbatim (networks) ===================== #


def main(args: Args):
    run_name = f"{args.env_id}__{args.noise_mode}_{args.noise_target}_rho{args.rho}__{args.seed}"
    writer = SummaryWriter(f"runs/{run_name}")
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device(args.device)

    env = make_env(args, args.seed)
    env.action_space.seed(args.seed)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    csv_file = None
    if args.csv_path:
        os.makedirs(os.path.dirname(args.csv_path) or ".", exist_ok=True)
        csv_file = open(args.csv_path, "w")
        csv_file.write("step,eval_return\n")

    actor = Actor(env).to(device)
    qf1 = SoftQNetwork(env).to(device)
    qf2 = SoftQNetwork(env).to(device)
    qf1_target = SoftQNetwork(env).to(device)
    qf2_target = SoftQNetwork(env).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    env.observation_space.dtype = np.float32
    rb = ReplayBuffer(args.buffer_size, env.observation_space, env.action_space, device,
                      n_envs=1, handle_timeout_termination=False)
    start_time = time.time()

    obs, _ = env.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        if args.eval_interval and global_step % args.eval_interval == 0:
            er = evaluate(actor, args.env_id, args.eval_episodes, device, args.seed)
            writer.add_scalar("charts/eval_return", er, global_step)
            if csv_file:
                csv_file.write(f"{global_step},{er}\n")
                csv_file.flush()

        # ALGO LOGIC: action selection
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            a, _, _ = actor.get_action(torch.Tensor(obs).unsqueeze(0).to(device))
            action = a[0].detach().cpu().numpy()

        next_obs, reward, termination, truncation, info = env.step(action)
        if "episode" in info:
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # single-env equivalent of CleanRL's buffer add (bootstrap on truncation -> done=termination)
        rb.add(obs, next_obs, action, reward, termination, [info])
        obs = next_obs
        if termination or truncation:
            obs, _ = env.reset()

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            # ===================== BEGIN CleanRL verbatim (update) ============ #
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(args.policy_frequency):
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            # ====================== END CleanRL verbatim (update) ============= #

            if global_step % 1000 == 0:
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    env.close()
    if args.eval_interval:
        er = evaluate(actor, args.env_id, args.eval_episodes, device, args.seed)
        writer.add_scalar("charts/eval_return", er, args.total_timesteps)
        print(f"final clean-eval return = {er:.1f}")
        if csv_file:
            csv_file.write(f"{args.total_timesteps},{er}\n")
    if csv_file:
        csv_file.close()
    writer.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
