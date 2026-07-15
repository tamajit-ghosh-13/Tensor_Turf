"""
train_rl_agent.py  —  COMPLETE, RUNNABLE reference trainer
==========================================================
This is a *working* example (not a skeleton).  Copy this whole ``starter/``
folder, tweak it, and submit ``agents.py`` + ``train_rl_agent.py``.

What it does
------------
Trains one small DQN per squad slot (0..10, GK included) on the Gymnasium
environment and saves 11 weight files ``agent_00_dqn.pt`` … ``agent_10_dqn.pt``
into the directory the organiser points ``SAVE_MODEL`` at.  The matching
``agents.py`` in this folder loads those files at match time.

It uses the **default** observation (20 floats) and action space (13 discrete),
so the network is ``STATE_DIM -> 64 -> 64 -> ACTION_DIM``.  None of that is
required by the platform — see TRAINING_GUIDE.md §3 and §8 to bring your own
observation / action space / algorithm.  If you change anything here, change
``agents.py`` to match.

Run locally
-----------
    pip install torch gymnasium numpy
    python train_rl_agent.py --episodes 20 --ticks 400 --seed 42
    # or a single slot while iterating:
    python train_rl_agent.py --slots 5 --episodes 30

The organiser runs it headless with env vars EPISODES / TICKS / SEED / SAVE_MODEL.
"""

from __future__ import annotations

import argparse
import collections
import os
import random
import sys

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("ERROR: this trainer needs torch — `pip install torch`")
    sys.exit(1)

import gymnasium as gym
try:
    import api.gym_env  # noqa: F401  — registers "FIFAWorldCupAgent-v0"
    from api.agents import STATE_DIM, ACTION_DIM  # defaults: 20 and 13
except ImportError:
    import fifa_ai_world_cup.gym_env  # type: ignore # noqa: F401
    from fifa_ai_world_cup.agents import STATE_DIM, ACTION_DIM  # type: ignore

SQUAD_SIZE = 11


# ---------------------------------------------------------------------------
# 1. The Q-network.  Any architecture is fine — this MUST match agents.py.
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 2. A minimal DQN agent (epsilon-greedy + replay buffer + target network).
# ---------------------------------------------------------------------------
class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, seed: int = 0):
        self.action_dim = action_dim
        self.gamma = 0.99
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.995
        self.batch_size = 64
        self.sync_every = 500
        self._steps = 0
        self.rng = random.Random(seed)

        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.buffer: collections.deque = collections.deque(maxlen=20000)

    def select_action(self, state) -> int:
        if self.rng.random() < self.eps:
            return self.rng.randrange(self.action_dim)
        with torch.no_grad():
            q = self.policy_net(torch.FloatTensor(np.asarray(state)).unsqueeze(0))
            return int(torch.argmax(q).item())

    def store(self, s, a, r, s2, done):
        self.buffer.append((np.asarray(s, dtype=np.float32), a, r,
                            np.asarray(s2, dtype=np.float32), done))

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = self.rng.sample(self.buffer, self.batch_size)
        s, a, r, s2, done = zip(*batch)
        s = torch.FloatTensor(np.stack(s))
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r).unsqueeze(1)
        s2 = torch.FloatTensor(np.stack(s2))
        done = torch.FloatTensor(done).unsqueeze(1)

        q = self.policy_net(s).gather(1, a)
        with torch.no_grad():
            q_next = self.target_net(s2).max(1, keepdim=True)[0]
            target = r + self.gamma * q_next * (1.0 - done)
        loss = nn.functional.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._steps += 1
        if self._steps % self.sync_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


# ---------------------------------------------------------------------------
# 3. Reward shaping.  Dense signal = goals + possession + forward progress.
#    (Rewrite freely — see examples/reward_examples.py for more ideas.)
# ---------------------------------------------------------------------------
def my_reward(prev_obs, action, obs, info, team) -> float:
    other = "B" if team == "A" else "A"
    prev = info.get("_prev_score", {"A": 0, "B": 0})
    now = info["score"]
    reward = 40.0 * (now[team] - prev[team]) - 40.0 * (now[other] - prev[other])

    any_ball = False
    for ag in info.get("agents", []):
        if ag.get("has_ball"):
            any_ball = True
            reward += 0.15
    # Move toward a loose ball when nobody on our side holds it.
    dists = [ag["ball_distance"] for ag in info.get("agents", [])
             if ag.get("ball_distance") is not None]
    if not any_ball and dists:
        prev_d = info.get("_prev_ball_dist")
        if prev_d is not None:
            reward += (prev_d - min(dists)) * 0.18
    if info.get("_prev_has_ball") and not any_ball:
        reward -= 0.8
    # Small nudge to SHOOT (9) / PASS (10), tiny penalty for spinning (8).
    acts = list(action) if hasattr(action, "__len__") else [int(action)]
    for a in acts:
        try:
            a = int(a)
        except (TypeError, ValueError):
            continue
        reward += 0.10 if a == 9 else 0.08 if a == 10 else -0.02 if a == 8 else 0.0
    return float(reward)


# ---------------------------------------------------------------------------
# 4. Train each requested slot and save its weights.
# ---------------------------------------------------------------------------
def train():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=int(os.environ.get("EPISODES", "20")))
    p.add_argument("--ticks", type=int, default=int(os.environ.get("TICKS", "400")))
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    p.add_argument("--slots", default=None,
                   help="comma-separated slots to train (default: all 11)")
    args = p.parse_args()

    slots = ([int(s) for s in args.slots.split(",")]
             if args.slots else list(range(SQUAD_SIZE)))
    save_dir = os.path.dirname(os.path.abspath(
        os.environ.get("SAVE_MODEL", "agent_dqn.pt")))
    os.makedirs(save_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Training slots {slots} — {args.episodes} episodes x {args.ticks} ticks each")

    for slot in slots:
        env = gym.make("FIFAWorldCupAgent-v0", slots=[slot],
                       opponent_style="aggressive", max_ticks=args.ticks,
                       reward_fn=my_reward, seed=args.seed + slot)
        agent = DQNAgent(STATE_DIM, ACTION_DIM, seed=args.seed + slot)
        for ep in range(1, args.episodes + 1):
            obs, _ = env.reset(seed=args.seed + slot * 1000 + ep)
            total, done, trunc = 0.0, False, False
            while not (done or trunc):
                a = agent.select_action(obs)
                obs2, r, done, trunc, _ = env.step(a)
                agent.store(obs, a, r, obs2, float(done or trunc))
                agent.train_step()
                obs, total = obs2, total + r
            print(f"  slot {slot:02d}  ep {ep:03d}/{args.episodes}  "
                  f"reward {total:+.2f}  eps {agent.eps:.2f}")
            sys.stdout.flush()
        out = os.path.join(save_dir, f"agent_{slot:02d}_dqn.pt")
        torch.save(agent.policy_net.state_dict(), out)
        print(f"  saved {out}")
        env.close()

    print(f"Done — {len(slots)} weight file(s) in {save_dir}")


if __name__ == "__main__":
    train()
