"""
train_template.py  (TEMPLATE — FILL IN THE TODOs)
==================================================
Skeleton training script.  Copy this file, fill in the TODOs, and submit it
as your ``train_rl_agent.py``.

The organiser runs this with env vars: EPISODES, TICKS, SEED (also passed as
``--episodes/--ticks/--seed``) and SAVE_MODEL — the DIRECTORY your weight files
go in. These values are the ones YOU choose in the web app's squad builder when
you submit: pick your own ticks/episodes/seed. Ticks and episodes are capped by
an organiser-set maximum (the squad builder shows the cap; a request over it is
rejected); the seed is entirely yours (it doesn't affect strength). Locally,
pass the same values with ``--episodes/--ticks/--seed`` (or the env vars).

Choosing FEWER / shorter episodes is fine if that's enough. The only HARD limit
is the training-time cap (shown in the web app / admin panel) — if your run
exceeds it, you are disqualified. So read the values you're given (they are your
choices) and stay under the clock.

NOTHING here is fixed by the platform — this DQN is just a working starting
point.  You are free to:
  * use ANY network architecture (depth/width/activations are your choice),
  * use ANY algorithm (DQN, PPO, actor-critic, evolutionary search, …) or any
    framework in the allowed list (torch, jax/flax, tensorflow, sklearn, …),
  * design your OWN observation and action space by passing ``feature_fn`` /
    ``action_fn`` / ``action_space`` to ``gym.make`` (see the commented example
    below and the Training Guide).

The ONE thing that must stay consistent is that whatever you train here, your
``agents.py`` must load and use to return valid engine action dicts at match
time.  The engine never inspects your weights or network shape — it only calls
``make_policy(...).decide(states, team)``.
"""

from __future__ import annotations
import argparse, collections, os, random, sys
import numpy as np

try:
    import torch, torch.nn as nn, torch.optim as optim
except ImportError:
    print("ERROR: pip install torch"); sys.exit(1)

import gymnasium as gym
try:
    import api.gym_env  # noqa: F401  (registers FIFAWorldCupAgent-v0)
    from api.agents import STATE_DIM, ACTION_DIM  # defaults: 20 and 13
except ImportError:
    import fifa_ai_world_cup.gym_env  # type: ignore # noqa: F401
    from fifa_ai_world_cup.agents import STATE_DIM, ACTION_DIM  # type: ignore

SQUAD_SIZE = 11


# TODO: Define your Q-network.  Any architecture is allowed — the default
#       reference shape is 20 -> 64 -> 64 -> 13 (ReLU), but you may change it
#       freely as long as your agents.py builds the SAME shape to load weights.
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: define layers
        raise NotImplementedError

    def forward(self, x):
        # TODO: forward pass
        raise NotImplementedError


# TODO: Define your DQN agent
class DQNAgent:
    def __init__(self):
        # TODO: policy_net, target_net, optimizer, replay buffer, hyperparams
        raise NotImplementedError

    def select_action(self, state):
        # TODO: epsilon-greedy
        raise NotImplementedError

    def store(self, s, a, r, s2, done):
        # TODO: store transition
        raise NotImplementedError

    def train_step(self):
        # TODO: mini-batch update
        raise NotImplementedError


# TODO: Define your reward function (see examples/reward_examples.py)
def my_reward(prev_obs, action, obs, info, team):
    # TODO: implement your reward shaping
    raise NotImplementedError


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=int(os.environ.get("EPISODES", "30")))
    parser.add_argument("--ticks", type=int, default=int(os.environ.get("TICKS", "400")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument("--slots", default=None)
    args = parser.parse_args()

    slots = [int(s) for s in args.slots.split(",")] if args.slots else list(range(SQUAD_SIZE))
    # The organiser sets SAVE_MODEL to the DIRECTORY your weight files go in.
    # Accept either a directory (the platform's case) or a file path (if you run
    # locally with SAVE_MODEL=weights/agent.pt) and resolve it to a folder.
    save_dir = os.environ.get("SAVE_MODEL", "weights")
    if os.path.splitext(save_dir)[1]:            # a file path -> use its folder
        save_dir = os.path.dirname(os.path.abspath(save_dir)) or "."
    os.makedirs(save_dir, exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    print(f"Training {len(slots)} slots, {args.episodes} episodes each")

    for slot in slots:
        # The default env uses the built-in 20-float obs and 13-action space.
        # To design your OWN, pass any of these (obs space is sized for you):
        #
        #   from gymnasium import spaces
        #   def my_features(state):            # raw state dict -> vector
        #       return [ ... ]                 # any length
        #   def my_action(a, state, pos, sho): # your action -> engine dict
        #       return { "action_type": "MOVE", ... }
        #   env = gym.make("FIFAWorldCupAgent-v0", slots=[slot],
        #                  feature_fn=my_features, action_fn=my_action,
        #                  action_space=spaces.Box(-1, 1, shape=(4,)),  # or Discrete(k)
        #                  reward_fn=my_reward, max_ticks=args.ticks)
        env = gym.make("FIFAWorldCupAgent-v0", slots=[slot],
                       opponent_style="aggressive", max_ticks=args.ticks,
                       reward_fn=my_reward, seed=args.seed + slot)
        agent = DQNAgent()
        for ep in range(1, args.episodes + 1):
            obs, info = env.reset(seed=args.seed + slot * 1000 + ep)
            total = 0.0; done = trunc = False
            while not (done or trunc):
                a = agent.select_action(obs)
                obs2, r, done, trunc, info = env.step(a)
                agent.store(obs, a, r, obs2, float(done or trunc))
                agent.train_step()
                obs = obs2; total += r
            print(f"  slot{slot:02d} ep {ep:03d}/{args.episodes}  reward {total:+.2f}")
            sys.stdout.flush()
        torch.save(agent.policy_net.state_dict(), os.path.join(save_dir, f"agent_{slot:02d}_dqn.pt"))
        print(f"  saved agent_{slot:02d}_dqn.pt")
        env.close()
    print(f"Done! {len(slots)} weight files in {save_dir}")

if __name__ == "__main__":
    train()
