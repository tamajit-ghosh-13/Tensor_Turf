from __future__ import annotations

import math
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
    import api.gym_env  # noqa: F401
    from api.agents import extract_features, ACTION_DIM
except ImportError:
    import fifa_ai_world_cup.gym_env  # type: ignore # noqa: F401
    from fifa_ai_world_cup.agents import extract_features, ACTION_DIM  # type: ignore

SQUAD_SIZE = 11
STATE_DIM = 36  # 20 default + 10 teammate features + 6 opponent features

def my_features(state: dict) -> list[float]:
    feats = extract_features(state)
    
    # 1. Teammate Radar (Top 5 closest)
    teammates = [obj for obj in state.get('visible_objects', []) if obj.get('type') == 'teammate']
    teammates.sort(key=lambda x: x.get('rel_distance', 999.0))
    for i in range(5):
        if i < len(teammates):
            tm = teammates[i]
            feats.append(tm.get('rel_distance', 0.0) / 40.0)
            feats.append(tm.get('rel_angle', 0.0) / 180.0)
        else:
            feats.extend([0.0, 0.0])
            
    # 2. Opponent Radar (Top 3 closest) — Brand New!
    opponents = [obj for obj in state.get('visible_objects', []) if obj.get('type') == 'opponent']
    opponents.sort(key=lambda x: x.get('rel_distance', 999.0))
    for i in range(3):
        if i < len(opponents):
            op = opponents[i]
            feats.append(op.get('rel_distance', 0.0) / 40.0)
            feats.append(op.get('rel_angle', 0.0) / 180.0)
        else:
            feats.extend([0.0, 0.0])
            
    return feats

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

def my_reward(prev_obs, action, obs, info, team, slot) -> float:
    # 1. The Terminal Anchor (Match Score)
    other = "B" if team == "A" else "A"
    prev = info.get("_prev_score", {"A": 0, "B": 0})
    now = info["score"]
    reward = 40.0 * (now[team] - prev[team]) - 40.0 * (now[other] - prev[other])

    # 2. Extract Current Agent's Data
    my_agent = next((a for a in info.get("agents", []) if a.get("slot") == slot), {})
    has_ball = my_agent.get("has_ball", False)
    
    # 3. Goalkeeper Logic (Slot 0)
    if slot == 0:
        # Penalise the keeper if they venture too far away from the ball's trajectory,
        # or if they are aggressively pressing when the ball is far away.
        ball_dist = my_agent.get("ball_distance")
        if ball_dist is not None and ball_dist > 15.0:
            # If the ball is far down the pitch, the keeper must stay put.
            # Give a small deduction if they select a MOVE action (0-7) instead of IDLE/DEFLECT
            acts = list(action) if hasattr(action, "__len__") else [int(action)]
            if any(0 <= int(a) <= 7 for a in acts):
                reward -= 0.1
        return float(reward)

    # 4. Outfield Players: Possession & Forward Progress
    ball_dist = my_agent.get("ball_distance")
    
    if has_ball:
        # Very small possession trickle to avoid reward hacking (was 0.25, now 0.05)
        reward += 0.05 
        
        # Solid reward for driving the ball up the pitch
        attack_dx = 1.0 if team == "A" else -1.0
        ori = my_agent.get("orientation", 0.0)
        reward += max(0.0, math.cos(math.radians(ori)) * attack_dx) * 0.4
        
    elif ball_dist is not None:
        # Swarm Prevention: Only reward closing down the ball if you are the closest
        all_dists = [a.get("ball_distance", 999.0) for a in info.get("agents", []) if a.get("ball_distance") is not None]
        if all_dists and ball_dist <= min(all_dists) + 0.5:
            prev_d = info.get("_prev_ball_dist")
            if prev_d is not None and ball_dist < prev_d:
                reward += (prev_d - ball_dist) * 0.2

    # 5. The Strike & Pass Incentives (Action Priors)
    acts = list(action) if hasattr(action, "__len__") else [int(action)]
    for a in acts:
        try:
            a_idx = int(a)
        except (TypeError, ValueError):
            continue
        
        if has_ball:
            if a_idx == 9:      # SHOOT
                # Find the opponent's goal in the visible objects
                visible_goals = [obj for obj in obs.get('visible_objects', []) if 'goal' in str(obj.get('type', ''))]
                
                if visible_goals:
                    closest_goal_dist = min(g.get('rel_distance', 999.0) for g in visible_goals)
                    if closest_goal_dist < 25.0:  # Only reward shooting when in a sensible attacking range!
                        reward += 2.0
                    else:
                        reward += 0.2  # Minor encouragement, but don't waste it from deep
                else:
                    reward -= 0.2  # Penalise shooting blindly when the goal isn't even in sight!
            
            elif a_idx == 10:   # PASS
                reward += 0.5
        elif a_idx == 8:        # ROTATE
            reward -= 0.02      # Tiny penalty for just spinning on the spot

    # 6. Team Penalty for Losing the Ball
    any_ball = any(a.get("has_ball") for a in info.get("agents", []))
    if info.get("_prev_has_ball") and not any_ball:
        reward -= 0.8

    return float(reward)

def train():
    class Args: pass
    args = Args()
    args.episodes = int(os.environ.get("EPISODES", "20"))
    args.ticks = int(os.environ.get("TICKS", "400"))
    args.seed = int(os.environ.get("SEED", "42"))
    args.slots = None
    args_list = sys.argv[1:]
    if "--episodes" in args_list: args.episodes = int(args_list[args_list.index("--episodes")+1])
    if "--ticks" in args_list: args.ticks = int(args_list[args_list.index("--ticks")+1])
    if "--seed" in args_list: args.seed = int(args_list[args_list.index("--seed")+1])
    if "--slots" in args_list: args.slots = args_list[args_list.index("--slots")+1]

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
                       feature_fn=my_features,
                       opponent_style="aggressive", max_ticks=args.ticks,
                       reward_fn=lambda p, a, o, i, t, s=slot: my_reward(p, a, o, i, t, s),
                       seed=args.seed + slot)
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
