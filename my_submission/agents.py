from __future__ import annotations

import os
from typing import Any, List, Optional

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    torch = None
    nn = None
    _TORCH = False

try:
    from api.agents import (
        extract_features,
        map_action,
        make_policy as make_reference_policy,
        ACTION_DIM,
    )
except ImportError:
    from fifa_ai_world_cup.agents import (  # type: ignore
        extract_features,
        map_action,
        make_policy as make_reference_policy,
        ACTION_DIM,
    )

SQUAD_SIZE = 11
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "weights")
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

if _TORCH:
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

class StarterPolicy:
    def __init__(self, squad: Optional[List[Any]] = None,
                 seed: Optional[int] = None):
        self.squad = list(squad) if squad else []
        self._fallback = make_reference_policy(self.squad, seed)
        self._nets: List[Any] = [None] * SQUAD_SIZE
        if _TORCH:
            self._load_weights()

    def _load_weights(self) -> None:
        loaded = 0
        for slot in range(SQUAD_SIZE):
            path = os.path.join(WEIGHTS_DIR, f"agent_{slot:02d}_dqn.pt")
            if not os.path.exists(path):
                continue
            try:
                net = QNetwork()
                net.load_state_dict(torch.load(path, map_location="cpu"))
                net.eval()
                self._nets[slot] = net
                loaded += 1
            except Exception as exc:
                print(f"[starter] failed to load {path}: {exc!r}")
        print(f"[starter] loaded {loaded}/{SQUAD_SIZE} trained slots from {WEIGHTS_DIR!r}")

    def decide(self, states: List[dict], team: str) -> List[dict]:
        actions = list(self._fallback.decide(states, team))
        if not _TORCH:
            return actions
        for i, state in enumerate(states):
            net = self._nets[i] if i < len(self._nets) else None
            if net is None:
                continue
            try:
                feat = my_features(state)
                with torch.no_grad():
                    q = net(torch.FloatTensor(feat).unsqueeze(0))
                    a_idx = int(torch.argmax(q).item())
                agent = self.squad[i] if i < len(self.squad) else None
                pos = (agent.x, agent.y) if agent is not None else None
                sho = agent.sho if agent is not None else 5.0
                actions[i] = map_action(a_idx, state, pos, sho)
            except Exception:
                pass
        return actions

def make_policy(squad: Optional[List[Any]] = None,
                seed: Optional[int] = None) -> StarterPolicy:
    return StarterPolicy(squad=squad, seed=seed)
