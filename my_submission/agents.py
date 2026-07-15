"""
agents.py  —  COMPLETE, RUNNABLE reference inference policy
===========================================================
This is a *working* example (not a skeleton).  It is the ONLY file the engine
requires from a submission.

The engine imports exactly one thing from here:

    make_policy(squad, seed) -> object with .decide(states, team) -> 11 actions

What this policy does
---------------------
Every tick it returns 11 engine action dicts:

  * For any slot that has a trained weight file (``agent_XX_dqn.pt`` produced by
    the matching ``train_rl_agent.py``), the action comes from the DQN:
    ``extract_features -> forward pass -> argmax -> map_action``.
  * For any slot without weights — or if PyTorch is unavailable — it falls back
    to the kit's built-in reference heuristic, so the team ALWAYS plays a sane,
    always-moving game even before you have trained anything.

It uses the **default** 20-float observation and 13-action space.  If you train
with your own ``feature_fn`` / ``action_fn`` / ``action_space`` (see
TRAINING_GUIDE.md §3), reproduce that same featurisation and action decoding
here so your trained weights behave identically at match time.
"""

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

# extract_features / map_action match the DEFAULT gym spaces; make_policy is the
# kit's reference heuristic, reused here as an always-sensible fallback.
try:
    from api.agents import (
        extract_features,
        map_action,
        make_policy as make_reference_policy,
        STATE_DIM,
        ACTION_DIM,
    )
except ImportError:
    from fifa_ai_world_cup.agents import (  # type: ignore
        extract_features,
        map_action,
        make_policy as make_reference_policy,
        STATE_DIM,
        ACTION_DIM,
    )

SQUAD_SIZE = 11
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "weights")


# ---------------------------------------------------------------------------
# Q-network — MUST match the one in train_rl_agent.py.
# ---------------------------------------------------------------------------
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
    """DQN where trained, reference heuristic everywhere else."""

    def __init__(self, squad: Optional[List[Any]] = None,
                 seed: Optional[int] = None):
        self.squad = list(squad) if squad else []
        # Always-available fallback: the kit's reference heuristic team policy.
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
            except Exception as exc:  # noqa: BLE001 — never crash a match on load
                print(f"[starter] failed to load {path}: {exc!r}")
        print(f"[starter] loaded {loaded}/{SQUAD_SIZE} trained slots "
              f"from {WEIGHTS_DIR!r}")

    def decide(self, states: List[dict], team: str) -> List[dict]:
        # Heuristic actions for everyone first (this is the fallback).
        actions = list(self._fallback.decide(states, team))
        if not _TORCH:
            return actions
        # Override any slot that has a trained network.
        for i, state in enumerate(states):
            net = self._nets[i] if i < len(self._nets) else None
            if net is None:
                continue
            try:
                feat = extract_features(state)
                with torch.no_grad():
                    q = net(torch.FloatTensor(feat).unsqueeze(0))
                    a_idx = int(torch.argmax(q).item())
                agent = self.squad[i] if i < len(self.squad) else None
                pos = (agent.x, agent.y) if agent is not None else None
                sho = agent.sho if agent is not None else 5.0
                actions[i] = map_action(a_idx, state, pos, sho)
            except Exception:  # noqa: BLE001 — fall back to the heuristic action
                pass
        return actions


def make_policy(squad: Optional[List[Any]] = None,
                seed: Optional[int] = None) -> StarterPolicy:
    """The engine's only entry point into this file."""
    return StarterPolicy(squad=squad, seed=seed)
