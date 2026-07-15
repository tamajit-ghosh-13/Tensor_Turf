"""
agents_template.py  (TEMPLATE — FILL IN THE TODOs)
====================================================
Skeleton inference script.  Copy this file, fill in the TODOs, and submit it
as your ``agents.py``.

This is the ONLY file the engine requires.  It must expose
``make_policy(squad, seed)`` returning an object whose
``decide(states, team)`` returns 11 engine action dicts every tick.

Everything else is your choice:
  * any network / model / framework (or none — a pure heuristic is fine),
  * any observation featurisation and any action decoding.
``extract_features`` and ``map_action`` are provided as CONVENIENCES so a
network trained with the default gym spaces transfers directly — but if you
trained with your own ``feature_fn`` / ``action_fn`` in ``train_rl_agent.py``,
reproduce that same featurisation / decoding here so weights stay compatible.

⚠ Two pitfalls this template prevents ⚠
---------------------------------------
1. GOALKEEPER:  DEFLECT (12) only sets ``deflect_angle`` — it does NOT move
   the keeper.  DIVE (11) triggers a cooldown that freezes the keeper.  A
   keeper that only DEFLECTs or DIVEs is static and concedes every shot.
   The keeper MUST MOVE (0 or 1) along the goal line to track the ball.
   See ``api/agents.py`` ``Policy._goalkeeper()`` for the full reference.

2. OFF-BALL:  Players that IDLE or ROTATE in place are useless.  An untrained
   slot should chase the ball if it is visible, or push forward otherwise.
   The fallback below does this so your team always plays, even before
   training.
"""

from __future__ import annotations
import os
from typing import Any, List, Optional

try:
    import torch
except ImportError:
    torch = None

# Convenience helpers for the DEFAULT obs/action space (you may ignore them and
# build your own observation/action decoding to match your trainer).
try:
    from api.agents import extract_features, map_action, STATE_DIM, ACTION_DIM
except ImportError:
    from fifa_ai_world_cup.agents import extract_features, map_action, STATE_DIM, ACTION_DIM  # type: ignore

SQUAD_SIZE = 11


def _compass(angle_deg: float) -> int:
    """Nearest of the 8 MOVE directions (0=E, 1=NE, 2=N, …) for an angle."""
    return int(round((angle_deg % 360.0) / 45.0)) % 8


# TODO: Define your Q-network.  It must match whatever you trained — the default
#       reference shape is 20 -> 64 -> 64 -> 13 (ReLU), but any shape works as
#       long as train_rl_agent.py saved weights for this exact architecture.
class QNetwork(torch.nn.Module if torch else object):
    def __init__(self):
        super().__init__()
        # TODO: define layers
        raise NotImplementedError

    def forward(self, x):
        # TODO: forward pass
        raise NotImplementedError


class MyPolicy:
    """Your inference policy.  Pure forward-pass — no env, no resets."""

    def __init__(self, squad: Optional[List[Any]] = None, seed: Optional[int] = None):
        self.squad = list(squad) if squad else []
        self._nets = [None] * SQUAD_SIZE
        self._load_weights()

    def _load_weights(self):
        """Load your 11 weight files from WEIGHTS_DIR."""
        if torch is None:
            print("[my_policy] torch not available — heuristic fallback")
            return
        wdir = os.environ.get("WEIGHTS_DIR", "weights")
        loaded = 0
        for slot in range(SQUAD_SIZE):
            path = os.path.join(wdir, f"agent_{slot:02d}_dqn.pt")
            if os.path.exists(path):
                # TODO: load weights into a QNetwork
                # net = QNetwork()
                # net.load_state_dict(torch.load(path, map_location="cpu"))
                # net.eval()
                # self._nets[slot] = net
                # loaded += 1
                raise NotImplementedError("Implement weight loading")
        print(f"[my_policy] loaded {loaded}/{SQUAD_SIZE} weights from {wdir}")

    def decide(self, states: List[dict], team: str) -> List[dict]:
        """Called every tick.  Return 11 action dicts."""
        actions: List[dict] = []
        for i, state in enumerate(states):
            # ---- Goalkeeper: MUST MOVE to track the ball. ----
            # DO NOT just return DEFLECT (12) or DIVE (11) every tick:
            #   * DEFLECT only sets deflect_angle — it does NOT move the GK.
            #   * DIVE triggers a cooldown that freezes the GK for several ticks.
            #   * A static GK concedes every shot.
            # See api/agents.py Policy._goalkeeper() for the full reference.
            if state.get("is_gk"):
                actions.append(self._goalkeeper(state, team))
                continue

            net = self._nets[i] if i < len(self._nets) else None
            if net is not None and torch is not None:
                # TODO: forward pass -> argmax -> map_action
                # feat = extract_features(state)
                # with torch.no_grad():
                #     q = net(torch.FloatTensor(feat).unsqueeze(0))
                #     a_idx = int(torch.argmax(q).item())
                # agent = self.squad[i]
                # actions.append(map_action(a_idx, state, (agent.x, agent.y), agent.sho))
                raise NotImplementedError("Implement the forward pass")
            else:
                # Fallback for untrained slots: chase the ball if visible,
                # otherwise push forward.  NEVER just IDLE — a team that
                # stands still cannot score or defend.
                actions.append(self._chase(state, team))
        return actions

    # ------------------------------------------------------------------
    # Minimal goalkeeper — MOVE along the goal line to track the ball.
    # For GKs, move_direction 0 and 1 are the two lateral directions (the
    # engine auto-faces the ball).  In vision_sectors, center_angle > 0
    # means the ball is to the GK's left → move_direction 0 in both teams.
    # Replace this with your own logic; see api/agents.py _goalkeeper().
    # ------------------------------------------------------------------
    def _goalkeeper(self, state: dict, team: str) -> dict:
        if state.get("has_ball"):
            return map_action(10, state, None, 5.0)   # PASS = distribute
        if state.get("cooldown_remaining", 0) > 0:
            return {"action_type": "IDLE"}
        for s in state.get("vision_sectors", []):
            if s.get("type") == "ball" and s.get("distance") is not None:
                ca = s.get("center_angle", 0.0)
                if abs(ca) < 5.0:
                    return {"action_type": "IDLE"}    # ball ahead — hold
                return {"action_type": "MOVE",
                        "move_direction": 0 if ca > 0 else 1,
                        "rotation_angle": 0.0,
                        "shoot_power_percentage": 0.0, "shoot_angle": 0.0}
        return {"action_type": "IDLE"}                 # ball not visible

    # ------------------------------------------------------------------
    # Minimal off-ball fallback — chase a visible ball, else push forward.
    # ------------------------------------------------------------------
    def _chase(self, state: dict, team: str) -> dict:
        for s in state.get("vision_sectors", []):
            if s.get("type") == "ball" and s.get("distance") is not None:
                ori = state.get("current_orientation", 0.0)
                target = ori + s.get("center_angle", 0.0)
                return map_action(_compass(target), state, None, 5.0)
        fwd = 0.0 if team == "A" else 180.0
        return map_action(_compass(fwd), state, None, 5.0)


def make_policy(squad: Optional[List[Any]] = None, seed: Optional[int] = None) -> MyPolicy:
    return MyPolicy(squad=squad, seed=seed)
