"""
demo_agent.py — a tiny, readable example agent
==============================================
A drop-in ``agents.py``: it exposes ``make_policy(squad, seed)`` returning a
policy whose ``decide(states, team)`` gives 11 engine action dicts each tick.

No training, no torch — pure rules you can read top to bottom. Use it as a
sparring opponent for ``evaluate.py`` (it is the default ``--b``), or copy it as
a starting point for your own ``agents.py``.

The idea in one line: chase the ball when you don't have it; pass/dribble
forward when you do; shoot near the opponent goal; the keeper tracks the
ball along the goal line and dives on close shots.

⚠ Goalkeeper note ⚠
-------------------
The keeper MUST MOVE (action indices 0 or 1) to track the ball along the goal
line.  DEFLECT (12) only sets ``deflect_angle`` — it does NOT move the keeper.
DIVE (11) triggers a cooldown that freezes the keeper for several ticks.
A keeper that only DEFLECTs or DIVEs is a static keeper that concedes every
shot.  See ``api/agents.py`` ``Policy._goalkeeper()`` for the full reference.
"""

from __future__ import annotations

from typing import Any, List, Optional

try:
    from api.agents import map_action
    from api.config import FIELD_X_MIN, FIELD_X_MAX
except ImportError:
    from fifa_ai_world_cup.agents import map_action  # type: ignore
    from fifa_ai_world_cup.config import FIELD_X_MIN, FIELD_X_MAX  # type: ignore


def _compass(angle_deg: float) -> int:
    """Nearest of the 8 MOVE directions (0=E, 1=NE, 2=N, …) for an angle."""
    return int(round((angle_deg % 360.0) / 45.0)) % 8


def _ball_bearing(state: dict) -> Optional[float]:
    """Absolute angle to the ball if any vision sector sees it, else None."""
    ori = state.get("current_orientation", 0.0)
    for s in state.get("vision_sectors", []):
        if s.get("type") == "ball" and s.get("distance") is not None:
            return ori + s.get("center_angle", 0.0)
    return None


class DemoPolicy:
    """A minimal centralized policy. Reads its own squad for positions."""

    def __init__(self, squad: Optional[List[Any]] = None,
                 seed: Optional[int] = None):
        # squad[i] is the live Agent for states[i] (same slot order).
        self.squad = list(squad) if squad else []

    def decide(self, states: List[dict], team: str) -> List[dict]:
        goal_x = FIELD_X_MAX if team == "A" else FIELD_X_MIN   # goal we attack
        forward = 0.0 if team == "A" else 180.0                # our attack heading
        actions = []
        for i, st in enumerate(states):
            ag = self.squad[i] if i < len(self.squad) else None
            pos = (ag.x, ag.y) if ag is not None else None
            sho = ag.sho if ag is not None else 5.0
            idx = self._choose(st, pos, goal_x, forward)
            actions.append(map_action(idx, st, pos, sho))
        return actions

    def _choose(self, st: dict, pos, goal_x: float, forward: float) -> int:
        if st.get("is_gk"):
            return self._goalkeeper(st, goal_x)
        if st.get("has_ball"):
            # Shoot when close to the opponent goal, else pass (dribbles forward
            # if there is no team-mate to pass to).
            if pos is not None and abs(goal_x - pos[0]) < 30.0:
                return 9                                    # SHOOT
            return 10                                       # PASS
        bearing = _ball_bearing(st)
        if bearing is not None:
            return _compass(bearing)                        # chase a visible ball
        return _compass(forward)                            # otherwise push up

    # ---- goalkeeper -----------------------------------------------------
    # The keeper MUST move along the goal line to track the ball.  Returning
    # DEFLECT (12) or DIVE (11) every tick is a common mistake — DEFLECT does
    # not move the keeper at all, and DIVE triggers a multi-tick cooldown.
    # For GKs, move_direction 0 and 1 are the two lateral directions along
    # the goal line (the engine auto-faces the ball, so rotation is moot).
    #
    # In vision_sectors, center_angle > 0 means the ball is to the GK's left
    # (which is +Y for Team A, -Y for Team B) — in both cases, move_direction
    # 0 is the correct lateral response.
    def _goalkeeper(self, st: dict, goal_x: float) -> int:
        if st.get("has_ball"):
            return 10                                       # PASS = distribute
        if st.get("cooldown_remaining", 0) > 0:
            return 8                                        # frozen after a dive
        for s in st.get("vision_sectors", []):
            if s.get("type") == "ball" and s.get("distance") is not None:
                ca = s.get("center_angle", 0.0)
                if abs(ca) < 5.0:
                    return 8                                # ball ahead — hold
                return 0 if ca > 0 else 1                   # track laterally
        return 8                                            # ball not visible


def make_policy(squad: Optional[List[Any]] = None,
                seed: Optional[int] = None) -> DemoPolicy:
    return DemoPolicy(squad, seed)
