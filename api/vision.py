"""
vision.py
=========
Directional vision cone and partial-observability state builder
(Section 4).

An agent can only perceive dynamic elements (other agents and the ball) that
satisfy BOTH:

  1. Distance limit:   d(P_i, P_target) <= VISION_RANGE (fixed 40.0 units)
  2. Angular aperture: |theta_relative - phi_i| <= VIS / 2

The engine builds a localized state dictionary for each agent every tick and
hands it to the participant policy (agents.py).  The structure matches the
SRS Section 4.2 example exactly.
"""

from __future__ import annotations

from typing import Any, Dict, List

from . import config as C
from .agent import Agent
from .ball import Ball
from .utils import angle_diff, distance, normalize_angle, relative_angle


def _in_cone(observer: Agent, target_x: float, target_y: float) -> bool:
    """Whether a target point is within the observer's vision cone."""
    dist = distance((observer.x, observer.y), (target_x, target_y))
    sight = getattr(observer, "vis_range", C.VISION_RANGE)
    if dist > sight:
        return False
    if dist < 1e-6:
        return True
    rel = relative_angle((observer.x, observer.y), observer.orientation,
                         (target_x, target_y))
    return abs(normalize_angle(rel)) <= (observer.vision_aperture() / 2.0)


def _build_vision_sectors(aperture: float,
                          visible: List[Dict[str, Any]],
                          n_sectors: int = 5) -> List[Dict[str, Any]]:
    """Bucket visible objects into ``n_sectors`` angular sectors of the cone.

    Each sector reports the *nearest* object inside it (distance + type) and
    the sector's centre angle (relative to the agent's heading).  This is the
    compact, fixed-width view consumed by ``rl_api.extract_features`` so a
    learned policy sees a constant-shape observation regardless of how many
    objects are on the pitch.
    """
    half = aperture / 2.0
    width = aperture / n_sectors if n_sectors else aperture
    sectors: List[Dict[str, Any]] = []
    for k in range(n_sectors):
        lo = -half + k * width
        hi = lo + width
        center = (lo + hi) / 2.0
        nearest: Dict[str, Any] = {"distance": None, "type": None,
                                   "center_angle": round(center, 3)}
        best_d = float("inf")
        for obj in visible:
            ang = obj["rel_angle"]
            in_sector = (lo <= ang < hi) or (k == n_sectors - 1 and ang == hi)
            if in_sector and obj["rel_distance"] < best_d:
                best_d = obj["rel_distance"]
                nearest = {
                    "distance": obj["rel_distance"],
                    "type": obj["type"],
                    "center_angle": round(center, 3),
                }
        sectors.append(nearest)
    return sectors


def build_agent_state(agent: Agent, all_agents: List[Agent],
                      ball: Ball, tick: int = 0,
                      max_ticks: int = 0) -> Dict[str, Any]:
    """Construct the partial-observability state dict for ``agent``.

    Mirrors SRS Section 4.2.  ``visible_objects`` lists every dynamic element
    currently inside the agent's vision cone (teammates, opponents and the
    ball), each described by relative distance and relative angle.  A compact
    ``vision_sectors`` view (fixed-width, one nearest object per angular sector)
    is also included for learned policies.  ``tick`` / ``max_ticks`` carry the
    match clock so a reward/feature function can reason about time remaining.
    """
    visible: List[Dict[str, Any]] = []

    for other in all_agents:
        if other.id == agent.id:
            continue
        if not _in_cone(agent, other.x, other.y):
            continue
        rel_dist = distance((agent.x, agent.y), (other.x, other.y))
        rel_ang = relative_angle((agent.x, agent.y), agent.orientation,
                                 (other.x, other.y))
        rel_type = "teammate" if other.team == agent.team else "opponent"
        visible.append({
            "id": other.id,
            "type": rel_type,
            "rel_distance": round(rel_dist, 3),
            "rel_angle": round(rel_ang, 3),
            "is_gk": other.is_gk,
            "is_star": other.is_star,
        })

    # The ball is always evaluated last and only if inside the cone.
    ball_visible = _in_cone(agent, ball.x, ball.y)
    if ball_visible:
        rel_dist = distance((agent.x, agent.y), (ball.x, ball.y))
        rel_ang = relative_angle((agent.x, agent.y), agent.orientation,
                                 (ball.x, ball.y))
        visible.append({
            "id": "Ball",
            "type": "ball",
            "rel_distance": round(rel_dist, 3),
            "rel_angle": round(rel_ang, 3),
        })

    state: Dict[str, Any] = {
        "self_id": agent.id,
        "team": agent.team,
        "is_gk": agent.is_gk,
        "archetype": agent.archetype,
        "is_star": agent.is_star,
        "current_orientation": round(agent.orientation, 3),
        "has_ball": agent.has_ball,
        "cooldown_remaining": agent.cooldown_remaining,
        "tick": tick,
        "max_ticks": max_ticks,
        # Own absolute position is NOT exposed (partial observability) except
        # for the goalkeeper, who is permitted local goal-line awareness so it
        # can position itself meaningfully.
        "visible_objects": visible,
        "vision_sectors": _build_vision_sectors(agent.vision_aperture(), visible),
    }
    if agent.is_gk:
        state["gk_x"] = round(agent.x, 3)
        state["gk_y"] = round(agent.y, 3)
        state["deflect_angle"] = round(agent.deflect_angle, 3)
    return state


def build_team_states(team_agents: List[Agent], all_agents: List[Agent],
                      ball: Ball, tick: int = 0,
                      max_ticks: int = 0) -> List[Dict[str, Any]]:
    """Build the list of localized state dicts for an entire team."""
    return [build_agent_state(a, all_agents, ball, tick, max_ticks)
            for a in team_agents]
