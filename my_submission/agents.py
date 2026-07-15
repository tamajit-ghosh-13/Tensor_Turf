"""
agents.py — England: A dominant centralized team policy.
================================================================
Pure heuristic, zero RL. Leverages full proprioception (reading all 11 squad
positions every tick) for coordinated attacking football.

Key design choices:
  - Aggressive shooting range (35 units, aimed away from visible GK)
  - High press: attackers/mids push into the final third when in possession
  - Counter-attack shape: compact midfield block when out of possession
  - Only the nearest outfielder chases a loose ball; rest hold shape
  - GK tracks ball Y precisely, dives only on close-range threats
  - Smart passing: always look for the most forward open teammate
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

try:
    from api.agents import map_action, extract_features, ACTION_DIM
    from api import config as C
    from api.utils import (
        angle_of_vector, clamp, distance,
        normalize_angle, vec_from_angle,
    )
except ImportError:
    from fifa_ai_world_cup.agents import map_action, extract_features, ACTION_DIM  # type: ignore
    from fifa_ai_world_cup import config as C  # type: ignore
    from fifa_ai_world_cup.utils import (  # type: ignore
        angle_of_vector, clamp, distance,
        normalize_angle, vec_from_angle,
    )

Point = Tuple[float, float]

# Action type constants
MOVE, ROTATE, SHOOT, IDLE = "MOVE", "ROTATE", "SHOOT", "IDLE"
DIVE, DEFLECT = "DIVE", "DEFLECT"

SQUAD_SIZE = 11

# ─── helpers ────────────────────────────────────────────────────────────────

def _compass(angle_deg: float) -> int:
    return int(round((angle_deg % 360.0) / 45.0)) % 8

def _rotate_toward(current: float, target: float) -> float:
    diff = (target - current + 180.0) % 360.0 - 180.0
    return max(-30.0, min(30.0, diff))

def _polar_to_global(origin: Point, heading: float,
                     rel_dist: float, rel_angle: float) -> Point:
    abs_ang = heading + rel_angle
    dx, dy = vec_from_angle(abs_ang, rel_dist)
    return (origin[0] + dx, origin[1] + dy)

def _kick_power(dist: float, sho: float) -> float:
    if sho <= 0:
        return 1.0
    return clamp(dist / (20.0 * sho) + 0.05, 0.08, 1.0)

def _shot_power(dist: float, sho: float) -> float:
    if sho <= 0:
        return 1.0
    return clamp(dist / (5.0 * sho) + 0.5, 0.55, 1.0)

def _opp_goal(team: str) -> Point:
    return (C.FIELD_X_MAX, C.FIELD_CENTER_Y) if team == "A" \
        else (C.FIELD_X_MIN, C.FIELD_CENTER_Y)

def _own_goal(team: str) -> Point:
    return (C.FIELD_X_MIN, C.FIELD_CENTER_Y) if team == "A" \
        else (C.FIELD_X_MAX, C.FIELD_CENTER_Y)

def _attack_angle(team: str) -> float:
    return 0.0 if team == "A" else 180.0

# ─── The Policy ─────────────────────────────────────────────────────────────

class SuperPolicy:
    """Centralized team policy with full proprioception."""

    def __init__(self, squad: Optional[List[Any]] = None,
                 seed: Optional[int] = None):
        self.squad: List[Any] = list(squad) if squad else []

    # ── main entry ──────────────────────────────────────────────────────
    def decide(self, states: List[Dict[str, Any]], team: str) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        have_squad = bool(self.squad) and len(self.squad) == len(states)

        # Team-level read
        ball = self._estimate_ball(states) if have_squad else None
        carrier_idx = next((i for i, s in enumerate(states)
                            if s.get("has_ball")), None)

        for i, state in enumerate(states):
            if state.get("is_gk"):
                actions.append(self._goalkeeper(state, team, ball, i))
            elif state.get("has_ball"):
                actions.append(self._carrier(state, team, ball, i))
            else:
                actions.append(self._outfield(state, team, ball,
                                              carrier_idx, i, have_squad))
        return actions

    # ── ball estimation ─────────────────────────────────────────────────
    def _estimate_ball(self, states: List[Dict[str, Any]]) -> Optional[Point]:
        best: Optional[Tuple[float, Point]] = None
        for i, s in enumerate(states):
            if i >= len(self.squad):
                break
            ag = self.squad[i]
            for obj in s.get("visible_objects", []):
                if obj.get("type") == "ball":
                    gp = _polar_to_global((ag.x, ag.y), ag.orientation,
                                          obj["rel_distance"], obj["rel_angle"])
                    if best is None or obj["rel_distance"] < best[0]:
                        best = (obj["rel_distance"], gp)
        return best[1] if best else None

    # ── visible objects helpers ──────────────────────────────────────────
    def _visible_globals(self, agent, state: Dict[str, Any],
                         only_type: Optional[str] = None):
        out = []
        for obj in state.get("visible_objects", []):
            if only_type is not None and obj.get("type") != only_type:
                continue
            gp = _polar_to_global((agent.x, agent.y), agent.orientation,
                                  obj["rel_distance"], obj["rel_angle"])
            out.append((obj, gp))
        return out

    def _agent(self, state: Dict[str, Any]):
        sid = state.get("self_id")
        for a in self.squad:
            if a.id == sid:
                return a
        return None

    # ── carrier ──────────────────────────────────────────────────────────
    def _carrier(self, state: Dict[str, Any], team: str,
                 ball: Optional[Point], i: int) -> Dict[str, Any]:
        agent = self._agent(state)
        opp = _opp_goal(team)
        own = (agent.x, agent.y) if agent else (50.0, 30.0)
        d_goal = distance(own, opp)
        goal_ang = angle_of_vector((opp[0] - own[0], opp[1] - own[1]))

        # 1) SHOOT when within 35 units of goal (aggressive range)
        if d_goal <= 35.0 and agent is not None:
            aim = self._aim_away_from_keeper(agent, state, opp)
            return {"action_type": SHOOT, "move_direction": 0,
                    "rotation_angle": _rotate_toward(agent.orientation, goal_ang),
                    "shoot_power_percentage": _shot_power(d_goal, agent.sho),
                    "shoot_angle": aim}

        # 2) PASS out of pressure (opponent within 2x tackle radius)
        if agent is not None:
            opps = self._visible_globals(agent, state, only_type="opponent")
            nearest_opp = min((distance(own, p) for _, p in opps), default=999.0)
            if nearest_opp < C.TACKLE_RADIUS * 2.5:
                tgt = self._best_pass_target(agent, state, team)
                if tgt is not None:
                    tp = tgt
                    ang = angle_of_vector((tp[0] - own[0], tp[1] - own[1]))
                    return {"action_type": SHOOT, "move_direction": 0,
                            "rotation_angle": _rotate_toward(agent.orientation, ang),
                            "shoot_power_percentage": _kick_power(distance(own, tp), agent.sho),
                            "shoot_angle": ang}

        # 3) PASS if a teammate is far forward and open (proactive passing)
        if agent is not None:
            tgt = self._best_pass_target(agent, state, team, min_forward=8.0)
            if tgt is not None:
                tp = tgt
                ang = angle_of_vector((tp[0] - own[0], tp[1] - own[1]))
                return {"action_type": SHOOT, "move_direction": 0,
                        "rotation_angle": _rotate_toward(agent.orientation, ang),
                        "shoot_power_percentage": _kick_power(distance(own, tp), agent.sho),
                        "shoot_angle": ang}

        # 4) DRIBBLE toward goal
        if agent is None:
            fwd = _attack_angle(team)
            ori = state.get("current_orientation", 0.0)
            return {"action_type": MOVE, "move_direction": _compass(fwd),
                    "rotation_angle": _rotate_toward(ori, fwd),
                    "shoot_power_percentage": 0.0, "shoot_angle": 0.0}
        return {"action_type": MOVE, "move_direction": _compass(goal_ang),
                "rotation_angle": _rotate_toward(agent.orientation, goal_ang),
                "shoot_power_percentage": 0.0, "shoot_angle": 0.0}

    def _aim_away_from_keeper(self, agent, state: Dict[str, Any],
                               opp_goal: Point) -> float:
        own = (agent.x, agent.y)
        center = angle_of_vector((opp_goal[0] - own[0], opp_goal[1] - own[1]))
        gk_y = None
        for obj, gp in self._visible_globals(agent, state, only_type="opponent"):
            if obj.get("is_gk"):
                gk_y = gp[1]
                break
        if gk_y is None:
            return center
        # Aim at the far post from the keeper
        target_y = (C.GOAL_Y_MIN + 1.0) if gk_y > C.FIELD_CENTER_Y \
            else (C.GOAL_Y_MAX - 1.0)
        return angle_of_vector((opp_goal[0] - own[0], target_y - own[1]))

    def _best_pass_target(self, agent, state: Dict[str, Any],
                           team: str, min_forward: float = 2.0) -> Optional[Point]:
        """Find the best forward, open teammate to pass to."""
        atk = 1.0 if team == "A" else -1.0
        own = (agent.x, agent.y)
        opp_g = _opp_goal(team)
        best_score = -1e9
        best_pos = None

        for obj, gp in self._visible_globals(agent, state, only_type="teammate"):
            if obj.get("is_gk"):
                continue
            forward = (gp[0] - own[0]) * atk
            if forward < min_forward:
                continue
            # Check openness (distance from nearest visible opponent)
            opp_min = 50.0
            for oobj, op in self._visible_globals(agent, state, only_type="opponent"):
                d = distance(gp, op)
                if d < opp_min:
                    opp_min = d
            if opp_min < 3.0:
                continue
            # Score: forward progress + openness - distance to goal
            score = forward * 1.2 + opp_min * 0.5 - distance(gp, opp_g) * 0.2
            if score > best_score:
                best_score = score
                best_pos = gp
        return best_pos

    # ── non-carrier outfield ─────────────────────────────────────────────
    def _outfield(self, state: Dict[str, Any], team: str,
                  ball: Optional[Point], carrier_idx: Optional[int],
                  i: int, have_squad: bool) -> Dict[str, Any]:
        agent = self._agent(state)

        # No squad reference: chase ball or push forward
        if agent is None or not have_squad:
            sec = self._nearest_ball_sector(state)
            ori = state.get("current_orientation", 0.0)
            if sec is not None and sec.get("distance") is not None:
                target = ori + sec["center_angle"]
                return {"action_type": MOVE, "move_direction": _compass(target),
                        "rotation_angle": _rotate_toward(ori, target),
                        "shoot_power_percentage": 0.0, "shoot_angle": 0.0}
            fwd = _attack_angle(team)
            return {"action_type": MOVE, "move_direction": _compass(fwd),
                    "rotation_angle": _rotate_toward(ori, fwd),
                    "shoot_power_percentage": 0.0, "shoot_angle": 0.0}

        own = (agent.x, agent.y)

        # WE HAVE POSSESSION → push into attacking support positions
        if carrier_idx is not None:
            target = self._attack_support(agent, team, ball)
            return self._move_toward(agent, target)

        # LOOSE BALL → only the nearest non-GK outfielder chases
        if ball is not None:
            nearest = min(
                range(len(self.squad)),
                key=lambda k: distance((self.squad[k].x, self.squad[k].y), ball)
                if not self.squad[k].is_gk else 1e9
            )
            if i == nearest:
                return self._move_toward(agent, ball)

        # NO POSSESSION → hold compact defensive shape
        target = self._defend_shape(agent, team, ball)
        return self._move_toward(agent, target)

    def _attack_support(self, agent, team: str,
                        ball: Optional[Point]) -> Point:
        """High press: push well forward when we have the ball."""
        atk = 1.0 if team == "A" else -1.0
        # Attackers (slots 8-10): push to the final third
        # Midfielders (slots 5-7): push to the halfway line
        # Defenders (slots 1-4): step up to halfway
        x = clamp(agent.home_x + atk * 15.0, 10.0, C.FIELD_X_MAX - 10.0)
        # Shift Y toward the ball to provide passing options
        if ball is not None:
            y = clamp(agent.home_y + (ball[1] - agent.home_y) * 0.3,
                      8.0, C.FIELD_Y_MAX - 8.0)
        else:
            y = agent.home_y
        return (x, y)

    def _defend_shape(self, agent, team: str,
                      ball: Optional[Point]) -> Point:
        """Compact defensive block between ball and own goal."""
        if ball is None:
            return (agent.home_x, agent.home_y)
        # Shift Y toward ball
        y = clamp(agent.home_y + (ball[1] - agent.home_y) * 0.45,
                  6.0, C.FIELD_Y_MAX - 6.0)
        # X: stay between ball and own goal, compact
        if team == "A":
            x = clamp(min(agent.home_x, ball[0] - 6.0), 4.0, 45.0)
        else:
            x = clamp(max(agent.home_x, ball[0] + 6.0),
                      C.FIELD_X_MAX - 45.0, C.FIELD_X_MAX - 4.0)
        return (x, y)

    def _move_toward(self, agent, target: Point) -> Dict[str, Any]:
        own = (agent.x, agent.y)
        d = distance(own, target)
        ang = angle_of_vector((target[0] - own[0], target[1] - own[1]))
        if d < 0.6:
            return {"action_type": IDLE, "move_direction": 0,
                    "rotation_angle": _rotate_toward(agent.orientation, ang),
                    "shoot_power_percentage": 0.0, "shoot_angle": 0.0}
        return {"action_type": MOVE, "move_direction": _compass(ang),
                "rotation_angle": _rotate_toward(agent.orientation, ang),
                "shoot_power_percentage": 0.0, "shoot_angle": 0.0}

    # ── goalkeeper ───────────────────────────────────────────────────────
    def _goalkeeper(self, state: Dict[str, Any], team: str,
                    ball: Optional[Point], i: int) -> Dict[str, Any]:
        if state.get("cooldown_remaining", 0) > 0:
            return {"action_type": IDLE}

        agent = self._agent(state)
        gy = state.get("gk_y", 30.0)
        gx = state.get("gk_x", (_own_goal(team)[0] + (3.0 if team == "A" else -3.0)))
        own = (gx, gy)
        target_y = clamp(ball[1] if ball else C.FIELD_CENTER_Y,
                         C.GOAL_Y_MIN - 1.0, C.GOAL_Y_MAX + 1.0)
        dy = target_y - gy

        # Distribute if GK has the ball
        if state.get("has_ball") and agent is not None:
            tgt = self._best_pass_target(agent, state, team)
            if tgt is not None:
                tp = tgt
                ang = angle_of_vector((tp[0] - own[0], tp[1] - own[1]))
                return {"action_type": SHOOT, "move_direction": 0,
                        "rotation_angle": _rotate_toward(state.get("current_orientation", 0.0), ang),
                        "shoot_power_percentage": _kick_power(distance(own, tp), agent.sho),
                        "shoot_angle": ang}
            # No pass target: boot it forward
            fwd = _attack_angle(team)
            return {"action_type": SHOOT, "move_direction": 0,
                    "rotation_angle": _rotate_toward(state.get("current_orientation", 0.0), fwd),
                    "shoot_power_percentage": 1.0, "shoot_angle": fwd}

        # Dive on close-range threat
        if ball is not None:
            og = _own_goal(team)
            on_side = (ball[0] < 20.0 and team == "A") or \
                      (ball[0] > 80.0 and team == "B")
            if on_side and distance(ball, og) < 14.0 and abs(dy) > 2.0:
                move_dir = 0 if (dy > 0 and team == "A") or (dy < 0 and team == "B") else 1
                return {"action_type": DIVE, "move_direction": move_dir,
                        "deflect_angle": _attack_angle(team)}

        # Track ball along goal line
        if abs(dy) < 0.5:
            return {"action_type": IDLE, "move_direction": 0,
                    "rotation_angle": 0.0, "shoot_power_percentage": 0.0,
                    "shoot_angle": 0.0, "deflect_angle": _attack_angle(team)}
        move_dir = 0 if (dy > 0 and team == "A") or (dy < 0 and team == "B") else 1
        return {"action_type": MOVE, "move_direction": move_dir,
                "rotation_angle": 0.0, "shoot_power_percentage": 0.0,
                "shoot_angle": 0.0, "deflect_angle": _attack_angle(team)}

    # ── utility ──────────────────────────────────────────────────────────
    @staticmethod
    def _nearest_ball_sector(state: Dict[str, Any]):
        for s in state.get("vision_sectors", []):
            if s.get("type") == "ball":
                return s
        return None


# ─── Factory (the engine's only import) ──────────────────────────────────
def make_policy(squad: Optional[List[Any]] = None,
                seed: Optional[int] = None) -> SuperPolicy:
    return SuperPolicy(squad=squad, seed=seed)
