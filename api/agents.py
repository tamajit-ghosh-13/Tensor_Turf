"""
agents.py
=========
The participant submission policy file.

You write your policy here.  The engine imports exactly one thing from this
module -- ``make_policy(squad=None, seed=None)`` -- which must return an
object exposing ``decide(self, states, team) -> list[action]`` (see
``agents_example.py`` for the precise contract).

This file ships as a **working starting point**, not a blank stub, so the
environment is immediately runnable with ``--style-a participant`` and so you
have a readable reference for the centralized-policy pattern.  It does two
things:

  1. If you have trained a DQN (see ``train_rl_agent.py``) AND PyTorch is
     installed AND the weight file for a role exists, that role's actions are
     produced by the network.  This is optional -- torch is imported lazily so
     the file runs fine without it.
  2. Otherwise (no torch, no weights, or a role you haven't trained yet) a
     solid centralized heuristic drives the squad.  The heuristic is built to
     *always keep moving*: the carrier dribbles at the opponent's goal, shoots
     in range, and passes out of pressure; team-mates push forward to support
     an attack or hold a compact defensive shape.  Nobody stands still spinning
     on the spot just because the ball is temporarily out of their vision cone
     -- the policy uses its own squad's positions (proprioception, see the
     README) to stay organized even when individual agents can't see the ball.

Replace the heuristic with your own logic / trained network as you develop.

────────────────────────────────────────────────────────────────────────────────
REFERENCE IMPLEMENTATIONS YOU SHOULD READ BEFORE WRITING YOUR OWN
────────────────────────────────────────────────────────────────────────────────
This file is NOT just a helper library — it is a complete, working policy with
correct implementations of every role.  Search for these method names:

  Policy._goalkeeper()   — ball-Y tracking, dive timing, distribution.
                           ⚠ DO NOT just return DEFLECT (12) or DIVE (11)
                           every tick.  DEFLECT does not move the GK; DIVE
                           triggers a cooldown.  The keeper MUST MOVE (0 or 1)
                           along the goal line.  See the method below.

  Policy._carrier()      — shoot in range, pass out of pressure, dribble.
  Policy._outfield()     — chase loose balls, support the attack, hold shape.
  Policy._safeguard()    — prevents a trained DQN from parking players.

The ``map_action`` and ``extract_features`` functions at the top of this file
are the convenience helpers you import; the ``Policy`` class below them is the
reference.  Read it before writing your own.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

try:
    from api import config as C
    from api.agent import Agent
    from api.utils import (
        angle_of_vector, clamp, distance, move_direction_index_to_angle,
        normalize_angle, vec_from_angle,
    )
except ImportError:
    from fifa_ai_world_cup import config as C  # type: ignore
    from fifa_ai_world_cup.agent import Agent  # type: ignore
    from fifa_ai_world_cup.utils import (  # type: ignore
        angle_of_vector, clamp, distance, move_direction_index_to_angle,
        normalize_angle, vec_from_angle,
    )

Point = Tuple[float, float]

# Action type constants (match the engine's action_type strings).
MOVE, ROTATE, SHOOT, IDLE = "MOVE", "ROTATE", "SHOOT", "IDLE"
DIVE, DEFLECT = "DIVE", "DEFLECT"


# ---------------------------------------------------------------------------
# Optional torch import -- training is the only hard dependency on torch.
# If torch is unavailable, ``_TORCH`` is None and the policy transparently
# falls back to the heuristic, so the match still runs.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - environment-dependent
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:  # noqa: BLE001 - torch is optional at runtime
    torch = None
    nn = None
    _TORCH = False


# ===========================================================================
# 1.  Observation / action helpers (shared by the heuristic and the DQN)
# ===========================================================================
def _rotate_toward(current: float, target: float) -> float:
    """Clamped (-30..30 deg) rotation that turns ``current`` toward ``target``."""
    diff = (target - current + 180.0) % 360.0 - 180.0
    return max(-30.0, min(30.0, diff))


def _compass(angle_deg: float) -> int:
    a = angle_deg % 360.0
    return int(round(a / 45.0)) % 8


def _polar_to_global(origin: Point, heading: float,
                     rel_dist: float, rel_angle: float) -> Point:
    abs_ang = heading + rel_angle
    dx, dy = vec_from_angle(abs_ang, rel_dist)
    return (origin[0] + dx, origin[1] + dy)


def _kick_power_for_distance(dist: float, sho: float) -> float:
    """Pass/clearance power: catchable pace that still covers ``dist``."""
    if sho <= 0:
        return 1.0
    return clamp(dist / (20.0 * sho) + 0.05, 0.08, 1.0)


def _shot_power_for_distance(dist: float, sho: float) -> float:
    """Shot-on-goal power: hard, so the ball beats the keeper."""
    if sho <= 0:
        return 1.0
    return clamp(dist / (5.0 * sho) + 0.5, 0.55, 1.0)


def _attack_angle(team: str) -> float:
    return 0.0 if team == "A" else 180.0


def _opp_goal(team: str) -> Point:
    return (C.TEAM_B_GOAL_X, C.FIELD_CENTER_Y) if team == "A" \
        else (C.TEAM_A_GOAL_X, C.FIELD_CENTER_Y)


def _own_goal(team: str) -> Point:
    return (C.TEAM_A_GOAL_X, C.FIELD_CENTER_Y) if team == "A" \
        else (C.TEAM_B_GOAL_X, C.FIELD_CENTER_Y)


# ===========================================================================
# 2.  DQN definition (only used when torch + trained weights are present)
# ===========================================================================
STATE_DIM = 20
# 13-action unified space (outfield + GK): 8 MOVE + ROTATE + SHOOT + PASS +
# DIVE + DEFLECT.  Outfield agents' DIVE/DEFLECT map to IDLE; GKs' SHOOT/PASS
# still work but DIVE/DEFLECT are keeper-specific.  This lets every slot
# (including the GK) train with the same interface.
ACTION_DIM = 13


def extract_features(state: Dict[str, Any]) -> "list[float]":
    """20-dim observation vector -- identical to ``train_rl_agent.py``.

    Keep this in sync with the trainer so a network trained there loads and
    runs here without surprise.
    """
    f: List[float] = []
    # Proprioception (5)
    f.append(1.0 if state.get("has_ball") else 0.0)
    f.append(float(state.get("cooldown_remaining", 0)) / 35.0)
    ori = math.radians(state.get("current_orientation", 0.0))
    f.append(math.sin(ori))
    f.append(math.cos(ori))
    f.append(float(state.get("gk_y", 30.0)) / 60.0)
    # Vision sectors (5 x 3 = 15)
    sectors = list(state.get("vision_sectors", []))
    while len(sectors) < 5:
        sectors.append({"distance": None, "type": None})
    for s in sectors[:5]:
        d = s.get("distance")
        f.append(float(d) / 40.0 if d is not None else 1.0)
        kind = s.get("type")
        f.append(1.0 if kind == "ball" else 0.0)
        f.append(1.0 if kind in ("teammate", "opponent") else 0.0)
    return f


def map_action(action_idx: int, state: Dict[str, Any],
               agent_pos: Optional[Point] = None, sho: float = 5.0) -> Dict[str, Any]:
    """Discrete action index -> engine action dict (13-action unified space).

    * 0..7  -- MOVE in compass direction ``idx``
    * 8     -- ROTATE (scan)
    * 9     -- SHOOT at the opponent goal (aimed away from a visible keeper).
               If ``agent_pos`` is supplied the shot is aimed precisely from
               the carrier's position; otherwise it shoots along the heading.
    * 10    -- PASS to the best visible forward team-mate (dribble if none).
               ``agent_pos`` enables precise pass angles.
    * 11    -- DIVE (goalkeeper only; outfielders -> IDLE)
    * 12    -- DEFLECT (goalkeeper only; outfielders -> IDLE)

    ``agent_pos`` is the controlled agent's absolute (x, y); it is available
    to a centralized policy (via its squad reference) and to the training loop
    (via ``TrainingEnv.controlled_agent``), but NOT inside the raw state dict,
    so callers that have it should pass it for accurate shooting/passing.
    """
    ori = state.get("current_orientation", 0.0)
    team = state.get("team", "A")
    is_gk = state.get("is_gk", False)
    if action_idx < 8:
        target = action_idx * 45.0
        return {
            "action_type": MOVE,
            "move_direction": action_idx,
            "rotation_angle": _rotate_toward(ori, target),
            "shoot_power_percentage": 0.0,
            "shoot_angle": 0.0,
        }
    if action_idx == 8:
        return {"action_type": ROTATE, "move_direction": 0,
                "rotation_angle": 15.0,
                "shoot_power_percentage": 0.0, "shoot_angle": 0.0}
    if action_idx == 9:
        return _shoot_at_goal(state, team, ori, agent_pos, sho)
    if action_idx == 10:
        # PASS to the best visible forward team-mate.
        ang = _pass_angle(state, team, ori, agent_pos)
        if ang is not None:
            return {"action_type": SHOOT, "move_direction": 0,
                    "rotation_angle": _rotate_toward(ori, ang),
                    "shoot_power_percentage": _kick_power_for_distance(15.0, sho),
                    "shoot_angle": ang}
        # No one to pass to: dribble forward.
        fwd = _attack_angle(team)
        return {"action_type": MOVE, "move_direction": _compass(fwd),
                "rotation_angle": _rotate_toward(ori, fwd),
                "shoot_power_percentage": 0.0, "shoot_angle": 0.0}
    if action_idx == 11:
        # DIVE — goalkeeper only.  Outfielders idle.
        if not is_gk:
            return {"action_type": IDLE, "move_direction": 0,
                    "rotation_angle": 0.0,
                    "shoot_power_percentage": 0.0, "shoot_angle": 0.0}
        # Dive toward the ball's bearing (left/right relative to own net).
        move_dir = _gk_dive_direction(state, team, agent_pos)
        return {"action_type": DIVE, "move_direction": move_dir,
                "deflect_angle": _attack_angle(team)}
    if action_idx == 12:
        # DEFLECT — goalkeeper only.  Outfielders idle.
        if not is_gk:
            return {"action_type": IDLE, "move_direction": 0,
                    "rotation_angle": 0.0,
                    "shoot_power_percentage": 0.0, "shoot_angle": 0.0}
        return {"action_type": DEFLECT, "move_direction": 0,
                "deflect_angle": _attack_angle(team)}
    # Fallback.
    return {"action_type": IDLE, "move_direction": 0,
            "rotation_angle": 0.0,
            "shoot_power_percentage": 0.0, "shoot_angle": 0.0}


def _gk_dive_direction(state: Dict[str, Any], team: str,
                       agent_pos: Optional[Point]) -> int:
    """Pick the GK dive direction (0=left, 1=right) from the ball bearing."""
    # Try to read the ball's relative bearing from the GK's vision.
    for obj in state.get("visible_objects", []):
        if obj.get("type") == "ball":
            rel_angle = obj.get("rel_angle", 0.0)
            # Team A GK: ball to the GK's +Y is "left" relative to own net
            # (defending X=0, facing East).  move_dir 0 = +Y for team A.
            if team == "A":
                return 0 if rel_angle > 0 else 1
            else:
                return 1 if rel_angle > 0 else 0
    # Ball not visible: dive right (arbitrary default).
    return 1


def _shoot_at_goal(state: Dict[str, Any], team: str, ori: float,
                   agent_pos: Optional[Point], sho: float) -> Dict[str, Any]:
    opp = _opp_goal(team)
    if agent_pos is None:
        # No absolute position: shoot along the current heading.
        return {"action_type": SHOOT, "move_direction": 0,
                "rotation_angle": 0.0,
                "shoot_power_percentage": 0.85, "shoot_angle": ori}
    # Aim at the goal corner farthest from a visible goalkeeper.
    target_y = opp[1]
    gk_y = _visible_keeper_y(state, ori, agent_pos)
    if gk_y is not None:
        target_y = (C.GOAL_Y_MIN + 1.0) if gk_y > C.FIELD_CENTER_Y \
            else (C.GOAL_Y_MAX - 1.0)
    shot_ang = angle_of_vector((opp[0] - agent_pos[0], target_y - agent_pos[1]))
    d = distance(agent_pos, opp)
    return {"action_type": SHOOT, "move_direction": 0,
            "rotation_angle": _rotate_toward(ori, shot_ang),
            "shoot_power_percentage": _shot_power_for_distance(d, sho),
            "shoot_angle": shot_ang}


def _pass_angle(state: Dict[str, Any], team: str, ori: float,
                agent_pos: Optional[Point]) -> Optional[float]:
    """Absolute angle to the best visible forward team-mate, or None."""
    atk = 1.0 if team == "A" else -1.0
    best = None
    best_score = -1e9
    for obj in state.get("visible_objects", []):
        if obj.get("type") != "teammate" or obj.get("is_gk"):
            continue
        rel = obj.get("rel_angle", 0.0)
        tm_abs = ori + rel
        align = math.cos(math.radians(_rotate_toward(_attack_angle(team), tm_abs)))
        if align < 0.3:
            continue
        # If we know our own position, prefer team-mates that are open (far
        # from the nearest visible opponent) and far forward.
        openness = 1.0
        if agent_pos is not None:
            tm_pos = _polar_to_global(agent_pos, ori,
                                      obj.get("rel_distance", 1.0), rel)
            forward = (tm_pos[0] - agent_pos[0]) * atk
            opp_min = _nearest_opponent_dist(state, ori, agent_pos, tm_pos)
            if opp_min < 3.0:
                continue
            openness = opp_min
            score = forward * 1.0 + openness * 0.4
        else:
            score = obj.get("rel_distance", 0.0) * align
        if score > best_score:
            best_score = score
            best = tm_abs % 360.0
    return best


def _visible_keeper_y(state: Dict[str, Any], ori: float,
                      agent_pos: Point) -> Optional[float]:
    for obj in state.get("visible_objects", []):
        if obj.get("type") == "opponent" and obj.get("is_gk"):
            gp = _polar_to_global(agent_pos, ori,
                                  obj.get("rel_distance", 1.0),
                                  obj.get("rel_angle", 0.0))
            return gp[1]
    return None


def _nearest_opponent_dist(state: Dict[str, Any], ori: float,
                           agent_pos: Point, target: Point) -> float:
    best = 1e9
    for obj in state.get("visible_objects", []):
        if obj.get("type") != "opponent":
            continue
        op = _polar_to_global(agent_pos, ori,
                              obj.get("rel_distance", 1.0),
                              obj.get("rel_angle", 0.0))
        best = min(best, distance(op, target))
    return best


if _TORCH:
    class QNetwork(nn.Module):  # type: ignore[name-defined]
        """MLP Q-network -- architecture must match ``train_rl_agent.py``."""

        def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(state_dim, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, action_dim),
            )

        def forward(self, x):  # type: ignore[override]
            return self.fc(x)
else:  # pragma: no cover - torch-less runtime
    class QNetwork:  # type: ignore[no-redef]
        """Placeholder so attribute lookups never fail when torch is absent."""

        def __init__(self, *a, **k):
            pass


# ===========================================================================
# 3.  The policy
# ===========================================================================
class Policy:
    """Centralized team policy.

    Holds a reference to its own ``squad`` (the live ``Agent`` objects the
    engine mutates each tick).  Reading your own squad's positions/attributes
    is proprioception -- explicitly permitted for a centralized policy (see the
    README) -- and is what lets the team stay organized even when individual
    agents momentarily can't see the ball in their vision cone.
    """

    def __init__(self, squad: Optional[List[Agent]] = None,
                 seed: Optional[int] = None):
        self.squad: List[Agent] = list(squad) if squad else []
        self.rng = random.Random(seed)
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.models: Dict[str, Any] = {}
        self._load_models()

    # ---- weight loading -------------------------------------------------
    def _load_models(self) -> None:
        """Load any trained DQN weights found next to this file / cwd."""
        if not _TORCH:
            return
        module_dir = os.path.dirname(os.path.abspath(__file__))
        for role in ("ATTACKER", "MIDFIELDER", "DEFENDER", "GK"):
            for cand in (
                f"{role.lower()}_dqn.pt",
                os.path.join(module_dir, f"{role.lower()}_dqn.pt"),
                os.path.join(module_dir, "..", f"{role.lower()}_dqn.pt"),
            ):
                if os.path.exists(cand):
                    try:
                        net = QNetwork(self.state_dim, self.action_dim)
                        net.load_state_dict(
                            torch.load(cand, map_location="cpu"))
                        net.eval()
                        self.models[role] = net
                        print(f"[policy] loaded trained model for {role}: {cand}")
                    except Exception as exc:  # noqa: BLE001
                        print(f"[policy] failed to load {cand}: {exc!r}")
                    break

    # ---- main entry point ----------------------------------------------
    def decide(self, states: List[Dict[str, Any]], team: str) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        # Re-sync the squad reference length (the engine never resizes a squad,
        # but a stray policy created without a squad should still do something).
        have_squad = bool(self.squad) and len(self.squad) == len(states)

        # Team-level situational read (uses the squad's absolute positions).
        ball = self._estimate_ball(states) if have_squad else None
        carrier_idx = next((i for i, s in enumerate(states)
                            if s.get("has_ball")), None)

        for i, state in enumerate(states):
            role = state.get("archetype", "MIDFIELDER")
            model = self.models.get(role)

            # 1) Trained network, if available for this role.
            if model is not None and _TORCH:
                try:
                    feat = extract_features(state)
                    with torch.no_grad():
                        q = model(torch.FloatTensor(feat).unsqueeze(0))
                        a_idx = int(torch.argmax(q).item())
                    agent = self._agent(state)
                    agent_pos = (agent.x, agent.y) if agent is not None else None
                    sho = agent.sho if agent is not None else 5.0
                    raw = map_action(a_idx, state, agent_pos, sho)
                    actions.append(self._safeguard(raw, state, team, i, ball,
                                                   carrier_idx, have_squad))
                    continue
                except Exception:  # noqa: BLE001 - never let inference crash a match
                    pass

            # 2) Heuristic fallback.
            if state.get("is_gk"):
                actions.append(self._goalkeeper(state, team, ball))
            elif state.get("has_ball"):
                actions.append(self._carrier(state, team, ball))
            else:
                actions.append(self._outfield(state, team, ball,
                                              carrier_idx, i, have_squad))
        return actions

    # ---- situational helpers -------------------------------------------
    def _estimate_ball(self, states: List[Dict[str, Any]]) -> Optional[Point]:
        """Reconstruct the ball's global position from whoever currently sees
        it (a team-mate, or the carrier whose ball sits at their feet)."""
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

    def _visible_globals(self, agent: Agent, state: Dict[str, Any],
                         only_type: Optional[str] = None):
        out = []
        for obj in state.get("visible_objects", []):
            if only_type is not None and obj.get("type") != only_type:
                continue
            gp = _polar_to_global((agent.x, agent.y), agent.orientation,
                                  obj["rel_distance"], obj["rel_angle"])
            out.append((obj, gp))
        return out

    # ---- anti-stationary safeguard for the DQN --------------------------
    def _safeguard(self, action: Dict[str, Any], state: Dict[str, Any],
                   team: str, i: int, ball: Optional[Point],
                   carrier_idx: Optional[int], have_squad: bool) -> Dict[str, Any]:
        """Prevent a poorly-trained network from parking players.

        The carrier's choices are always trusted (it is the decision-maker
        with the ball).  For every other outfield player, only MOVE actions
        are trusted from the network -- a ROTATE (spinning on the spot) or a
        SHOOT/PASS (a useless whiff without the ball, which also freezes the
        agent for the tick via the movement-shot mutex) is replaced with the
        heuristic's purposeful outfield decision (support the attack, chase a
        loose ball, or hold shape).  This guarantees the team is never a
        static cluster, no matter how under-trained the network is.
        """
        if state.get("is_gk") or state.get("has_ball"):
            return action
        at = action.get("action_type", "IDLE") if isinstance(action, dict) else "IDLE"
        if at == "MOVE":
            return action
        # Replace any non-MOVE non-carrier choice (ROTATE/SHOOT/IDLE/...) with
        # the heuristic outfield decision, which always translates the agent.
        return self._outfield(state, team, ball, carrier_idx, i, have_squad)

    # ---- carrier --------------------------------------------------------
    def _carrier(self, state: Dict[str, Any], team: str,
                 ball: Optional[Point]) -> Dict[str, Any]:
        agent = self._agent(state)
        opp = _opp_goal(team)
        own = (agent.x, agent.y) if agent else (50.0, 30.0)
        d_goal = distance(own, opp)
        goal_ang = angle_of_vector((opp[0] - own[0], opp[1] - own[1]))

        # Shoot when in range.
        if d_goal <= 30.0 and agent is not None:
            aim = self._aim_away_from_keeper(agent, state, opp)
            return {"action_type": SHOOT, "move_direction": 0,
                    "rotation_angle": _rotate_toward(agent.orientation, goal_ang),
                    "shoot_power_percentage": _shot_power_for_distance(d_goal, agent.sho),
                    "shoot_angle": aim}

        # Pass out of immediate pressure.
        if agent is not None:
            opps = self._visible_globals(agent, state, only_type="opponent")
            nearest = min((distance(own, p) for _, p in opps), default=999.0)
            if nearest < C.TACKLE_RADIUS * 2.2:
                tgt = self._best_pass_target(agent, state, team)
                if tgt is not None:
                    tp, _ = tgt
                    ang = angle_of_vector((tp[0] - own[0], tp[1] - own[1]))
                    return {"action_type": SHOOT, "move_direction": 0,
                            "rotation_angle": _rotate_toward(agent.orientation, ang),
                            "shoot_power_percentage": _kick_power_for_distance(distance(own, tp), agent.sho),
                            "shoot_angle": ang}

        # Otherwise dribble at the goal.
        if agent is None:
            # No squad reference: use the state's heading + forward compass.
            fwd = _attack_angle(team)
            return {"action_type": MOVE, "move_direction": _compass(fwd),
                    "rotation_angle": _rotate_toward(state.get("current_orientation", 0.0), fwd),
                    "shoot_power_percentage": 0.0, "shoot_angle": 0.0}
        return {"action_type": MOVE, "move_direction": _compass(goal_ang),
                "rotation_angle": _rotate_toward(agent.orientation, goal_ang),
                "shoot_power_percentage": 0.0, "shoot_angle": 0.0}

    def _aim_away_from_keeper(self, agent: Agent, state: Dict[str, Any],
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
        target_y = (C.GOAL_Y_MIN + 1.0) if gk_y > C.FIELD_CENTER_Y \
            else (C.GOAL_Y_MAX - 1.0)
        return angle_of_vector((opp_goal[0] - own[0], target_y - own[1]))

    def _best_pass_target(self, agent: Agent, state: Dict[str, Any], team: str):
        opp = _opp_goal(team)
        own = (agent.x, agent.y)
        atk = 1.0 if team == "A" else -1.0
        best = None
        for obj, gp in self._visible_globals(agent, state, only_type="teammate"):
            if obj.get("is_gk"):
                continue
            forward = (gp[0] - own[0]) * atk
            if forward < 2.0:
                continue
            opps = self._visible_globals(agent, state, only_type="opponent")
            opp_min = min((distance(gp, op) for _, op in opps), default=50.0)
            if opp_min < 3.0:
                continue
            score = forward + opp_min * 0.6 - distance(gp, opp) * 0.3
            if best is None or score > best[0]:
                best = (score, gp, obj.get("id"))
        return (best[1], best[2]) if best else None

    # ---- non-carrier outfield ------------------------------------------
    def _outfield(self, state: Dict[str, Any], team: str,
                  ball: Optional[Point], carrier_idx: Optional[int], i: int,
                  have_squad: bool) -> Dict[str, Any]:
        agent = self._agent(state)

        # No squad reference: chase the ball if we can see it, else drift
        # forward (never just spin -- that is the bug this fixes).
        if agent is None or not have_squad:
            sec = self._nearest_ball_sector(state)
            ori = state.get("current_orientation", 0.0)
            if sec is not None and sec.get("distance") is not None:
                target = ori + sec["center_angle"]
                return {"action_type": MOVE, "move_direction": _compass(target),
                        "rotation_angle": _rotate_toward(ori, target),
                        "shoot_power_percentage": 0.0, "shoot_angle": 0.0}
            # Ball not visible: drift toward the attack instead of spinning.
            fwd = _attack_angle(team)
            return {"action_type": MOVE, "move_direction": _compass(fwd),
                    "rotation_angle": _rotate_toward(ori, fwd),
                    "shoot_power_percentage": 0.0, "shoot_angle": 0.0}

        own = (agent.x, agent.y)

        # We have possession -> push into forward support.
        if carrier_idx is not None:
            target = self._support_position(agent, team)
            return self._move_toward(agent, target)

        # Loose ball -> the nearest out-fielder chases it.
        if ball is not None:
            nearest = min(range(len(self.squad)),
                          key=lambda k: distance((self.squad[k].x, self.squad[k].y), ball)
                          if not self.squad[k].is_gk else 1e9)
            if i == nearest:
                return self._move_toward(agent, ball)

        # Otherwise hold a compact defensive shape between ball and own goal.
        target = self._hold_position(agent, team, ball)
        return self._move_toward(agent, target)

    def _support_position(self, agent: Agent, team: str) -> Point:
        atk = 1.0 if team == "A" else -1.0
        x = clamp(agent.home_x + atk * 12.0, 8.0, C.FIELD_X_MAX - 8.0)
        return (x, agent.home_y)

    def _hold_position(self, agent: Agent, team: str,
                       ball: Optional[Point]) -> Point:
        if ball is None:
            return (agent.home_x, agent.home_y)
        y = clamp(agent.home_y + (ball[1] - agent.home_y) * 0.4,
                  6.0, C.FIELD_Y_MAX - 6.0)
        if team == "A":
            x = clamp(min(agent.home_x, ball[0] - 8.0), 4.0, 45.0)
        else:
            x = clamp(max(agent.home_x, ball[0] + 8.0),
                      C.FIELD_X_MAX - 45.0, C.FIELD_X_MAX - 4.0)
        return (x, y)

    def _move_toward(self, agent: Agent, target: Point) -> Dict[str, Any]:
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

    # ---- goalkeeper -----------------------------------------------------
    def _goalkeeper(self, state: Dict[str, Any], team: str,
                    ball: Optional[Point]) -> Dict[str, Any]:
        if state.get("cooldown_remaining", 0) > 0:
            return {"action_type": IDLE}
        agent = self._agent(state)
        # A keeper always knows its own goal-line position.
        gy = state.get("gk_y", 30.0)
        gx = state.get("gk_x", (_own_goal(team)[0] + (3.0 if team == "A" else -3.0)))
        own = (gx, gy)
        target_y = clamp(ball[1] if ball else C.FIELD_CENTER_Y,
                         C.GOAL_Y_MIN - 1.0, C.GOAL_Y_MAX + 1.0)
        dy = target_y - gy

        # Distribute if the keeper has the ball.
        if state.get("has_ball") and agent is not None:
            tgt = self._best_pass_target(agent, state, team)
            if tgt is not None:
                tp, _ = tgt
                ang = angle_of_vector((tp[0] - own[0], tp[1] - own[1]))
                return {"action_type": SHOOT, "move_direction": 0,
                        "rotation_angle": _rotate_toward(state.get("current_orientation", 0.0), ang),
                        "shoot_power_percentage": _kick_power_for_distance(distance(own, tp), agent.sho),
                        "shoot_angle": ang}
            fwd = _attack_angle(team)
            return {"action_type": SHOOT, "move_direction": 0,
                    "rotation_angle": _rotate_toward(state.get("current_orientation", 0.0), fwd),
                    "shoot_power_percentage": 1.0, "shoot_angle": fwd}

        # Dive if a ball is bearing down on the goal.
        if ball is not None:
            og = _own_goal(team)
            on_side = (ball[0] < 20.0 and team == "A") or \
                      (ball[0] > 80.0 and team == "B")
            if on_side and distance(ball, og) < 14.0 and abs(dy) > 2.0:
                move_dir = 0 if (dy > 0 and team == "A") or (dy < 0 and team == "B") else 1
                return {"action_type": DIVE, "move_direction": move_dir,
                        "deflect_angle": _attack_angle(team)}

        # Otherwise shuffle along the goal line to track the ball.
        if abs(dy) < 0.5:
            return {"action_type": IDLE, "move_direction": 0,
                    "rotation_angle": 0.0, "shoot_power_percentage": 0.0,
                    "shoot_angle": 0.0, "deflect_angle": _attack_angle(team)}
        move_dir = 0 if (dy > 0 and team == "A") or (dy < 0 and team == "B") else 1
        return {"action_type": MOVE, "move_direction": move_dir,
                "rotation_angle": 0.0, "shoot_power_percentage": 0.0,
                "shoot_angle": 0.0, "deflect_angle": _attack_angle(team)}

    # ---- small utilities ------------------------------------------------
    def _agent(self, state: Dict[str, Any]) -> Optional[Agent]:
        sid = state.get("self_id")
        for a in self.squad:
            if a.id == sid:
                return a
        return None

    @staticmethod
    def _nearest_ball_sector(state: Dict[str, Any]):
        for s in state.get("vision_sectors", []):
            if s.get("type") == "ball":
                return s
        return None


# ===========================================================================
# 4.  Factory (the engine's only import from this module)
# ===========================================================================
def make_policy(squad: Optional[List[Agent]] = None,
                seed: Optional[int] = None) -> Policy:
    """Factory hook imported by the engine / tournament."""
    return Policy(squad=squad, seed=seed)
