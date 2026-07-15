"""
actions.py
==========
Action execution for the FIFA AI World Cup action space (Section 5).

Outfield action interface (Section 5.1):
    { action_type, move_direction, rotation_angle,
      shoot_power_percentage, shoot_angle }
    action_type in {MOVE, ROTATE, SHOOT, IDLE}

Goalkeeper action interface (Section 5.2):
    { action_type, move_direction, deflect_angle }
    action_type in {MOVE, ROTATE, SHOOT, DIVE, DEFLECT, IDLE}

Key rules implemented here:
  * Movement-Shot Mutex (6.1): a SHOOT zeros the agent's movement this tick.
  * Rotation is clamped to [-30, 30] degrees per tick (5.1).
  * Goalkeeper dive burst (6.3) and mandatory cooldown.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Tuple

from . import config as C
from .agent import Agent
from .ball import Ball
from .utils import (
    clamp,
    move_direction_index_to_angle,
    normalize_angle,
    vec_from_angle,
)

# Outfield action types
MOVE = "MOVE"
ROTATE = "ROTATE"
SHOOT = "SHOOT"
IDLE = "IDLE"
# Goalkeeper-only action types
DIVE = "DIVE"
DEFLECT = "DEFLECT"


def normalize_action(agent: Agent, raw: Dict[str, Any] | None) -> Dict[str, Any]:
    """Coerce a raw policy action into a safe, complete action dict."""
    if not isinstance(raw, dict):
        return {"action_type": IDLE}

    action = dict(raw)
    action_type = str(action.get("action_type", IDLE)).upper()
    if agent.is_gk and action_type not in (MOVE, ROTATE, SHOOT, DIVE, DEFLECT, IDLE):
        action_type = IDLE
    if not agent.is_gk and action_type not in (MOVE, ROTATE, SHOOT, IDLE):
        action_type = IDLE
    action["action_type"] = action_type

    # Common numeric fields with safe defaults / clamping.
    action["move_direction"] = int(action.get("move_direction", 0) or 0)
    action["rotation_angle"] = clamp(
        float(action.get("rotation_angle", 0.0) or 0.0),
        -C.ROTATION_CLAMP, C.ROTATION_CLAMP,
    )
    action["shoot_power_percentage"] = clamp(
        float(action.get("shoot_power_percentage", 0.0) or 0.0), 0.0, 1.0
    )
    action["shoot_angle"] = float(action.get("shoot_angle", 0.0) or 0.0)
    action["deflect_angle"] = float(action.get("deflect_angle", 0.0) or 0.0)
    # Distinguish a shot (uses shot_power attribute) from a pass (uses
    # pass_range attribute).  Anything not explicitly "pass" is a shot.
    kick_type = str(action.get("kick_type", "shot")).lower()
    action["kick_type"] = "pass" if kick_type == "pass" else "shot"
    return action


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clamp_to_pitch(agent: Agent) -> None:
    agent.x = clamp(agent.x, C.FIELD_X_MIN + 0.5, C.FIELD_X_MAX - 0.5)
    agent.y = clamp(agent.y, C.FIELD_Y_MIN + 0.5, C.FIELD_Y_MAX - 0.5)


def _gk_lateral_direction(agent: Agent, move_direction: int) -> float:
    """Sign of Y movement for a goalkeeper MOVE/DIVE.

    move_direction: 0 = Left relative to net, 1 = Right relative to net.
    Team A GK (defends X=0, faces East):  Left -> +Y, Right -> -Y.
    Team B GK (defends X=100, faces West): Left -> -Y, Right -> +Y.
    """
    if agent.team == "A":
        return 1.0 if move_direction == 0 else -1.0
    else:
        return -1.0 if move_direction == 0 else 1.0


def _face_ball(agent: Agent, ball: Ball) -> None:
    """Goalkeepers automatically track the ball with their heading."""
    from .utils import angle_of_vector
    if (agent.x, agent.y) != (ball.x, ball.y):
        agent.orientation = angle_of_vector(
            (ball.x - agent.x, ball.y - agent.y)
        )


# ---------------------------------------------------------------------------
# Action application
# ---------------------------------------------------------------------------
def apply_action(agent: Agent, raw_action: Dict[str, Any], ball: Ball,
                 rng: random.Random) -> Dict[str, Any]:
    """Execute a single agent's action for the current tick.

    Returns a small telemetry dict (``{"shot": bool, "dove": bool}``) that the
    engine uses for interception / ball-follow bookkeeping.
    """
    telemetry = {"shot": False, "dove": False, "moved": False}
    action = normalize_action(agent, raw_action)
    atype = action["action_type"]

    # Goalkeepers in cooldown are frozen (Section 6.3): no rotate/move/pass.
    if agent.is_gk and agent.cooldown_remaining > 0:
        # Only DEFLECT (setting an angle) is permitted during freeze; we still
        # honour it because it is purely informational state.
        if atype == DEFLECT:
            agent.deflect_angle = action["deflect_angle"]
        _face_ball(agent, ball)
        return telemetry

    if agent.is_gk:
        _face_ball(agent, ball)

    if atype == IDLE:
        return telemetry

    if atype == ROTATE:
        agent.orientation = normalize_angle(
            agent.orientation + action["rotation_angle"]
        )
        return telemetry

    if atype == MOVE:
        _apply_move(agent, action, ball)
        telemetry["moved"] = True
        return telemetry

    if atype == SHOOT:
        # Movement-Shot Mutex (6.1): kicker is stationary this tick.  The
        # telemetry ``shot`` flag reflects a *real* kick (ball actually left
        # the agent), so a whiff without possession is not mistaken for a
        # touch by the engine's last-touch / offside bookkeeping.
        telemetry["shot"] = _apply_shoot(agent, action, ball, rng)
        return telemetry

    if atype == DEFLECT:  # GK only
        agent.deflect_angle = action["deflect_angle"]
        return telemetry

    if atype == DIVE:  # GK only
        _apply_dive(agent, action, ball)
        telemetry["dove"] = True
        return telemetry

    return telemetry


def _apply_move(agent: Agent, action: Dict[str, Any], ball: Ball) -> None:
    """MOVE: travel in ``move_direction`` and rotate the heading."""
    # Rotation is always permitted alongside a MOVE.
    agent.orientation = normalize_angle(
        agent.orientation + action["rotation_angle"]
    )

    # Dribbling (carrying the ball) uses the slower dribble speed; free running
    # uses the running speed.
    speed = agent.dribble_speed() if agent.has_ball else agent.move_speed()
    if agent.is_gk:
        sign = _gk_lateral_direction(agent, action["move_direction"])
        agent.y += sign * speed
    else:
        angle = move_direction_index_to_angle(action["move_direction"])
        dx, dy = vec_from_angle(angle, speed)
        agent.x += dx
        agent.y += dy

    _clamp_to_pitch(agent)


def _apply_shoot(agent: Agent, action: Dict[str, Any], ball: Ball,
                 rng: random.Random) -> bool:
    """SHOOT: kick the ball; the kicker is stationary this tick (6.1).

    Returns True if the ball was actually kicked (the agent had possession),
    False on a whiff (no possession) so the engine does not record a touch.
    """
    if not agent.has_ball:
        # Cannot shoot without possession; treat as a whiff (no-op).
        return False

    power = action["shoot_power_percentage"]
    angle = action["shoot_angle"]
    # A pass draws its ball speed from pass_range; a shot from shot_power and
    # carries extra spread so it is not an automatic goal.
    is_pass = action.get("kick_type") == "pass"
    power_attr = agent.prg if is_pass else agent.sho
    noise = agent.pass_noise() + (0.0 if is_pass else C.SHOT_EXTRA_NOISE)
    ball.kick(angle, power, power_attr, noise, rng)
    agent.has_ball = False
    ball.possessed_by = None
    return True


def _apply_dive(agent: Agent, action: Dict[str, Any], ball: Ball) -> None:
    """DIVE: goalkeeper bursts for 1 tick (distance scales with agility)."""
    sign = _gk_lateral_direction(agent, action["move_direction"])
    agent.y += sign * agent.gk_dive_speed()
    _clamp_to_pitch(agent)
    agent.cooldown_remaining = agent.dive_cooldown()
