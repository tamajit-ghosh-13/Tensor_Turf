"""
ball.py
=======
Independent ball physics class (Section 2.2).

The ball is a non-agent physical entity governed by:
  * Velocity friction decay (mu = 0.96).
  * A stop threshold (||V|| < 0.05 -> (0, 0)).
  * Boundary bounce-back (walls retain 0.8 of the impacted component).
  * Goal detection within Y in [25, 35] at the X extremes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from . import config as C
from .utils import distance, vec_magnitude


@dataclass
class Ball:
    """The single ball entity shared by both teams."""

    x: float = C.FIELD_CENTER_X
    y: float = C.FIELD_CENTER_Y
    vx: float = 0.0
    vy: float = 0.0
    possessed_by: Optional[str] = None   # agent.id when carried, else None

    # ---- queries -----------------------------------------------------------
    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def velocity(self) -> Tuple[float, float]:
        return (self.vx, self.vy)

    @property
    def speed(self) -> float:
        return vec_magnitude(self.velocity)

    @property
    def is_free(self) -> bool:
        return self.possessed_by is None

    def distance_to(self, point) -> float:
        return distance(self.position, point)

    # ------------------------------------------------------------------ physics
    def step_physics(self) -> Optional[Dict[str, Any]]:
        """Advance the ball one tick and resolve boundaries.

        Returns a small descriptor when the ball leaves normal play this tick:
          * ``{"type": "goal", "team": "A"|"B"}`` — a goal was scored.
          * ``{"type": "out", "boundary": <str>, "x": float, "y": float}`` —
            the ball fully crossed a line (only when set pieces are enabled),
            where ``boundary`` is one of ``byline_left`` / ``byline_right`` /
            ``touchline_top`` / ``touchline_bottom``.
        Otherwise returns ``None``.  The ball is only stepped when it is *free*;
        a possessed ball is repositioned by the engine to follow its carrier.
        """
        if not self.is_free:
            return None

        # 1. Integrate position.
        self.x += self.vx
        self.y += self.vy

        event = self._resolve_boundaries()

        # 2. Friction decay (Section 2.2).  Applied after movement & bounces.
        self.vx *= C.BALL_FRICTION
        self.vy *= C.BALL_FRICTION
        if self.speed < C.BALL_STOP_THRESHOLD:
            self.vx = 0.0
            self.vy = 0.0

        return event

    def _resolve_boundaries(self) -> Optional[Dict[str, Any]]:
        """Detect goals, out-of-play and (legacy) wall bounces.

        Goals are always detected.  When ``SET_PIECES_ENABLED`` is True a line
        crossing outside the goal mouth stops the ball and returns an ``out``
        descriptor for the engine to restart with a set piece; otherwise the
        legacy energy-retaining wall bounce is used.
        """
        # --- X walls (left / right): goals take priority over everything ---
        if self.x <= C.FIELD_X_MIN and C.GOAL_Y_MIN <= self.y <= C.GOAL_Y_MAX:
            self._reset_kickoff()
            return {"type": "goal", "team": "B"}  # Team B scores in A's goal
        if self.x >= C.FIELD_X_MAX and C.GOAL_Y_MIN <= self.y <= C.GOAL_Y_MAX:
            self._reset_kickoff()
            return {"type": "goal", "team": "A"}  # Team A scores in B's goal

        if not C.SET_PIECES_ENABLED:
            return self._legacy_bounce()

        # --- Out of play (set pieces enabled) ------------------------------
        # Byline crossings are reported before touchline crossings so a ball
        # that exits near a corner is treated as a corner / goal kick.
        if self.x <= C.FIELD_X_MIN:
            return self._out("byline_left", C.FIELD_X_MIN)
        if self.x >= C.FIELD_X_MAX:
            return self._out("byline_right", C.FIELD_X_MAX)
        if self.y <= C.FIELD_Y_MIN:
            return self._out("touchline_bottom", None, C.FIELD_Y_MIN)
        if self.y >= C.FIELD_Y_MAX:
            return self._out("touchline_top", None, C.FIELD_Y_MAX)
        return None

    def _out(self, boundary: str, clamp_x: Optional[float] = None,
             clamp_y: Optional[float] = None) -> Dict[str, Any]:
        """Freeze the ball on the line it crossed and describe the exit."""
        exit_x = clamp_x if clamp_x is not None else \
            min(max(self.x, C.FIELD_X_MIN), C.FIELD_X_MAX)
        exit_y = clamp_y if clamp_y is not None else \
            min(max(self.y, C.FIELD_Y_MIN), C.FIELD_Y_MAX)
        self.x, self.y = exit_x, exit_y
        self.vx = self.vy = 0.0
        return {"type": "out", "boundary": boundary, "x": exit_x, "y": exit_y}

    def _legacy_bounce(self) -> None:
        """Original energy-retaining wall bounce (set pieces disabled)."""
        if self.y <= C.FIELD_Y_MIN:
            self.y = C.FIELD_Y_MIN
            self.vy = -self.vy * C.WALL_BOUNCE_FACTOR
        elif self.y >= C.FIELD_Y_MAX:
            self.y = C.FIELD_Y_MAX
            self.vy = -self.vy * C.WALL_BOUNCE_FACTOR
        if self.x <= C.FIELD_X_MIN:
            self.x = C.FIELD_X_MIN
            self.vx = -self.vx * C.WALL_BOUNCE_FACTOR
        elif self.x >= C.FIELD_X_MAX:
            self.x = C.FIELD_X_MAX
            self.vx = -self.vx * C.WALL_BOUNCE_FACTOR
        return None

    def _reset_kickoff(self) -> None:
        """Reset the ball to the centre after a goal (Section 2.2)."""
        self.x = C.FIELD_CENTER_X
        self.y = C.FIELD_CENTER_Y
        self.vx = 0.0
        self.vy = 0.0
        self.possessed_by = None

    def reset(self, x: float = C.FIELD_CENTER_X,
              y: float = C.FIELD_CENTER_Y) -> None:
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.possessed_by = None

    def kick(self, angle_deg: float, power: float, sho: float,
             pass_noise: float, rng) -> None:
        """Impart velocity to the ball from a kick.

        ``power`` is the shoot_power_percentage in [0, 1]; ``sho`` is the
        kicker's Shot Power attribute; ``pass_noise`` is the angular spread
        (degrees) reduced by the kicker's PASS attribute.
        """
        from .utils import vec_from_angle
        # Section 3.1 SHO: scales initial velocity on kicks.
        speed = power * sho * C.SHOOT_SPEED_SCALE
        noisy_angle = angle_deg
        if pass_noise > 0.0 and speed > 0.0:
            noisy_angle += rng.uniform(-pass_noise, pass_noise)
        dx, dy = vec_from_angle(noisy_angle, speed)
        self.vx = dx
        self.vy = dy
        self.possessed_by = None
