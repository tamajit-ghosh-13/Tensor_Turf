"""
utils.py
========
Vector and angle math helpers used throughout the simulation.

All public helpers accept and return *degrees* at the boundary so the
participant-facing API is consistent with the SRS notation.
"""

from __future__ import annotations

import math
from typing import Tuple

Point = Tuple[float, float]
Vec = Tuple[float, float]


# ---------------------------------------------------------------------------
# Angle utilities (degrees)
# ---------------------------------------------------------------------------
def normalize_angle(angle: float) -> float:
    """Normalize an angle in degrees to the (-180, 180] range."""
    a = (angle + 180.0) % 360.0 - 180.0
    if a == -180.0:
        a = 180.0
    return a


def angle_diff(a: float, b: float) -> float:
    """Smallest signed difference a - b in degrees, wrapped to (-180, 180]."""
    return normalize_angle(a - b)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# Vector utilities
# ---------------------------------------------------------------------------
def distance(p: Point, q: Point) -> float:
    """Euclidean distance d(p, q) = sqrt((px-qx)^2 + (py-qy)^2)."""
    return math.hypot(p[0] - q[0], p[1] - q[1])


def vec_from_angle(deg: float, mag: float = 1.0) -> Vec:
    """Unit-length direction vector for an angle measured in degrees.

    Convention (matches SRS compass): 0deg = East (+X), 90deg = North (+Y).
    """
    r = math.radians(deg)
    return (mag * math.cos(r), mag * math.sin(r))


def angle_of_vector(v: Vec) -> float:
    """Absolute angle (degrees, 0..360) of a vector, East = 0."""
    if v[0] == 0.0 and v[1] == 0.0:
        return 0.0
    return math.degrees(math.atan2(v[1], v[0])) % 360.0


def relative_angle(observer_pos: Point, observer_heading: float,
                   target_pos: Point) -> float:
    """Angle of the target relative to the observer's heading, in degrees.

    Returns a value in (-180, 180].  0 means "directly ahead".
    """
    dx = target_pos[0] - observer_pos[0]
    dy = target_pos[1] - observer_pos[1]
    abs_angle = angle_of_vector((dx, dy))
    return angle_diff(abs_angle, observer_heading)


def add_vec(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1])


def scale_vec(v: Vec, s: float) -> Vec:
    return (v[0] * s, v[1] * s)


def vec_magnitude(v: Vec) -> float:
    return math.hypot(v[0], v[1])


def move_direction_index_to_angle(idx: int) -> float:
    """Convert a compass move_direction index [0,7] to an absolute angle.

    0 = 0deg (East), 1 = 45deg (NE), 2 = 90deg (North), etc. (SRS 5.1).
    """
    idx = idx % 8
    return idx * 45.0


def point_in_goal_zone(x: float, y: float, side: str) -> bool:
    """Whether point (x,y) lies within a goal zone (used for goal detection)."""
    from . import config as C
    if side == "A":  # Team A's goal at X = 0
        return x <= C.TEAM_A_GOAL_X and C.GOAL_Y_MIN <= y <= C.GOAL_Y_MAX
    else:            # Team B's goal at X = 100
        return x >= C.TEAM_B_GOAL_X and C.GOAL_Y_MIN <= y <= C.GOAL_Y_MAX
