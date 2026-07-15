"""
skills.py
=========
Participant squad design: the **knob** schema, the per-player point **budget**,
custom **formations**, and construction of a live squad from a ``squad.json``
spec.

A submission's ``squad.json`` is a list of exactly 11 player entries. Slot 0 is
the goalkeeper; slots 1..10 are outfielders. Each entry carries a home position
``(x, y)`` — in the team's own-half frame, so any formation is expressible —
plus that player's knob allocation:

    [
      {"role": "GK", "x": 4, "y": 30,
       "reflexes": 80, "positioning": 80, "agility": 60,
       "handling": 50, "distribution": 30},              # sum <= GK_BUDGET (300)
      {"role": "DEFENDER", "x": 16, "y": 12,
       "running": 70, "dribbling": 40, "shot_power": 20,
       "pass_accuracy": 60, "pass_range": 50,
       "tackling": 100, "vision": 80},                    # sum <= OUTFIELD_BUDGET (420)
      ... 9 more outfielders ...
    ]

Knobs are integers in ``[KNOB_MIN, KNOB_MAX]`` and convert to internal 1..10
attributes via ``knob_to_attr``. Validation is strict: a spec that busts the
budget, omits a knob, or places a player out of bounds raises ``ValueError`` —
the backend surfaces that to the participant at upload time.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from . import config as C
from .agent import Agent, ARCHETYPES, FORMATION
from .utils import clamp


# --------------------------------------------------------------------------- #
# Knob -> attribute conversions
# --------------------------------------------------------------------------- #
def knob_to_attr(knob: float) -> float:
    """Map a 0..100 knob to a 1..10 physics attribute."""
    return round(1.0 + 9.0 * (knob / 100.0), 3)


def knob_to_vision_deg(knob: float) -> float:
    return round(C.VIS_MIN + (C.VIS_MAX - C.VIS_MIN) * (knob / 100.0), 2)


def knob_to_vision_range(knob: float) -> float:
    return round(C.VIS_RANGE_MIN
                 + (C.VIS_RANGE_MAX - C.VIS_RANGE_MIN) * (knob / 100.0), 2)


# --------------------------------------------------------------------------- #
# Formation presets (Team A own-half frame: x in [0,50], attacking +X)
# --------------------------------------------------------------------------- #
# Each preset is 11 (x, y) home positions: slot 0 = GK, then the outfield shape.
FORMATION_PRESETS: Dict[str, List[tuple]] = {
    "4-3-3": [(4, 30), (16, 12), (16, 24), (16, 36), (16, 48),
              (32, 18), (32, 30), (32, 42), (46, 20), (48, 30), (46, 40)],
    "4-4-2": [(4, 30), (16, 12), (16, 24), (16, 36), (16, 48),
              (32, 12), (32, 24), (32, 36), (32, 48), (46, 24), (46, 36)],
    "3-5-2": [(4, 30), (16, 18), (16, 30), (16, 42),
              (30, 8), (30, 22), (32, 30), (30, 38), (30, 52), (46, 24), (46, 36)],
    "3-4-3": [(4, 30), (16, 18), (16, 30), (16, 42),
              (32, 12), (32, 24), (32, 36), (32, 48), (46, 20), (48, 30), (46, 40)],
    "4-2-3-1": [(4, 30), (16, 12), (16, 24), (16, 36), (16, 48),
                (28, 22), (28, 38), (40, 16), (40, 30), (40, 44), (50, 30)],
}
_OUTFIELD_ROLES_BY_SLOT = FORMATION  # index -> GK/DEFENDER/MIDFIELDER/ATTACKER


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _check_knobs(entry: Dict[str, Any], keys: tuple, budget: int,
                 where: str) -> int:
    total = 0
    for k in keys:
        if k not in entry:
            raise ValueError(f"{where}: missing knob {k!r}")
        v = entry[k]
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(f"{where}: knob {k!r} must be a number")
        if not (C.KNOB_MIN <= v <= C.KNOB_MAX):
            raise ValueError(
                f"{where}: knob {k!r}={v} out of range "
                f"[{C.KNOB_MIN}, {C.KNOB_MAX}]")
        total += v
    if total > budget:
        raise ValueError(
            f"{where}: knob total {total} exceeds budget {budget}")
    return total


def _check_position(entry: Dict[str, Any], is_gk: bool, where: str) -> None:
    try:
        x, y = float(entry["x"]), float(entry["y"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{where}: needs numeric 'x' and 'y'")
    x_lo, x_hi = ((C.GK_HOME_X_MIN, C.GK_HOME_X_MAX) if is_gk
                  else (C.HOME_X_MIN, C.HOME_X_MAX))
    if not (x_lo <= x <= x_hi):
        raise ValueError(f"{where}: x={x} out of bounds [{x_lo}, {x_hi}]")
    if not (C.HOME_Y_MIN <= y <= C.HOME_Y_MAX):
        raise ValueError(
            f"{where}: y={y} out of bounds [{C.HOME_Y_MIN}, {C.HOME_Y_MAX}]")


def validate_squad_spec(spec: List[Dict[str, Any]]) -> None:
    """Raise ``ValueError`` if ``spec`` is not a legal 11-player squad."""
    if not isinstance(spec, list) or len(spec) != C.SQUAD_SIZE:
        raise ValueError(f"squad must be a list of {C.SQUAD_SIZE} players")
    # Slot 0 is the goalkeeper; slots 1..10 are outfielders.
    _check_knobs(spec[0], C.GK_KNOBS, C.GK_BUDGET, "slot 0 (GK)")
    _check_position(spec[0], is_gk=True, where="slot 0 (GK)")
    if str(spec[0].get("role", "GK")).upper() != "GK":
        raise ValueError("slot 0 must be the goalkeeper (role 'GK')")
    for i in range(1, C.SQUAD_SIZE):
        where = f"slot {i}"
        if str(spec[i].get("role", "")).upper() == "GK":
            raise ValueError(f"{where}: only slot 0 may be the goalkeeper")
        _check_knobs(spec[i], C.OUTFIELD_KNOBS, C.OUTFIELD_BUDGET, where)
        _check_position(spec[i], is_gk=False, where=where)


# --------------------------------------------------------------------------- #
# Spec -> squad
# --------------------------------------------------------------------------- #
def _apply_outfield_knobs(agent: Agent, e: Dict[str, Any]) -> None:
    agent.spd = knob_to_attr(e["running"])
    agent.drb = knob_to_attr(e["dribbling"])
    agent.sho = knob_to_attr(e["shot_power"])
    agent.pass_ = knob_to_attr(e["pass_accuracy"])
    agent.prg = knob_to_attr(e["pass_range"])
    agent.def_ = knob_to_attr(e["tackling"])
    agent.vis = knob_to_vision_deg(e["vision"])
    agent.vis_range = knob_to_vision_range(e["vision"])


def _apply_gk_knobs(agent: Agent, e: Dict[str, Any]) -> None:
    agent.ref = knob_to_attr(e["reflexes"])
    agent.pos = knob_to_attr(e["positioning"])
    agent.gk_agility = knob_to_attr(e["agility"])
    agent.gk_handling = knob_to_attr(e["handling"])
    agent.gk_dist = knob_to_attr(e["distribution"])
    # A keeper still needs a passing baseline for its distribution kicks.
    agent.pass_ = agent.gk_dist
    agent.sho = agent.gk_dist


def spec_to_squad(team: str, spec: List[Dict[str, Any]],
                  validate: bool = True) -> List[Agent]:
    """Build a live 11-agent squad for ``team`` from a validated ``squad.json``.

    Home positions are given in the own-half frame and mirrored for team B.
    """
    if validate:
        validate_squad_spec(spec)

    squad: List[Agent] = []
    for i, e in enumerate(spec):
        is_gk = (i == 0)
        archetype = "GK" if is_gk else str(
            e.get("role", _OUTFIELD_ROLES_BY_SLOT[i])).upper()
        if archetype not in ARCHETYPES:
            archetype = _OUTFIELD_ROLES_BY_SLOT[i]
        agent = Agent(id=f"Team{team}_Player_{i + 1}", team=team, index=i,
                      archetype=archetype, is_gk=is_gk)
        if is_gk:
            _apply_gk_knobs(agent, e)
        else:
            _apply_outfield_knobs(agent, e)

        hx = clamp(float(e["x"]), C.HOME_X_MIN, C.HOME_X_MAX)
        hy = clamp(float(e["y"]), C.HOME_Y_MIN, C.HOME_Y_MAX)
        if is_gk:
            hx = clamp(float(e["x"]), C.GK_HOME_X_MIN, C.GK_HOME_X_MAX)
        if team == "B":
            hx = C.FIELD_X_MAX - hx    # mirror about the halfway line
        agent.home_x, agent.home_y = hx, hy
        agent.x, agent.y = hx, hy
        agent.home_set = True
        squad.append(agent)
    return squad


# --------------------------------------------------------------------------- #
# Default / helper specs
# --------------------------------------------------------------------------- #
def default_spec(formation: str = "4-3-3") -> List[Dict[str, Any]]:
    """A balanced, budget-legal spec: every knob at cap / n (evenly split)."""
    positions = FORMATION_PRESETS.get(formation, FORMATION_PRESETS["4-3-3"])
    out_val = C.OUTFIELD_BUDGET // len(C.OUTFIELD_KNOBS)   # 60
    gk_val = C.GK_BUDGET // len(C.GK_KNOBS)                 # 60
    spec: List[Dict[str, Any]] = []
    for i, (x, y) in enumerate(positions):
        if i == 0:
            entry: Dict[str, Any] = {"role": "GK", "x": x, "y": y}
            entry.update({k: gk_val for k in C.GK_KNOBS})
        else:
            entry = {"role": _OUTFIELD_ROLES_BY_SLOT[i], "x": x, "y": y}
            entry.update({k: out_val for k in C.OUTFIELD_KNOBS})
        spec.append(entry)
    return spec


def spec_summary(spec: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Small human/UI summary: per-player totals and remaining budget."""
    out = []
    for i, e in enumerate(spec):
        keys = C.GK_KNOBS if i == 0 else C.OUTFIELD_KNOBS
        budget = C.GK_BUDGET if i == 0 else C.OUTFIELD_BUDGET
        used = sum(e.get(k, 0) for k in keys)
        out.append({"slot": i, "role": e.get("role"),
                    "used": used, "budget": budget,
                    "remaining": budget - used})
    return {"players": out}
