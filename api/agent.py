"""
agent.py
========
Agent dataclass and squad generation with star-player balance guardrails.

Section 3 (Agent Attributes & Squad Allocations):
  * Each participant gets a squad of 11 unique agents.
  * Outfield players (positions 1-10) have SHO, SPD, DEF, PASS (1.0-10.0)
    and VIS (30.0-120.0 degrees).
  * The Goalkeeper (position 11) has REF, POS, PASS.
  * Exactly one non-goalkeeper is the "Star Player" with average rating >= 6.0
    and one priority stat pegged at 10.0 by archetype.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

from . import config as C
from .utils import clamp


# Outfield position archetypes map to the attribute that gets boosted to 10.0
# when the holder is the designated star player (Section 3.3).
ARCHETYPES = {
    "GK":        "REF",   # goalkeepers are never stars (SRS: non-GK role)
    "DEFENDER":  "DEF",
    "MIDFIELDER":"PASS",
    "ATTACKER":  "SHO",
}

# A 4-3-3 formation: 1 GK + 4 defenders + 3 midfielders + 3 attackers.
FORMATION = [
    "GK",
    "DEFENDER", "DEFENDER", "DEFENDER", "DEFENDER",
    "MIDFIELDER", "MIDFIELDER", "MIDFIELDER",
    "ATTACKER", "ATTACKER", "ATTACKER",
]


@dataclass
class Agent:
    """A single reinforcement-learning entity on the pitch."""

    id: str
    team: str                 # "A" or "B"
    index: int                # 0..10 within the squad
    archetype: str            # GK / DEFENDER / MIDFIELDER / ATTACKER
    is_gk: bool

    # Outfield attributes (Section 3.1).  Goalkeepers keep these at defaults.
    # These are internal 1..10 values derived from the participant's knobs.
    sho: float = 5.0       # shot power       (knob: shot_power)
    spd: float = 5.0       # running speed    (knob: running)
    def_: float = 5.0      # tackling/defend  (knob: tackling)
    pass_: float = 5.0     # passing accuracy (knob: pass_accuracy)
    vis: float = 75.0      # vision aperture  (knob: vision -> degrees)
    drb: float = 5.0       # dribbling speed  (knob: dribbling)
    prg: float = 5.0       # pass range/power (knob: pass_range)
    vis_range: float = C.VISION_RANGE  # sight distance (knob: vision -> units)

    # Goalkeeper attributes (Section 3.2).  Outfield players keep defaults.
    ref: float = 5.0        # reflexes / dive recovery  (knob: reflexes)
    pos: float = 5.0        # positioning / save radius (knob: positioning)
    gk_agility: float = 5.0  # dive burst distance      (knob: agility)
    gk_handling: float = 5.0  # catch-vs-parry on saves (knob: handling)
    gk_dist: float = 5.0    # distribution power/accuracy (knob: distribution)

    is_star: bool = False

    # ---- Dynamic (per-match) state ----------------------------------------
    x: float = 0.0
    y: float = 0.0
    orientation: float = 0.0     # heading in degrees, 0 = East
    has_ball: bool = False
    cooldown_remaining: int = 0
    deflect_angle: float = 0.0   # GK: angle to redirect blocked shots

    # The home position the agent tries to return to when idle.
    home_x: float = 0.0
    home_y: float = 0.0
    # True when home_x/home_y come from a participant's custom formation, so
    # place_kickoff restores those instead of the built-in default shape.
    home_set: bool = False

    # ------------------------------------------------------------------ ratings
    def average_rating(self) -> float:
        """Average of the agent's *relevant* attributes (used for star guard)."""
        if self.is_gk:
            return (self.ref + self.pos + self.pass_) / 3.0
        return (self.sho + self.spd + self.def_ + self.pass_ + self.vis / 12.0) / 5.0

    def move_speed(self) -> float:
        """Running speed: displacement per tick when NOT carrying the ball."""
        if self.is_gk:
            return C.GK_BASE_SPEED
        return C.BASE_MOVE_SPEED * self.spd

    def dribble_speed(self) -> float:
        """Dribbling speed: displacement per tick WHILE carrying the ball."""
        if self.is_gk:
            return C.GK_BASE_SPEED
        return C.BASE_MOVE_SPEED * self.drb

    def gk_dive_speed(self) -> float:
        """Goalkeeper dive burst distance, scaled by the agility knob."""
        return C.GK_DIVE_MULTIPLIER * C.GK_BASE_SPEED * (self.gk_agility / 5.0)

    def gk_catch_probability(self) -> float:
        """Probability a GK save is cleanly caught (else parried loose)."""
        return max(0.15, min(0.95, 0.2 + 0.075 * self.gk_handling))

    def intercept_radius(self) -> float:
        """Passive interception envelope (Section 6.2)."""
        return C.INTERCEPT_RADIUS_BASE * (1.0 + self.def_ / 10.0)

    def gk_block_radius(self) -> float:
        """Automatic save radius around the keeper (POS), widened for balance."""
        # Mirrors the intercept formula but keyed on POS and scaled up so the
        # keeper can realistically deny shots (GK_SAVE_RADIUS_SCALE).
        return (C.INTERCEPT_RADIUS_BASE * (1.0 + self.pos / 10.0)
                * C.GK_SAVE_RADIUS_SCALE)

    def dive_speed(self) -> float:
        """Burst speed for a goalkeeper dive (Section 6.3)."""
        return C.GK_DIVE_MULTIPLIER * C.GK_BASE_SPEED

    def dive_cooldown(self) -> int:
        """Mandatory freeze ticks after a dive: max(5, round(35 - REF))."""
        return max(5, round(35 - self.ref))

    def pass_noise(self) -> float:
        """Angular spread (degrees) added to kicks, reduced by PASS."""
        return max(0.0, C.PASS_NOISE_BASE - self.pass_)

    def vision_aperture(self) -> float:
        """Field-of-view aperture in degrees (the VIS attribute)."""
        return self.vis

    # ------------------------------------------------------------------ resets
    def reset_dynamic_state(self) -> None:
        self.has_ball = False
        self.cooldown_remaining = 0
        self.orientation = self._default_orientation()
        self.deflect_angle = self._default_orientation()

    def _default_orientation(self) -> float:
        """Team A faces East (toward B's goal), Team B faces West."""
        return 0.0 if self.team == "A" else 180.0

    def snapshot_position(self):
        return (self.x, self.y)


# ---------------------------------------------------------------------------
# Squad generation
# ---------------------------------------------------------------------------
def _rand_attr(rng: random.Random, low: float = C.ATTR_MIN,
             high: float = C.ATTR_MAX) -> float:
    """Random attribute in [low, high] rounded to one decimal."""
    return round(rng.uniform(low, high), 1)


def _make_outfield(team: str, index: int, archetype: str,
                   is_star: bool, rng: random.Random) -> Agent:
    agent = Agent(
        id=f"Team{team}_Player_{index + 1}",
        team=team,
        index=index,
        archetype=archetype,
        is_gk=False,
    )

    # Base random capabilities (Section 3 intro: "randomly initialized with
    # base capabilities").
    agent.sho = _rand_attr(rng)
    agent.spd = _rand_attr(rng)
    agent.def_ = _rand_attr(rng)
    agent.pass_ = _rand_attr(rng)
    agent.vis = round(rng.uniform(C.VIS_MIN, C.VIS_MAX), 1)

    if is_star:
        agent.is_star = True
        boost_attr = ARCHETYPES[archetype]
        # Peg the priority stat at maximum capacity (10.0).
        setattr(agent, boost_attr, C.STAR_BOOST_VALUE)
        # Guarantee an average rating >= 6.0 across the standard attributes.
        _enforce_star_floor(agent)

    return agent


def _enforce_star_floor(agent: Agent) -> None:
    """Raise the star player's other attributes until avg rating >= 6.0."""
    while agent.average_rating() < C.STAR_AVG_MIN:
        # Raise the weakest non-boosted outfield attribute.
        attrs = ["sho", "spd", "def_", "pass_"]
        # Skip the boosted attribute (already 10.0).
        boost_attr = ARCHETYPES[agent.archetype]
        candidates = [a for a in attrs if a != boost_attr]
        weakest = min(candidates, key=lambda a: getattr(agent, a))
        current = getattr(agent, weakest)
        setattr(agent, weakest, min(C.ATTR_MAX, round(current + 0.5, 1)))
        # Also widen vision if it is pulling the average down.
        if agent.vis < C.VIS_MAX:
            agent.vis = min(C.VIS_MAX, round(agent.vis + 5.0, 1))
        if all(getattr(agent, a) >= C.ATTR_MAX for a in candidates):
            break


def _make_goalkeeper(team: str, index: int, rng: random.Random) -> Agent:
    agent = Agent(
        id=f"Team{team}_Player_{index + 1}",
        team=team,
        index=index,
        archetype="GK",
        is_gk=True,
        ref=_rand_attr(rng),
        pos=_rand_attr(rng),
        pass_=_rand_attr(rng),
        # Goalkeepers keep outfield attributes at neutral defaults.
    )
    return agent


def generate_squad(team: str, star_index: Optional[int] = None,
                   rng: Optional[random.Random] = None) -> List[Agent]:
    """Generate a full 11-agent squad for ``team`` ("A" or "B").

    Exactly one outfield player is the designated star (Section 3.3).  Pass a
    ``rng`` for reproducible rosters; if omitted the module-level RNG is used.
    """
    rng = rng if rng is not None else random
    squad: List[Agent] = []
    outfield_indices = [i for i, role in enumerate(FORMATION) if role != "GK"]
    if star_index is None:
        star_index = rng.choice(outfield_indices)

    for i, role in enumerate(FORMATION):
        if role == "GK":
            squad.append(_make_goalkeeper(team, i, rng))
        else:
            squad.append(_make_outfield(team, i, role, is_star=(i == star_index),
                                        rng=rng))
    return squad


# ---------------------------------------------------------------------------
# Kickoff formation
# ---------------------------------------------------------------------------
# Formation coordinates for Team A (attacking East, defending X=0).
# Team B is mirrored horizontally about the centre line (x -> 100 - x).
_TEAM_A_FORMATION = [
    # (index, x, y)   - GK + 4-3-3
    (0,  3.0, 30.0),    # GK
    (1, 18.0, 12.0),    # LB
    (2, 18.0, 24.0),    # CB
    (3, 18.0, 36.0),    # CB
    (4, 18.0, 48.0),    # RB
    (5, 35.0, 18.0),    # LM
    (6, 35.0, 30.0),    # CM
    (7, 35.0, 42.0),    # RM
    (8, 45.0, 20.0),    # LW
    (9, 47.0, 30.0),    # ST
    (10, 45.0, 40.0),   # RW
]


def place_kickoff(squad: List[Agent], team: str) -> None:
    """Position a squad in its kickoff configuration and reset dynamic state.

    A squad built from a participant's custom formation (``home_set``) is
    restored to those home positions; otherwise the built-in default 4-3-3 is
    applied (and becomes each agent's home).
    """
    if squad and all(a.home_set for a in squad):
        for agent in squad:
            agent.x, agent.y = agent.home_x, agent.home_y
            agent.reset_dynamic_state()
        return

    for idx, x, y in _TEAM_A_FORMATION:
        agent = squad[idx]
        if team == "A":
            agent.x, agent.y = x, y
        else:
            # Mirror horizontally about the centre line.
            agent.x, agent.y = C.FIELD_X_MAX - x, y
        agent.home_x, agent.home_y = agent.x, agent.y
        agent.reset_dynamic_state()


def squad_value(squad: List[Agent]) -> int:
    """Transfer-market value of a squad (star = 200, others = 100)."""
    return sum(C.STAR_PLAYER_COST if a.is_star else C.NORMAL_PLAYER_COST
               for a in squad)
