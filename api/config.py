"""
config.py
=========
Global simulation constants for the FIFA AI World Cup environment.

All spatial units are abstract "pitch units".  Time is measured in discrete
simulation *ticks*.  Angles are expressed in degrees at the API boundary
(participant-facing) and converted to radians internally where convenient.
"""

import math

# ---------------------------------------------------------------------------
# 2.1 Spatial Dimensions
# ---------------------------------------------------------------------------
FIELD_X_MIN: float = 0.0
FIELD_X_MAX: float = 100.0
FIELD_Y_MIN: float = 0.0
FIELD_Y_MAX: float = 60.0

FIELD_WIDTH: float = FIELD_X_MAX - FIELD_X_MIN   # 100.0
FIELD_HEIGHT: float = FIELD_Y_MAX - FIELD_Y_MIN  # 60.0

FIELD_CENTER_X: float = (FIELD_X_MIN + FIELD_X_MAX) / 2.0  # 50.0
FIELD_CENTER_Y: float = (FIELD_Y_MIN + FIELD_Y_MAX) / 2.0  # 30.0

# Goal zones (Section 2.1)
GOAL_Y_MIN: float = 25.0
GOAL_Y_MAX: float = 35.0
GOAL_DEPTH: float = GOAL_Y_MAX - GOAL_Y_MIN  # 10.0

# Team A defends the left goal (X = 0.0), attacks toward the right.
# Team B defends the right goal (X = 100.0), attacks toward the left.
TEAM_A_GOAL_X: float = FIELD_X_MIN   # 0.0
TEAM_B_GOAL_X: float = FIELD_X_MAX   # 100.0

# ---------------------------------------------------------------------------
# 2.2 Ball Physics Model
# ---------------------------------------------------------------------------
BALL_FRICTION: float = 0.96          # velocity decay coefficient mu
BALL_STOP_THRESHOLD: float = 0.05    # ||V|| < this -> velocity set to (0, 0)
WALL_BOUNCE_FACTOR: float = 0.8      # energy retained on wall bounce

# ---------------------------------------------------------------------------
# 3.1 Agent attribute ranges
# ---------------------------------------------------------------------------
ATTR_MIN: float = 1.0
ATTR_MAX: float = 10.0

VIS_MIN: float = 30.0   # Vision attribute lower bound (degrees)
VIS_MAX: float = 120.0  # Vision attribute upper bound (degrees)

VIS_RANGE_MIN: float = 20.0   # per-agent sight distance lower bound (units)
VIS_RANGE_MAX: float = 60.0   # per-agent sight distance upper bound (units)

# ---------------------------------------------------------------------------
# 3.4 Skill knobs & per-player budget (participant squad design)
# ---------------------------------------------------------------------------
# Participants allocate points across a fixed set of knobs per player, subject
# to a per-player budget so nobody can max everything (forced specialisation).
# Each knob is an integer in [KNOB_MIN, KNOB_MAX]; a knob converts to an
# internal 1..10 attribute via ``1 + 9 * (knob / 100)`` and feeds the physics.
KNOB_MIN: int = 10
KNOB_MAX: int = 100
BUDGET_RATIO: float = 0.6   # cap = ratio * n_knobs * KNOB_MAX

# Outfield knobs (7) and goalkeeper knobs (5).
OUTFIELD_KNOBS = (
    "running", "dribbling", "shot_power", "pass_accuracy",
    "pass_range", "tackling", "vision",
)
GK_KNOBS = (
    "reflexes", "positioning", "agility", "handling", "distribution",
)
OUTFIELD_BUDGET: int = int(BUDGET_RATIO * len(OUTFIELD_KNOBS) * KNOB_MAX)  # 420
GK_BUDGET: int = int(BUDGET_RATIO * len(GK_KNOBS) * KNOB_MAX)              # 300

# Home-position bounds for a custom formation (Team A own-half frame; the away
# squad is mirrored about the halfway line).  Outfielders may push up to just
# past halfway; the keeper stays near its own goal.
HOME_X_MIN: float = 3.0
HOME_X_MAX: float = 55.0
HOME_Y_MIN: float = 3.0
HOME_Y_MAX: float = 57.0
GK_HOME_X_MIN: float = 2.0
GK_HOME_X_MAX: float = 12.0

# Star-player guardrails (Section 3.3)
STAR_AVG_MIN: float = 6.0
STAR_BOOST_VALUE: float = 10.0

# ---------------------------------------------------------------------------
# Movement & action constants
# ---------------------------------------------------------------------------
BASE_MOVE_SPEED: float = 0.15        # units/tick baseline (Section 3.1 SPD)

# Rotation clamp per tick (Section 5.1)
ROTATION_CLAMP: float = 30.0

# Movement-shot mutex (Section 6.1): an agent that shoots is stationary.
# Outfield intercept (Section 6.2)
INTERCEPT_RADIUS_BASE: float = 1.2

# Goalkeeper (Section 6.3)
GK_DIVE_MULTIPLIER: float = 2.5
# The SRS lists only REF / POS / PASS for goalkeepers.  The dive distance is
# "2.5 x their maximum base speed attribute"; goalkeepers have no SPD stat so
# we use an implicit baseline speed for keeper locomotion.
GK_IMPLICIT_SPD: float = 5.0
GK_BASE_SPEED: float = BASE_MOVE_SPEED * GK_IMPLICIT_SPD  # 0.75 units/tick

# ---------------------------------------------------------------------------
# 4. State Space & Partial Observability
# ---------------------------------------------------------------------------
VISION_RANGE: float = 40.0   # fixed distance limit of the vision cone

# ---------------------------------------------------------------------------
# Shooting / passing mechanics
# ---------------------------------------------------------------------------
# Initial ball velocity on a kick scales with SHO and power.  Kept moderate so
# a keeper can actually react to a shot rather than it tunnelling past in a
# single tick (see the balance pass).
SHOOT_SPEED_SCALE: float = 0.85   # v0 = power * SHO * scale

# Extra angular spread (degrees) applied to shots on top of the kicker's pass
# noise, so long-range rockets are not automatic goals.  Passes are unaffected.
SHOT_EXTRA_NOISE: float = 5.0

# Goalkeepers cover a larger save envelope than an outfield interceptor — the
# main dial that keeps scorelines realistic.
GK_SAVE_RADIUS_SCALE: float = 1.7

# Angular noise on every kick (pass/shot).  Spread = (11 - PASS) degrees,
# clamped to be non-negative.  (Section 3.1 PASS table.)
PASS_NOISE_BASE: float = 11.0

# Possession dribble offset (ball sits slightly ahead of the carrier).
DRIBBLE_OFFSET: float = 1.0

# Tackle envelope: an opponent within this distance of the carrier can
# dislodge the ball (kept small so possession is meaningful).
TACKLE_RADIUS: float = 1.6

# ---------------------------------------------------------------------------
# Set pieces & out-of-play restarts (corners / goal kicks / throw-ins)
# ---------------------------------------------------------------------------
# When the ball fully crosses a boundary line it is *dead*.  Instead of bouncing
# off the wall the engine stops it and restarts play with instant possession
# (the nearest eligible player of the restarting team collects the ball).  The
# restart type depends on which line was crossed and which team last touched it:
#   * touchline (top/bottom, the Y walls)  -> throw-in to the team that did NOT
#     touch it last.
#   * byline / goal line (left/right X walls, outside the goal mouth):
#       - defending team touched last -> CORNER to the attacking team.
#       - attacking team touched last -> GOAL KICK to the defending team.
SET_PIECES_ENABLED: bool = True

# How far inside the corner the ball is placed for a corner kick.
CORNER_INSET: float = 2.0
# Goal-kick spot: on the six-yard-ish line, in front of the goal.
GOAL_KICK_INSET_X: float = 8.0
# Throw-in: the ball is placed just inside the touchline it went out on.
THROW_IN_INSET_Y: float = 1.0
# A dead-ball restart briefly makes the taker's team the only side allowed to
# reach the ball, so a set piece cannot be instantly stolen at point-blank
# range.  Purely internal (no new action types); the RL loop is unchanged.
SET_PIECE_PROTECT_TICKS: int = 12

# ---------------------------------------------------------------------------
# Offside (full positional model, evaluated at the moment a pass is played)
# ---------------------------------------------------------------------------
# A team-mate of the kicker is in an offside *position* if, at the instant the
# ball is played, they are in the opponent half AND ahead of both the ball and
# the second-last defender (the last outfielder in front of the keeper counts
# as the second-last defender when the keeper is the last).  Being *level* with
# the second-last defender or the ball (within OFFSIDE_LEVEL_TOL) is onside.
# An offside is only *called* when such a player becomes actively involved —
# they are the first attacker to touch the ball after the pass.  The phase is
# reset (no offside) if a defender deliberately plays the ball in between.
OFFSIDE_ENABLED: bool = True
OFFSIDE_LEVEL_TOL: float = 0.5   # units of "level with" tolerance on the X axis

# ---------------------------------------------------------------------------
# 7. Tournament & economy
# ---------------------------------------------------------------------------
TOURNAMENT_SIZE: int = 64
TOURNAMENT_ROUNDS: int = 6  # log2(64)

WIN_BASE_REWARD: int = 100       # W_base
GOAL_BONUS: int = 15             # G_bonus per goal scored by the winner
NORMAL_PLAYER_COST: int = 100    # 1.0 * W_base
STAR_PLAYER_COST: int = 200      # 2.0 * W_base
SQUAD_SIZE: int = 11

# ---------------------------------------------------------------------------
# Match configuration
# ---------------------------------------------------------------------------
DEFAULT_MATCH_TICKS: int = 3600   # ~ a full simulated match
KICKOFF_RESET_TICKS: int = 0      # goals reset immediately

# Match clock. Regulation is two halves of 45:00 = 90:00 total, mapped uniformly
# onto however many regulation ticks a match is played with (the tick count
# itself is unchanged -> balance is preserved). Play is re-centred (kickoff) at
# the half-time midpoint. If still level after 90:00, a golden-goal extra-time
# period (ET_MINUTES) is played, then a penalty shootout. Ticks -> MM:SS via
# REGULATION_SECONDS / regulation_ticks.
HALF_MINUTES: int = 45
STOPPAGE_MINUTES: int = 0          # no added time — a clean 45 + 45
HALVES: int = 2
REGULATION_SECONDS: int = (HALF_MINUTES + STOPPAGE_MINUTES) * HALVES * 60  # 5400
ET_MINUTES: int = 30               # golden-goal extra time
ET_FRACTION: float = (ET_MINUTES * 60) / REGULATION_SECONDS  # 0.3 of regulation


def tick_to_clock(tick: int, reg_ticks: int) -> str:
    """Format a simulation tick as an ``MM:SS`` match clock string."""
    reg_ticks = max(1, int(reg_ticks))
    total = int(round(tick * REGULATION_SECONDS / reg_ticks))
    return f"{total // 60:02d}:{total % 60:02d}"


def tick_half(tick: int, reg_ticks: int) -> str:
    """Which period a tick falls in: ``"1"``, ``"2"`` or ``"ET"``."""
    reg_ticks = max(1, int(reg_ticks))
    if tick > reg_ticks:
        return "ET"
    return "1" if tick <= reg_ticks / 2 else "2"

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
RENDER_SCALE: int = 9             # pixels per pitch unit
RENDER_FPS: int = 60

# Match-replay GIF produced by the compute worker for every tournament match
# (organizer-side). Tuned for a shareable, reasonably small full-match clip:
# capture every Nth tick, downscale, and play back at a modest fps. The frame
# cap bounds both memory and the final file size on long matches.
REPLAY_FRAME_STRIDE: int = 4      # capture every 4th tick
REPLAY_SCALE: float = 0.5         # downscale the saved GIF to half-resolution
REPLAY_FPS: int = 15              # GIF playback frame rate
REPLAY_MAX_FRAMES: int = 600      # hard cap on captured frames (size/memory)

# Distinct, accessible team colors (no indigo/blue per house style).
TEAM_A_COLOR: tuple = (220, 38, 38)      # red
TEAM_B_COLOR: tuple = (16, 122, 87)      # emerald green
BALL_COLOR: tuple = (250, 250, 250)
FIELD_COLOR: tuple = (34, 92, 58)
FIELD_LINE_COLOR: tuple = (235, 240, 230)
STAR_RING_COLOR: tuple = (250, 204, 21)

# Helper: convert degrees <-> radians
def deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def rad2deg(r: float) -> float:
    return r * 180.0 / math.pi
