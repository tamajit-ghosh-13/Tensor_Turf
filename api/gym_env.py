"""
gym_env.py
==========
OpenAI Gymnasium environment for the FIFA AI World Cup — **per-agent** training.

A single env instance controls a **variable number of squad slots** (1 to 11)
simultaneously.  Train one agent at a time, a subset (e.g. the 3 attackers),
or all 11 (including the goalkeeper) in one run.

Registration
------------
The env is registered on import::

    import gymnasium as gym
    import gym_env  # noqa: F401  (registers FIFAWorldCupAgent-v0)
    env = gym.make("FIFAWorldCupAgent-v0", slots=[0,5,9], opponent_style="aggressive")

Spaces — the built-in defaults (depend on ``len(slots)``)
---------------------------------------------------------
* ``n == 1`` → ``observation_space = Box(20,)``, ``action_space = Discrete(13)``
* ``n > 1`` → ``observation_space = Box(20*n,)``, ``action_space = MultiDiscrete([13]*n)``

The **13-action unified space** (outfield + GK):
    0..7  MOVE (compass)
    8     ROTATE (scan)
    9     SHOOT (at goal, away from a visible keeper)
    10    PASS  (to the best visible forward team-mate)
    11    DIVE     (GK only; outfielders -> IDLE)
    12    DEFLECT  (GK only; outfielders -> IDLE)

Bring your own observation & action space
-----------------------------------------
The 20-float obs and 13-action space are **only defaults**.  You are free to
design your own — pass any of these to ``gym.make`` / the constructor:

* ``feature_fn(state) -> sequence[float]`` — your own observation vector from
  the raw engine ``state`` dict (any length; the obs space is sized for you).
* ``action_fn(raw_action, state, agent_pos, sho) -> action dict`` — turn what
  your policy emits (a discrete index, a continuous vector, …) into an engine
  action dict.
* ``action_space`` — any Gymnasium space for a *single* agent (``Discrete(k)``,
  ``Box(...)`` for continuous control, …).  It is composed automatically when
  you train more than one slot at once.
* ``obs_dim`` / ``obs_low`` / ``obs_high`` — override obs sizing / bounds.

This means any algorithm (DQN, PPO, actor-critic, evolutionary search, …) and
any obs/action design works.  The engine only requires that your ``agents.py``
returns valid action dicts at match time, so keep it consistent with whatever
you train here.

The env wraps the **real** match engine (``fifa_ai_world_cup.game.Game``) so
physics, vision cones and the action contract are identical to a real match —
a policy trained here transfers directly to inference.

Pluggable reward
----------------
Pass ``reward_fn(prev_obs, action, obs, info, team) -> float`` to ``gym.make``
(or to the constructor).  If ``None``, a sensible default dense reward is used.
Participants import the env and pass their own ``reward_fn`` — this is how they
model the reward function.  ``info`` carries everything a custom reward needs:
``score``, ``tick``, ``phase``, and a per-controlled-agent dict with
``has_ball``, ``ball_distance``, ``ball_visible``, ``archetype``, ``is_gk``.

Head-to-head at inference time
------------------------------
Training produces per-slot weight files (``agent_00_dqn.pt`` …
``agent_10_dqn.pt``).  At match time ``agents_trained.py`` loads them and does
pure forward-pass inference (no env, no resets) — see ``TRAINING.md``.
"""

from __future__ import annotations

import math
import os
import random
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM = True
except Exception:  # pragma: no cover
    gym = None
    spaces = None
    _GYM = False

# When inside the `api` package, relative imports work automatically.
# (The old path-manipulation block broke the package context.)
# If running standalone (not as part of a package), add the parent to sys.path.
if __package__ is None or __package__ == "":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(_HERE)
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from .agent import generate_squad  # noqa: E402
from .game import Game  # noqa: E402
from .agents import (  # noqa: E402
    extract_features,
    map_action,
    STATE_DIM,
    ACTION_DIM,
)

SQUAD_SIZE = 11
# RewardFn signature: (prev_obs, action, obs, info, team) -> float
RewardFn = Callable[[np.ndarray, Any, np.ndarray, Dict[str, Any], str], float]


class FIFAWorldCupAgentEnv(gym.Env if _GYM else object):
    """Gymnasium env controlling a variable subset of a team's squad slots.

    Parameters
    ----------
    slots : list[int], optional
        Squad slot indices (0..10) to control simultaneously.  ``[0]`` trains
        just the GK; ``[5]`` trains one outfielder; ``[0,5,9]`` trains three
        at once; ``list(range(11))`` trains the whole team.  Default: all 11.
    opponent_style : str
        Reference-AI style for the opposing team (``balanced`` / ``aggressive``
        / ``defensive`` / ``random``).
    teammate_style : str
        Reference-AI style driving the squad slots you are NOT controlling.
    max_ticks : int
        Episode length in simulation ticks.
    reward_fn : callable, optional
        ``reward_fn(prev_obs, action, obs, info, team) -> float``.  If ``None``
        a default dense reward is used (see ``default_reward``).
    seed : int, optional
        Base RNG seed; each ``reset`` derives a fresh seed from it.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    # ---- lifecycle ------------------------------------------------------
    def __init__(
        self,
        slots: Optional[List[int]] = None,
        opponent_style: str = "aggressive",
        teammate_style: str = "balanced",
        max_ticks: int = 1800,
        reward_fn: Optional[RewardFn] = None,
        seed: Optional[int] = None,
        feature_fn: Optional[Callable[[Dict[str, Any]], Any]] = None,
        action_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        obs_dim: Optional[int] = None,
        action_space: Optional[Any] = None,
        obs_low: Optional[float] = None,
        obs_high: Optional[float] = None,
    ):
        if not _GYM:
            raise RuntimeError("gymnasium is not installed; pip install gymnasium")
        super().__init__()
        self.slots = list(slots) if slots is not None else list(range(SQUAD_SIZE))
        self.n = len(self.slots)
        self.opponent_style = opponent_style
        self.teammate_style = teammate_style
        self.max_ticks = int(max_ticks)
        self.reward_fn = reward_fn or default_reward
        self._base_seed = seed
        self._rng = random.Random(seed)

        # --- Pluggable observation / action interface (see module docstring) --
        # Defaults reproduce the built-in 20-float obs and 13-action space.
        self._custom_features = feature_fn is not None
        self.feature_fn = feature_fn or extract_features
        self.action_fn = action_fn or map_action
        self._base_action_space = action_space
        self._obs_dim = (int(obs_dim) if obs_dim is not None
                         else (None if self._custom_features else STATE_DIM))
        self._obs_low = (float(obs_low) if obs_low is not None
                         else (-1.0 if not self._custom_features else -np.inf))
        self._obs_high = (float(obs_high) if obs_high is not None
                          else (2.0 if not self._custom_features else np.inf))

        self.game: Optional[Game] = None
        self._shim = None
        self._prev_obs: Optional[np.ndarray] = None
        self._prev_score: Dict[str, int] = {"A": 0, "B": 0}
        self._prev_ball_dist: Optional[float] = None
        self._prev_has_ball: bool = False

        self._build_spaces()

    # ---- space construction / action-batching helpers -------------------
    def _build_spaces(self) -> None:
        if self._obs_dim is None:
            self._obs_dim = self._infer_obs_dim()
        d = self._obs_dim
        shape = (d,) if self.n == 1 else (d * self.n,)
        self.observation_space = spaces.Box(
            low=self._obs_low, high=self._obs_high, shape=shape, dtype=np.float32)

        base = self._base_action_space or spaces.Discrete(ACTION_DIM)
        self._single_action_space = base
        if isinstance(base, spaces.Discrete):
            self._act_kind = "discrete"
            self.action_space = base if self.n == 1 \
                else spaces.MultiDiscrete([base.n] * self.n)
        elif isinstance(base, spaces.Box):
            self._act_kind = "box"
            self._act_dim = int(np.prod(base.shape))
            if self.n == 1:
                self.action_space = base
            else:
                low = np.tile(np.asarray(base.low, dtype=np.float32).reshape(-1), self.n)
                high = np.tile(np.asarray(base.high, dtype=np.float32).reshape(-1), self.n)
                self.action_space = spaces.Box(low=low, high=high, dtype=base.dtype)
        else:
            self._act_kind = "other"
            self.action_space = base if self.n == 1 \
                else spaces.Tuple([base] * self.n)

    def _infer_obs_dim(self) -> int:
        """Probe ``feature_fn`` once (on a throwaway game) to size the obs space."""
        seed = self._base_seed if self._base_seed is not None else 0
        self._make_game(int(seed))
        try:
            states = self._team_states()
            return len(list(self.feature_fn(states[self.slots[0]])))
        finally:
            self.game = None
            self._shim = None

    def _slice_action(self, action: Any, j: int) -> Any:
        """Extract the j-th controlled slot's raw action from a batched action."""
        if self._act_kind == "discrete":
            return int(action) if self.n == 1 \
                else int(np.asarray(action).reshape(-1)[j])
        if self._act_kind == "box":
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
            return arr if self.n == 1 \
                else arr[j * self._act_dim:(j + 1) * self._act_dim]
        return action if self.n == 1 else action[j]

    def _make_game(self, ep_seed: int) -> None:
        squad_a = generate_squad("A", rng=random.Random(ep_seed))
        squad_b = generate_squad("B", rng=random.Random(ep_seed + 1))
        from .test_agents import make_policy
        opp_policy = make_policy(self.opponent_style, squad_b, seed=ep_seed + 2)
        # Team A's policy is a shim: controlled slots use env-supplied actions,
        # the other slots run the teammate baseline.
        self._shim = _SlotShim(squad_a, set(self.slots), self.teammate_style,
                               ep_seed + 3)
        self.game = Game(
            "Trainee", squad_a, self._shim,
            "Opponent", squad_b, opp_policy,
            max_ticks=self.max_ticks,
            rng=random.Random(ep_seed + 4),
        )

    # ---- reset / step ---------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        ep_seed = seed if seed is not None else self._rng.randrange(2**31)
        # Allow re-configuring slots at reset (re-builds spaces if changed).
        if options and "slots" in options:
            self.slots = list(options["slots"])
            self.n = len(self.slots)
            self._build_spaces()

        self._make_game(ep_seed)
        self._prev_obs = None
        self._prev_score = dict(self.game.score)
        self._prev_ball_dist = None
        self._prev_has_ball = False
        obs = self._observe()
        info = self._info()
        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.game is None:
            raise RuntimeError("call reset() before step()")

        # Build the 11 action dicts for team A.
        states = self._team_states()
        squad = self.game.squad_a
        action_dicts: List[Dict[str, Any]] = []
        controlled = set(self.slots)
        for i in range(SQUAD_SIZE):
            if i in controlled:
                raw = self._slice_action(action, self.slots.index(i))
                st = states[i]
                ag = squad[i]
                action_dicts.append(self.action_fn(raw, st, (ag.x, ag.y), ag.sho))
            else:
                action_dicts.append(None)  # shim fills from baseline
        self._shim.set_actions(action_dicts)
        self.game.step()

        obs = self._observe()
        info = self._info()
        # Inject prev-state tracking so reward_fn can compute deltas.
        info["_prev_score"] = dict(self._prev_score)
        info["_prev_ball_dist"] = self._prev_ball_dist
        info["_prev_has_ball"] = self._prev_has_ball
        reward = self.reward_fn(self._prev_obs if self._prev_obs is not None else obs,
                                action, obs, info, "A")
        # Update prev-state for next tick.
        self._prev_score = dict(self.game.score)
        any_has_ball = any(a["has_ball"] for a in info.get("agents", []))
        ball_dists = [a["ball_distance"] for a in info.get("agents", [])
                      if a["ball_distance"] is not None]
        self._prev_ball_dist = min(ball_dists) if ball_dists else None
        self._prev_has_ball = any_has_ball
        self._prev_obs = obs
        terminated = self.game._winner is not None and self.game.phase != "regulation"
        truncated = self.game.tick >= self.max_ticks
        return obs, float(reward), terminated, truncated, info

    # ---- observation / info --------------------------------------------
    def _observe(self) -> np.ndarray:
        """Flat observation from the controlled slots' states (via feature_fn)."""
        states = self._team_states()
        if self.n == 1:
            return np.asarray(self.feature_fn(states[self.slots[0]]),
                              dtype=np.float32)
        d = self._obs_dim
        flat = np.zeros(d * self.n, dtype=np.float32)
        for j, slot in enumerate(self.slots):
            flat[j * d:(j + 1) * d] = \
                np.asarray(self.feature_fn(states[slot]), dtype=np.float32)
        return flat

    def _team_states(self):
        from .vision import build_team_states
        return build_team_states(self.game.squad_a, self.game.all_agents,
                                 self.game.ball, tick=self.game.tick,
                                 max_ticks=self.game.max_ticks)

    def _info(self) -> Dict[str, Any]:
        states = self._team_states()
        per_agent = []
        for slot in self.slots:
            st = states[slot]
            ball_sector = next((s for s in st.get("vision_sectors", [])
                                if s.get("type") == "ball"), None)
            per_agent.append({
                "slot": slot,
                "archetype": st.get("archetype"),
                "is_gk": st.get("is_gk", False),
                "has_ball": st.get("has_ball", False),
                "ball_visible": ball_sector is not None
                                and ball_sector.get("distance") is not None,
                "ball_distance": (ball_sector.get("distance")
                                  if ball_sector else None),
                "orientation": st.get("current_orientation", 0.0),
            })
        return {
            "tick": self.game.tick,
            "score": dict(self.game.score),
            "phase": self.game.phase,
            "agents": per_agent,
        }

    # ---- optional rendering (not used during training) -----------------
    def render(self):
        return None

    def close(self):
        self.game = None


# ---------------------------------------------------------------------------
# Slot shim: controlled slots use env-supplied actions; the rest run baseline
# ---------------------------------------------------------------------------
class _SlotShim:
    """Policy shim: blends env-supplied actions (controlled slots) with a
    baseline teammate policy (uncontrolled slots)."""

    def __init__(self, squad, controlled_slots: set, teammate_style: str,
                 seed: int):
        self.squad = squad
        self.controlled = controlled_slots
        from .test_agents import make_policy
        self._baseline = make_policy(teammate_style, squad, seed=seed)
        self._pending: List[Optional[Dict[str, Any]]] = []

    def set_actions(self, actions: List[Optional[Dict[str, Any]]]):
        self._pending = actions

    def decide(self, states, team):
        # Run the baseline for everyone first.
        if hasattr(self._baseline, "decide"):
            base = list(self._baseline.decide(states, team))
        else:
            base = list(self._baseline(states, team))
        # Override the controlled slots with the env-supplied actions.
        for i in range(len(states)):
            if i in self.controlled and i < len(self._pending) \
                    and self._pending[i] is not None:
                base[i] = self._pending[i]
        self._pending = []
        return base


# ---------------------------------------------------------------------------
# Default dense reward (participants replace this with their own)
# ---------------------------------------------------------------------------
def default_reward(prev_obs: np.ndarray, action: Any, obs: np.ndarray,
                   info: Dict[str, Any], team: str) -> float:
    """A sensible default dense reward.  Override by passing ``reward_fn``.

    Sums team-level signals:
      * ±40 per goal scored / conceded (from ``info["_prev_score"]`` delta)
      * +0.15 per controlled agent in possession
      * +0.35 × forward-heading for the carrier (Team A attacks +X)
      * +0.18 × (decrease in nearest ball distance) when not in possession
      * −0.8 for losing possession this tick
      * small + for SHOOT / PASS actions, small − for ROTATE
    """
    new = info["score"]
    other = "B" if team == "A" else "A"
    prev = info.get("_prev_score", {"A": 0, "B": 0})
    reward = 40.0 * (new[team] - prev[team]) - 40.0 * (new[other] - prev[other])

    attack_dx = 1.0 if team == "A" else -1.0
    any_has_ball = False
    for a in info.get("agents", []):
        if a["has_ball"]:
            any_has_ball = True
            reward += 0.15
            ori = a.get("orientation", 0.0)
            reward += max(0.0, math.cos(math.radians(ori)) * attack_dx) * 0.35
    # Ball-proximity shaping for non-carriers.
    ball_dists = [a["ball_distance"] for a in info.get("agents", [])
                  if a["ball_distance"] is not None]
    if not any_has_ball and ball_dists:
        d = min(ball_dists)
        prev_d = info.get("_prev_ball_dist")
        if prev_d is not None:
            reward += (prev_d - d) * 0.18
    # Possession-loss penalty.
    if info.get("_prev_has_ball") and not any_has_ball:
        reward -= 0.8
    # Action bonuses for controlled agents.
    if hasattr(action, "__len__"):
        acts = list(action)
    else:
        acts = [int(action)]
    for a_idx in acts:
        if a_idx == 9:      # SHOOT
            reward += 0.10
        elif a_idx == 10:   # PASS
            reward += 0.08
        elif a_idx == 8:    # ROTATE
            reward -= 0.02
    return float(reward)


# ---------------------------------------------------------------------------
# Registration: ``gym.make("FIFAWorldCupAgent-v0")`` works after importing.
# ---------------------------------------------------------------------------
if _GYM:
    try:
        gym.register(
            id="FIFAWorldCupAgent-v0",
            entry_point="api.gym_env:FIFAWorldCupAgentEnv",
        )
    except Exception:
        pass  # already registered
