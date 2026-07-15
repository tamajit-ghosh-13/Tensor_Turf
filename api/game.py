"""
game.py
=======
The environment runner (organizer side).  Implements the continuous physics
simulation, local agent sightlines, policy execution and match resolution.

Tick execution loop (SRS Section 6):
    1. Parse Agent Decisions (Actions from Team A & B)
    2. Update Agent States (Movement / Rotations / Cooldowns)
    3. Calculate Ball Physics (Velocity Decay / Friction)
    4. Resolve Interceptions & Defensive Blocks
    5. Check Boundary Collisions & Goal Evaluations

Additional rules implemented here:
  * Movement-Shot Mutex (6.1).
  * Defensive Interceptions (6.2).
  * Goalkeeper Dive Penalty (6.3).
  * Possession follow-through (dribbling) and tackling.
  * Kickoff reset after a goal (2.2).
  * Overtime + attribute-based penalty shootout for knockout resolution.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import config as C
from .agent import Agent, place_kickoff
from .ball import Ball
from .utils import (
    angle_of_vector,
    distance,
    vec_from_angle,
)
from .vision import build_team_states
from .actions import apply_action, DEFLECT, SHOOT


PolicyLike = Any  # callable(states, team) -> list[dict] OR object with .decide


class Game:
    """A single match between two teams."""

    def __init__(
        self,
        team_a_name: str,
        squad_a: List[Agent],
        policy_a: PolicyLike,
        team_b_name: str,
        squad_b: List[Agent],
        policy_b: PolicyLike,
        max_ticks: int = C.DEFAULT_MATCH_TICKS,
        rng: Optional[random.Random] = None,
        renderer: Any = None,
        overtime_ticks: int = 900,
        seed: Optional[int] = None,
    ):
        self.team_names = {"A": team_a_name, "B": team_b_name}
        self.squad_a = squad_a
        self.squad_b = squad_b
        self.policy_a = policy_a
        self.policy_b = policy_b
        self.max_ticks = max_ticks
        self.overtime_ticks = overtime_ticks
        self.rng = rng or random.Random(seed)
        self.renderer = renderer

        self.ball = Ball()
        self.score = {"A": 0, "B": 0}
        self.tick = 0
        self.phase = "regulation"   # regulation -> overtime -> shootout
        self.events: List[Dict[str, Any]] = []
        self.last_shooter_id: Optional[str] = None
        self._gk_deflect_this_tick: set = set()
        self._winner: Optional[str] = None
        # Half-time re-centre fires once, when regulation crosses its midpoint.
        self._halftime_done: bool = False

        # Last player/team to touch the ball (for corners vs. goal kicks and
        # for the offside "deliberate play" reset).
        self.last_touch_id: Optional[str] = None
        self.last_touch_team: Optional[str] = None

        # Offside phase currently being watched (set when a pass is played).
        #   {"team", "offside_ids": set[str], "ball_x": float}
        self._offside_watch: Optional[Dict[str, Any]] = None

        # Set-piece possession protection: the restarting team briefly cannot
        # be tackled/intercepted so the restart is not stolen point-blank.
        #   {"team": str, "until_tick": int}
        self._set_piece_protect: Optional[Dict[str, Any]] = None

        # Live, FIFA-style match statistics for the result / frontend verdict.
        self.stats = {
            "corners": {"A": 0, "B": 0},
            "goal_kicks": {"A": 0, "B": 0},
            "throw_ins": {"A": 0, "B": 0},
            "offsides": {"A": 0, "B": 0},
            "shots": {"A": 0, "B": 0},
            "shots_on_target": {"A": 0, "B": 0},
            "passes": {"A": 0, "B": 0},
            "tackles": {"A": 0, "B": 0},
            "possession_ticks": {"A": 0, "B": 0},
        }
        # The most recent kick awaiting a receiver (for pass completion).
        self._last_kick: Optional[Dict[str, Any]] = None

        self._kickoff()

    # ------------------------------------------------------------------ setup
    @property
    def all_agents(self) -> List[Agent]:
        return self.squad_a + self.squad_b

    def _agent_by_id(self, agent_id: str) -> Optional[Agent]:
        for a in self.all_agents:
            if a.id == agent_id:
                return a
        return None

    def _kickoff(self) -> None:
        place_kickoff(self.squad_a, "A")
        place_kickoff(self.squad_b, "B")
        self.ball.reset()
        self.last_touch_id = None
        self.last_touch_team = None
        self._offside_watch = None
        self._set_piece_protect = None
        self._last_kick = None

    # ------------------------------------------------------------------ run
    def run(self) -> Dict[str, Any]:
        """Run the full match (regulation + overtime + shootout if needed)."""
        self.phase = "regulation"
        while self.tick < self.max_ticks and self._winner is None:
            self.step()

        # Overtime (golden goal) if drawn.
        if self._winner is None and self.score["A"] == self.score["B"]:
            self.phase = "overtime"
            ot_end = self.tick + self.overtime_ticks
            self._kickoff()
            while self.tick < ot_end and self._winner is None:
                self.step()
                if self.score["A"] != self.score["B"]:
                    self._winner = "A" if self.score["A"] > self.score["B"] else "B"
                    break

        # Penalty shootout if still drawn.
        if self._winner is None and self.score["A"] == self.score["B"]:
            self.phase = "shootout"
            self._winner = self._penalty_shootout()

        if self._winner is None:
            # Fallback: decide on goal difference then coin flip.
            if self.score["A"] != self.score["B"]:
                self._winner = "A" if self.score["A"] > self.score["B"] else "B"
            else:
                self._winner = self.rng.choice(["A", "B"])

        return self.result()

    # ------------------------------------------------------------------ step
    def step(self) -> Optional[str]:
        """Advance the simulation by one tick.  Returns scoring team or None."""
        self.tick += 1
        self.last_shooter_id = None
        self._gk_deflect_this_tick.clear()

        # Half-time: at the regulation midpoint, re-centre both squads and the
        # ball to a fresh kickoff (once). Overtime/shootout are excluded via the
        # phase guard; a match played to fewer ticks still halves cleanly.
        if (self.phase == "regulation" and not self._halftime_done
                and self.tick >= self.max_ticks // 2):
            self._halftime_done = True
            self.events.append({"tick": self.tick, "type": "halftime"})
            self._kickoff()

        # Possession accounting: credit the team holding the ball this tick.
        if self.ball.possessed_by:
            holder = self._agent_by_id(self.ball.possessed_by)
            if holder is not None:
                self.stats["possession_ticks"][holder.team] += 1

        # 1. Parse agent decisions: build partial states, query policies.
        states_a = build_team_states(self.squad_a, self.all_agents, self.ball)
        states_b = build_team_states(self.squad_b, self.all_agents, self.ball)
        actions_a = self._call_policy(self.policy_a, states_a, "A")
        actions_b = self._call_policy(self.policy_b, states_b, "B")

        # Track DEFLECT intent for goalkeepers (used in interception step).
        for agent, action in zip(self.squad_a, actions_a):
            if self._action_type(action) == DEFLECT and agent.is_gk:
                self._gk_deflect_this_tick.add(agent.id)
        for agent, action in zip(self.squad_b, actions_b):
            if self._action_type(action) == DEFLECT and agent.is_gk:
                self._gk_deflect_this_tick.add(agent.id)

        # 2. Update agent states (movement / rotations / cooldowns / shoots).
        for agent, action in zip(self.squad_a, actions_a):
            tel = apply_action(agent, action, self.ball, self.rng)
            if tel["shot"]:
                self.last_shooter_id = agent.id
                self._on_kick(agent)
        for agent, action in zip(self.squad_b, actions_b):
            tel = apply_action(agent, action, self.ball, self.rng)
            if tel["shot"]:
                self.last_shooter_id = agent.id
                self._on_kick(agent)

        # 2b. Possession follow-through: a carried ball tracks its carrier.
        if self.ball.possessed_by:
            carrier = self._agent_by_id(self.ball.possessed_by)
            if carrier is not None and carrier.has_ball:
                self._carry_ball(carrier)
            else:
                # Desync guard: possession lost if carrier no longer holds.
                self.ball.possessed_by = None

        # 3 + 5. Ball physics (friction) and boundary collisions / goal eval.
        event = self.ball.step_physics()
        if event is not None:
            if event["type"] == "goal":
                self._record_goal(event["team"])
                if self.renderer:
                    self.renderer.capture(self)
                return event["team"]
            if event["type"] == "out":
                # Dead ball: restart with a corner / goal kick / throw-in.
                self._restart_set_piece(event)
                if self.renderer:
                    self.renderer.capture(self)
                return None

        # 4. Resolve interceptions & defensive blocks.
        self._resolve_interceptions()

        # Tackling: an opponent can dislodge a carried ball.
        self._resolve_tackles()

        # Expire the set-piece protection window.
        if (self._set_piece_protect is not None
                and self.tick >= self._set_piece_protect["until_tick"]):
            self._set_piece_protect = None

        # End-of-tick cooldown decrement (Section 6.3 freeze timer).
        for a in self.all_agents:
            if a.cooldown_remaining > 0:
                a.cooldown_remaining -= 1

        if self.renderer:
            self.renderer.capture(self)

        return None

    # -------------------------------------------------------------- policies
    @staticmethod
    def _action_type(action: Any) -> str:
        if isinstance(action, dict):
            return str(action.get("action_type", "IDLE")).upper()
        return "IDLE"

    def _call_policy(self, policy: PolicyLike, states: List[Dict[str, Any]],
                     team: str) -> List[Dict[str, Any]]:
        """Invoke a participant policy defensively.

        A policy that raises or returns malformed output is penalised with a
        squad-wide IDLE so the match never crashes.
        """
        try:
            if hasattr(policy, "decide"):
                actions = policy.decide(states, team)
            else:
                actions = policy(states, team)
        except Exception as exc:  # noqa: BLE001 - intentional safety net
            self.events.append({
                "tick": self.tick, "type": "policy_error",
                "team": team, "error": repr(exc),
            })
            actions = None
        if not isinstance(actions, (list, tuple)) or len(actions) != len(states):
            actions = [{"action_type": "IDLE"} for _ in states]
        return list(actions)

    # ------------------------------------------------------------ possession
    def _carry_ball(self, carrier: Agent) -> None:
        dx, dy = vec_from_angle(carrier.orientation, C.DRIBBLE_OFFSET)
        self.ball.x = carrier.x + dx
        self.ball.y = carrier.y + dy
        self.ball.vx = 0.0
        self.ball.vy = 0.0

    def _protected_from(self, agent: Agent) -> bool:
        """Whether ``agent`` may not touch the ball during a set-piece window.

        During the brief protection window after a restart, only the restarting
        team may reach the ball, so the set piece cannot be stolen point-blank.
        """
        prot = self._set_piece_protect
        return prot is not None and agent.team != prot["team"]

    def _resolve_interceptions(self) -> None:
        """Section 6.2: nearest eligible interceptor gains possession."""
        if not self.ball.is_free:
            return

        candidates: List[Tuple[float, Agent]] = []
        for a in self.all_agents:
            if a.id == self.last_shooter_id:
                continue
            if a.is_gk and a.cooldown_remaining > 0:
                continue
            if self._protected_from(a):
                continue
            radius = a.gk_block_radius() if a.is_gk else a.intercept_radius()
            d = distance((a.x, a.y), (self.ball.x, self.ball.y))
            if d <= radius:
                candidates.append((d, a))

        if not candidates:
            return

        candidates.sort(key=lambda c: c[0])
        _, agent = candidates[0]

        # Offside: an attacker flagged when the pass was played becoming the
        # first team-mate to touch the ball is penalised before they collect.
        if self._touch_is_offside(agent):
            self._award_offside(agent)
            return

        # Goalkeeper active DEFLECT: punch the ball away instead of catching.
        if agent.is_gk and agent.id in self._gk_deflect_this_tick:
            speed = max(self.ball.speed * 0.6, 3.0)
            dx, dy = vec_from_angle(agent.deflect_angle, speed)
            self.ball.vx, self.ball.vy = dx, dy
            self.ball.possessed_by = None
            self._register_touch(agent)
            return

        # Goalkeeper handling: a fast shot is caught (held) with probability
        # gk_catch_probability(), otherwise parried loose upfield.  Slow balls
        # are simply collected.
        if (agent.is_gk and self.ball.speed > 2.0
                and self.rng.random() >= agent.gk_catch_probability()):
            speed = max(self.ball.speed * 0.5, 2.5)
            dx, dy = vec_from_angle(agent.deflect_angle, speed)
            self.ball.vx, self.ball.vy = dx, dy
            self.ball.possessed_by = None
            self._register_touch(agent)
            self.events.append({"tick": self.tick, "type": "gk_parry",
                                "by": agent.id})
            return

        # Otherwise the agent collects the ball.
        agent.has_ball = True
        self.ball.possessed_by = agent.id
        self.ball.vx = 0.0
        self.ball.vy = 0.0
        self._carry_ball(agent)
        self._register_touch(agent)

    def _resolve_tackles(self) -> None:
        """Dislodge the ball from a carrier pressured by an opponent."""
        if self.ball.possessed_by is None:
            return
        carrier = self._agent_by_id(self.ball.possessed_by)
        if carrier is None or not carrier.has_ball:
            self.ball.possessed_by = None
            return

        for a in self.all_agents:
            if a.team == carrier.team or a.id == carrier.id:
                continue
            if a.is_gk and a.cooldown_remaining > 0:
                continue
            if self._protected_from(a):
                continue
            d = distance((a.x, a.y), (carrier.x, carrier.y))
            if d <= C.TACKLE_RADIUS:
                # Contest: tackler's tackling (def_) vs carrier's dribbling (drb).
                p_win = min(0.6, max(0.03, 0.12 + 0.03 * (a.def_ - carrier.drb)))
                if self.rng.random() < p_win:
                    carrier.has_ball = False
                    self.ball.possessed_by = None
                    angle = self.rng.uniform(0.0, 360.0)
                    dx, dy = vec_from_angle(angle, 1.5)
                    self.ball.vx, self.ball.vy = dx, dy
                    self.stats["tackles"][a.team] += 1
                    self._register_touch(a)
                    self.events.append({
                        "tick": self.tick, "type": "tackle",
                        "by": a.id, "from": carrier.id,
                    })
                    return

    # ------------------------------------------------------------ touches
    def _classify_shot(self, kicker: Agent) -> None:
        """Tally a kick as a shot / shot-on-target when aimed at the goal.

        Uses the ball's post-kick velocity to project where the ball would
        cross the opponent goal line.  A kick counts as a *shot* when it is
        played toward the opponent goal from within range and roughly on frame;
        a *shot on target* additionally crosses inside the goal mouth.
        """
        team = kicker.team
        goal_x = C.TEAM_B_GOAL_X if team == "A" else C.TEAM_A_GOAL_X
        vx, vy = self.ball.vx, self.ball.vy
        dx = goal_x - self.ball.x
        if vx == 0.0 or dx == 0.0 or (dx > 0) != (vx > 0):
            return  # not heading toward the opponent goal
        t = dx / vx
        if t <= 0:
            return
        y_at = self.ball.y + vy * t
        dist_to_goal = distance((self.ball.x, self.ball.y),
                                (goal_x, C.FIELD_CENTER_Y))
        on_frame = (C.GOAL_Y_MIN - 10.0) <= y_at <= (C.GOAL_Y_MAX + 10.0)
        if dist_to_goal <= 45.0 and on_frame:
            self.stats["shots"][team] += 1
            if C.GOAL_Y_MIN <= y_at <= C.GOAL_Y_MAX:
                self.stats["shots_on_target"][team] += 1

    def _register_touch(self, agent: Agent) -> None:
        """Record the last player/team to touch the ball and settle offside.

        Any touch closes the current offside phase: an opponent touching the
        ball is a deliberate defensive play (reset), and a team-mate that was
        onside simply receives (the phase ends).  An *offside* touch is handled
        separately by ``_touch_is_offside`` before possession is granted.
        """
        # Completed pass: a team-mate (not the kicker) receives the kick.
        if (self._last_kick is not None
                and agent.team == self._last_kick["team"]
                and agent.id != self._last_kick["id"]):
            self.stats["passes"][agent.team] += 1
        self._last_kick = None

        self.last_touch_id = agent.id
        self.last_touch_team = agent.team
        self._offside_watch = None

    def _touch_is_offside(self, agent: Agent) -> bool:
        """True if ``agent`` receiving now is an offside offence."""
        watch = self._offside_watch
        if watch is None:
            return False
        if agent.team != watch["team"]:
            return False
        return agent.id in watch["offside_ids"]

    # ------------------------------------------------------------ offside
    @staticmethod
    def _other(team: str) -> str:
        return "B" if team == "A" else "A"

    def _squad(self, team: str) -> List[Agent]:
        return self.squad_a if team == "A" else self.squad_b

    def _on_kick(self, kicker: Agent) -> None:
        """Record the kick as a touch and open an offside phase.

        The offside line is snapshotted at the instant the ball is played: any
        team-mate ahead of both the ball and the second-last defender, in the
        opponent half, is flagged.  If one of them is the first attacker to
        touch the resulting ball, ``_resolve_interceptions`` calls offside.
        """
        self.last_touch_id = kicker.id
        self.last_touch_team = kicker.team
        self._last_kick = {"team": kicker.team, "id": kicker.id}
        self._classify_shot(kicker)

        team = kicker.team
        attack_sign = 1.0 if team == "A" else -1.0  # +X for A, -X for B
        ball_x = self.ball.x

        if not C.OFFSIDE_ENABLED:
            self._offside_watch = None
            return

        # Second-last defender line (opponents ordered by defensive depth).
        opp = self._squad(self._other(team))
        depths = sorted((o.x for o in opp),
                        reverse=(team == "A"))  # deepest first
        # Index 1 == the second-last defender (the last is usually the keeper).
        line_x = depths[1] if len(depths) >= 2 else (
            depths[0] if depths else ball_x)

        tol = C.OFFSIDE_LEVEL_TOL
        offside_ids = set()
        for mate in self._squad(team):
            if mate.id == kicker.id:
                continue
            # In the opponent half?
            if (mate.x - C.FIELD_CENTER_X) * attack_sign <= 0:
                continue
            # Ahead of the ball?
            if (mate.x - ball_x) * attack_sign <= tol:
                continue
            # Ahead of the second-last defender (level is onside)?
            if (mate.x - line_x) * attack_sign <= tol:
                continue
            offside_ids.add(mate.id)

        self._offside_watch = {
            "team": team, "offside_ids": offside_ids,
            "ball_x": ball_x, "line_x": line_x,
        }

    def _award_offside(self, offside_agent: Agent) -> None:
        """Penalise an offside: indirect free kick to the defending team."""
        team_off = offside_agent.team
        defending = self._other(team_off)
        self.stats["offsides"][team_off] += 1
        spot = (
            min(max(offside_agent.x, C.FIELD_X_MIN + 3.0), C.FIELD_X_MAX - 3.0),
            min(max(offside_agent.y, C.FIELD_Y_MIN + 3.0), C.FIELD_Y_MAX - 3.0),
        )
        self.events.append({
            "tick": self.tick, "type": "offside",
            "against": team_off, "player": offside_agent.id,
            "restart_team": defending,
        })
        self._place_restart(defending, spot, prefer_gk=False)

    # ---------------------------------------------------------- set pieces
    def _restart_set_piece(self, event: Dict[str, Any]) -> None:
        """Turn an out-of-play event into a corner / goal kick / throw-in."""
        boundary = event["boundary"]
        ex, ey = event["x"], event["y"]
        lt = self.last_touch_team

        if boundary in ("touchline_top", "touchline_bottom"):
            # Throw-in to the team that did NOT put it out.
            team = self._other(lt) if lt else "A"
            inset = C.THROW_IN_INSET_Y
            y = (C.FIELD_Y_MAX - inset) if boundary == "touchline_top" \
                else (C.FIELD_Y_MIN + inset)
            spot = (min(max(ex, C.FIELD_X_MIN + 3.0), C.FIELD_X_MAX - 3.0), y)
            kind, prefer_gk = "throw_in", False
            self.stats["throw_ins"][team] += 1

        else:
            # Byline crossing: corner (defender out) or goal kick (attacker out).
            # byline_left is Team A's goal line; byline_right is Team B's.
            defending = "A" if boundary == "byline_left" else "B"
            attacking = self._other(defending)
            near_top = ey >= C.FIELD_CENTER_Y
            if lt == defending:
                # Defender put it behind -> corner to the attacking team.
                team = attacking
                gx = (C.FIELD_X_MIN + C.CORNER_INSET) if boundary == "byline_left" \
                    else (C.FIELD_X_MAX - C.CORNER_INSET)
                gy = (C.FIELD_Y_MAX - C.CORNER_INSET) if near_top \
                    else (C.FIELD_Y_MIN + C.CORNER_INSET)
                spot = (gx, gy)
                kind, prefer_gk = "corner", False
                self.stats["corners"][team] += 1
            else:
                # Attacker (or unknown) put it behind -> goal kick to defenders.
                team = defending
                gx = (C.FIELD_X_MIN + C.GOAL_KICK_INSET_X) \
                    if boundary == "byline_left" \
                    else (C.FIELD_X_MAX - C.GOAL_KICK_INSET_X)
                spot = (gx, C.FIELD_CENTER_Y)
                kind, prefer_gk = "goal_kick", True
                self.stats["goal_kicks"][team] += 1

        self.events.append({
            "tick": self.tick, "type": kind, "team": team,
            "boundary": boundary,
        })
        self._place_restart(team, spot, prefer_gk=prefer_gk)

    def _place_restart(self, team: str, spot: Tuple[float, float],
                       prefer_gk: bool) -> None:
        """Hand instant possession of a dead ball to ``team`` at ``spot``.

        Player positions are untouched (play resumes live); only the ball and
        possession are reset.  A short protection window keeps opponents from
        stealing the restart at point-blank range.
        """
        # Clear any stale possession flags across the pitch.
        for a in self.all_agents:
            a.has_ball = False
        self.ball.reset(spot[0], spot[1])

        squad = self._squad(team)
        eligible = [a for a in squad if a.cooldown_remaining <= 0]
        if not eligible:
            eligible = list(squad)

        def sort_key(a: Agent):
            # Goal kicks favour the keeper; everything else the nearest player.
            gk_pref = 0 if (prefer_gk and a.is_gk) else 1
            return (gk_pref, distance((a.x, a.y), spot))

        taker = min(eligible, key=sort_key)
        # The taker steps onto the mark to take the set piece (instant
        # possession: only the taker relocates; play resumes live from there).
        taker.x, taker.y = spot
        taker.orientation = 0.0 if team == "A" else 180.0
        taker.has_ball = True
        self.ball.possessed_by = taker.id
        self._carry_ball(taker)

        self.last_touch_id = taker.id
        self.last_touch_team = team
        self._offside_watch = None
        self._last_kick = None
        self._set_piece_protect = {
            "team": team,
            "until_tick": self.tick + C.SET_PIECE_PROTECT_TICKS,
        }

    # ----------------------------------------------------------------- goals
    def _record_goal(self, scoring_team: str) -> None:
        self.score[scoring_team] += 1
        shooter = self._agent_by_id(self.last_shooter_id) \
            if self.last_shooter_id else None
        self.events.append({
            "tick": self.tick, "type": "goal",
            "team": scoring_team,
            "team_name": self.team_names[scoring_team],
            "scorer": shooter.id if shooter else None,
            "score": dict(self.score),
        })
        # Golden-goal check during overtime ends the match immediately.
        if self.phase == "overtime":
            self._winner = scoring_team
        # Reset to a kickoff configuration (Section 2.2).
        self._kickoff()

    # ------------------------------------------------------------ shootout
    def _penalty_shootout(self) -> str:
        """Best-of-5 attribute-driven penalty shootout, then sudden death."""
        self.events.append({"tick": self.tick, "type": "shootout_start"})
        gk_a = next(a for a in self.squad_a if a.is_gk)
        gk_b = next(a for a in self.squad_b if a.is_gk)
        shooters_a = [a for a in self.squad_a if not a.is_gk]
        shooters_b = [a for a in self.squad_b if not a.is_gk]
        self.rng.shuffle(shooters_a)
        self.rng.shuffle(shooters_b)

        score_a = score_b = 0
        for i in range(5):
            if i < len(shooters_a):
                if self._penalty(shooters_a[i], gk_b):
                    score_a += 1
            if i < len(shooters_b):
                if self._penalty(shooters_b[i], gk_a):
                    score_b += 1

        # Sudden death.
        sd = 0
        while score_a == score_b:
            a = shooters_a[sd % len(shooters_a)]
            b = shooters_b[sd % len(shooters_b)]
            if self._penalty(a, gk_b):
                score_a += 1
            if self._penalty(b, gk_a):
                score_b += 1
            sd += 1
            if sd > 20:  # safety valve
                break

        winner = "A" if score_a > score_b else "B"
        self.events.append({
            "tick": self.tick, "type": "shootout_result",
            "score_a": score_a, "score_b": score_b, "winner": winner,
        })
        return winner

    def _penalty(self, shooter: Agent, keeper: Agent) -> bool:
        """Resolve one penalty kick; returns True if a goal was scored."""
        # Shooter accuracy improves with SHO/PASS; keeper save with REF/POS.
        shooter_skill = (shooter.sho + shooter.pass_) / 2.0 / 10.0
        keeper_skill = (keeper.ref + keeper.pos) / 2.0 / 10.0
        # Net probability of scoring.
        p_goal = 0.55 + 0.30 * (shooter_skill - keeper_skill)
        p_goal = max(0.20, min(0.85, p_goal))
        scored = self.rng.random() < p_goal
        self.events.append({
            "tick": self.tick, "type": "penalty",
            "shooter": shooter.id, "keeper": keeper.id,
            "scored": scored,
        })
        return scored

    # -------------------------------------------------------------- results
    def result(self) -> Dict[str, Any]:
        winner = self._winner or (
            "A" if self.score["A"] > self.score["B"] else "B"
        )
        loser = "B" if winner == "A" else "A"
        return {
            "team_a": self.team_names["A"],
            "team_b": self.team_names["B"],
            "score_a": self.score["A"],
            "score_b": self.score["B"],
            "winner": winner,
            "winner_name": self.team_names[winner],
            "loser_name": self.team_names[loser],
            "winner_goals": self.score[winner],
            "loser_goals": self.score[loser],
            "ticks": self.tick,
            "phase": self.phase,
            "stats": {k: dict(v) for k, v in self.stats.items()},
            "events": list(self.events),
        }
