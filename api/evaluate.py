"""
evaluate.py
===========
Organizer-side evaluation harness for the FIFA AI World Cup competition.

This is the piece a backend calls to turn two *participant submissions* into a
match **verdict** — a self-contained JSON document the frontend/DB can store
and render (score, winner, and real-FIFA-style stats: possession, shots, shots
on target, corners, offsides, passes, tackles).

A submission is a folder containing:

    <submission>/
      train_rl_agent.py     # trains 11 DQNs, saves agent_00_dqn.pt .. agent_10_dqn.pt
      agents.py             # exposes make_policy(squad, seed) -> policy with .decide

Pipeline (per submission):
  1. (optional) run ``train_rl_agent.py`` in a sandboxed subprocess with the
     organiser env vars (SAVE_MODEL / WEIGHTS_DIR / EPISODES / TICKS / SEED).
  2. import ``agents.py`` and grab ``make_policy`` (weights loaded from
     ``WEIGHTS_DIR``).  A broken/malicious submission is caught and forfeits
     gracefully to a safe idle policy — the platform never crashes.
  3. play the match on the real engine (corners, offside, shootout all active).

CLI
---
    # single match, no training (assumes weights already present, or heuristic)
    python -m fifa_ai_world_cup.evaluate match \\
        --sub-a submissions/alice --name-a "Alice FC" \\
        --sub-b submissions/bob   --name-b "Bob Utd" \\
        --out verdict.json

    # two-legged tie with training first
    python -m fifa_ai_world_cup.evaluate tie --train \\
        --sub-a submissions/alice --sub-b submissions/bob --out tie.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from . import config as C
from .agent import generate_squad, Agent
from .game import Game
from .skills import spec_to_squad, validate_squad_spec

PolicyFactory = Callable[[List[Agent], Optional[int]], Any]


def load_squad_spec(sub_dir: str) -> Optional[List[Dict[str, Any]]]:
    """Read and validate a submission's ``squad.json`` (None if absent/invalid).

    An invalid squad forfeits to the default balanced squad rather than
    crashing the match; the reason is logged to stderr.
    """
    path = os.path.join(sub_dir, "squad.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            spec = json.load(fh)
        validate_squad_spec(spec)
        return spec
    except Exception as exc:  # noqa: BLE001
        print(f"[evaluate] invalid squad.json in {sub_dir!r}: {exc!r}",
              file=sys.stderr)
        return None


def _build_squad(team: str, spec: Optional[List[Dict[str, Any]]],
                 seed: int) -> List[Agent]:
    if spec is not None:
        return spec_to_squad(team, spec, validate=False)
    return generate_squad(team, rng=random.Random(seed))


# ===========================================================================
# Submission loading (sandboxed)
# ===========================================================================
class _IdlePolicy:
    """A forfeiting policy: every agent idles.  Used when a submission fails."""

    def decide(self, states, team):
        return [{"action_type": "IDLE"} for _ in states]


class _SafePolicy:
    """Wrap a participant policy so it can never crash — or stall — a match.

    * A raised exception forfeits only that tick (idle fallback).
    * ``budget_seconds`` caps the *cumulative* wall-clock spent in ``decide``
      across the match. Once a submission burns its budget (e.g. an oversized
      model that is too slow at inference) it forfeits the rest of the match by
      idling, and ``forfeited`` is set so the caller can record it. 0 disables.
    """

    def __init__(self, inner: Any, budget_seconds: float = 0.0):
        self._inner = inner
        self._fallback = _IdlePolicy()
        self._budget = float(budget_seconds or 0.0)
        self._spent = 0.0
        self.forfeited = False

    def decide(self, states, team):
        if self.forfeited:
            return self._fallback.decide(states, team)
        t0 = time.monotonic()
        result = None
        try:
            out = (self._inner.decide(states, team)
                   if hasattr(self._inner, "decide")
                   else self._inner(states, team))
            if isinstance(out, (list, tuple)) and len(out) == len(states):
                result = list(out)
        except Exception:  # noqa: BLE001 - defensive: forfeit this tick only
            pass
        self._spent += time.monotonic() - t0
        if self._budget and self._spent > self._budget:
            self.forfeited = True
        return result if result is not None else self._fallback.decide(states, team)


def load_policy_factory(sub_dir: str, weights_dir: Optional[str] = None,
                        decide_budget: float = 0.0) -> PolicyFactory:
    """Import ``agents.py`` from a submission and return its ``make_policy``.

    The returned factory always yields a ``_SafePolicy``.  If the submission is
    missing, unimportable, or ``make_policy`` raises, the factory falls back to
    a safe idle policy so the caller can proceed with a graceful forfeit.
    ``decide_budget`` (seconds) caps the total inference compute per match.
    """
    agents_path = os.path.join(sub_dir, "agents.py")

    def _forfeit_factory(squad=None, seed=None) -> _SafePolicy:
        return _SafePolicy(_IdlePolicy(), decide_budget)

    if not os.path.isfile(agents_path):
        return _forfeit_factory

    # Each submission's weights live in a PER-SIDE directory. We pin WEIGHTS_DIR
    # now (so a submission reading it at import time is correct) AND again inside
    # the factory below, right before make_policy runs. Both matters because a
    # match builds *both* sides' factories before invoking either — so without
    # the re-assert at invocation, whichever side was loaded last would leak its
    # WEIGHTS_DIR to both, and any agents.py that reads it inside make_policy
    # (the reference + template both do) would load the wrong side's weights.
    # Falling back to a per-submission "weights" subdir keeps a weightless side
    # isolated instead of inheriting the other side's directory.
    weights_root = weights_dir or os.path.join(sub_dir, "weights")
    os.environ["WEIGHTS_DIR"] = weights_root

    mod_name = f"_submission_{uuid.uuid4().hex}"
    try:
        spec = importlib.util.spec_from_file_location(mod_name, agents_path)
        module = importlib.util.module_from_spec(spec)
        # Let the submission import sibling files (e.g. its own train module).
        sys.path.insert(0, sub_dir)
        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        finally:
            if sys.path and sys.path[0] == sub_dir:
                sys.path.pop(0)
        make_policy = getattr(module, "make_policy")
    except Exception as exc:  # noqa: BLE001
        print(f"[evaluate] submission {sub_dir!r} failed to load: {exc!r}",
              file=sys.stderr)
        return _forfeit_factory

    def _factory(squad=None, seed=None) -> _SafePolicy:
        # Re-assert THIS side's weights dir right before building the policy, so
        # each side loads its own weights even though both factories exist first.
        os.environ["WEIGHTS_DIR"] = weights_root
        try:
            return _SafePolicy(make_policy(squad, seed), decide_budget)
        except Exception as exc:  # noqa: BLE001
            print(f"[evaluate] make_policy failed for {sub_dir!r}: {exc!r}",
                  file=sys.stderr)
            return _SafePolicy(_IdlePolicy(), decide_budget)

    return _factory


def run_training(sub_dir: str, weights_dir: str, *, episodes: int = 20,
                 ticks: int = 400, seed: int = 42, timeout: int = 3600
                 ) -> Dict[str, Any]:
    """Run a submission's ``train_rl_agent.py`` in a sandboxed subprocess.

    The organiser controls training via env vars (``SAVE_MODEL`` /
    ``WEIGHTS_DIR`` / ``EPISODES`` / ``TICKS`` / ``SEED``).  Returns a small
    report; never raises (a failed/absent trainer simply means the submission
    plays with whatever weights exist, or the heuristic fallback).
    """
    os.makedirs(weights_dir, exist_ok=True)
    script = os.path.join(sub_dir, "train_rl_agent.py")
    if not os.path.isfile(script):
        return {"trained": False, "reason": "no train_rl_agent.py"}

    env = dict(os.environ)
    env.update({
        "SAVE_MODEL": weights_dir, "WEIGHTS_DIR": weights_dir,
        "EPISODES": str(episodes), "TICKS": str(ticks), "SEED": str(seed),
    })
    try:
        proc = subprocess.run(
            [sys.executable, script, "--episodes", str(episodes),
             "--ticks", str(ticks), "--seed", str(seed)],
            cwd=sub_dir, env=env, capture_output=True, text=True,
            timeout=timeout,
        )
        return {
            "trained": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
    except subprocess.TimeoutExpired:
        return {"trained": False, "reason": f"timeout after {timeout}s"}
    except Exception as exc:  # noqa: BLE001
        return {"trained": False, "reason": repr(exc)}


def _proc_vram_mb(pid: int) -> int:
    """Per-process GPU memory in MB via nvidia-smi; 0 if no GPU / not found."""
    if not shutil.which("nvidia-smi"):
        return 0
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return 0
    total = 0
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
            try:
                total += int(float(parts[1]))
            except ValueError:
                pass
    return total


def _proc_tree(pid: int) -> List[int]:
    """``pid`` plus all its descendant pids (so RAM covers spawned workers)."""
    try:
        out = subprocess.run(["ps", "-eo", "pid=,ppid="],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return [pid]
    children: Dict[int, List[int]] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    seen: List[int] = []
    stack = [pid]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.append(p)
        stack.extend(children.get(p, []))
    return seen


def _proc_rss_mb(pid: int) -> int:
    """Resident memory (RAM) in MB for a process tree via ``ps``; 0 on error.

    Unlike VRAM this needs no GPU, so it is the meaningful memory cap on a
    CPU host: a submission that trains an oversized model is caught here.
    """
    pids = _proc_tree(pid)
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", ",".join(str(p) for p in pids)],
            capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return 0
    total_kb = sum(int(tok) for tok in out.split() if tok.strip().isdigit())
    return total_kb // 1024


def run_training_guarded(sub_dir: str, weights_dir: str, *, max_seconds: int,
                         max_vram_mb: int = 0, max_ram_mb: int = 0,
                         episodes: int = 20, ticks: int = 400, seed: int = 42,
                         poll: float = 1.0) -> Dict[str, Any]:
    """Run ``train_rl_agent.py`` under time + VRAM + RAM caps.

    Returns a report including ``disqualified`` (True if a cap was breached),
    ``reason``, ``elapsed_seconds``, ``peak_vram_mb`` and ``peak_ram_mb``. A
    trainer that merely fails or is absent is NOT a disqualification — only
    breaching a cap is. VRAM only bites on a GPU host; the RAM cap is the
    equivalent guardrail on a CPU host.
    """
    os.makedirs(weights_dir, exist_ok=True)
    report = {"trained": False, "disqualified": False, "reason": None,
              "elapsed_seconds": 0.0, "peak_vram_mb": 0, "peak_ram_mb": 0,
              "returncode": None}
    script = os.path.join(sub_dir, "train_rl_agent.py")
    if not os.path.isfile(script):
        report["reason"] = "no train_rl_agent.py"
        return report

    env = dict(os.environ)
    env.update({"SAVE_MODEL": weights_dir, "WEIGHTS_DIR": weights_dir,
                "EPISODES": str(episodes), "TICKS": str(ticks), "SEED": str(seed)})
    out_path = os.path.join(weights_dir, "_train.out")
    start = time.monotonic()
    peak = 0
    peak_ram = 0
    with open(out_path, "w") as out_fh:
        proc = subprocess.Popen(
            [sys.executable, script, "--episodes", str(episodes),
             "--ticks", str(ticks), "--seed", str(seed)],
            cwd=sub_dir, env=env, stdout=out_fh, stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                elapsed = time.monotonic() - start
                if elapsed > max_seconds:
                    proc.kill()
                    report.update(disqualified=True,
                                  reason=f"training exceeded time limit "
                                         f"({max_seconds}s)")
                    break
                if max_vram_mb:
                    used = _proc_vram_mb(proc.pid)
                    peak = max(peak, used)
                    if used > max_vram_mb:
                        proc.kill()
                        report.update(disqualified=True,
                                      reason=f"training exceeded VRAM limit "
                                             f"({max_vram_mb} MB; used {used} MB)")
                        break
                if max_ram_mb:
                    ram = _proc_rss_mb(proc.pid)
                    peak_ram = max(peak_ram, ram)
                    if ram > max_ram_mb:
                        proc.kill()
                        report.update(disqualified=True,
                                      reason=f"training exceeded RAM limit "
                                             f"({max_ram_mb} MB; used {ram} MB)")
                        break
                time.sleep(poll)
            proc.wait(timeout=10)
        except Exception as exc:  # noqa: BLE001
            report["reason"] = repr(exc)
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    report["elapsed_seconds"] = round(time.monotonic() - start, 1)
    report["peak_vram_mb"] = peak
    report["peak_ram_mb"] = peak_ram
    report["returncode"] = proc.returncode
    if not report["disqualified"]:
        report["trained"] = proc.returncode == 0
    return report


# ===========================================================================
# Match play + verdict
# ===========================================================================
def _pct(a: int, b: int) -> Dict[str, float]:
    total = a + b
    if total <= 0:
        return {"home": 50.0, "away": 50.0}
    return {"home": round(100.0 * a / total, 1), "away": round(100.0 * b / total, 1)}


def _hv(d: Dict[str, int]) -> Dict[str, int]:
    """Re-key an {"A","B"} stat as {"home","away"}."""
    return {"home": d.get("A", 0), "away": d.get("B", 0)}


def verdict_from_result(result: Dict[str, Any], name_home: str,
                        name_away: str, match_id: Optional[str] = None,
                        reg_ticks: Optional[int] = None
                        ) -> Dict[str, Any]:
    """Convert an engine result (A/B) into the home/away frontend verdict.

    ``reg_ticks`` is the regulation length used for the match; it sets the
    tick -> MM:SS clock scale so goals report a match clock, not a raw tick.
    """
    st = result["stats"]
    winner = "home" if result["winner"] == "A" else "away"
    reg = int(reg_ticks or C.DEFAULT_MATCH_TICKS)

    shootout = None
    for e in result.get("events", []):
        if e.get("type") == "shootout_result":
            shootout = {"home": e.get("score_a", 0), "away": e.get("score_b", 0)}

    goals = [
        {"tick": e["tick"],
         "clock": C.tick_to_clock(e["tick"], reg),
         "half": C.tick_half(e["tick"], reg),
         "team": "home" if e["team"] == "A" else "away",
         "scorer": e.get("scorer")}
        for e in result.get("events", []) if e.get("type") == "goal"
    ]

    return {
        "match_id": match_id or uuid.uuid4().hex,
        "teams": {"home": name_home, "away": name_away},
        "score": {"home": result["score_a"], "away": result["score_b"]},
        "winner": winner,
        "winner_name": name_home if winner == "home" else name_away,
        "decided_by": result["phase"],   # regulation | overtime | shootout
        "shootout": shootout,
        "duration_ticks": result["ticks"],
        "duration": C.tick_to_clock(result["ticks"], reg),
        "stats": {
            "possession_pct": _pct(st["possession_ticks"]["A"],
                                   st["possession_ticks"]["B"]),
            "shots": _hv(st["shots"]),
            "shots_on_target": _hv(st["shots_on_target"]),
            "passes": _hv(st["passes"]),
            "corners": _hv(st["corners"]),
            "offsides": _hv(st["offsides"]),
            "throw_ins": _hv(st["throw_ins"]),
            "goal_kicks": _hv(st["goal_kicks"]),
            "tackles": _hv(st["tackles"]),
        },
        "goals": goals,
        "generated_at": None,   # backend stamps a real timestamp
    }


def play_match(factory_home: PolicyFactory, name_home: str,
               factory_away: PolicyFactory, name_away: str, *,
               seed: int = 7, ticks: int = 1800,
               overtime_ticks: Optional[int] = None,
               spec_home: Optional[List[Dict[str, Any]]] = None,
               spec_away: Optional[List[Dict[str, Any]]] = None,
               match_id: Optional[str] = None,
               renderer: Any = None) -> Dict[str, Any]:
    """Play one decisive match (home = A, away = B) and return its verdict.

    ``spec_home`` / ``spec_away`` are the teams' ``squad.json`` specs (custom
    formation + knobs); ``None`` falls back to a random generated squad. Extra
    time (golden goal) defaults to ET_FRACTION of regulation so it maps to
    ET_MINUTES on the match clock. Pass a headless ``renderer`` to capture the
    match for a replay GIF; the caller owns saving/closing it afterwards.
    """
    if overtime_ticks is None:
        overtime_ticks = int(round(ticks * C.ET_FRACTION))
    squad_a = _build_squad("A", spec_home, seed * 7 + 1)
    squad_b = _build_squad("B", spec_away, seed * 7 + 2)
    policy_a = factory_home(squad_a, seed)
    policy_b = factory_away(squad_b, seed + 1)
    game = Game(name_home, squad_a, policy_a, name_away, squad_b, policy_b,
                max_ticks=ticks, overtime_ticks=overtime_ticks, seed=seed,
                renderer=renderer)
    result = game.run()
    verdict = verdict_from_result(result, name_home, name_away, match_id,
                                  reg_ticks=ticks)
    # Record sides that exhausted their inference-compute budget (too slow).
    forfeit = {"home": getattr(policy_a, "forfeited", False),
               "away": getattr(policy_b, "forfeited", False)}
    if forfeit["home"] or forfeit["away"]:
        verdict["compute_forfeit"] = forfeit
    return verdict


def play_tie(factory_a: PolicyFactory, name_a: str,
             factory_b: PolicyFactory, name_b: str, *,
             seed: int = 7, ticks: int = 1800, legs: int = 2,
             spec_a: Optional[List[Dict[str, Any]]] = None,
             spec_b: Optional[List[Dict[str, Any]]] = None,
             tie_id: Optional[str] = None) -> Dict[str, Any]:
    """Two-legged tie with side swap; aggregate goals, shootout decider on tie."""
    tie_id = tie_id or uuid.uuid4().hex
    leg_verdicts: List[Dict[str, Any]] = []

    # Leg 1: A home, B away.
    leg_verdicts.append(play_match(factory_a, name_a, factory_b, name_b,
                                   seed=seed, ticks=ticks,
                                   spec_home=spec_a, spec_away=spec_b,
                                   match_id=f"{tie_id}-leg1"))
    agg_a = leg_verdicts[0]["score"]["home"]
    agg_b = leg_verdicts[0]["score"]["away"]

    if legs >= 2:
        # Leg 2: B home, A away (swap sides for fairness).
        v2 = play_match(factory_b, name_b, factory_a, name_a,
                        seed=seed + 101, ticks=ticks,
                        spec_home=spec_b, spec_away=spec_a,
                        match_id=f"{tie_id}-leg2")
        leg_verdicts.append(v2)
        agg_a += v2["score"]["away"]   # A played away in leg 2
        agg_b += v2["score"]["home"]

    if agg_a > agg_b:
        winner, decided_by = name_a, "aggregate"
    elif agg_b > agg_a:
        winner, decided_by = name_b, "aggregate"
    else:
        # Aggregate tie -> a decider that goes to a penalty shootout.
        decider = play_match(factory_a, name_a, factory_b, name_b,
                             seed=seed + 202, ticks=max(300, ticks // 4),
                             overtime_ticks=300,
                             spec_home=spec_a, spec_away=spec_b,
                             match_id=f"{tie_id}-decider")
        leg_verdicts.append(decider)
        winner = name_a if decider["winner"] == "home" else name_b
        decided_by = decider["decided_by"]  # overtime | shootout

    return {
        "tie_id": tie_id,
        "teams": {"a": name_a, "b": name_b},
        "aggregate": {"a": agg_a, "b": agg_b},
        "winner": winner,
        "decided_by": decided_by,
        "legs": leg_verdicts,
    }


# ===========================================================================
# High-level: evaluate two submission folders
# ===========================================================================
def evaluate_submissions(sub_a: str, name_a: str, sub_b: str, name_b: str, *,
                         train: bool = False, mode: str = "match",
                         seed: int = 7, ticks: int = 1800, legs: int = 2,
                         decide_budget: float = 0.0,
                         weights_a: Optional[str] = None,
                         weights_b: Optional[str] = None,
                         work_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load (optionally train) two submissions and produce a verdict.

    ``decide_budget`` (seconds, 0 = off) caps each side's cumulative inference
    compute per match, so an oversized/too-slow model forfeits instead of
    stalling play. Pass ``weights_a`` / ``weights_b`` to point each side at an
    existing trained-weights folder (its ``WEIGHTS_DIR``) so two already-trained
    models can play head-to-head without retraining; ``train=True`` overrides
    them by training each side fresh first.
    """
    work_dir = work_dir or os.path.join(os.getcwd(), "_eval_work")
    os.makedirs(work_dir, exist_ok=True)
    wd_a = os.path.join(work_dir, "weights_a")
    wd_b = os.path.join(work_dir, "weights_b")

    train_report = {}
    if train:
        train_report["a"] = run_training(sub_a, wd_a, seed=seed)
        train_report["b"] = run_training(sub_b, wd_b, seed=seed + 1)
        weights_a, weights_b = wd_a, wd_b

    factory_a = load_policy_factory(sub_a, weights_a, decide_budget)
    factory_b = load_policy_factory(sub_b, weights_b, decide_budget)
    spec_a = load_squad_spec(sub_a)
    spec_b = load_squad_spec(sub_b)

    if mode == "tie":
        verdict = play_tie(factory_a, name_a, factory_b, name_b,
                           seed=seed, ticks=ticks, legs=legs,
                           spec_a=spec_a, spec_b=spec_b)
    else:
        verdict = play_match(factory_a, name_a, factory_b, name_b,
                             seed=seed, ticks=ticks,
                             spec_home=spec_a, spec_away=spec_b)
    if train_report:
        verdict["training"] = train_report
    return verdict


# ===========================================================================
# CLI
# ===========================================================================
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sub-a", required=True, help="path to submission A folder")
    p.add_argument("--sub-b", required=True, help="path to submission B folder")
    p.add_argument("--name-a", default="Team A")
    p.add_argument("--name-b", default="Team B")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--ticks", type=int, default=1800)
    p.add_argument("--train", action="store_true",
                   help="run each submission's train_rl_agent.py first")
    p.add_argument("--out", default=None, help="write the verdict JSON here")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fifa-evaluate",
        description="Evaluate FIFA AI World Cup submissions into a verdict.")
    sub = parser.add_subparsers(dest="command", required=True)

    mp = sub.add_parser("match", help="single decisive match")
    _add_common(mp)

    tp = sub.add_parser("tie", help="two-legged tie (side swap + shootout decider)")
    _add_common(tp)
    tp.add_argument("--legs", type=int, default=2, choices=[1, 2])

    args = parser.parse_args(argv)
    verdict = evaluate_submissions(
        args.sub_a, args.name_a, args.sub_b, args.name_b,
        train=args.train, mode=args.command, seed=args.seed,
        ticks=args.ticks, legs=getattr(args, "legs", 2))

    text = json.dumps(verdict, indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"[evaluate] wrote verdict -> {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
