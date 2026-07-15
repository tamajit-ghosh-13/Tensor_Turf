#!/usr/bin/env python3
"""
evaluate.py — play your submission against an opponent and print the result
===========================================================================
Runs ONE full match on the same engine the organiser uses to judge the
tournament (``api/evaluate.py``), so the verdict — score, winner and the
FIFA-style stats — is exactly what you would get on the server. The engine
source is included so you can study the physics, scoring and goalkeeper
mechanics.  This thin wrapper is all you need to drive it.

Usage
-----
    # your submission folder vs the bundled demo agent
    python evaluate.py --a ./my_submission

    # two folders — or single agents.py files — head to head
    python evaluate.py --a ./me/agents.py --b ./examples/demo_agent.py \
                       --name-a "My FC" --name-b "Demo" --ticks 1800 --seed 7

    # two TRAINED models against each other — point each side at its own weights
    python evaluate.py --a ./v1 --weights-a ./v1/weights \
                       --b ./v2 --weights-b ./v2/weights --name-a V1 --name-b V2

Each side may be a submission FOLDER (with ``agents.py`` [+ ``squad.json``]) or a
single ``.py`` file. Pass ``--weights-a`` / ``--weights-b`` to load each side's
trained ``agent_00_dqn.pt .. agent_10_dqn.pt`` (each side loads its OWN weights).
Add ``--out result.json`` to also save the full verdict.

Notes
-----
* Regulation is 90:00 (two 45:00 halves) with a re-centre at half-time, then
  golden-goal extra time and a penalty shootout if still level — same as a real
  tie. The clock in the verdict is ``MM:SS`` up to 90:00.
* This does NOT train. Your ``agents.py`` loads trained weights only if a
  ``WEIGHTS_DIR`` is populated; run your ``train_rl_agent.py`` first for that.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:                      # so `api` imports when run anywhere
    sys.path.insert(0, HERE)

from api.evaluate import evaluate_submissions  # compiled engine (not readable)

DEMO = os.path.join(HERE, "examples", "demo_agent.py")


def _as_submission_dir(path: str, tmp_root: str, tag: str) -> str:
    """Return a folder containing ``agents.py``; wrap a bare ``.py`` if needed."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        if not os.path.isfile(os.path.join(path, "agents.py")):
            sys.exit(f"error: {path!r} has no agents.py")
        return path
    if os.path.isfile(path) and path.endswith(".py"):
        d = os.path.join(tmp_root, tag)
        os.makedirs(d, exist_ok=True)
        shutil.copyfile(path, os.path.join(d, "agents.py"))
        sibling = os.path.join(os.path.dirname(path), "squad.json")
        if os.path.isfile(sibling):
            shutil.copyfile(sibling, os.path.join(d, "squad.json"))
        return d
    sys.exit(f"error: {path!r} is not a folder or a .py file")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Evaluate a submission against an opponent.")
    p.add_argument("--a", required=True, help="your submission folder or agents.py")
    p.add_argument("--b", default=DEMO,
                   help="opponent folder or .py (default: the bundled demo agent)")
    p.add_argument("--name-a", default="Team A")
    p.add_argument("--name-b", default="Team B")
    p.add_argument("--weights-a", default=None,
                   help="folder with side A's trained agent_00_dqn.pt .. "
                        "agent_10_dqn.pt (its WEIGHTS_DIR). Omit for heuristic play.")
    p.add_argument("--weights-b", default=None,
                   help="folder with side B's trained weights")
    p.add_argument("--ticks", type=int, default=1800, help="regulation length")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default=None, help="write the full verdict JSON here")
    args = p.parse_args()

    tmp = tempfile.mkdtemp(prefix="fifa_eval_")
    try:
        sub_a = _as_submission_dir(args.a, tmp, "a")
        sub_b = _as_submission_dir(args.b, tmp, "b")
        verdict = evaluate_submissions(
            sub_a, args.name_a, sub_b, args.name_b,
            seed=args.seed, ticks=args.ticks,
            weights_a=args.weights_a, weights_b=args.weights_b,
            work_dir=os.path.join(tmp, "work"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    s = verdict["score"]
    st = verdict.get("stats", {})
    poss = st.get("possession_pct", {})
    print("\n================  RESULT  ================")
    print(f"  {verdict['teams']['home']}  {s['home']} - {s['away']}  "
          f"{verdict['teams']['away']}")
    print(f"  winner: {verdict['winner_name']}  "
          f"(decided by {verdict['decided_by']})")
    print(f"  duration: {verdict['duration']}   "
          f"possession: {poss.get('home', '?')}% / {poss.get('away', '?')}%")
    print(f"  shots: {st.get('shots', {})}   "
          f"on target: {st.get('shots_on_target', {})}")
    print("=========================================\n")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(verdict, fh, indent=2)
        print(f"full verdict written to {args.out}")
    else:
        print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
