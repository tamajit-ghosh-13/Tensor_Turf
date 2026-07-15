#!/usr/bin/env python3
"""Validate a submission LOCALLY before uploading — no server, no VM load.

Run from the participant kit (so the ``api`` package is importable):

    python validate_submission.py                 # checks ./ (agents.py, squad.json)
    python validate_submission.py path/to/my_sub  # checks another folder

It runs three checks:
  1. Static  — required files present; squad.json obeys the budget/formation
     rules (same rules the server enforces at upload).
  2. Load    — imports your agents.py, calls make_policy(squad, seed), and
     confirms the returned policy exposes .decide(states, team).
  3. Smoke   — plays a short match (your policy vs a reference opponent) so a
     crash in .decide surfaces here, on your machine, not in the tournament.

Exit code 0 = all good; 1 = something failed (details printed).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # make the sibling `api` package importable

# --- limits (organiser defaults; the web validator uses the live values) ---- #
MAX_TRAIN_SECONDS = 3600          # 1 hour
MAX_VRAM_MB = 4096                # 4 GB (GPU only)
MAX_RAM_MB = 4096                 # 4 GB system memory (the CPU-host cap)
MAX_FILE_MB = 10                  # per file (agents.py, train_rl_agent.py)
MAX_TOTAL_MB = 20                 # combined
# Must match the server's list AND requirements.txt — only libraries actually
# installed on the training/inference machines are allowed (importing anything
# else would crash your submission on the VM).
ALLOWED_LIBRARIES = {
    "torch", "numpy", "gymnasium", "api", "fifa_ai_world_cup",
    "tensorflow", "keras", "jax", "jaxlib", "flax", "optax", "chex",
    "sklearn", "scipy", "pandas",
    "__future__", "math", "random", "json", "os", "sys", "time",
    "collections", "itertools", "functools", "dataclasses", "typing",
    "abc", "copy", "heapq", "enum", "statistics", "bisect", "re",
}

# --- squad rules (mirror of the engine / backend; keep in sync) ------------ #
SQUAD_SIZE = 11
KNOB_MIN, KNOB_MAX = 10, 100
OUTFIELD_KNOBS = ("running", "dribbling", "shot_power", "pass_accuracy",
                  "pass_range", "tackling", "vision")
GK_KNOBS = ("reflexes", "positioning", "agility", "handling", "distribution")
OUTFIELD_BUDGET, GK_BUDGET = 420, 300
HOME_X, GK_HOME_X, HOME_Y = (3.0, 55.0), (2.0, 12.0), (3.0, 57.0)

GREEN, RED, YEL, RST = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def ok(msg):   print(f"{GREEN}  ok {RST}{msg}")
def bad(msg):  print(f"{RED} FAIL{RST} {msg}")
def warn(msg): print(f"{YEL} warn{RST} {msg}")


def imports_of(code):
    """Top-level module names imported by ``code`` (['<syntax-error>'] if bad)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["<syntax-error>"]
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level and not node.module:
                continue  # relative sibling import (own module) — allowed
            if node.module:
                mods.append(node.module.split(".")[0])
    return mods


def proc_vram_mb(pid):
    if not shutil.which("nvidia-smi"):
        return 0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return 0
    total = 0
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 2 and p[0].isdigit() and int(p[0]) == pid:
            try:
                total += int(float(p[1]))
            except ValueError:
                pass
    return total


def proc_rss_mb(pid):
    """Resident memory (RAM) in MB for a process tree via ``ps``; 0 on error."""
    try:
        tree = subprocess.run(["ps", "-eo", "pid=,ppid="],
                              capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return 0
    kids = {}
    for line in tree.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            kids.setdefault(int(parts[1]), []).append(int(parts[0]))
    seen, stack = [], [pid]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.append(p); stack.extend(kids.get(p, []))
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", ",".join(map(str, seen))],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return 0
    return sum(int(t) for t in out.split() if t.strip().isdigit()) // 1024


def measure_training(target):
    """Run train_rl_agent.py; return
    (elapsed, peak_vram, peak_ram, time_breach, vram_breach, ram_breach).

    Training time here is your LOCAL machine's time — only informative, since
    the tournament measures time on the SERVER. VRAM and RAM are roughly
    hardware-independent (they track your model size), so a breach here means
    you'd breach on the server too.
    """
    script = os.path.join(target, "train_rl_agent.py")
    weights = tempfile.mkdtemp()
    env = dict(os.environ)
    env.update({"SAVE_MODEL": weights, "WEIGHTS_DIR": weights,
                "EPISODES": "20", "TICKS": "400", "SEED": "42"})
    start = time.monotonic()
    peak = peak_ram = 0
    time_breach = vram_breach = ram_breach = False
    proc = subprocess.Popen([sys.executable, script], cwd=target, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while proc.poll() is None:
        el = time.monotonic() - start
        if el > MAX_TRAIN_SECONDS:  # bound the local run at the server budget
            proc.kill(); time_breach = True; break
        used = proc_vram_mb(proc.pid)
        peak = max(peak, used)
        if used > MAX_VRAM_MB:
            proc.kill(); vram_breach = True; break
        ram = proc_rss_mb(proc.pid)
        peak_ram = max(peak_ram, ram)
        if ram > MAX_RAM_MB:
            proc.kill(); ram_breach = True; break
        time.sleep(1.0)
    try:
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        pass
    return (round(time.monotonic() - start, 1), peak, peak_ram,
            time_breach, vram_breach, ram_breach)


def check_squad_entry(e, keys, budget, xr, where, errs):
    total = 0
    for k in keys:
        v = e.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            errs.append(f"{where}: knob {k!r} must be a number"); continue
        if not (KNOB_MIN <= v <= KNOB_MAX):
            errs.append(f"{where}: knob {k!r}={v} out of [{KNOB_MIN},{KNOB_MAX}]")
        total += v if isinstance(v, (int, float)) else 0
    if total > budget:
        errs.append(f"{where}: knob total {total} exceeds budget {budget}")
    try:
        x, y = float(e["x"]), float(e["y"])
    except (KeyError, TypeError, ValueError):
        errs.append(f"{where}: needs numeric 'x' and 'y'"); return
    if not (xr[0] <= x <= xr[1]):
        errs.append(f"{where}: x={x} out of bounds {xr}")
    if not (HOME_Y[0] <= y <= HOME_Y[1]):
        errs.append(f"{where}: y={y} out of bounds {HOME_Y}")


def validate_squad_json(text):
    errs = []
    try:
        spec = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        return [f"squad.json is not valid JSON: {exc}"]
    players = spec.get("players", spec) if isinstance(spec, dict) else spec
    if not isinstance(players, list) or len(players) != SQUAD_SIZE:
        return [f"squad.json must list exactly {SQUAD_SIZE} players"]
    for i, e in enumerate(players):
        if not isinstance(e, dict):
            errs.append(f"player {i}: must be an object"); continue
        if i == 0:
            check_squad_entry(e, GK_KNOBS, GK_BUDGET, GK_HOME_X, "GK (slot 0)", errs)
        else:
            check_squad_entry(e, OUTFIELD_KNOBS, OUTFIELD_BUDGET, HOME_X,
                              f"player {i}", errs)
    return errs


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    measure = "--train" in sys.argv
    target = os.path.abspath(args[0]) if args else os.getcwd()
    print(f"Validating submission in: {target}\n")
    failed = False

    # 1. Static -------------------------------------------------------------
    print("[1/4] Static checks — files, size & libraries")
    agents_path = os.path.join(target, "agents.py")
    if not os.path.isfile(agents_path):
        bad("agents.py is REQUIRED and was not found."); return 1
    ok("agents.py present")
    train_path = os.path.join(target, "train_rl_agent.py")
    if os.path.isfile(train_path):
        ok("train_rl_agent.py present (optional)")
    else:
        warn("no train_rl_agent.py — your agents.py heuristic will play as-is")

    # file sizes vs limits
    mb = 1024 * 1024
    a_sz = os.path.getsize(agents_path)
    t_sz = os.path.getsize(train_path) if os.path.isfile(train_path) else 0
    total = a_sz + t_sz
    print(f"       sizes: agents.py {a_sz/mb:.2f} MB, "
          f"train_rl_agent.py {t_sz/mb:.2f} MB, total {total/mb:.2f} MB "
          f"(limits {MAX_FILE_MB}/{MAX_FILE_MB}/{MAX_TOTAL_MB} MB)")
    if a_sz > MAX_FILE_MB * mb:
        failed = True; bad(f"agents.py exceeds {MAX_FILE_MB} MB")
    if t_sz > MAX_FILE_MB * mb:
        failed = True; bad(f"train_rl_agent.py exceeds {MAX_FILE_MB} MB")
    if total > MAX_TOTAL_MB * mb:
        failed = True; bad(f"total exceeds {MAX_TOTAL_MB} MB")
    if total <= MAX_TOTAL_MB * mb and a_sz <= MAX_FILE_MB * mb and t_sz <= MAX_FILE_MB * mb:
        ok("file sizes within limits")

    # allowed libraries
    lib_bad = False
    for p in (agents_path, train_path):
        if not os.path.isfile(p):
            continue
        with open(p) as fh:
            for mod in imports_of(fh.read()):
                if mod == "<syntax-error>":
                    failed = True; lib_bad = True
                    bad(f"{os.path.basename(p)}: Python syntax error")
                elif mod not in ALLOWED_LIBRARIES:
                    failed = True; lib_bad = True
                    bad(f"{os.path.basename(p)}: disallowed library {mod!r}")
    if not lib_bad:
        ok("all imports are in the allowed library list")

    # strictly only .py files (+ the generated squad.json) may be submitted
    stray = [f for f in os.listdir(target)
             if os.path.isfile(os.path.join(target, f))
             and f != "squad.json" and not f.endswith(".py")]
    if stray:
        failed = True
        bad(f"disallowed file(s) — only .py + squad.json are accepted: {stray}")
    else:
        ok("only .py files (+ squad.json) present")

    squad_path = os.path.join(target, "squad.json")
    if os.path.isfile(squad_path):
        with open(squad_path) as fh:
            errs = validate_squad_json(fh.read())
        if errs:
            failed = True
            for e in errs[:10]:
                bad(e)
        else:
            ok("squad.json valid (formation + knob budgets)")
    else:
        warn("no squad.json — a default squad will be used")

    # The load + smoke checks need the compiled engine, which is built for a
    # specific Python. A version mismatch shows up as "bad magic number".
    try:
        from api.agent import generate_squad  # noqa: F401
    except ImportError as exc:
        if "bad magic number" in str(exc):
            print()
            warn("The kit's compiled engine (api/*.pyc) does not match your "
                 "Python version, so the load + match checks can't run here.")
            print(f"       Use the Python version the kit was built for "
                  f"(Python 3.12). You're on {sys.version.split()[0]}.")
            print(f"       Static checks are still valid. Re-run under 3.12 to "
                  f"verify your policy actually runs.")
            return 1
        bad(f"could not import the kit engine: {exc!r}")
        return 1

    # 2. Load ---------------------------------------------------------------
    print("\n[2/4] Load agents.py")
    try:
        from api.agent import generate_squad
        squad = generate_squad("A", rng=random.Random(0))
        sys.path.insert(0, target)
        spec = importlib.util.spec_from_file_location("_participant_agents",
                                                      agents_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "make_policy"):
            bad("agents.py must define make_policy(squad, seed)."); return 1
        policy = mod.make_policy(squad, 0)
        if not hasattr(policy, "decide"):
            bad("make_policy() must return an object with a .decide method.")
            return 1
        ok("make_policy(squad, seed) returned a policy with .decide()")
    except Exception as exc:  # noqa: BLE001
        bad(f"agents.py failed to load / make_policy raised: {exc!r}")
        return 1

    # 3. Smoke match --------------------------------------------------------
    print("\n[3/4] Smoke match (short)")
    try:
        from api.game import Game
        squad_a = generate_squad("A", rng=random.Random(1))
        squad_b = generate_squad("B", rng=random.Random(2))
        pol_a = mod.make_policy(squad_a, 1)
        pol_b = mod.make_policy(squad_b, 2)
        game = Game("You", squad_a, pol_a, "Clone", squad_b, pol_b,
                    max_ticks=300, seed=1)
        result = game.run()
        s = result.get("score", {})
        ok(f"played 300 ticks with no crash "
           f"(score You {s.get('A', 0)}–{s.get('B', 0)} Clone)")
    except Exception as exc:  # noqa: BLE001
        failed = True
        bad(f"your policy crashed during the match: {exc!r}")

    # 4. Training time + memory (RAM / VRAM) -------------------------------
    print("\n[4/4] Training resource usage")
    if not os.path.isfile(train_path):
        warn("no train_rl_agent.py — nothing to measure")
    elif not measure:
        warn("skipped — re-run with --train to measure training time, RAM & VRAM "
             f"(caps: {MAX_TRAIN_SECONDS}s, {MAX_RAM_MB} MB RAM, {MAX_VRAM_MB} MB VRAM)")
    else:
        print(f"       running train_rl_agent.py (server caps: "
              f"{MAX_TRAIN_SECONDS}s, {MAX_RAM_MB} MB RAM, {MAX_VRAM_MB} MB VRAM)…")
        (elapsed, peak, peak_ram, time_breach,
         vram_breach, ram_breach) = measure_training(target)
        gpu = "" if shutil.which("nvidia-smi") else " (no GPU — VRAM shown as 0)"
        print(f"       LOCAL training time: {elapsed}s (server budget "
              f"{MAX_TRAIN_SECONDS}s)")
        print(f"       peak RAM: {peak_ram} MB / {MAX_RAM_MB} MB   "
              f"peak VRAM: {peak} MB / {MAX_VRAM_MB} MB{gpu}")
        # RAM/VRAM track model size -> a breach here is a real disqualifier.
        if ram_breach:
            failed = True
            bad(f"peak RAM exceeds {MAX_RAM_MB} MB — you would be "
                f"DISQUALIFIED on the server")
        else:
            ok("RAM within the limit")
        if vram_breach:
            failed = True
            bad(f"peak VRAM exceeds {MAX_VRAM_MB} MB — you would be "
                f"DISQUALIFIED on the server")
        else:
            ok("VRAM within the limit")
        # Time is only DQ'd on the SERVER; local time just guides you.
        if time_breach:
            warn("training hit the time budget on THIS machine — only the "
                 "SERVER's measured time counts for disqualification, so a "
                 "slow local machine is fine (but check your server headroom)")
        else:
            print(f"       (local time is informational; the server enforces "
                  f"the {MAX_TRAIN_SECONDS}s cap)")

    print()
    if failed:
        print(f"{RED}VALIDATION FAILED — fix the issues above before uploading.{RST}")
        return 1
    print(f"{GREEN}VALIDATION PASSED — your submission loads and runs.{RST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
