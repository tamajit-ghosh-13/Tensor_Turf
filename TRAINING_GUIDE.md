# FIFA AI World Cup — Participant Training Guide

This guide tells you everything you need to know to train your RL agents and
prepare your inference script for the tournament.

---

## 1. What you're building

You write up to **two files** (only `agents.py` is required):

1. **`train_rl_agent.py`** *(optional)* — trains your agents using the
   Gymnasium environment and saves whatever weights/artifacts your `agents.py`
   will load.  The reference approach trains 11 DQNs (one per squad slot) into
   `agent_00_dqn.pt` … `agent_10_dqn.pt`, but this is **your choice** — see §8.

2. **`agents.py`** *(required)* — at match time, loads your artifacts and
   returns actions.  Exposes `make_policy(squad, seed)`.

The organiser runs your `train_rl_agent.py` to produce the weights, then
runs your `agents.py` in a head-to-head match against another participant.

> **Nothing about your algorithm or network is fixed by the platform.**  Use
> DQN, PPO, actor-critic, evolutionary search, or a pure heuristic; use torch,
> JAX/Flax, TensorFlow, scikit-learn, or none.  The engine never inspects your
> weights — it only calls `make_policy(...).decide(states, team)` and expects
> valid action dicts back.  The **only** hard rule is that your trainer and your
> `agents.py` agree with each other (see §8).

---

## 2. The package layout you receive

```
participant-kit/
├── api/
│   ├── __init__.py
│   ├── agents.py          # PUBLIC API + REFERENCE POLICY (read Policy._goalkeeper etc.)
│   ├── agent.pyi          # Agent dataclass stub (attributes you can read)
│   ├── gym_env.py         # The Gymnasium environment (FIFAWorldCupAgent-v0) — full source
│   ├── game.pyc           # Match engine (compiled — no source visible)
│   ├── evaluate.pyc       # Match evaluation engine (compiled — same one the organiser uses)
│   ├── skills.pyc         # squad.json validation / build (compiled)
│   ├── ball.pyc           # Ball physics (compiled)
│   ├── actions.pyc        # Action execution (compiled)
│   ├── vision.pyc         # Vision cone (compiled)
│   ├── config.pyc         # Constants (compiled)
│   ├── utils.pyc          # Math helpers (compiled)
│   ├── test_agents.pyc    # Reference AIs (compiled)
│   └── *.pyi              # Type stubs for each .pyc (signatures + docstrings for IDE)
├── examples/
│   ├── starter/              # ⭐ COMPLETE, RUNNABLE submission — copy & submit
│   │   ├── agents.py         #    inference policy (DQN + heuristic fallback)
│   │   ├── train_rl_agent.py #    trains 11 DQNs into agent_00..10_dqn.pt
│   │   └── README.md         #    step-by-step quick start
│   ├── demo_agent.py         # a tiny, readable heuristic agent (sparring opponent)
│   ├── reward_examples.py    # 3 example reward functions (copy + modify)
│   ├── train_template.py     # training script skeleton (fill in the TODOs)
│   └── agents_template.py    # inference script skeleton (fill in the TODOs)
├── evaluate.py            # ▶ run a match: your submission vs an opponent
├── validate_submission.py # ▶ check your submission before uploading
└── TRAINING_GUIDE.md      # this file
```

> **Fastest path:** copy `examples/starter/agents.py` +
> `examples/starter/train_rl_agent.py`, run `python validate_submission.py`
> on them, and submit. They already play a full game (heuristic fallback) even
> before you train, then use your DQN weights once trained. The `*_template.py`
> files are blank skeletons for building from scratch instead.

The `api/` folder is the engine package.  Import from it:
```python
from api.agents import extract_features, map_action, STATE_DIM, ACTION_DIM
import api.gym_env  # registers the Gymnasium env
```

Internal modules (`.pyc` files) are compiled bytecode — they run but you
cannot read their source.  Each has a `.pyi` stub file with signatures +
docstrings for your IDE.

---

## 3. The Gymnasium environment

### Registration

```python
import gymnasium as gym
import api.gym_env  # noqa: F401  (registers FIFAWorldCupAgent-v0)

env = gym.make("FIFAWorldCupAgent-v0",
               slots=[0],                    # which squad slot to control (0..10)
               opponent_style="aggressive",  # opponent AI: balanced/aggressive/defensive/random
               max_ticks=400,                # episode length
               reward_fn=my_reward)          # your reward function
```

### Slots — which agents you control

| `slots` | What it trains | Observation | Action |
|---|---|---|---|
| `[0]` | Just the GK | `Box(20,)` | `Discrete(13)` |
| `[5]` | One outfielder | `Box(20,)` | `Discrete(13)` |
| `[0,5,9]` | GK + MID + ATT jointly | `Box(60,)` | `MultiDiscrete([13,13,13])` |
| `list(range(11))` | Whole team jointly | `Box(220,)` | `MultiDiscrete([13]*11)` |

**Recommended**: train each slot individually (`slots=[k]` for k in 0..10).

### reset / step

```python
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
```

### Bring your own observation & action space (optional)

The 20-float obs and 13-action space in §4–§5 are just the **defaults**.  Pass
your own to `gym.make` to train on any observation and any action space:

```python
from gymnasium import spaces

def my_features(state):              # raw engine state dict -> vector (any len)
    return [ ... ]

def my_action(raw, state, agent_pos, sho):   # your action -> engine action dict
    return {"action_type": "MOVE", "move_direction": int(raw), ...}

env = gym.make("FIFAWorldCupAgent-v0", slots=[5],
               feature_fn=my_features,          # obs space is sized automatically
               action_fn=my_action,
               action_space=spaces.Box(-1, 1, shape=(4,)),  # or Discrete(k)
               reward_fn=my_reward)
```

| Argument | Purpose |
|---|---|
| `feature_fn(state) -> seq[float]` | your observation (any length) |
| `action_fn(raw, state, pos, sho) -> dict` | decode your action into an engine action dict |
| `action_space` | any single-agent Gym space (`Discrete`, `Box`, …); composed automatically for multi-slot training |
| `obs_dim`, `obs_low`, `obs_high` | override obs sizing / bounds |

Whatever obs/action design you train with, reproduce the same decoding in your
`agents.py` so the policy behaves identically at match time.

---

## 4. The default 13-action space

| Index | Action | Notes |
|---|---|---|
| 0–7 | MOVE | Compass direction (0=E, 1=NE, 2=N, …, 7=SE). **For GKs, 0 and 1 are lateral along the goal line** (the engine auto-faces the ball). |
| 8 | ROTATE | Scan (turn 15°). |
| 9 | SHOOT | At the opponent goal, aimed away from a visible keeper. |
| 10 | PASS | To the best visible forward team-mate. |
| 11 | DIVE | GK only. Bursts laterally at dive speed, then enters a **cooldown** (frozen for several ticks). Use sparingly — only when a shot is close. |
| 12 | DEFLECT | GK only. Sets `deflect_angle` but **does NOT move the GK**. A static GK concedes every shot. |

Use `map_action(action_idx, state, agent_pos, sho)` to translate an index.

> **⚠ Goalkeeper warning — read this or concede every shot ⚠**
>
> DEFLECT (12) only sets the keeper's deflect angle — it does **not** move the
> keeper. DIVE (11) triggers a cooldown that freezes the keeper for several
> ticks. A keeper that only DEFLECTs or DIVEs every tick is a **static keeper
> that concedes every shot**.
>
> To track the ball, the keeper must **MOVE (0 or 1)** along the goal line.
> For GKs, `move_direction` 0 = one lateral direction, 1 = the other (the
> engine auto-faces the ball, so rotation is irrelevant). In `vision_sectors`,
> `center_angle > 0` means the ball is to the GK's left → `move_direction` 0
> is the correct response for both teams.
>
> See `api/agents.py` → `Policy._goalkeeper()` for the full reference
> implementation (ball-Y tracking, dive timing, distribution).

---

## 5. The default observation (20 features per agent)

```
Index  Feature                     Range
-----  --------------------------  -----
0      has_ball                    0 or 1
1      cooldown_remaining / 35     0..1
2      sin(orientation)            -1..1
3      cos(orientation)            -1..1
4      gk_y / 60                   0..1
5-19   5 vision sectors x 3 each   dist/40, is_ball, is_player
```

Use `extract_features(state)` to get this vector.

---

## 6. The `info` dict (for your reward function)

```python
info = {
    "tick": int,
    "score": {"A": int, "B": int},
    "phase": "regulation" | "overtime" | "shootout",
    "agents": [
        {
            "slot": int, "archetype": str, "is_gk": bool,
            "has_ball": bool, "ball_visible": bool,
            "ball_distance": float | None, "orientation": float,
        }, ...
    ],
    "_prev_score": {"A": int, "B": int},
    "_prev_ball_dist": float | None,
    "_prev_has_ball": bool,
}
```

---

## 7. Writing your reward function

Signature: `def my_reward(prev_obs, action, obs, info, team) -> float`

See `examples/reward_examples.py` for 3 starter rewards.  Combine and extend.

---

## 8. Network architecture — your choice (just be self-consistent)

There is **no platform-imposed architecture.**  The engine does not load your
weights — your own `agents.py` does.  So the only requirement is that the model
your `agents.py` builds matches the one your `train_rl_agent.py` saved.

The reference DQN uses **20 → 64 → 64 → 13** (MLP + ReLU) because that matches
the default 20-float obs / 13-action space.  If you change the obs/action space
(§3) or the algorithm, change both files together — anything goes as long as
they agree.  You may also skip training entirely and ship a heuristic
`agents.py`.

---

## 9. Training all 11 agents

Fill in `examples/train_template.py` and submit it as `train_rl_agent.py`.
- Save weights as `agent_00_dqn.pt` … `agent_10_dqn.pt`.
- Save them in the **directory** pointed to by `os.environ["SAVE_MODEL"]` (the
  organiser creates it for you; it is a folder, not a file path).
- **You choose `TICKS`, `EPISODES` and `SEED`** in the web app's squad builder
  when you submit — they're then delivered to your trainer as those env vars
  **and** as `--episodes/--ticks/--seed` (with `SAVE_MODEL`). **Ticks and
  episodes are capped** by an organiser-set maximum (the squad builder shows the
  cap; the platform default max is **`TICKS ≤ 1800`**, **`EPISODES ≤ 20`**) — a
  request above the cap is **rejected at upload**. **Seed is entirely yours**
  (uncapped — it has no effect on competitive strength). Training for **fewer /
  shorter** episodes than your chosen budget is fine. The one limit that is
  enforced at run time is the **training-time cap** (`max_train_seconds`, shown
  in the app) — exceed it and you're **disqualified**. Read the values you're
  given (they are your own choices) rather than hard-coding, and stay under the
  clock. Locally, pass your values with `--episodes/--ticks/--seed`.

---

## 10. Writing your inference script

Fill in `examples/agents_template.py` and submit it as `agents.py`.
- Expose `make_policy(squad, seed)`.
- Load weights from `os.environ["WEIGHTS_DIR"]`.
- With the **default** spaces, each agent is: `extract_features → forward pass
  → argmax → map_action`.  With a custom space, use your own featurisation and
  action decoding (matching your trainer) instead.
- No env, no resets — pure forward-pass.

---

## 11. Submission checklist

- [ ] `agents.py` — exposes `make_policy(squad, seed)` (**required**)
- [ ] `train_rl_agent.py` *(optional)* — saves your weights into `SAVE_MODEL`
- [ ] Your trainer and `agents.py` use the **same** obs/action space + model shape
- [ ] `agents.py` loads its artifacts from `WEIGHTS_DIR`
- [ ] Only allowed libraries are imported (see the validator / admin panel)
- [ ] `python validate_submission.py` passes before you upload

---

## 12. Tips for a strong team

1. Train all 11 agents (including the GK).
2. Shape the reward densely (goals + possession + proximity + forward progress).
3. Give the GK its own reward (saves, positioning).
4. Train enough episodes per slot to converge — the platform default is 20 per
   slot, but tune within your `max_train_seconds` budget.
5. Train against `aggressive` opponents.
6. Focus your reward on positioning and ball-winning.

---

## 13. Testing locally

Use **Python 3.12** (the engine ships as 3.12 bytecode) and install the pinned,
VM-matching versions so local results match the server exactly:

```bash
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # exact versions used on the GPU/eval boxes
python validate_submission.py ./my_submission
python train_rl_agent.py --episodes 5 --ticks 200 --seed 42   # quick local run
```

---

## 14. Evaluate a match locally

Play a real match on the **same hidden engine the organiser uses** and get the
full verdict (score, winner, FIFA-style stats). The engine ships as compiled
bytecode (`api/evaluate.pyc`) — you can't read it, but you can run it through
`evaluate.py`:

```bash
# your submission folder vs the bundled demo agent
python evaluate.py --a ./my_submission

# two folders, or single agents.py files, head to head
python evaluate.py --a ./my_submission --b examples/demo_agent.py \
                   --name-a "My FC" --name-b "Demo" --ticks 1800 --seed 7
```

Each side may be a submission **folder** (`agents.py` [+ `squad.json`]) or a
single `.py` file. Add `--out result.json` to save the full verdict.
`examples/demo_agent.py` is a tiny, readable heuristic — a good first opponent
and a compact example of the `decide()` contract.

> `evaluate.py` does **not** train — it loads trained weights only if a
> `WEIGHTS_DIR` is already populated. Run `train_rl_agent.py` first for that.

### Match rules (what the verdict reflects)

* Regulation is **90:00** — two **45:00** halves, with a **kickoff re-centre at
  half-time** (all players return to their formation, ball to the centre spot).
* If level after 90:00: **golden-goal extra time**, then a **penalty shootout**.
* `decided_by` in the verdict is `regulation`, `overtime`, or `shootout`; each
  goal carries a `clock` (`MM:SS`, up to 90:00) and a `half` (`1`/`2`/`ET`).

---

## 15. Which Python?

The compiled engine (`api/*.pyc`) is built for **Python 3.12** — run the kit,
`evaluate.py`, and `validate_submission.py` with 3.12.

Good luck!
