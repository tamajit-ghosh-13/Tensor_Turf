# Starter submission — a complete, working example

Unlike `../agents_template.py` and `../train_template.py` (skeletons with
`TODO`s), the two files in **this** folder are **filled in and runnable**. They
are the fastest way to get a real, playing team on the board — copy them, run
the validator, and submit.

```
starter/
├── agents.py            # REQUIRED — inference policy (make_policy)
├── train_rl_agent.py    # OPTIONAL — trains 11 DQNs into agent_00..10_dqn.pt
└── README.md            # this file
```

## What they do

| File | Behaviour |
|---|---|
| `train_rl_agent.py` | Trains one small DQN per squad slot on the Gym env and saves `agent_00_dqn.pt` … `agent_10_dqn.pt`. Uses the **default** 20-float obs / 13-action space, so the net is `20 → 64 → 64 → 13`. |
| `agents.py` | Loads those weights at match time. Trained slots play from the DQN; every other slot (and the whole team, before you've trained anything) falls back to the kit's built-in reference heuristic — so it **always plays a sane game**. |

The two files are deliberately **consistent**: the `QNetwork` shape and the
obs/action decoding are identical on both sides. That is the one rule the
platform cares about — the engine never inspects your weights, it only calls
`make_policy(...).decide(states, team)`.

## Quick start

```bash
# 1. Install the training stack (only train_rl_agent.py needs it).
pip install torch gymnasium numpy

# 2. Train (start small while iterating). Weights land next to SAVE_MODEL.
export SAVE_MODEL=./weights/agent.pt
python train_rl_agent.py --episodes 20 --ticks 400 --seed 42
# ...or a single slot:
python train_rl_agent.py --slots 5 --episodes 30

# 3. Sanity-check inference (agents.py loads ./weights via WEIGHTS_DIR).
export WEIGHTS_DIR=./weights

# 4. Validate exactly what the platform will check, then upload.
python ../../validate_submission.py .
```

`agents.py` runs **with or without** trained weights and **with or without**
torch installed — no weights just means the heuristic drives every slot.

## Make it yours

1. **Reward** — edit `my_reward` in `train_rl_agent.py` (see
   `../reward_examples.py`). This is where most of your gains come from.
2. **Network / algorithm** — change `QNetwork` (or swap DQN for PPO, JAX, …).
   If you change the shape, change it in **both** files.
3. **Observation / action space** — pass `feature_fn` / `action_fn` /
   `action_space` to `gym.make` (see `TRAINING_GUIDE.md` §3), and reproduce the
   same featurisation / decoding in `agents.py`.
4. **Squad** — design your formation and per-player knobs in `squad.json` (via
   the web squad builder); it is submitted alongside these files.

## Limits (enforced at upload + on the training server)

- Only `.py` files (plus the generated `squad.json`) may be uploaded.
- `agents.py` and `train_rl_agent.py`: ≤ 10 MB each, ≤ 20 MB total.
- Imports must be on the allowed list (torch, numpy, gymnasium, jax/flax,
  tensorflow, sklearn, scipy, pandas, the stdlib, and `api`).
- Training must finish within the server's time cap and stay under the VRAM cap
  (both shown by the validator and in the web UI). Exceeding them on the
  training server disqualifies the run.
