# Graph Report - .  (2026-07-15)

## Corpus Check
- Corpus is ~31,366 words - fits in a single context window. You may not need a graph.

## Summary
- 473 nodes · 978 edges · 29 communities (19 shown, 10 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 14 edges (avg confidence: 0.62)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Module Actions|Module Actions]]
- [[_COMMUNITY_Module Agents|Module Agents]]
- [[_COMMUNITY_Module Agent|Module Agent]]
- [[_COMMUNITY_Module Evaluate|Module Evaluate]]
- [[_COMMUNITY_Module Game|Module Game]]
- [[_COMMUNITY_Module Gym|Module Gym]]
- [[_COMMUNITY_Module Submission|Module Submission]]
- [[_COMMUNITY_Module Agents|Module Agents]]
- [[_COMMUNITY_Module Train|Module Train]]
- [[_COMMUNITY_Module Skills|Module Skills]]
- [[_COMMUNITY_Module Submission|Module Submission]]
- [[_COMMUNITY_Module Demo|Module Demo]]
- [[_COMMUNITY_Module Starter|Module Starter]]
- [[_COMMUNITY_Module Ball|Module Ball]]
- [[_COMMUNITY_Module Starter|Module Starter]]
- [[_COMMUNITY_Module Reward|Module Reward]]
- [[_COMMUNITY_Module Agents|Module Agents]]
- [[_COMMUNITY_Module Guide|Module Guide]]
- [[_COMMUNITY_Module Participant|Module Participant]]
- [[_COMMUNITY_Module Jax|Module Jax]]
- [[_COMMUNITY_Module Tensorflow|Module Tensorflow]]
- [[_COMMUNITY_Module Torch|Module Torch]]
- [[_COMMUNITY_Module Guide|Module Guide]]
- [[_COMMUNITY_Module Guide|Module Guide]]
- [[_COMMUNITY_Module Guide|Module Guide]]
- [[_COMMUNITY_Module Guide|Module Guide]]
- [[_COMMUNITY_Module Guide|Module Guide]]
- [[_COMMUNITY_Module Guide|Module Guide]]

## God Nodes (most connected - your core abstractions)
1. `Agent` - 69 edges
2. `Game` - 39 edges
3. `Ball` - 26 edges
4. `map_action()` - 22 edges
5. `Policy` - 21 edges
6. `distance()` - 20 edges
7. `FIFAWorldCupAgentEnv` - 15 edges
8. `apply_action()` - 14 edges
9. `clamp()` - 13 edges
10. `vec_from_angle()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `Starter train_rl_agent.py` --semantically_similar_to--> `train_rl_agent.py`  [INFERRED] [semantically similar]
  examples/starter/README.md → TRAINING_GUIDE.md
- `Starter agents.py` --semantically_similar_to--> `agents.py`  [INFERRED] [semantically similar]
  examples/starter/README.md → TRAINING_GUIDE.md
- `main()` --calls--> `generate_squad()`  [EXTRACTED]
  validate_submission.py → api/agent.py
- `my_features()` --calls--> `extract_features()`  [EXTRACTED]
  my_submission/train_rl_agent.py → api/agents.py
- `main()` --calls--> `Game`  [EXTRACTED]
  validate_submission.py → api/game.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Agent Actions** — training_guide_action_move, training_guide_action_rotate, training_guide_action_shoot, training_guide_action_pass, training_guide_action_dive, training_guide_action_deflect [EXTRACTED 1.00]
- **Machine Learning Frameworks** — requirements_torch, requirements_tensorflow, requirements_jax [INFERRED 0.85]

## Communities (29 total, 10 thin omitted)

### Community 0 - "Module Actions"
Cohesion: 0.05
Nodes (58): apply_action(), _apply_dive(), _apply_move(), _apply_shoot(), _clamp_to_pitch(), _face_ball(), _gk_lateral_direction(), normalize_action() (+50 more)

### Community 1 - "Module Agents"
Cohesion: 0.12
Nodes (33): _attack_angle(), _compass(), _gk_dive_direction(), _kick_power_for_distance(), map_action(), _nearest_opponent_dist(), _opp_goal(), _own_goal() (+25 more)

### Community 2 - "Module Agent"
Cohesion: 0.06
Nodes (28): Agent, _enforce_star_floor(), generate_squad(), _make_goalkeeper(), _make_outfield(), place_kickoff(), Random, _rand_attr() (+20 more)

### Community 3 - "Module Evaluate"
Cohesion: 0.08
Nodes (39): _add_common(), _build_squad(), evaluate_submissions(), _hv(), _IdlePolicy, load_policy_factory(), load_squad_spec(), main() (+31 more)

### Community 4 - "Module Game"
Cohesion: 0.09
Nodes (20): Game, Any, Random, Run the full match (regulation + overtime + shootout if needed)., Advance the simulation by one tick.  Returns scoring team or None., Invoke a participant policy defensively.          A policy that raises or return, Whether ``agent`` may not touch the ball during a set-piece window.          Dur, Section 6.2: nearest eligible interceptor gains possession. (+12 more)

### Community 5 - "Module Gym"
Cohesion: 0.12
Nodes (15): make_policy(), Factory hook imported by the engine / tournament., default_reward(), FIFAWorldCupAgentEnv, Any, gym_env.py ========== OpenAI Gymnasium environment for the FIFA AI World Cup — *, Gymnasium env controlling a variable subset of a team's squad slots.      Parame, Probe ``feature_fn`` once (on a throwaway game) to size the obs space. (+7 more)

### Community 6 - "Module Submission"
Cohesion: 0.13
Nodes (12): extract_features(), 20-dim observation vector -- identical to ``train_rl_agent.py``.      Keep this, make_policy(), my_features(), Any, QNetwork, StarterPolicy, DQNAgent (+4 more)

### Community 7 - "Module Agents"
Cohesion: 0.11
Nodes (15): _compass(), make_policy(), MyPolicy, Any, QNetwork, agents_template.py  (TEMPLATE — FILL IN THE TODOs) =============================, # TODO: forward pass -> argmax -> map_action, Nearest of the 8 MOVE directions (0=E, 1=NE, 2=N, …) for an angle. (+7 more)

### Community 8 - "Module Train"
Cohesion: 0.11
Nodes (14): DQNAgent, QNetwork, train_template.py  (TEMPLATE — FILL IN THE TODOs) ==============================, # TODO: Define your Q-network.  Any architecture is allowed — the default, # TODO: define layers, # TODO: forward pass, # TODO: Define your DQN agent, # TODO: policy_net, target_net, optimizer, replay buffer, hyperparams (+6 more)

### Community 9 - "Module Skills"
Cohesion: 0.20
Nodes (18): _apply_gk_knobs(), _apply_outfield_knobs(), _check_knobs(), _check_position(), default_spec(), knob_to_attr(), knob_to_vision_deg(), knob_to_vision_range() (+10 more)

### Community 10 - "Module Submission"
Cohesion: 0.24
Nodes (13): bad(), check_squad_entry(), imports_of(), main(), measure_training(), ok(), proc_rss_mb(), proc_vram_mb() (+5 more)

### Community 11 - "Module Demo"
Cohesion: 0.21
Nodes (8): _ball_bearing(), _compass(), DemoPolicy, make_policy(), Any, Nearest of the 8 MOVE directions (0=E, 1=NE, 2=N, …) for an angle., Absolute angle to the ball if any vision sector sees it, else None., A minimal centralized policy. Reads its own squad for positions.

### Community 12 - "Module Starter"
Cohesion: 0.26
Nodes (4): DQNAgent, QNetwork, train_rl_agent.py  —  COMPLETE, RUNNABLE reference trainer =====================, train()

### Community 13 - "Module Ball"
Cohesion: 0.22
Nodes (6): Any, Freeze the ball on the line it crossed and describe the exit., Original energy-retaining wall bounce (set pieces disabled)., Reset the ball to the centre after a goal (Section 2.2)., Advance the ball one tick and resolve boundaries.          Returns a small descr, Detect goals, out-of-play and (legacy) wall bounces.          Goals are always d

### Community 14 - "Module Starter"
Cohesion: 0.24
Nodes (6): make_policy(), Any, QNetwork, The engine's only entry point into this file., DQN where trained, reference heuristic everywhere else., StarterPolicy

### Community 15 - "Module Reward"
Cohesion: 0.22
Nodes (8): reward_examples.py  (EXAMPLES — NOT A COMPLETE SOLUTION) =======================, Reward goals scored, penalise goals conceded, small possession bonus., Dense reward for getting closer to the ball when not in possession., Reward the carrier for heading toward the opponent's goal., # TODO: combine these (and add your own terms) into a single reward function., reward_forward(), reward_goals(), reward_proximity()

### Community 16 - "Module Agents"
Cohesion: 0.29
Nodes (4): QNetwork, MLP Q-network -- architecture must match ``train_rl_agent.py``., Placeholder so attribute lookups never fail when torch is absent., Load any trained DQN weights found next to this file / cwd.

### Community 17 - "Module Guide"
Cohesion: 0.29
Nodes (8): Starter agents.py, Starter train_rl_agent.py, agents.py, FIFAWorldCupAgent-v0, Gymnasium environment, make_policy, Network Architecture, train_rl_agent.py

## Knowledge Gaps
- **14 isolated node(s):** `participant-kit`, `make_policy`, `FIFAWorldCupAgent-v0`, `MOVE Action`, `ROTATE Action` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Agent` connect `Module Agent` to `Module Actions`, `Module Agents`, `Module Evaluate`, `Module Game`, `Module Gym`, `Module Skills`, `Module Agents`?**
  _High betweenness centrality (0.293) - this node is a cross-community bridge._
- **Why does `Game` connect `Module Game` to `Module Actions`, `Module Agent`, `Module Evaluate`, `Module Gym`, `Module Submission`?**
  _High betweenness centrality (0.158) - this node is a cross-community bridge._
- **Why does `Ball` connect `Module Actions` to `Module Agents`, `Module Game`, `Module Ball`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `Agent` (e.g. with `Policy` and `QNetwork`) actually correct?**
  _`Agent` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `Game` (e.g. with `_IdlePolicy` and `_SafePolicy`) actually correct?**
  _`Game` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Policy` (e.g. with `Agent` and `._call_policy()`) actually correct?**
  _`Policy` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `FIFA AI World Cup - Reinforcement Learning Tournament Environment.  A multi-agen`, `actions.py ========== Action execution for the FIFA AI World Cup action space (S`, `Coerce a raw policy action into a safe, complete action dict.` to the rest of the system?**
  _165 weakly-connected nodes found - possible documentation gaps or missing edges._