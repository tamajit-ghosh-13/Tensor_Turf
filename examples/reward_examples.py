"""
reward_examples.py  (EXAMPLES — NOT A COMPLETE SOLUTION)
========================================================
Three example reward functions for the FIFAWorldCupAgent-v0 Gymnasium env.
Copy one of these into your training script and modify it — this is where
your reward engineering happens.

The reward function signature is:
    def my_reward(prev_obs, action, obs, info, team) -> float

See TRAINING_GUIDE.md for what's in ``info``.
"""

import math


# ---------------------------------------------------------------------------
# Example 1: Goal-focused reward (sparse + mild possession bonus)
# ---------------------------------------------------------------------------
def reward_goals(prev_obs, action, obs, info, team):
    """Reward goals scored, penalise goals conceded, small possession bonus."""
    other = "B" if team == "A" else "A"
    prev = info.get("_prev_score", {"A": 0, "B": 0})
    new = info["score"]
    r = 40.0 * (new[team] - prev[team]) - 40.0 * (new[other] - prev[other])
    for a in info.get("agents", []):
        if a["has_ball"]:
            r += 0.1
    return r


# ---------------------------------------------------------------------------
# Example 2: Ball-proximity shaping (dense — encourages closing on the ball)
# ---------------------------------------------------------------------------
def reward_proximity(prev_obs, action, obs, info, team):
    """Dense reward for getting closer to the ball when not in possession."""
    r = 0.0
    any_has_ball = any(a["has_ball"] for a in info.get("agents", []))
    if not any_has_ball:
        ball_dists = [a["ball_distance"] for a in info.get("agents", [])
                      if a["ball_distance"] is not None]
        if ball_dists:
            d = min(ball_dists)
            prev_d = info.get("_prev_ball_dist")
            if prev_d is not None:
                r += (prev_d - d) * 0.2
    return r


# ---------------------------------------------------------------------------
# Example 3: Forward-progress reward (for the ball carrier)
# ---------------------------------------------------------------------------
def reward_forward(prev_obs, action, obs, info, team):
    """Reward the carrier for heading toward the opponent's goal."""
    r = 0.0
    attack_dx = 1.0 if team == "A" else -1.0
    for a in info.get("agents", []):
        if a["has_ball"]:
            ori = a.get("orientation", 0.0)
            r += max(0.0, math.cos(math.radians(ori)) * attack_dx) * 0.4
    return r


# ---------------------------------------------------------------------------
# TODO: combine these (and add your own terms) into a single reward function.
# ---------------------------------------------------------------------------
