"""
train_rl_agent.py — No-op trainer for the England policy.
================================================================
The England policy is a pure centralized heuristic that requires no training.
This stub satisfies the platform's import interface.
"""

from __future__ import annotations

import sys


def train():
    """No training needed — pure heuristic policy."""
    print("England policy — no training required.")
    print("The policy uses full proprioception and coordinated heuristics.")
    print("Done — 0 weight file(s) needed.")


if __name__ == "__main__":
    train()

