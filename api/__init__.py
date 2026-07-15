"""
FIFA AI World Cup - Reinforcement Learning Tournament Environment.

A multi-agent soccer simulation environment for a FIFA World Cup-themed
hackathon.  Organizers run the central environment (``game.py``) which handles
continuous physics, local agent sightlines, policy execution and match
rendering.  Participants design structural policies (``agents.py``) controlling
individual agents under continuous execution criteria.

Modules
-------
config          - Global simulation constants.
utils           - Vector / angle math helpers.
agent           - Agent dataclass and squad generation (star-player guardrails).
ball            - Independent ball physics class.
vision          - Directional vision cone and partial-observability state builder.
actions         - Action execution (move / rotate / shoot / dive / deflect).
game            - The environment runner and tick execution loop.
skills          - squad.json validation and squad construction.
evaluate        - Match evaluation engine (the one the organiser judges with;
                  drive it via the top-level ``evaluate.py`` script).
agents          - PARTICIPANT-WRITTEN policy (blank; this is your file).
agents_example  - Reference implementation showing the required format.
training        - Restricted single-agent environment for developing/testing
                  a policy before wiring it into ``agents.py``.
test_agents     - Baked-in demo/reference policies used by the CLI & tests.
economy         - Match economy, credits and transfer window.
tournament      - 64-team single-elimination knockout championship.
render          - Pygame renderer with headless frame/GIF export, including
                  per-goal highlight clips.
main            - Command-line entry point.
"""

__version__ = "1.1.0"
