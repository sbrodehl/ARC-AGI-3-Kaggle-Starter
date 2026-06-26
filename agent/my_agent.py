"""Your ARC-AGI-3 agent. This is the *only* file you should normally edit.

`scripts/build_notebook.py` splices the contents of this file into the
Kaggle submission notebook, so your local dev loop and your Kaggle
submission stay in lock-step:

    [edit my_agent.py] → [make play-local] → [make submit]

The default body below is a port of the Stochastic Goose / random_agent
sample — a known-good baseline that produces a valid submission and
proves your end-to-end pipeline works. Replace `choose_action` with your
real strategy.

Contract (enforced by the ARC-AGI-3-Agents framework):
  - Subclass `agents.agent.Agent`.
  - Class must be named `MyAgent` (the notebook's __init__.py registers it).
  - Implement `is_done(frames, latest_frame) -> bool`.
  - Implement `choose_action(frames, latest_frame) -> GameAction`.
"""
import random
import time
import re
import os
import sys
from random import choice
from typing import Any

from arcengine import FrameData, GameAction, GameState

# When run inside the ARC-AGI-3-Agents framework (locally or on Kaggle)
# the `agents` package is on sys.path, so this import resolves.
from agents.agent import Agent

from openai import OpenAI, APIConnectionError


VLLM_BASE_URL = os.getenv("VLLM_BASE", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "local")

try:
    VLLM_CLIENT = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    VLLM_MODEL = VLLM_CLIENT.models.list().data.pop().id  # use the first model available
except (ConnectionError, APIConnectionError):
    raise RuntimeError("LLM backend not available!")

CANDIDATE_ACTIONS = [a for a in GameAction if a is not GameAction.RESET]
SIMPLE_ACTIONS = [a for a in CANDIDATE_ACTIONS if a.is_simple()]
PROMPT = f"""Return the following line verbatim:
ACTION={{RANDOM_ACTION}}""".strip()


class MyAgent(Agent):
    """Picks legal actions uniformly at random."""

    # Upper bound on actions per game; the framework also enforces global limits.
    MAX_ACTIONS = int(os.getenv("MAX_ACTIONS", "50"))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Seed per game_id so replays from the same game are reproducible but
        # different games explore independently.
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        random.seed(seed)

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    @staticmethod
    def is_done(frames: list[FrameData], latest_frame: FrameData) -> bool:
        # Stop once we win. Don't stop on GAME_OVER — we want to RESET and retry.
        return latest_frame.state is GameState.WIN

    @staticmethod
    def _llm_choose_action(prompt: str) -> str:
        try:
            resp = VLLM_CLIENT.chat.completions.create(
                model=VLLM_MODEL,
                messages=[
                    {"role": "system", "content": "Return only valid ARC-AGI 3 action outputs."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2
            )
        except (ConnectionError, APIConnectionError):
            raise RuntimeError("VLLM backend not available!")
        return resp.choices[0].message.content.strip()

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # First call or after a death → reset the level.
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        rnd_action = choice(SIMPLE_ACTIONS)
        llm_response = self._llm_choose_action(prompt=PROMPT.format(RANDOM_ACTION=rnd_action.name))

        # --- Parse model output ---
        # Accept variants like "ACTION=ACTION4" or "ACTION: ACTION4"
        def get_value(pattern: str, default=None):
            m = re.search(pattern, llm_response, flags=re.IGNORECASE | re.MULTILINE)
            return m.group(1).strip() if m else default

        action_str = get_value(r"^ACTION\s*[:=]\s*([A-Z0-9_]+)\s*$", default=None)

        parsed_action = GameAction.RESET
        if action_str:
            # action_str should match GameAction enum member name (e.g., ACTION4)
            for a in CANDIDATE_ACTIONS:
                if a.name.upper() == action_str.upper():
                    parsed_action = a
                    break

        if parsed_action.name != rnd_action.name:
            print("[-] AI fucked up.", file=sys.stderr)
            parsed_action = rnd_action

        # --- Convert parsed output to GameAction object ---
        action = parsed_action
        action.reasoning = {"why": "vllm simple action"}
        return action
