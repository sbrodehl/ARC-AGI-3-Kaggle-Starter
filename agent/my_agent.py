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
from __future__ import annotations

import random
import time
import re
import os
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
ACTION_NAMES = [a.name for a in CANDIDATE_ACTIONS]
SIMPLE_ACTION_NAMES = [a.name for a in CANDIDATE_ACTIONS if a.is_simple()]
COMPLEX_ACTION_NAMES = [a.name for a in CANDIDATE_ACTIONS if a.is_complex()]
PROMPT = f"""Return the following line verbatim:
ACTION={{RANDOM_ACTION}}""".strip()


class MyAgent(Agent):
    """Picks legal actions uniformly at random. Replace with your strategy."""

    # Upper bound on actions per game; the framework also enforces global limits.
    MAX_ACTIONS = int(os.getenv("MAX_ACTIONS", "80"))

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
                    {"role": "user", "content": prompt.format(RANDOM_ACTION=choice(SIMPLE_ACTION_NAMES))},
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

        llm_response = self._llm_choose_action(prompt=PROMPT)

        # --- Parse model output ---
        # Accept variants like "ACTION=ACTION4" or "ACTION: ACTION4"
        def get_value(pattern: str, default=None):
            m = re.search(pattern, llm_response, flags=re.IGNORECASE | re.MULTILINE)
            return m.group(1).strip() if m else default

        action_str = get_value(r"^ACTION\s*[:=]\s*([A-Z0-9_]+)\s*$", default=None)
        x_str = get_value(r"^X\s*[:=]\s*(\d+)\s*$", default=None)
        y_str = get_value(r"^Y\s*[:=]\s*(\d+)\s*$", default=None)

        parsed_action = None
        if action_str:
            # action_str should match GameAction enum member name (e.g., ACTION4)
            for a in CANDIDATE_ACTIONS:
                if a.name.upper() == action_str.upper():
                    parsed_action = a
                    break

        # TODO: needs fall back option! (e.g. random)
        if parsed_action is None:
            # TODO DEBUG: raise error early
            raise RuntimeError("No action found in LLM response: \n" + llm_response)

        # --- Convert parsed output to GameAction object ---
        action = parsed_action
        if action.is_complex():
            # If missing/invalid coords, fall back to random complex coords
            try:
                # assume random value if pattern didn't match
                x = int(x_str) if x_str is not None else random.randint(0, 63)
                y = int(y_str) if y_str is not None else random.randint(0, 63)
                # clamp values to reasonable range
                x = max(0, min(63, x))
                y = max(0, min(63, y))
            except Exception:
                raise RuntimeError("Unable to convert coordinates for complex action.")

            action.set_data({"x": x, "y": y})
            action.reasoning = {"why": "vllm complex action"}
        else:
            action.reasoning = {"why": "vllm simple action"}
        return action
