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
import os
import json
import logging
from typing import Any

from arcengine import FrameData, GameAction, GameState

# When run inside the ARC-AGI-3-Agents framework (locally or on Kaggle)
# the `agents` package is on sys.path, so this import resolves.
from agents.agent import Agent

logger = logging.getLogger()


try:
    from openai import OpenAI, APIConnectionError
    VLLM_BASE_URL = os.getenv("VLLM_BASE", "http://localhost:8000/v1")
    VLLM_API_KEY = os.getenv("VLLM_API_KEY", "local")
    VLLM_CLIENT = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    VLLM_MODEL = VLLM_CLIENT.models.list().data.pop().id  # use the first model available
    _USE_LLM = True
    print("[+] vLLM available!")
except Exception:
    _USE_LLM = False
    print("[-] vLLM not available!")


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

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # Stop once we win. Don't stop on GAME_OVER — we want to RESET and retry.
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # First call or after a death → reset the level.
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        if _USE_LLM:
            return self._llm_action(frames, latest_frame)
        return self._random_action()

    def _llm_action(self, frames, latest_frame) -> GameAction:
        candidates = [a for a in GameAction if a is not GameAction.RESET]
        valid_values = [a.value for a in candidates]

        prompt = (
            f"You are playing ARC-AGI-3 game '{self.game_id}'.\n"
            f"Move number: {len(frames)}\n"
            f"Current frame: {latest_frame}\n\n"
            f"Valid actions: {valid_values}\n"
            f"Respond ONLY with JSON: {{\"action\": \"<value>\", \"reasoning\": \"<why>\"}}"
        )
        try:
            resp = VLLM_CLIENT.chat.completions.create(
                model=VLLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1,
            )
            parsed = json.loads(resp.choices[0].message.content.strip())
            action = GameAction(parsed["action"])
            action.reasoning = parsed.get("reasoning", "llm")
            return action
        except Exception as e:
            action = self._random_action()
            action.reasoning = f"llm fallback: {e}"
            return action

    @staticmethod
    def _random_action() -> GameAction:
        candidates = [a for a in GameAction if a is not GameAction.RESET]
        action = random.choice(candidates)
        if action.is_complex():
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
        action.reasoning = "random fallback"
        return action
