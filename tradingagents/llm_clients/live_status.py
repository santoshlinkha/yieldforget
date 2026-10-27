"""Short live status lines from an agent run: which agent is thinking, which tool it called, what failed.

A LangChain callback handler. LangGraph tags each call with the graph node that made it
(``metadata["langgraph_node"]``, e.g. "Market Analyst"), so every line names the agent. ``say`` gets
plain text like "Market Analyst: fetching get_stock_data" and must be cheap (it runs on the model's
thread). Nothing about the prompt or the answer is shown: only who is working and on what.
"""

from __future__ import annotations

import time

from langchain_core.callbacks import BaseCallbackHandler

FRIENDLY = {"tools_market": "Market Analyst", "tools_social": "Sentiment Analyst", "tools_news": "News Analyst",
            "tools_fundamentals": "Fundamentals Analyst"}


class LiveStatus(BaseCallbackHandler):
    def __init__(self, say):
        self.say = say
        self.started: dict = {}

    @staticmethod
    def _who(metadata) -> str:
        node = (metadata or {}).get("langgraph_node") or "agent"
        return FRIENDLY.get(node, node.replace("_", " ").strip())

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        self.started[run_id] = time.time()
        self.say(f"{self._who(metadata)}: thinking")

    def on_llm_start(self, serialized, prompts, *, run_id, metadata=None, **kwargs):
        self.started[run_id] = time.time()
        self.say(f"{self._who(metadata)}: thinking")

    def on_llm_end(self, response, *, run_id, metadata=None, **kwargs):
        took = time.time() - self.started.pop(run_id, time.time())
        self.say(f"{self._who(metadata)}: answered in {took:.0f}s")

    def on_llm_error(self, error, *, run_id, metadata=None, **kwargs):
        self.started.pop(run_id, None)
        self.say(f"{self._who(metadata)}: model error ({type(error).__name__})")

    def on_tool_start(self, serialized, input_str, *, metadata=None, **kwargs):
        self.say(f"{self._who(metadata)}: fetching {(serialized or {}).get('name') or 'data'}")

    def on_tool_error(self, error, *, metadata=None, **kwargs):
        self.say(f"{self._who(metadata)}: a data source failed ({type(error).__name__})")
