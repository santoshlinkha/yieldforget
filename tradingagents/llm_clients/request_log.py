"""A record of every model request an agent run makes and what came back, for reading afterwards.

``RequestLog`` is a LangChain callback: before each request it notes the agent (the graph node), the
model, the messages sent (system prompt, the conversation, tool results) and the tools offered; after
it, the reply (text and tool calls) and its token counts. Each finished request is one JSON line in the
log file, next to the run's reports. API keys are never in the messages, so never in the log.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:  # pragma: no cover
    BaseCallbackHandler = object

FILE = "requests.jsonl"


def _message(m) -> dict:
    content = m.content
    if isinstance(content, list):   # Gemini answers in parts: keep the text, note anything else
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content
                          if not isinstance(p, dict) or p.get("type") in (None, "text"))
    out = {"role": getattr(m, "type", type(m).__name__), "content": content if isinstance(content, str)
           else json.dumps(content, default=str)}
    calls = getattr(m, "tool_calls", None)
    if calls:
        out["tool_calls"] = [{"name": c.get("name"), "args": c.get("args")} for c in calls]
    if getattr(m, "name", None):
        out["name"] = m.name
    if getattr(m, "tool_call_id", None):
        out["tool_call_id"] = m.tool_call_id
    return out


def _tools(params: dict) -> list[str]:
    names = []
    for t in params.get("tools") or []:
        f = t.get("function", t) if isinstance(t, dict) else {}
        names.append(f.get("name") or str(t)[:60])
    return names


class RequestLog(BaseCallbackHandler):
    """Appends one JSON line per model request to ``folder / requests.jsonl``."""

    def __init__(self, folder: Path):
        self.path = Path(folder) / FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._open: dict = {}
        self._lock = threading.Lock()
        self.n = 0

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        params = kwargs.get("invocation_params") or {}
        meta = metadata or {}
        # An analyst runs as its own small graph whose node is just "agent": name it by the outer node.
        outer = str(meta.get("langgraph_checkpoint_ns") or "").split("|")[0].split(":")[0]
        self._open[run_id] = {"agent": outer or meta.get("langgraph_node"),
                              "model": params.get("model") or params.get("model_name"),
                              "tools_offered": _tools(params),
                              "sent": [_message(m) for m in (messages[0] if messages else [])],
                              "at": datetime.now(UTC).isoformat(timespec="seconds"), "_t": time.monotonic()}

    def on_llm_end(self, response, *, run_id, **kwargs):
        entry = self._open.pop(run_id, None)
        if entry is None:
            return
        gen = (response.generations or [[None]])[0][0]
        msg = getattr(gen, "message", None)
        usage = getattr(msg, "usage_metadata", None) or {}
        entry.update(received=_message(msg) if msg is not None else {"content": getattr(gen, "text", "")},
                     input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                     seconds=round(time.monotonic() - entry.pop("_t"), 1))
        self._write(entry)

    def on_llm_error(self, error, *, run_id, **kwargs):
        entry = self._open.pop(run_id, None)
        if entry is not None:
            entry.pop("_t", None)
            self._write({**entry, "error": str(error)[:500]})

    def _write(self, entry: dict):
        with self._lock:
            self.n += 1
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"n": self.n, **entry}, default=str) + "\n")


def read(path: Path) -> list[dict]:
    path = Path(path)
    if path.is_dir():
        path = path / FILE
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
