"""The request log: one line per model request with the agent, messages sent, tools offered and reply."""

import uuid
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tradingagents.llm_clients.request_log import RequestLog, read

pytestmark = pytest.mark.unit


def test_a_request_and_its_reply_are_written_together(tmp_path):
    log = RequestLog(tmp_path)
    run = uuid.uuid4()
    sent = [SystemMessage("You are the market analyst."), HumanMessage("AMAT"),
            ToolMessage("Date,Close\n2026-10-23,377.24", tool_call_id="t1", name="get_stock_data")]
    log.on_chat_model_start({}, [sent], run_id=run, metadata={"langgraph_node": "Market Analyst"},
                            invocation_params={"model": "gemini-3.8-flash",
                                               "tools": [{"type": "function", "function": {"name": "get_indicators"}}]})
    reply = AIMessage("", tool_calls=[{"name": "get_indicators", "args": {"indicator": "rsi"}, "id": "c1"}],
                      usage_metadata={"input_tokens": 900, "output_tokens": 40, "total_tokens": 940})
    log.on_llm_end(SimpleNamespace(generations=[[SimpleNamespace(message=reply)]]), run_id=run)
    (row,) = read(tmp_path)
    assert row["agent"] == "Market Analyst" and row["model"] == "gemini-3.8-flash"
    assert row["tools_offered"] == ["get_indicators"]
    assert [m["role"] for m in row["sent"]] == ["system", "human", "tool"] and row["sent"][2]["name"] == "get_stock_data"
    assert row["received"]["tool_calls"] == [{"name": "get_indicators", "args": {"indicator": "rsi"}}]
    assert (row["input_tokens"], row["output_tokens"]) == (900, 40)


def test_an_error_is_recorded_and_a_missing_log_reads_empty(tmp_path):
    log = RequestLog(tmp_path)
    run = uuid.uuid4()
    log.on_chat_model_start({}, [[HumanMessage("x")]], run_id=run, invocation_params={"model": "m"})
    log.on_llm_error(RuntimeError("429 quota"), run_id=run)
    assert read(tmp_path)[0]["error"] == "429 quota"
    assert read(tmp_path / "nowhere") == []
