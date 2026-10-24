"""The AI moat reading is checked against the filing, and the filing's business section is found."""

import pytest

from tradingagents.dataflows.vendors.sec_edgar import _business_text
from tradingagents.research.moat_ai import verify

pytestmark = pytest.mark.unit


def test_a_quote_not_in_the_filing_loses_its_rating():
    filing = "Our customers rely on the ecosystem. Switching to another platform requires migrating years of data."
    answer = {"sources": [
        {"kind": "switching_costs", "strength": "strong", "evidence": "switching to another platform requires migrating  years of data"},
        {"kind": "network_effects", "strength": "moderate", "evidence": "every new user makes it better for all"},
        {"kind": "cost_advantage", "strength": "none", "evidence": ""}]}
    out = {s["kind"]: s for s in verify(answer, filing)["sources"]}
    assert out["switching_costs"]["verified"] and out["switching_costs"]["strength"] == "strong"
    assert not out["network_effects"]["verified"] and out["network_effects"]["strength"] == "unclear"
    assert out["cost_advantage"]["strength"] == "none"


def test_the_business_section_is_found_past_the_contents_and_cross_references():
    body = "We make widgets. Competition is intense. " * 200
    html = (f"<html><body><div style='display:none'><ix:header>us-gaap:Hidden facts</ix:header></div>"
            f"<p>INDEX Item 1. Business 3 Item 1A. Risk Factors 9</p>"
            f"<p>as described in “Item 1. Business” of this report</p>"
            f"<h2>ITEM 1. B USINESS</h2><p>{body}</p><h2>ITEM 1A. RISK FACTORS</h2><p>Risks.</p></body></html>")
    text = _business_text(html)
    assert text.startswith("ITEM 1. B USINESS") and "Competition is intense" in text and "Risks." not in text
    assert "Hidden facts" not in text
