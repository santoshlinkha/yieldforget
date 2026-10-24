"""What the optimizer scans besides your own holdings, in groups you switch on or off.

- ``etfs``: one liquid ETF per corner of the market. Long-lived funds are
  preferred, since the stress loss needs a long price history, and
  near-duplicates (VOO and IVV, SHY and VGSH) are left out because they only
  split one position in two. Bonds and gold are here because under a tight loss
  limit they are the only assets whose worst drop is small enough to hold in size.
- ``sp500``: today's S&P 500 members, downloaded and cached for a week.
- ``themes``: disruptive themes (quantum, robotics, space, AI, nuclear, genomics,
  crypto, EVs, drones) as stocks and theme ETFs, plus small-cap index funds.
  A free, current list of every small cap (the Russell 2000) is not available,
  so small caps come in through the index funds and the theme stocks, most of
  which are small or mid caps.

Single stocks, and young ones especially, carry crash estimates of 50-95%, so
under a tight loss limit they rarely win any of the budget. The optimizer says
so in each one's reason.
"""

from __future__ import annotations

import json
from pathlib import Path

from tradingagents.dataflows.vendors.index_members import get_sp500_members

ETFS: dict[str, str] = {
    # US stocks
    "VTI": "US total stock market",
    "VOO": "S&P 500",
    "QQQ": "Nasdaq-100",
    "SPYG": "S&P 500 growth",
    "VTV": "US large-cap value",
    "SCHD": "US dividend stocks",
    "VIG": "US dividend growers",
    "USMV": "US minimum-volatility stocks",
    "QUAL": "US quality stocks",
    "MTUM": "US momentum stocks",
    "RSP": "S&P 500 equal weight",
    "IJR": "US small caps",
    "AVUV": "US small-cap value",
    # International
    "VXUS": "International stocks",
    "VEA": "Developed-market stocks",
    "VWO": "Emerging-market stocks",
    # Sectors
    "XLK": "Technology sector",
    "SOXX": "Semiconductors",
    "XLV": "Health care sector",
    "XLF": "Financials sector",
    "XLE": "Energy sector",
    "XLU": "Utilities sector",
    "XLP": "Consumer staples sector",
    "XLI": "Industrials sector",
    # Real assets
    "VNQ": "US real estate (REITs)",
    "GLD": "Gold",
    "DBC": "Commodities",
    # Bonds
    "BIL": "1–3 month Treasury bills",
    "SHY": "1–3 year Treasuries",
    "IEF": "7–10 year Treasuries",
    "TLT": "20+ year Treasuries",
    "TIP": "Inflation-protected Treasuries",
    "BND": "US total bond market",
    "LQD": "Investment-grade corporate bonds",
    "HYG": "High-yield corporate bonds",
    # Income and leveraged
    "JEPI": "Covered-call equity income",
    "UPRO": "3x S&P 500 (leveraged)",
    "TQQQ": "3x Nasdaq-100 (leveraged)",
    "SOXL": "3x semiconductors (leveraged)",
}

THEMES: dict[str, str] = {
    # Quantum computing
    "IONQ": "IonQ (quantum computing)",
    "RGTI": "Rigetti Computing (quantum computing)",
    "QBTS": "D-Wave Quantum (quantum computing)",
    "QUBT": "Quantum Computing Inc. (quantum computing)",
    "QTUM": "Defiance Quantum ETF (quantum and AI)",
    # Robotics and automation
    "SYM": "Symbotic (warehouse robotics)",
    "SERV": "Serve Robotics (delivery robots)",
    "BOTZ": "Global X Robotics & AI ETF",
    "ROBO": "ROBO Global Robotics & Automation ETF",
    # Space
    "RKLB": "Rocket Lab (space launch)",
    "ASTS": "AST SpaceMobile (satellite broadband)",
    "LUNR": "Intuitive Machines (lunar landers)",
    "PL": "Planet Labs (earth imaging)",
    "RDW": "Redwire (space infrastructure)",
    "ARKX": "ARK Space Exploration ETF",
    "UFO": "Procure Space ETF",
    # AI and software
    "AI": "C3.ai (enterprise AI)",
    "SOUN": "SoundHound AI (voice AI)",
    "PATH": "UiPath (automation software)",
    "SNOW": "Snowflake (data cloud)",
    "NET": "Cloudflare (edge network)",
    # Nuclear and uranium
    "OKLO": "Oklo (advanced nuclear)",
    "SMR": "NuScale Power (small reactors)",
    "NNE": "Nano Nuclear Energy (micro reactors)",
    "LEU": "Centrus Energy (nuclear fuel)",
    "CCJ": "Cameco (uranium)",
    "URA": "Global X Uranium ETF",
    # Genomics
    "CRSP": "CRISPR Therapeutics (gene editing)",
    "NTLA": "Intellia Therapeutics (gene editing)",
    "BEAM": "Beam Therapeutics (base editing)",
    "ARKG": "ARK Genomic Revolution ETF",
    # Crypto and fintech
    "MSTR": "Strategy (bitcoin treasury)",
    "SOFI": "SoFi Technologies (digital bank)",
    "AFRM": "Affirm (buy now, pay later)",
    # Electric vehicles and batteries
    "RIVN": "Rivian (electric vehicles)",
    "LIT": "Global X Lithium & Battery ETF",
    # Drones and air mobility
    "AVAV": "AeroVironment (drones)",
    "KTOS": "Kratos (drones, defense tech)",
    "JOBY": "Joby Aviation (air taxis)",
    "ACHR": "Archer Aviation (air taxis)",
    # Broad disruptive innovation
    "ARKK": "ARK Innovation ETF",
    # Small caps
    "IWM": "Russell 2000 small caps",
    "VB": "US small caps (Vanguard)",
    "IWO": "Russell 2000 growth",
}

THEME_GROUPS: dict[str, list[str]] = {
    "Quantum computing": ["IONQ", "RGTI", "QBTS", "QUBT", "QTUM"],
    "Robotics": ["SYM", "SERV", "BOTZ", "ROBO"],
    "Space": ["RKLB", "ASTS", "LUNR", "PL", "RDW", "ARKX", "UFO"],
    "AI software": ["AI", "SOUN", "PATH", "SNOW", "NET"],
    "Nuclear": ["OKLO", "SMR", "NNE", "LEU", "CCJ", "URA"],
    "Genomics": ["CRSP", "NTLA", "BEAM", "ARKG"],
    "Crypto & fintech": ["MSTR", "SOFI", "AFRM"],
    "EVs & batteries": ["RIVN", "LIT"],
    "Drones & air taxis": ["AVAV", "KTOS", "JOBY", "ACHR"],
    "Small caps": ["IWM", "VB", "IWO"],
}

GROUPS = {
    "etfs": "ETFs: stocks, sectors, bonds, gold, leveraged",
    "sp500": "S&P 500 stocks",
    "themes": "Disruptive themes and small caps",
}
DEFAULT_GROUPS = ["etfs", "sp500", "themes"]

# Names for funds the user may hold that no group lists.
EXTRA_NAMES: dict[str, str] = {
    "SGOV": "0–3 month Treasury bills (cash)",
    "QQQM": "Nasdaq-100",
    "ARR": "ARMOUR Residential, mortgage REIT",
}

_sp500_names: dict[str, str] = {}

# Back-compat for callers that read the ETF list under its old name.
DEFAULT_UNIVERSE = ETFS


def describe(ticker: str) -> str:
    return (ETFS.get(ticker) or THEMES.get(ticker) or _sp500_names.get(ticker)
            or EXTRA_NAMES.get(ticker) or "")


def read_config(path: str | Path | None) -> dict:
    """``{"groups": [...], "extra": [...]}`` from ``path``, or the defaults.

    The older ``{"tickers": [...]}`` form is read as the ETF group replaced by
    that list.
    """
    if not (path and Path(path).exists()):
        return {"groups": list(DEFAULT_GROUPS), "extra": []}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"universe file {path} is not usable: {exc}") from exc
    if "tickers" in data and "groups" not in data:
        return {"groups": [], "extra": [t.strip().upper() for t in data["tickers"] if t.strip()]}
    groups = [g for g in data.get("groups", DEFAULT_GROUPS) if g in GROUPS]
    return {"groups": groups, "extra": [t.strip().upper() for t in data.get("extra", []) if t.strip()]}


def tickers_for(config: dict, cache_dir: str | Path | None = None) -> list[str]:
    """The scan list a config describes, in group order, without duplicates."""
    out: list[str] = []
    for group in config.get("groups", []):
        if group == "etfs":
            out += list(ETFS)
        elif group == "themes":
            out += list(THEMES)
        elif group == "sp500":
            members = get_sp500_members(cache_dir)
            _sp500_names.update(members)
            out += list(members)
    out += config.get("extra", [])
    return list(dict.fromkeys(out))


def load_universe(path: str | Path | None, cache_dir: str | Path | None = None) -> list[str]:
    return tickers_for(read_config(path), cache_dir)
