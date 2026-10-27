"""The policy study's report as Markdown: measured numbers only, straight from the stored results."""

from __future__ import annotations

from pathlib import Path


def _pct(x, d=2):
    return "n/a" if x is None or x != x else f"{x * 100:+.{d}f}%"


def write(report: dict, path: Path) -> Path:
    from .policy_study import DESIGN_VERSION, HOLDOUT_START

    st = report["status"]
    lines = [f"# Policy study ({DESIGN_VERSION})", "",
             f"Trial {report.get('trial')}; runtime {report.get('seconds')} s; decision dates every {report['every']} "
             f"sessions; survivorship: {report['survivorship']}. Holdout from {HOLDOUT_START.date()}.", "",
             f"**Status: {st['state']}.** {st.get('why') or 'Every promotion check passed.'}", "",
             "## Outcome probabilities (Brier skill = 1 - Brier / Brier of the training base rates; > 0 beats them)", "",
             "| Model : policy : period | Rows | Log loss (base) | P(target) skill | P(stop) skill | P(expiry) skill | P(no fill) skill | Mean-excess IC | q10 coverage |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key, c in report["classification"].items():
        pc = c["per_class"]
        sk = lambda k: "–" if k not in pc or pc[k]["skill"] is None else f"{pc[k]['skill']:+.4f}"  # noqa: E731
        lines.append(f"| {key} | {c['n']:,} | {c['log_loss']:.4f} ({c['log_loss_base']:.4f}) | {sk('target')} | {sk('stop')} | "
                     f"{sk('expiry')} | {sk('nofill')} | {c['mean_excess_ic']:+.4f} | {c['q10_coverage']:.1%} |")
    lines += ["", "Skill by fold (LightGBM, next close; all classes together):", ""]
    dev = report["classification"].get("lgbm:next_close:development", {})
    lines.append(", ".join(f"{k}: {v['skill']:+.4f} (n={v['n']:,})" for k, v in dev.get("by_fold", {}).items()) or "n/a")
    if dev.get("by_regime"):
        lines += ["", "By VIX regime (development): " + ", ".join(
            f"{k} {v['skill']:+.4f} (n={v['n']:,})" for k, v in dev["by_regime"].items())]
    lines += ["", "Reliability, P(stop first), LightGBM next close, holdout (forecast vs observed by decile):", ""]
    hold = report["classification"].get("lgbm:next_close:holdout")
    if hold:
        lines.append("; ".join(f"{b['forecast']:.2f}->{b['observed']:.2f} (n={b['n']})"
                               for b in hold["per_class"]["stop"]["reliability"]))
    lines += ["", "## Strategies: mean excess over cash per decision date (5 slots, empty slot = cash)", "",
              "Windows overlap (20-25 sessions, every 10), so the interval is a block bootstrap. Not annualized.", "",
              "| Strategy : period | Dates | Mean excess | 95% interval | Share > 0 | 5th pct | Worst |", "|---|---|---|---|---|---|---|"]
    for key, s in report["strategies"].items():
        lines.append(f"| {key} | {s['dates']} | {_pct(s['mean_excess'])} | [{_pct(s['ci95'][0])}, {_pct(s['ci95'][1])}] | "
                     f"{s['share_positive']:.0%} | {_pct(s['p05'])} | {_pct(s['worst'])} |")
    lines += ["", "Margins tried on development (selection count = 4 per model): " + "; ".join(
        f"{k}: chose {v['chosen']:.3f} from {v['tried']}" for k, v in report["margins"].items()), "",
        "The existing rule here is the deployed ranker and its expected-net gate (>= 0.5%), without its P(profit) "
        "and P(large loss) gates. SPY: bought at the next close, held 20 sessions, over cash.", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
