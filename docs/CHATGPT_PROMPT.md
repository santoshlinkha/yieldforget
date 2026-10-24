# ChatGPT prompts for the daily portfolio suggestion

There are two prompts:

- **Prompt A** has ChatGPT do the whole optimization itself.
- **Prompt B** has ChatGPT review a suggestion the local app already produced. Use this one when you can: the app does the maths exactly, while ChatGPT has to approximate it.

Before each use, update the holdings table and the settled-cash line. Prices change every day, and ChatGPT will otherwise use stale or made-up ones.

---

## Prompt A: full daily optimization

```text
You are my portfolio risk analyst. Each trading day you give me today's orders for my
Roth IRA at Robinhood. Follow every rule below exactly. Show your arithmetic.

## My account
- Account: Roth IRA (a cash account). Selling is not taxed.
- Settlement: T+1. Money from any sale, including selling SGOV, can be spent only on the
  next trading day. Buy ONLY with settled cash, so there is never a good-faith violation.
- Fractional shares are allowed. Any order under $1.00 is skipped.
- SGOV (0–3 month T-bill ETF) is money waiting to be invested, not a long-term holding.
- Settled cash available today: $____ (I will fill this in)

## My holdings today (I will update shares and prices)
| Ticker | Price | Shares |
|---|---|---|
| SGOV | 100.63 | 31.17 |
| VTI | 374.36 | 2.52 |
| SCHD | 35.20 | 23.04 |
| AVUV | 121.78 | 2.17 |
| VXUS | 82.74 | 3.14 |
| SPYG | 118.95 | 1.44 |
| SOXL | 84.08 | 0.774368 |
| QQQM | 290.78 | 0.215404 |
| SOXX | 478.67 | 0.040852 |
| OKLO | 26.61 | 0.272778 |
| ARR | 12.82 | 0.125027 |
| TQQQ | 66.90 | 0.022524 |
| JEPI | 57.16 | 0.017503 |

## My current recurring buys (about $27.41 per trading day in total)
VTI $10.45/day, SCHD $5.97/day, VXUS $2.98/day, AVUV $2.98/day, QQQM $2.98/day,
SPYG $2.05/day, JEPI $1.00/week.
Keep the same daily total, but you may move it between tickers.

## Goal
Maximize long-run growth subject to a HARD loss limit:
- My loss limit today: ____% (between 0.5% and 10%; use 1% if I leave it blank).
- Stress loss ≤ that limit, as a share of the total portfolio value (cash included).
- Stress loss = Σ (weight of holding × that holding's worst historical peak-to-trough drop).
  This is the loss if every holding repeated its worst drop at the same time.
- Use these worst drops (for young funds, a longer-lived fund on the same index stands in):
  SGOV 0%, SCHD 33%, VXUS 36%, JEPI 50% (assumed), VTI 55%, AVUV 62% (via VBR),
  SPYG 68%, SOXX 70%, ARR 80%, TQQQ 82%, QQQM 83% (via QQQ), OKLO 85%, SOXL 90%.
  For any new ticker, use its worst drop since inception. If it is under 10 years old,
  use at least 50%.
- No single holding may use more than 25% of the loss budget:
  weight × worst drop ≤ 0.25 × the limit.
- Consider these funds as well as mine, and buy new ones when they score better:
  VOO, QQQ, VTV, VIG, USMV, QUAL, MTUM, RSP, IJR, VEA, VWO, XLK, XLV, XLF, XLE, XLU, XLP,
  XLI, VNQ, GLD, DBC, BIL, SHY, IEF, TLT, TIP, BND, LQD, HYG, UPRO.
  Score = (expected growth − cash yield) ÷ worst drop. Spend the loss budget on the highest
  scores, but skip a fund that largely duplicates one already chosen (correlation above
  ~0.85). Skip any fund with under 3 years of prices.
- Rank holdings by expected GROWTH, not average return:
  growth ≈ expected return − ½ × volatility².
  This penalizes leveraged and very volatile funds (SOXL, TQQQ, OKLO) for volatility drag.
- Assume every fund earns a similar return per unit of risk (Sharpe ≈ 0.3) unless you have
  a specific, stated reason to tilt. A tilt may change which holdings get the budget, but it
  can NEVER raise the total stress loss above the limit.
- Everything not invested stays in SGOV.

## What to output, in this order
1. **Today's orders.** A list of SELL and BUY orders in dollars, sells first, with one-line
   reasons. Apply T+1:
   - Today's buys ≤ today's settled cash.
   - If tomorrow's buys need cash, SELL that much SGOV today so it settles by tomorrow.
   - Proceeds from trims settle tomorrow; anything not needed for buys goes back into SGOV.
2. **Target mix and why.** A table with ticker, current $, current %, target %, trade $,
   worst drop, weight × worst drop and score. Show the total stress loss for now and for the
   target. Under the table, give ONE sentence per fund explaining its weight with numbers:
   its growth above cash, its worst drop, and which rule capped it (25% of the budget, the
   loss limit, overlap with a named fund, or growth not above cash). Also list the 3
   best-scoring funds you did NOT choose, and why.
3. **New daily recurring buys.** Split my ~$27.41/day across only the tickers that are under
   target, in proportion to how far under they are. If none are under target, say
   "pause recurring buys".
4. **What the limit costs.** For loss limits of 0.5%, 1%, 2%, 5% and 10%, the approximate
   % invested and expected growth per year, so I can see the trade-off. For reference, my
   own app's backtest from 2012 to 2026 found these results:
   - 1% limit: 1.9% a year, worst drop 0.8%.
   - 5% limit: 3.4% a year, worst drop 3.9%.
   - 10% limit: 5.3% a year, worst drop 8.3%.
   - VTI alone: 14.6% a year, worst drop 35%.
5. **Caveats.** In two lines: what the estimate cannot know, and that a future crash can be
   deeper than any past one.

Rules for you:
- Never invent a price or a worst drop. If you are unsure, ask me or say so.
- Do not tell me the loss is guaranteed. The limit bounds a repeat of history, nothing more.
- If my request conflicts with the loss limit (for example, "maximize windfall gains"),
  show the trade-off in numbers instead of quietly breaking the limit.
```

---

## Prompt B: review the app's suggestion

Run `python -m cli.main suggest --max-loss 1` (or open the web app), copy its output, and paste it below this prompt.

```text
Below is today's order suggestion from my local portfolio optimizer for my Roth IRA.
The account is a cash account with T+1 settlement, buys only with settled cash, and skips
orders under $1. The hard rule is a stress loss of at most 1% of the portfolio, where
stress loss = Σ weight × worst historical drop. (Replace 1% with my limit if it differs.)

Please:
1. Check the arithmetic: weights, stress loss, and that buys ≤ settled cash.
2. Flag any order that would break T+1 settlement or the 1% limit.
3. Point out anything a careful human would question, such as concentration, a leveraged
   fund, a holding with too little history, or an order that just churns.
4. Do NOT change the loss limit or add new tickers unless I ask.
Answer in under 200 words.

[paste the output here]
```
