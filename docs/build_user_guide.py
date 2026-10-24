"""Build docs/Portfolio_App_User_Guide.pdf: how to use the app, which models it uses, how it makes a plan.

Run from the TradingAgents folder (reportlab is not part of the app's environment):
    uv run --no-project --with reportlab python docs/build_user_guide.py
Screenshots come from docs/img/ (taken from the running app).
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = Path(__file__).parent
OUT = HERE / "Portfolio_App_User_Guide.pdf"
IMG = HERE / "img"

INK = colors.HexColor("#0f172a")
MUTED = colors.HexColor("#475569")
ACCENT = colors.HexColor("#2563eb")
LINE = colors.HexColor("#e2e8f0")
SOFT = colors.HexColor("#f1f5f9")
WARN_BG = colors.HexColor("#fff7ed")
WARN = colors.HexColor("#9a3412")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=18, leading=22,
                    textColor=INK, spaceBefore=6, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=17,
                    textColor=INK, spaceBefore=12, spaceAfter=5)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=10.5, leading=14,
                    textColor=ACCENT, spaceBefore=8, spaceAfter=3)
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.6, leading=13.6,
                      textColor=INK, spaceAfter=5, alignment=TA_LEFT)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.4, leading=11.5, textColor=MUTED)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=8.6, leading=11.4, spaceAfter=0)
CELLB = ParagraphStyle("CellB", parent=CELL, fontName="Helvetica-Bold")
CODE = ParagraphStyle("Code", parent=BODY, fontName="Courier", fontSize=8.4, leading=11, textColor=INK,
                      backColor=SOFT, borderPadding=(5, 6, 5, 6), spaceBefore=3, spaceAfter=8)
NOTE = ParagraphStyle("Note", parent=BODY, backColor=WARN_BG, textColor=WARN, borderPadding=(6, 8, 6, 8),
                      spaceBefore=4, spaceAfter=10)
TITLE = ParagraphStyle("Title", parent=H1, fontSize=26, leading=31, spaceAfter=6)
SUB = ParagraphStyle("Sub", parent=BODY, fontSize=12, leading=16, textColor=MUTED)


def p(text, style=BODY):
    return Paragraph(text, style)


def bullets(items, style=BODY):
    return ListFlowable([ListItem(p(i, style), leftIndent=12, value="•") for i in items],
                        bulletType="bullet", start="•", leftIndent=12, bulletFontSize=8, spaceAfter=6)


def numbered(items):
    return ListFlowable([ListItem(p(i), leftIndent=14) for i in items], bulletType="1", leftIndent=14,
                        bulletFontSize=9, spaceAfter=6)


def table(rows, widths, header=True):
    data = [[p(c, CELLB if (header and r == 0) else CELL) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
             ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), SOFT), ("LINEBELOW", (0, 0), (-1, 0), 0.8, MUTED)]
    t.setStyle(TableStyle(style))
    return t


def shot(name, width=6.9 * inch, caption=None, max_height=4.6 * inch):
    path = IMG / name
    if not path.exists():
        return Spacer(1, 0)
    img = Image(str(path))
    scale = min(width / img.imageWidth, max_height / img.imageHeight)
    img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
    parts = [img]
    if caption:
        parts.append(p(caption, SMALL))
    return KeepTogether(parts + [Spacer(1, 6)])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.8 * inch, 0.5 * inch, "Portfolio Optimizer: user guide")
    canvas.drawRightString(7.7 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.restoreState()


W = 6.9 * inch
story = []

# --- cover ---
story += [Spacer(1, 1.2 * inch), p("Portfolio Optimizer", TITLE),
          p("User guide: how to use it, which models it uses, and how it makes a plan", SUB),
          Spacer(1, 0.3 * inch),
          p("For a Robinhood Roth IRA split into an <b>Investing</b> and a <b>Trading</b> account. "
            "The app runs on your own computer, plans each trading day's orders, records what you actually did, "
            "and keeps an honest record of how its advice would have performed. <b>It never places an order</b>: "
            "you place them in the Robinhood app."),
          Spacer(1, 0.2 * inch),
          table([["Section", "What it covers"],
                 ["1. Start and stop", "Running the web app"],
                 ["2. The two accounts", "Investing, Trading, transfers, cash and settlement"],
                 ["3. A normal day", "From opening the app to recording your fills"],
                 ["4. The pages", "What each tab and the ticker page show"],
                 ["5. Settings", "Risk level, stress measure, growth target, forecasts"],
                 ["6. How a plan is made", "From prices to orders, step by step"],
                 ["7. The models", "Risk, expected-return, volatility and AI models, and the evidence for each"],
                 ["8. Records, data and costs", "Where everything is kept; what is downloaded and paid for"],
                 ["9. Command line", "The same things without the browser"],
                 ["10. Limits", "What the app cannot know or promise"]],
                [1.9 * inch, 5.0 * inch]),
          Spacer(1, 0.25 * inch),
          p("Plans are estimates from past prices, not promises. A future crash can be deeper than any past one. "
            "Nothing here is financial advice.", NOTE),
          PageBreak()]

# --- 1. start ---
story += [p("1. Start and stop", H1),
          p("Open PowerShell (or any terminal) and run:"),
          p("cd C:\\Users\\santo\\Desktop\\random_projects\\Robinhood\\TradingAgents<br/>"
            ".venv\\Scripts\\python.exe -m cli.main web", CODE),
          p("The app opens at <b>http://127.0.0.1:8765/</b>. Press <b>Ctrl+C</b> in that window to stop it. "
            "Only one copy can run at a time; if the port is taken, a copy is already running: open the address, "
            "or stop it first:"),
          p("Get-NetTCPConnection -LocalPort 8765 -State Listen | ForEach-Object { Stop-Process -Id $_.OwningProcess }",
            CODE),
          bullets(["Use <font face='Courier'>python.exe -m cli.main</font>; the <font face='Courier'>tradingagents</font> "
                   "shortcut is broken on this machine.",
                   "<font face='Courier'>--port 8800</font> runs it on another port.",
                   "Add <font face='Courier'>?sleeve=trading</font> to the address to open the Trading account directly."]),
          ]

# --- 2. accounts ---
story += [p("2. The two accounts", H1),
          p("Robinhood sees <b>one</b> Roth IRA. The app divides it into two bookkeeping accounts (\"sleeves\"); "
            "Robinhood's rules (settlement, what an IRA may trade) apply to the whole IRA."),
          table([["", "Investing", "Trading"],
                 ["Purpose", "The long-term portfolio", "An experimental allocation, started with $1,000"],
                 ["Money in", "Your recurring contributions ($27.61 a trading day by default)",
                  "Only transfers you make from Investing"],
                 ["After a loss", "-", "Never topped up automatically; its value moves only with its own results and transfers"],
                 ["Settings", "Its own risk level, measure, growth target", "Its own, separately"]],
                [1.1 * inch, 2.9 * inch, 2.9 * inch]),
          Spacer(1, 6),
          p("Choose the account with the <b>Investing / Trading</b> tabs at the top right; each tab shows that "
            "account's value. Every page then shows that account's plan. The <b>Account</b> card on the Overview "
            "shows both and the whole IRA."),
          p("Cash, settlement and transfers", H3),
          bullets(["<b>Settled cash</b> can be spent now; <b>unsettled</b> cash is sale proceeds waiting one trading "
                   "day (T+1). Buys only ever use settled cash, so there is never a good-faith violation.",
                   "<b>SGOV</b> (0-3 month Treasury bills) is money waiting to be invested, not a long-term holding.",
                   "A <b>transfer</b> moves shares or settled cash between the accounts. The IRA's total does not change, "
                   "and the transfer is not counted as profit for either account. It is bookkeeping: nothing is "
                   "bought or sold at Robinhood.",
                   "If Trading is empty, the Overview offers to fund it from Investing (settled cash first, then SGOV) "
                   "with a button. It only proposes what Trading still lacks."]),
          ]

# --- 3. a normal day ---
story += [p("3. A normal day", H1),
          numbered([
              "<b>Open the app</b> after the market closes (or the next morning). Prices update from the last "
              "<i>completed</i> trading day; nothing intraday is used.",
              "<b>Pick the account</b> (Investing or Trading) and read the <b>Overview</b>: the orders for the next "
              "trading session, what follows once sales settle, and why each one is there.",
              "<b>Save the plan</b> if you intend to follow it: <i>Save as official recommendation</i>. A plan on "
              "screen is only a preview; saving makes it the record for its session (see section 8).",
              "<b>Place the orders yourself</b> in the Robinhood app. The sells come first; the buys use their "
              "proceeds the next session, once they settle.",
              "<b>Record what happened</b>: each fill, or import Robinhood's activity CSV (section 10). Holdings in "
              "the app change only through what you record, never because a plan was saved or accepted.",
              "<b>Recurring buys</b>: if the Auto-invest card lists changes, update your recurring investments in "
              "the Robinhood app to match."]),
          p("Saving or accepting a recommendation is not trading. The app never sends orders.", NOTE),
          ]

# --- 4. pages ---
story += [PageBreak(), p("4. The pages", H1),
          shot("overview.png", caption="Overview: risk summary and account tabs at the top; orders on the left; "
                                       "paper record, growth chart, auto-invest, account and data freshness on the right."),
          table([["Page", "What it shows"],
                 ["Overview", "Value, stress loss, expected growth, share invested, daily buys; the orders for the next "
                              "session with reasons; the paper record; the growth-vs-risk chart; <b>Auto-invest</b> "
                              "(recurring buys to set in Robinhood and how they differ from yours); the accounts; "
                              "<b>Data freshness</b> (prices, last official recommendation, where expected returns come from)."],
                 ["Performance", "The paper record since day one: \"followed the plan\" against \"kept as is\", value chart, "
                                 "day by day, and the paper holdings. It covers the whole IRA."],
                 ["Allocation", "Every fund held or chosen: now, target, the order actually placed (small trades under "
                                "0.5% of the account are <i>skipped</i>), worst drop, expected growth, and why."],
                 ["Screener", "Every fund scanned, ranked by expected growth per unit of crash risk."],
                 ["Forecasts", "Tick horizons and models and run them at once, side by side with each model's evidence; "
                               "each fund's forecast (mean, 5-95% range, chance of a gain); the <b>live record</b> (saved "
                               "forecasts scored after the fact); the walk-forward comparison of all models. Section 8 "
                               "explains how to read it all."],
                 ["Settings", "Holdings, recurring buys, <b>Risk and goals</b> per account, AI ratings, what to scan."],
                 ["Ticker page", "Click any ticker. Opens in a new tab: price and chart against SPY, news newest first, "
                                 "Wall Street ratings, price targets and EPS estimates, a <b>quality checklist</b> (economic "
                                 "moat, financial strength, profitability and cash flow, valuation), earnings surprises, "
                                 "dividends, splits and share count, financial statements, your position and the plan's "
                                 "target, forecasts, saved analysis reports."]],
                [1.2 * inch, 5.7 * inch]),
          Spacer(1, 8),
          shot("allocation.png", caption="Allocation: the Trade column is the order actually placed; small trades are marked skip."),
          shot("ticker.png", caption="A ticker page (AAPL): ratings, estimates and forecasts at the top; the quality "
                                     "checklist, earnings, dividends and statements below."),
          ]

# --- 5. settings ---
story += [PageBreak(), p("5. Settings: risk and goals", H1),
          p("Settings > <b>Risk and goals</b> holds each account's choices; the bar at the top shows them for the "
            "account in view, with a <b>Change</b> button that opens this card."),
          shot("settings.png", max_height=2.5 * inch),
          p("Risk level (1 to 10)", H3),
          p("Each level caps the <b>stress loss</b>: the most the account may lose in the stress test."),
          table([["Level", "Name", "Stress-loss cap", "Level", "Name", "Stress-loss cap"],
                 ["1", "Capital preservation", "1%", "6", "Balanced", "20%"],
                 ["2", "Very conservative", "2.5%", "7", "Growth", "30%"],
                 ["3", "Conservative", "5%", "8", "Aggressive", "40%"],
                 ["4", "Cautious", "10%", "9", "Very aggressive", "50%"],
                 ["5", "Moderate", "15%", "10", "Maximum growth", "no cap"]],
                [0.5 * inch, 1.5 * inch, 1.1 * inch, 0.5 * inch, 1.5 * inch, 1.1 * inch]),
          Spacer(1, 6),
          p("How stress loss is measured", H3),
          bullets(["<b>Worst case</b> (default, recommended): what the account would lose if every holding repeated its "
                   "own worst historical drop at the same time. In a 2008-2026 backtest it never broke its cap.",
                   "<b>Past crashes replayed</b>: the worst real crash since 1998 (stocks, long bonds, gold, commodities) "
                   "replayed on the mix, so holdings that did not fall together are not assumed to. It invests more at the "
                   "same level, but in the same backtest it went past its cap by up to 1.5 times."]),
          p("Growth target", H3),
          p("Optional. With a target, the plan takes the <b>least</b> risk expected to reach it, never more than the level "
            "allows. If the level cannot reach it, the plan says <i>out of reach</i> and stops at the level's cap. "
            "Expected growth is an estimate; backtests realized 0.3-1.3 points a year less than expected."),
          p("Expected returns: standard rule or forecasts", H3),
          p("Forecasts > <i>Use forecasts in the Investing plan</i> switches the source of expected returns from the "
            "standard rule to a saved forecast model (section 7). In backtests no forecast beat the standard rule after "
            "costs, so the standard rule is the recommended setting. The Overview's Data freshness card always says which "
            "one is in use."),
          ]

# --- 6. how a plan is made ---
story += [PageBreak(), p("6. How a plan is made", H1),
          p("Every time a page needs a plan, the app runs these steps for the chosen account. The result is reused until "
            "prices, your records or your settings change."),
          table([["Step", "What happens"],
                 ["1. Prices", "Daily adjusted closes (dividends and splits included) from Yahoo Finance, kept on disk; only "
                               "sessions not yet stored are downloaded. Only completed sessions are used."],
                 ["2. Funds to consider", "The account's holdings, plus the scan list: about 40 long-lived ETFs, the S&amp;P 500 "
                                          "and \"disruptive theme\" stocks and funds. A fund needs 3 years of prices. The 60 "
                                          "with the best expected growth per unit of crash risk go on to the optimizer."],
                 ["3. Risk figures", "For each fund: volatility and correlations over 10 years, and its worst drop ever "
                                     "(a young fund borrows an older fund's history, or is assumed to be able to fall 50%)."],
                 ["4. Expected returns", "Standard rule: volatility × a Sharpe ratio blended between the fund's own history "
                                         "(at most half) and a common 0.3, less half the variance (what compounding loses). "
                                         "Or a forecast model, if switched on (section 7)."],
                 ["5. Optimize", "Choose weights that maximize expected return minus a risk penalty "
                                 "(mean-variance, risk aversion 3), subject to: stress loss within the level's cap; no fund "
                                 "above 35% of the account; no fund using more than a quarter of the cap at its own worst drop; "
                                 "long only. Whatever is not invested stays in SGOV."],
                 ["6. Growth target", "If one is set, the cap is lowered as far as it can go while the plan still expects the "
                                      "target (a bisection over the cap)."],
                 ["7. Orders", "Trades under 0.5% of the account are skipped. Sales today; buys the next session from their "
                               "settled proceeds (T+1). New money comes out of SGOV at your daily contribution pace; an account "
                               "without contributions (Trading) uses SGOV for the whole gap. Orders under $1 are dropped."],
                 ["8. Auto-invest", "Your daily contribution split across the target mix, at Robinhood's frequencies (every "
                                    "market day, week, 2 weeks or month), with a $1 minimum per order assumed."]],
                [1.35 * inch, 5.55 * inch]),
          Spacer(1, 8),
          p("The optimizer, in one line", H3),
          p("maximize  expected_return · w  -  (3/2) · w<super>T</super> Cov w<br/>"
            "subject to  0 &lt;= w &lt;= caps,  sum(w) &lt;= 1,  stress loss(w) &lt;= cap", CODE),
          p("It is solved by ADMM (the method the OSQP solver uses), and the stress limit and per-fund caps are then enforced "
            "exactly. Each reason on the Overview and Allocation pages quotes the numbers that decided a fund's weight."),
          ]

# --- 7. models ---
story += [PageBreak(), p("7. The models", H1),
          p("Risk models (what \"stress loss\" means)", H2),
          table([["Model", "How it works", "Evidence (2008-2026 walk-forward)"],
                 ["Worst case (default)", "Sum over holdings of weight × that fund's worst historical drop.",
                  "Never broke its cap in 274 one-year checks."],
                 ["Past crashes replayed", "Each crash in which SPY, TLT, GLD or DBC fell past a threshold, replayed on the "
                                           "mix; a fund absent from a crash is estimated from its downside beta.",
                  "Broke its cap by up to 1.5×; no extra return at equal realized risk."]],
                [1.4 * inch, 3.0 * inch, 2.5 * inch]),
          p("Expected-return models", H2),
          p("The standard rule is used unless you switch to forecasts. The forecast models were all trained and tested the "
            "same way: walk-forward from 2012, refitted every year on data known at the time, on the ETF universe."),
          table([["Model", "Type", "Inputs"],
                 ["Standard rule (vol_prior)", "Volatility × shrunk Sharpe", "Each fund's volatility and its own Sharpe history"],
                 ["Momentum", "Transparent baseline", "12-month return, skipping the last month"],
                 ["Elastic Net", "Regularized linear regression", "The 22 features below"],
                 ["XGBoost", "Gradient-boosted trees (regression, ranking, quantiles)", "The 22 features below"],
                 ["LSTM", "Small neural network (about 10,000 weights)", "Last 60 daily returns of the fund and SPY, plus the features"],
                 ["Chronos-2", "Amazon's pretrained forecaster, run locally, zero-shot", "Only the fund's last 1,024 daily prices"]],
                [1.5 * inch, 2.4 * inch, 3.0 * inch]),
          Spacer(1, 6),
          p("<b>Features</b> (all as of each date, ranked across funds): momentum over 1, 3, 6 and 12 months; one-week reversal; "
            "volatility (all and downside); beta to SPY; distance from the 52-week high and the 200-day average; the past "
            "year's deepest fall; high-low range; dollar volume, its trend, and illiquidity. Market conditions: VIX and its "
            "slope, the 10-year yield's change, the T-bill rate, SPY's recent return and volatility. For stocks there are also "
            "analyst actions (price-target upside, net upgrades) and earnings surprises from Yahoo's dated histories.", SMALL),
          p("What each forecast gives", H3),
          p("For each fund and horizon (5, 20 or 60 trading days): the expected log return above cash, the median, a 5-95% "
            "range (from the model's errors on data it had not seen), the chance of a gain, and how much the forecast itself "
            "depends on which history the model saw (model uncertainty)."),
          p("What the evidence says", H3),
          table([["Measure (ETF universe, 2012-2026)", "Result"],
                 ["Ranking funds against each other (IC, 0 = chance)",
                  "Learned models beat chance and the standard rule: about 0.11 at 20 days and up to 0.21 at 60 days, vs 0.07 and 0.12"],
                 ["Timing the whole market", "No model can (correlation about zero or negative)"],
                 ["Chance-of-gain figures", "Worse than simply using the historical base rate"],
                 ["Portfolio after trading costs", "No forecast beat the standard rule; the best roughly tied it with 5× the trading"],
                 ["Against simply holding VTI or SPY", "Every optimizer setting trailed them over 2012-2026 (a strong US bull market)"]],
                [2.6 * inch, 4.3 * inch]),
          Spacer(1, 6),
          p("Volatility and covariance", H3),
          p("For volatility, GARCH(1,1) forecasts best at every horizon and EWMA beats the optimizer's 10-year figure. For the "
            "optimizer's covariance, the 10-year sample covariance did better than Ledoit-Wolf, factor or EWMA estimates, so it "
            "is kept. Full results: docs/FORECASTING.md and docs/BACKTEST_RISK_MODELS.md."),
          p("AI analysts (optional, paid)", H3),
          p("The TradingAgents multi-agent analysts (by default one, the market analyst) can rate your holdings Buy to Sell, "
            "which tilts expected returns. They run only when you press <i>Run AI ratings</i>, within a budget ($1 a day and "
            "$10 a month by default), and only for models with a configured price. Results are cached; page loads never call "
            "them. They are off by default and currently unfunded."),
          ]

# --- 8. reading forecasts ---
story += [PageBreak(), p("8. How to interpret forecasts", H1),
          p("A forecast here is a small, uncertain tilt between funds, not a prediction of where a price will be. This "
            "section shows how to read one, using real rows from the Forecasts page on October 23, 2026."),
          shot("forecasts.png", max_height=4.2 * inch,
               caption="Forecasts: choose horizons and models, run them at once, compare them side by side, then read one "
                       "run's forecasts. Here a 5-session batch; the notes column flags runs no walk-forward trial backs."),
          p("What the number means", H2),
          bullets(["<b>A log return above cash.</b> +2.00% means the fund is expected to beat SGOV (T-bills) by about 2% "
                   "over the horizon. It is not the fund's total return, and it is not a yearly rate.",
                   "<b>The horizon is in trading sessions:</b> 5 is about a week, 20 about a month, 60 about three months. "
                   "+1.6% over 60 sessions is roughly 6.5% a year above cash <i>if</i> it repeated four times, which it may not.",
                   "<b>The ticker page shows two kinds of number.</b> The standard rule is a <i>yearly</i> expected growth; the "
                   "model forecasts below it are for their horizon. Do not compare them directly."]),
          p("Reading one row", H2),
          table([["Column", "VOO (60 sessions, XGBoost quantiles)", "How to read it"],
                 ["Mean", "+0.52%", "The expected return above cash over the horizon: the forecast itself."],
                 ["Median", "+0.52%", "The middle outcome. Far from the mean means lopsided risk."],
                 ["5% / 95%", "-8.28% / +8.95%", "Nine times in ten the outcome should fall in this range. The range is "
                                                  "about 33 times wider than the mean: the tilt is small next to the noise."],
                 ["P(gain)", "61%", "The chance of beating zero in total. Treat it loosely: in tests these were worse "
                                    "than simply using how often prices rose in the past."],
                 ["Model unc.", "±1.42%", "How much the mean moves when the model is refitted on resampled history. A "
                                         "mean smaller than this is mostly noise from the fit itself."]],
                [1.0 * inch, 2.1 * inch, 3.8 * inch]),
          Spacer(1, 6),
          p("Compare a leveraged fund in the same run: OKLO at +5.33%, range -37.33% to +34.67%. A bigger mean, but a far "
            "wider range. The top of a forecast list is often the most volatile funds, because a model that ranks by expected "
            "return gives the biggest numbers to the things that move most. Look at the range before the mean."),
          p("Use forecasts to rank, not to time", H2),
          p("Across 2012-2026 the models ranked funds against each other better than chance, but none could tell when the "
            "whole market would rise or fall. So read a run as an ordering on one date: which funds look relatively better "
            "than the others. The plan uses forecasts the same way (only the differences between funds move money), and only "
            "after scaling them by each model's measured calibration."),
          p("The evidence columns", H2),
          table([["Column", "What it is", "Good sign"],
                 ["IC (95%)", "Rank correlation between forecasts and outcomes on each date, averaged; 0 is chance. "
                              "The range is its 95% confidence interval.", "Above 0.05, and the whole range above 0"],
                 ["p (Holm)", "The chance the IC is luck, after counting every model tried in the same study.",
                  "Under 0.05"],
                 ["Calib. (within date)", "Whether the size of the differences between funds was right: 1 is right, "
                                          "above 1 too timid, below 1 too bold, below 0 backwards.", "Between 0.5 and 1.5"],
                 ["Cover.", "Share of outcomes inside the 5-95% range.", "Near 90%"],
                 ["Net top fifth", "What the top fifth of funds by forecast earned above the average, per period, after "
                                   "10 bp a side of trading costs.", "Positive"],
                 ["Agrees", "Rank correlation of this run's forecasts with the other models' at the same horizon.",
                  "High agreement among models with good evidence is reassuring; a lone outlier deserves doubt"],
                 ["best evidence", "Tag on the model with the highest IC among those with Holm p under 0.05.", ""]],
                [1.3 * inch, 3.9 * inch, 1.7 * inch]),
          Spacer(1, 6),
          p("Flags in the Notes column", H2),
          table([["Flag", "Meaning"],
                 ["no walk-forward trial backs this model at this horizon",
                  "Nobody has tested it this way. The plan gives it no weight. Read it as an experiment."],
                 ["trained on ETFs and this ticker was not among them",
                  "A single stock forecast by a fund model. Treat it as a forecast for an unfamiliar kind of asset."],
                 ["gives every ticker the same forecast, so it ranks nothing",
                  "The model kept only market-wide inputs. It says nothing about one fund versus another."],
                 ["under a year of prices", "Some inputs (12-month momentum) are missing; the forecast leans on the rest."],
                 ["survivorship-biased (stock models)", "Trained on today's S&amp;P 500, the companies that survived. Its "
                                                        "results look better than they would have in real time."]],
                [2.6 * inch, 4.3 * inch]),
          Spacer(1, 6),
          p("The live record", H2),
          p("Every saved run is scored once its horizon has passed, on prices no model can have seen in training. This is the "
            "fairest test, and the only clean one for Chronos-2, which may have seen past prices in pretraining. The app "
            "records Chronos-2's forecasts at each new close when it starts. A handful of scored dates says almost nothing; "
            "wait for a few dozen before reading much into the IC there."),
          p("Rules of thumb", H2),
          numbered(["Prefer models whose IC range is above 0 with Holm p under 0.05 at that horizon. Today that is mostly the "
                    "60-session ones: XGBoost ranking and XGBoost on ETFs (IC about 0.21-0.22), XGBoost ranking on stocks "
                    "(about 0.07).",
                    "Look at the 5-95% range and model uncertainty before the mean. A mean inside the model uncertainty is noise.",
                    "Read forecasts across funds on one date, never as a call on where the market is going.",
                    "When well-backed models agree, the ordering is sturdier; when they disagree, neither should move you much.",
                    "Remember the portfolio test: after trading costs, no forecast beat the standard rule in backtests. The "
                    "standard rule stays the default for that reason."]),
          p("Forecasts are model estimates, not outcomes, and not advice. The app never places orders.", NOTE),
          ]

# --- 9. records ---
story += [PageBreak(), p("9. Records, data and costs", H1),
          table([["What", "Where", "Notes"],
                 ["Holdings, trades, transfers, settings, recommendations, forecast runs",
                  "portfolio.sqlite3 (next to my_portfolio.json)",
                  "Append-only: a correction is a new record pointing at the old one. Backed up before any upgrade."],
                 ["Official recommendations", "The database, plus a readable report in recommendations/",
                  "Saved for a trading session; a later save for the same session supersedes, never edits"],
                 ["Paper record (Performance)", "paper_tracker.json", "Updated after each close; never restarted by accident"],
                 ["Prices, news, analyst and earnings histories", "~/.tradingagents/cache/optimizer/",
                  "Downloaded once, then only what is new (news reused for 3 hours, histories for a week)"],
                 ["Forecast models", "~/.tradingagents/cache/optimizer/forecast_models/", "Each run's model file and its hash"]],
                [2.1 * inch, 2.2 * inch, 2.6 * inch]),
          Spacer(1, 8),
          bullets(["<b>Preview vs official.</b> A plan on screen is not stored. <i>Save as official recommendation</i> stores it "
                   "whole, with the code version, settings and the prices it used, so it can be reproduced later.",
                   "<b>Which one counts.</b> A recommendation saved before 3:45 pm New York time is for that day's session; later, "
                   "for the next session. Its orders cannot fill before the session after its last price.",
                   "<b>Costs.</b> Prices, news and data are free (Yahoo Finance, SEC EDGAR, FRED). Only the optional AI analysts "
                   "cost money, within the budget, and every request is recorded (<font face='Courier'>book ai</font>).",
                   "<b>Computer.</b> Numerical work runs locally; the GPU (RTX 3050 Ti, 4 GB) is used only where it is measurably "
                   "faster (large tree models, Chronos-2), one job at a time, falling back to the CPU."]),
          ]

# --- 10. CLI ---
story += [p("10. Command line", H1),
          p("Everything in the web app can also be done from the TradingAgents folder. Prefix each command with "
            "<font face='Courier'>.venv\\Scripts\\python.exe -m cli.main</font>."),
          table([["Command", "What it does"],
                 ["web", "Start the web app"],
                 ["suggest [--sleeve trading] [--save]", "Print the plan and orders; --save makes it official"],
                 ["suggest --risk 4 --model worst_case --target-growth 8", "Change the account's settings, then plan"],
                 ["book status", "Holdings, settled and unsettled cash, value of each account and the IRA"],
                 ["book fill buy VTI 0.5 330.12", "Record a fill you placed (--sleeve trading for Trading)"],
                 ["book contribute 27.61", "Record new money deposited into the IRA"],
                 ["book import-csv activity.csv [--commit]", "Preview, then import, Robinhood's activity export (duplicates skipped)"],
                 ["book transfer investing trading --ticker SGOV --quantity 2", "Move shares between the accounts"],
                 ["book fund-trading [--confirm]", "Preview, then make, the transfers that fund Trading"],
                 ["book recommendations / book decide ID accepted", "List official recommendations; record a decision"],
                 ["book ai", "AI spending against the budget"],
                 ["track", "Update the paper record (the web app also does this)"]],
                [3.1 * inch, 3.8 * inch]),
          ]

# --- 11. limits ---
story += [p("11. Limits", H1),
          bullets(["<b>History is not the future.</b> Stress losses replay the past; a new kind of crash can be deeper.",
                   "<b>Expected growth runs high.</b> Backtests realized 0.3-1.3 points a year less than the plan expected.",
                   "<b>Survivorship.</b> The S&amp;P 500 list is today's members; backtests use ETFs to limit this bias.",
                   "<b>Yahoo's analyst and earnings histories</b> are its lists as of today; corrections cannot be detected.",
                   "<b>Instruments.</b> Recommendations cover long stocks and ETFs. Options (level 2 if approved), shorting, "
                   "futures and crypto are research only in this IRA (docs/CAPABILITIES.md).",
                   "<b>Sleeves are bookkeeping.</b> Robinhood sees one IRA; its rules apply to the whole account.",
                   "<b>No orders are placed.</b> The app plans and records; you trade."]),
          Spacer(1, 10),
          p("More detail in the repository: docs/RUNNING.md (commands), docs/FORECASTING.md (models and results), "
            "docs/BACKTEST_RISK_MODELS.md (risk models), docs/CAPABILITIES.md (what the IRA allows), docs/AUDIT.md.", SMALL),
          ]

doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                        topMargin=0.75 * inch, bottomMargin=0.8 * inch, title="Portfolio Optimizer: user guide",
                        author="Portfolio Optimizer", subject="How to use the app, its models, and how it makes a plan")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
