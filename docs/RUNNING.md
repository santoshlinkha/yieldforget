# Running the portfolio app

Run every command from the `TradingAgents` folder. Use `.venv/Scripts/python.exe -m cli.main`,
because the `tradingagents.exe` shortcut is broken on this machine.

## Records

Everything is kept in `portfolio.sqlite3` next to `my_portfolio.json`. On first open, the
old files are copied into the database byte for byte, your holdings become opening
positions in the **Investing** sleeve, and the risk level becomes Investing's setting.
After that the database is the record. `my_portfolio.json` is rewritten from it after
every change, for code that still reads the file, and should not be edited by hand.

- A backup is taken before any schema migration, in `backups/` next to the database.
- Copy `portfolio.sqlite3` anywhere to back it up by hand; it is one self-contained file.
- Events, recommendations, decisions and orders are append-only; a correction is a new
  record that points at the original.

## Daily use

```powershell
# Preview today's plan for a sleeve (nothing is stored)
.venv/Scripts/python.exe -m cli.main suggest
.venv/Scripts/python.exe -m cli.main suggest --sleeve trading

# Save it as the official recommendation for its trading session
.venv/Scripts/python.exe -m cli.main suggest --save

# The web app (same records): http://localhost:8765
.venv/Scripts/python.exe -m cli.main web
```

A recommendation saved before 3:45 pm New York time is for that day's session; one saved
later is for the next session. Its orders cannot fill before the session after its last
close. Saving a second recommendation for the same session supersedes the first (both are
kept). **Saving or accepting a recommendation is not trading.** Holdings change only when
you record what happened.

## Recording what happened

```powershell
.venv/Scripts/python.exe -m cli.main book status                       # every sleeve and the account
.venv/Scripts/python.exe -m cli.main book fill buy VTI 0.5 330.12      # a fill you placed
.venv/Scripts/python.exe -m cli.main book contribute 27.61             # new money into the IRA
.venv/Scripts/python.exe -m cli.main book import-csv activity.csv      # preview a Robinhood export
.venv/Scripts/python.exe -m cli.main book import-csv activity.csv --commit
.venv/Scripts/python.exe -m cli.main book recommendations
.venv/Scripts/python.exe -m cli.main book decide 8f9ec831 accepted
```

Sale proceeds settle the next session. Buys can only spend settled, unreserved cash, so a
manual fill that the books cannot pay for is refused, unless you add `--broker-record`
because the broker's statement shows it happened.

## The Trading sleeve

Trading starts empty. Fund it yourself from what Investing actually holds:

```powershell
.venv/Scripts/python.exe -m cli.main book fund-trading                 # preview: about $1,000 from settled cash, then SGOV
.venv/Scripts/python.exe -m cli.main book fund-trading --confirm       # make those transfers
.venv/Scripts/python.exe -m cli.main book transfer investing trading --ticker VTI --quantity 1   # anything else
```

A transfer moves shares (at that day's price) or settled cash between sleeves. The account
total does not change, and the transfer counts as a capital flow for each sleeve, not as
profit. Trading is never topped up after a loss; its value changes only with its own
results and the transfers you make. The sleeves are bookkeeping: the broker sees one IRA.

## AI costs

No paid AI call runs unless you ask for one (`suggest --ai`, or the Rate or Explain
buttons), and only within budget:

- the default budget is $1 a day and $10 a month, with 200 and 2,000 requests;
- a model with no configured price cannot be called (Settings > AI, or `POST /api/settings`
  with `{"scope": "global", "values": {"ai_prices": {"openai/<model>": {"input": 1.25, "output": 10}}}}`);
- results are cached by their inputs, so asking again is free;
- one analyst by default, on your holdings plus a shortlist of 10;
- `book ai` shows the spending; `TRADINGAGENTS_NO_PAID_AI=1` switches all paid calls off.

## Tests

```powershell
.venv/Scripts/python.exe -m pytest tests -q
```

`test_suite_isolation::test_yfinance_keeps_its_cache_out_of_the_users_home` fails on this
machine whatever the code does. All other tests pass.
