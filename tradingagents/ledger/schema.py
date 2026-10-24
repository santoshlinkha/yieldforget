"""Database migrations, in order. Never edit a released migration; add the next one.

Records of what happened are append-only: triggers refuse UPDATE and DELETE on
events, recommendations, decisions, orders and their status logs. A correction
is a new row that names the one it replaces, so the original stays readable.
"""

_IMMUTABLE = ["events", "event_voids", "recommendations", "recommendation_status", "decisions",
              "orders", "order_status", "legacy_imports", "price_chunks", "settings_history", "ai_usage"]


def _guards(table: str) -> str:
    return (f"CREATE TRIGGER {table}_no_update BEFORE UPDATE ON {table} "
            f"BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END;\n"
            f"CREATE TRIGGER {table}_no_delete BEFORE DELETE ON {table} "
            f"BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END;\n")


MIGRATIONS: dict[int, str] = {
    1: """
CREATE TABLE accounts (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    broker TEXT NOT NULL,
    account_type TEXT NOT NULL,           -- roth_ira
    cash_account INTEGER NOT NULL,        -- 1: buys need settled cash (T+1)
    created_at TEXT NOT NULL
);
CREATE TABLE sleeves (
    account_id TEXT NOT NULL REFERENCES accounts(id),
    name TEXT NOT NULL CHECK (name IN ('investing', 'trading')),
    created_at TEXT NOT NULL,
    PRIMARY KEY (account_id, name)
);

-- Every change to holdings or cash. quantity and cash are signed changes to the sleeve.
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_uid TEXT NOT NULL UNIQUE,       -- dedupe key: broker reference or content hash
    account_id TEXT NOT NULL REFERENCES accounts(id),
    sleeve TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN (
        'opening_position', 'opening_cash', 'contribution', 'withdrawal', 'transfer',
        'fill', 'dividend', 'interest', 'fee', 'corporate_action', 'adjustment',
        'forced_liquidation', 'assignment', 'exercise', 'expiration')),
    trade_date TEXT NOT NULL,             -- the session the event belongs to
    settle_date TEXT NOT NULL,            -- when its cash becomes spendable
    ticker TEXT,
    quantity REAL NOT NULL DEFAULT 0,
    price REAL,
    cash REAL NOT NULL DEFAULT 0,
    transfer_id TEXT,                     -- both legs of an internal transfer
    order_id TEXT,
    source TEXT NOT NULL,                 -- manual, csv, migration, paper
    source_ref TEXT,
    note TEXT,
    supersedes INTEGER REFERENCES events(id),
    recorded_at TEXT NOT NULL,
    FOREIGN KEY (account_id, sleeve) REFERENCES sleeves(account_id, name)
);
CREATE INDEX events_by_sleeve ON events (account_id, sleeve, trade_date);
CREATE TABLE event_voids (
    event_id INTEGER PRIMARY KEY REFERENCES events(id),
    voided_at TEXT NOT NULL,
    reason TEXT NOT NULL,
    replaced_by INTEGER REFERENCES events(id)
);

-- Cash set aside: open buy orders and option collateral. Released, not deleted.
CREATE TABLE reservations (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    sleeve TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('order', 'collateral')),
    amount REAL NOT NULL CHECK (amount >= 0),
    ref TEXT,
    created_at TEXT NOT NULL,
    released_at TEXT
);

-- Official recommendations; previews are never stored here.
CREATE TABLE recommendations (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    sleeve TEXT NOT NULL,
    session TEXT NOT NULL,                -- trading session the plan is for
    created_at TEXT NOT NULL,
    published_at TEXT NOT NULL,
    info_cutoff TEXT NOT NULL,            -- last close the inputs include
    market_data_at TEXT NOT NULL,         -- when those prices were fetched
    earliest_execution TEXT NOT NULL,     -- first session an order may fill
    supersedes TEXT REFERENCES recommendations(id),
    code_version TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    config_json TEXT NOT NULL,
    state_json TEXT NOT NULL,             -- holdings and cash the plan started from
    targets_json TEXT NOT NULL,           -- target weights, cash included
    orders_json TEXT NOT NULL,            -- proposed orders, today and next session
    metrics_json TEXT NOT NULL,           -- expected growth and risk, before and after
    reasons_json TEXT NOT NULL,
    universe_json TEXT NOT NULL,
    ai_json TEXT NOT NULL,
    price_tickers_json TEXT NOT NULL,     -- whose snapshot reproduces the inputs
    artifact_sha256 TEXT NOT NULL,
    artifact BLOB NOT NULL                -- the whole plan, zlib JSON
);
CREATE INDEX recommendations_by_session ON recommendations (account_id, sleeve, session, published_at);
CREATE TABLE recommendation_status (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recommendation_id TEXT NOT NULL REFERENCES recommendations(id),
    status TEXT NOT NULL CHECK (status IN ('official', 'superseded', 'withdrawn')),
    at TEXT NOT NULL,
    note TEXT
);
CREATE TABLE decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recommendation_id TEXT NOT NULL REFERENCES recommendations(id),
    decision TEXT NOT NULL CHECK (decision IN ('accepted', 'rejected', 'deferred', 'partial')),
    decided_at TEXT NOT NULL,
    note TEXT
);
CREATE TABLE orders (
    id TEXT PRIMARY KEY,
    recommendation_id TEXT REFERENCES recommendations(id),
    account_id TEXT NOT NULL,
    sleeve TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    ticker TEXT NOT NULL,
    amount REAL,                          -- dollars, or
    quantity REAL,                        -- shares or contracts
    limit_price REAL,
    session TEXT NOT NULL,
    created_at TEXT NOT NULL,
    note TEXT
);
CREATE TABLE order_status (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL REFERENCES orders(id),
    status TEXT NOT NULL CHECK (status IN ('proposed', 'placed', 'partially_filled', 'filled', 'cancelled', 'expired', 'rejected')),
    at TEXT NOT NULL,
    note TEXT
);

-- Prices as fetched, so a past recommendation's inputs can be rebuilt exactly.
-- A ticker's series at time T: its last 'base' chunk created by T plus the 'append' chunks after it.
CREATE TABLE price_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('base', 'append')),
    created_at TEXT NOT NULL,
    first_date TEXT NOT NULL,
    last_date TEXT NOT NULL,
    rows INTEGER NOT NULL,
    payload BLOB NOT NULL                 -- zlib JSON [[date, close], ...]
);
CREATE INDEX price_chunks_by_ticker ON price_chunks (ticker, created_at);

-- Settings per scope: 'global', 'investing', 'trading'. Current value, plus the full history.
CREATE TABLE settings (
    scope TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (scope, key)
);
CREATE TABLE settings_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    changed_at TEXT NOT NULL,
    source TEXT NOT NULL
);

-- Paid AI: results cache, one row per job key with an atomic claim, and every request's usage.
CREATE TABLE ai_cache (
    key TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    info_cutoff TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    output_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE ai_jobs (
    key TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('claimed', 'done', 'failed')),
    owner TEXT NOT NULL,
    reserved_usd REAL NOT NULL DEFAULT 0,
    reserved_requests INTEGER NOT NULL DEFAULT 0,
    claimed_at TEXT NOT NULL,
    heartbeat_at TEXT NOT NULL,
    finished_at TEXT,
    error TEXT
);
CREATE TABLE ai_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_key TEXT,
    task TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    at TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER,
    cost_usd REAL,                        -- NULL when the model's price is not configured
    status TEXT NOT NULL,
    retry INTEGER NOT NULL DEFAULT 0,
    note TEXT
);

-- Short claims on background work, so two processes never run the same day twice.
CREATE TABLE work_claims (
    key TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    claimed_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

-- Files from before the database, kept byte for byte.
CREATE TABLE legacy_imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    imported_at TEXT NOT NULL,
    payload BLOB NOT NULL
);
""" + "".join(_guards(t) for t in _IMMUTABLE),
    2: """
-- Every forecasting experiment run, whatever it found, so the number of trials is known.
CREATE TABLE research_trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    name TEXT NOT NULL,
    universe TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    first_test TEXT NOT NULL,
    last_test TEXT NOT NULL,
    config_json TEXT NOT NULL,
    code_version TEXT NOT NULL,
    scores_json TEXT NOT NULL,
    by_period_json TEXT NOT NULL,
    seconds REAL NOT NULL
);
""" + _guards("research_trials"),
    3: """
-- A forecast run: one model, one horizon, one information cutoff, and its saved model files.
CREATE TABLE forecast_runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    model TEXT NOT NULL,
    model_version TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    info_cutoff TEXT NOT NULL,
    universe TEXT NOT NULL,
    feature_schema TEXT NOT NULL,
    train_start TEXT NOT NULL,
    train_end TEXT NOT NULL,
    config_json TEXT NOT NULL,
    calibration_json TEXT NOT NULL,     -- the walk-forward trial it is backed by, and its scores
    backend TEXT NOT NULL,
    seconds REAL NOT NULL,
    peak_memory_mb REAL,
    code_version TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL
);
CREATE TABLE forecasts (
    run_id TEXT NOT NULL REFERENCES forecast_runs(id),
    ticker TEXT NOT NULL,
    mean REAL NOT NULL,
    median REAL NOT NULL,
    q05 REAL NOT NULL,
    q95 REAL NOT NULL,
    p_gain REAL NOT NULL,
    sigma REAL NOT NULL,
    estimation_sd REAL,
    cash_h REAL NOT NULL,
    flags_json TEXT NOT NULL,
    PRIMARY KEY (run_id, ticker)
);
""" + _guards("forecast_runs") + _guards("forecasts"),
    4: """
-- Stock research: one batch, all its stocks on one information cutoff.
CREATE TABLE stock_runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    info_cutoff TEXT NOT NULL,            -- the last close the inputs include
    earliest_execution TEXT NOT NULL,     -- the next session: nothing can trade at the cutoff's own close
    requested_json TEXT NOT NULL,         -- tickers asked for, and where they came from
    rejected_json TEXT NOT NULL,          -- tickers turned away, with reasons
    universe_size INTEGER NOT NULL,
    horizons_json TEXT NOT NULL,
    code_version TEXT NOT NULL,
    feature_schema TEXT NOT NULL,
    thresholds_version TEXT NOT NULL,
    thresholds_json TEXT NOT NULL,
    models_json TEXT NOT NULL,            -- per horizon and target: model, artifact hash, backing trial
    provenance_json TEXT NOT NULL,        -- prices through, filings and economic vintages used, news collected
    config_json TEXT NOT NULL,
    costs_json TEXT NOT NULL,             -- compute time, device, and any paid AI calls
    seconds REAL NOT NULL
);
CREATE TABLE stock_forecasts (
    run_id TEXT NOT NULL REFERENCES stock_runs(id),
    ticker TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    status TEXT NOT NULL,                 -- ok | abstain
    reasons_json TEXT NOT NULL,           -- data quality and evidence: why to trust it less, or not at all
    payload_json TEXT NOT NULL,           -- every forecast figure
    PRIMARY KEY (run_id, ticker, horizon)
);
-- Headlines as collected: publication and first-seen times, duplicates linked, never rewritten.
CREATE TABLE stock_news (
    id TEXT PRIMARY KEY,                  -- hash of the normalized headline and source
    ticker TEXT NOT NULL,
    title TEXT NOT NULL,
    publisher TEXT,
    url TEXT,
    published_at TEXT,
    first_seen_at TEXT NOT NULL,
    duplicate_of TEXT,
    kind TEXT NOT NULL,                   -- fact | opinion | unclear
    events_json TEXT NOT NULL,
    sentiment REAL,                       -- positive minus negative, -1 to 1; NULL when not scored
    sentiment_model TEXT
);
CREATE INDEX stock_news_ticker ON stock_news (ticker, published_at);
-- Reviewable stock recommendations, and what was decided about them: separate records.
CREATE TABLE stock_ideas (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES stock_runs(id),
    created_at TEXT NOT NULL,
    sleeve TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE stock_idea_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    idea_id TEXT NOT NULL REFERENCES stock_ideas(id),
    ticker TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    decision TEXT NOT NULL,               -- accepted | rejected | deferred
    note TEXT
);
-- What happened, recorded once each forecast's horizon has passed.
CREATE TABLE stock_outcomes (
    run_id TEXT NOT NULL REFERENCES stock_runs(id),
    ticker TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    matured_on TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    realized_json TEXT NOT NULL,
    PRIMARY KEY (run_id, ticker, horizon)
);
""" + "".join(_guards(t) for t in ("stock_runs", "stock_forecasts", "stock_news", "stock_ideas",
                                   "stock_idea_decisions", "stock_outcomes")),
}
