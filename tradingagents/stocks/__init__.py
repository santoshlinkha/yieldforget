"""Stock-only research: multi-ticker forecasts of returns, trend, breakouts, drawdowns and news.

Funds have their own forecasting (``tradingagents.research``); this package covers
single stocks only, and reuses that package's walk-forward machinery, models and
trial registry. See ``targets`` for what is forecast, ``pipeline`` for a batch run.
"""
