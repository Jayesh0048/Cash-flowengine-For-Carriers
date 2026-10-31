# Tech Truckers Cash Flow Engine

This is a FastAPI service that turns truck-load and operating-expense data into a cash ledger and 30/60/90-day liquidity forecast. It supports CSV ingestion today and has optional TMS sync and webhook endpoints for live integrations.

## What it does

- Reads loads and expenses from CSV files.
- Converts each load into a factoring advance (92%) and a future reserve release (5%); the remaining 3% is the factoring fee.
- Stores inflows and outflows in SQLite (or PostgreSQL when `DATABASE_URL` is changed).
- Produces per-fleet forecast summaries for `acc1`, `acc2`, or both (`all`).
- Serves API documentation at `/docs` and the bundled dashboard at `/dashboard`.

## Run locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
.venv/bin/python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`, or `http://127.0.0.1:8000/dashboard`.

Load the supplied CSV data with:

```bash
.venv/bin/python -m app.engine.ingestion
```

Then request a forecast, for example:

```bash
curl 'http://127.0.0.1:8000/api/v1/forecast/summary?days=30&account_id=acc1'
```

## Train the forecasting model

The model trains only from `SETTLED` ledger events. This prevents pending
invoices or planned expenses from being treated as known outcomes. Train the
portfolio model with:

```bash
.venv/bin/python -m app.ml_models.train_v1 --account-id all
```

An account-specific model can be trained with `acc1` or `acc2`. It is saved
separately and the API prefers it only when it passes validation; otherwise it
uses the validated portfolio model or the account's rolling baseline.

## Configuration

Copy `.env.example` to `.env` and set real values only for integrations you use. `TMS_USER_EMAIL`, `TMS_USER_PASSWORD`, and `TMS_AUTH_COOKIE` are intentionally blank by default. Do not put passwords, tokens, or cookies in source files.

## Current scope and production work still needed

The forecast is a planning estimate, not an accounting balance: the opening cash balances, fallback revenue pace, factoring terms, and burn rate are code defaults. Before relying on it operationally, move those values to account-specific configuration, validate the CSV field mapping against the TMS exports, add database migrations and authentication/authorization, restrict CORS, protect the webhook with a verified signature, and use a managed database with backups. The live TMS endpoints also need confirmed API contracts and valid credentials.

## Verify

```bash
.venv/bin/pytest -q
```
