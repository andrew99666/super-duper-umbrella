# Search Term Relevancy Assistant

A production-ready FastAPI web application that connects to the Google Ads API, inspects campaign landing
pages, and uses OpenAI's ChatGPT models to score paid search queries for relevancy. The tool highlights
irrelevant search terms, suggests negative keywords, and supports exporting the review as CSV (or applying
negatives behind a feature flag).

## Features

- Google OAuth web flow with encrypted refresh-token storage in SQLite
- Campaign picker with HTMX-powered analysis workflow and Tailwind UI
- GAQL helpers targeting `campaign_search_term_view` with automatic fallback
- Landing-page scraping with robots.txt support and OpenAI summarisation cache
- Batch LLM relevancy scoring and schema-validated JSON parsing
- Suggestion review table with quick filters, CSV export, and optional API application hook
- Dockerised deployment using Python 3.11, FastAPI, SQLAlchemy, and uvicorn

## Prerequisites

1. **Google Cloud project** with the Google Ads API enabled.
   - Visit <https://console.cloud.google.com/> and create/select a project.
   - Enable the Google Ads API for that project.
2. **Google Ads developer token.**
   - Request one from the Google Ads UI (Tools & Settings → API Center).
   - Ensure it is approved for at least test usage.
3. **OAuth client credentials.**
   - In the Cloud Console, create OAuth credentials of type **Web application**.
   - Add `http://localhost:8000/oauth/callback` (or your `BASE_URL`) as an authorized redirect URI.
   - Record the client ID and secret.
4. **Login-customer ID (optional).**
   - If you operate through a manager account, capture the manager's customer ID without dashes.
   - Otherwise, the tool will default to the first accessible customer returned by the API.
5. **OpenAI API key.**
   - Create a key from <https://platform.openai.com/> with access to Chat Completions.
6. **Docker** and **docker compose** installed locally.

## Environment variables

Copy `.env.example` to `.env` and populate the following:

```env
APP_SECRET_KEY=replace-with-a-long-random-string
BASE_URL=http://localhost:8000
DATABASE_URL=sqlite:///./data/app.db
GOOGLE_ADS_DEVELOPER_TOKEN=your_dev_token
GOOGLE_ADS_OAUTH_CLIENT_ID=your_oauth_client_id
GOOGLE_ADS_OAUTH_CLIENT_SECRET=your_oauth_client_secret
GOOGLE_ADS_API_VERSION=
GOOGLE_ADS_LOGIN_CUSTOMER_ID=optional_manager_id
OPENAI_API_KEY=sk-...
FEATURE_APPLY_NEGATIVES=false
```

Leave `GOOGLE_ADS_API_VERSION` empty to use the latest version supported by the installed `google-ads` library.
Only set it (for example, `v21`) if Google retires the default and you need to pin to a specific release.

All sensitive values are read from environment variables at runtime. Refresh tokens are encrypted at rest
using a key derived from `APP_SECRET_KEY`.

## Running locally with Docker

```bash
# Build the container and start the stack (FastAPI + SQLite volume)
# (Optional) Ensure the data directory exists so SQLite can create the database file
mkdir -p data  # use `mkdir data` on Windows PowerShell
docker compose up --build
```

The application will be available at <http://localhost:8000> by default. Uvicorn runs with a single worker
inside the container. File changes in `app/` are mounted into the container for rapid iteration.

## Usage workflow

1. Visit `http://localhost:8000` and click **Sign in with Google**.
2. Complete the OAuth consent flow. The refresh token is stored encrypted in SQLite.
3. Pick the operating customer (if multiple) and choose one or more campaigns plus a reporting date range.
4. Click **Run analysis**. The app pulls search terms via `campaign_search_term_view` (falling back to
   `search_term_view` when necessary) and collects landing-page URLs from ad final URLs plus
   `landing_page_view`.
5. Landing pages are fetched (respecting robots.txt), parsed, summarised via OpenAI, and cached.
6. Search terms are batched (≤200 per call) and sent to OpenAI for relevancy scoring and negative keyword
   recommendations. Results are validated against a strict JSON schema.
7. Review the suggestion table, toggle approvals (or auto-select high-confidence irrelevants), and export the
   approved list as CSV. Optionally enable `FEATURE_APPLY_NEGATIVES=true` and extend `/apply-negatives` to
   push negatives via the Google Ads API.

## Development

- Python version: **3.11**
- Frameworks: FastAPI, SQLAlchemy, HTMX, Tailwind CSS via CDN
- Database: SQLite (file-backed, persisted to `data/app.db` by default)
- Tests: run with `pytest`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Notes

- `campaign_search_term_view` is used whenever available to include Performance Max search terms. The code
  automatically retries with `search_term_view` if the preferred view is not supported in the target API
  version. If Google sunsets an API version, bump the `google-ads` dependency (or set `GOOGLE_ADS_API_VERSION`
  to a currently supported release) and rebuild the Docker image.
- Landing-page URLs are sourced from both `ad_group_ad.ad.final_urls` and `landing_page_view` to capture
  expanded final URLs. Duplicate URLs are cached and summaries refreshed on a rolling basis.
- OpenAI calls use conservative temperature settings (≤0.2), exponential back-off, and strict schema
  validation with retries to guard against malformed JSON.
- Applying negatives via the Google Ads API is scaffolded behind a feature flag to prevent accidental account
  changes.
