# Frontend

Browser UI for the creature simulation, served locally via FastAPI.

## What Is Here

- `index.html`: simulation UI and client logic
- `main.py`: local FastAPI server for serving the frontend files
- `pyproject.toml` / `uv.lock`: Python dependency config

## Prerequisites

- Python 3.12+
- [Astral `uv`](https://docs.astral.sh/uv/)

## Run Frontend

From the `frontend/` directory:

```powershell
uv sync
uv run main.py
```

Frontend URL: `http://127.0.0.1:8000`

Health check: `http://127.0.0.1:8000/health`

## Backend Dependency

The frontend expects the Rust backend running at `http://127.0.0.1:8787`.

Start backend from repo root:

```powershell
cd backend
cargo run
```

## Backend URL Config (if needed)

If you run backend elsewhere, update these constants in `index.html`:

- `BACKEND_HTTP_URL`
- `BACKEND_WS_URL`
- `BACKEND_EVAL_WS_URL`

## Typical Dev Workflow

1. Terminal 1: start backend (`cd backend && cargo run`)
2. Terminal 2: start frontend (`cd frontend && uv sync && uv run main.py`)
3. Open `http://127.0.0.1:8000`
