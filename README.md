# Breve Creatures Clone

![Project Image](images/image.png)

A browser-based evolutionary creature simulation.

- Frontend: HTML/JS app served locally by FastAPI (`main.py`)
- Backend: Rust simulation/evaluation service (`sim-backend`)

## Prerequisites

- Python 3.12+
- [Astral `uv`](https://docs.astral.sh/uv/) (for Python env + deps)
- Rust toolchain (`rustup`, `cargo`)

If you do not have them installed:

- Install `uv` (Windows example): `winget install --id=astral-sh.uv -e`
- Install Rust (recommended): https://rustup.rs/

On Windows, if Rust build fails, install **Visual Studio Build Tools (C++ workload)**.

## Quick Start

Run backend and frontend in separate terminals.

1. Start backend

```powershell
cd sim-backend
cargo run
```

Backend listens on `http://127.0.0.1:8787`.

2. Start frontend server (from repo root)

```powershell
uv sync
uv run python main.py
```

Frontend listens on `http://127.0.0.1:8000`.

3. Open the app

- `http://127.0.0.1:8000`

## Health Checks

- Frontend: `http://127.0.0.1:8000/health`
- Backend: `http://127.0.0.1:8787/health`
