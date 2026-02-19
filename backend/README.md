# backend

Rust simulation backend for Breve Creatures. This binary now serves both API endpoints and the frontend UI.

## Run

```powershell
cd backend
cargo run
```

Server binds to `127.0.0.1:8787`.

UI entrypoint: `GET /`

Optional env var:

- `SIM_MAX_CONCURRENT_JOBS` to cap concurrent simulation workers (defaults to `available_cores - 1`, minimum `1`).

## Endpoints

- `GET /health`
- `GET /api/trial/ws` (WebSocket)
  - First client message must be a JSON trial request:
    - `genome`
    - `seed`
    - optional: `durationSeconds`, `dt`, `snapshotHz`
  - Streams:
    - `trial_started`
    - `snapshot`
    - `trial_complete`
- `POST /api/eval/generation`
  - Body:
    - `genomes` (array)
    - `seeds` (array)
    - optional: `durationSeconds`, `dt`
  - Returns per-genome aggregate fitness/descriptor summaries.
- `GET /api/eval/ws` (WebSocket)
  - First client message must be a JSON generation eval request:
    - `genomes` (array)
    - `seeds` (array)
    - optional: `durationSeconds`, `dt`
  - Streams:
    - `generation_started`
    - `attempt_trial_started`
    - `attempt_complete`
    - `generation_complete`
