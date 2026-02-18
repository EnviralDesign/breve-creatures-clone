# sim-backend

Rust simulation backend for Breve Creatures.

## Run

```powershell
cd sim-backend
cargo run
```

Server binds to `127.0.0.1:8787`.

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
