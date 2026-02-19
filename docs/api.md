# API and Streaming Contracts

The `breve-creatures` executable serves both the UI and the simulation APIs.

## HTTP

- `GET /health`
- `POST /api/eval/generation`
  - Body:
    - `genomes` (array)
    - `seeds` (array)
    - optional: `durationSeconds`, `dt`
  - Returns per-genome aggregate fitness and descriptor summaries.

## WebSocket

- `GET /api/trial/ws`
  - First client message must be a JSON trial request:
    - `genome`
    - `seed`
    - optional: `durationSeconds`, `dt`, `snapshotHz`
  - Streams:
    - `trial_started`
    - `snapshot`
    - `trial_complete`

- `GET /api/eval/ws`
  - First client message must be a JSON generation eval request:
    - `genomes` (array)
    - `seeds` (array)
    - optional: `durationSeconds`, `dt`
  - Streams:
    - `generation_started`
    - `attempt_trial_started`
    - `attempt_complete`
    - `generation_complete`
