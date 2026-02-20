# API and Streaming Contracts

The `breve-creatures` executable serves both UI assets and backend simulation APIs.

## HTTP

- `GET /health`

- `GET /api/evolution/state`
  - Returns current backend-owned evolution status.
  - Includes:
    - `generation`, `populationSize`, `pendingPopulationSize`
    - `currentAttemptIndex`, `currentTrialIndex`, `trialCount`
    - `bestEverScore`, `currentScore`, `paused`
    - `runSpeed`
    - `fastForwardRemaining`, `fastForwardActive`
    - `injectionQueueCount`
    - `currentGenome`, `bestGenome`

- `GET /api/evolution/history`
  - Returns backend-owned fitness history points.
  - Each point includes:
    - `generation`
    - `bestFitness`
    - `attemptFitnesses` (all candidate fitness values for that generation)

- `POST /api/evolution/control`
  - Body:
    - `action`
      - `pause`
      - `resume`
      - `toggle_pause`
      - `restart`
      - `set_population_size` (requires `populationSize`)
      - `set_run_speed` (requires `runSpeed`)
      - `queue_fast_forward` (requires `fastForwardGenerations`)
      - `stop_fast_forward`
  - Returns updated evolution status.

- `GET /api/evolution/genome/current`
  - Returns the current candidate genome.

- `GET /api/evolution/genome/best`
  - Returns best-so-far genome.

- `POST /api/evolution/genome/import`
  - Queues imported genomes for next-generation injection.
  - Backend consumes up to one queued injected genome per generation rollover.
  - Body:
    - optional: `genome` (single genome)
    - optional: `genomes` (array)
    - optional: `mutationMode` (`none` | `light`, default `none`)
  - Returns:
    - `addedCount`
    - `queuedCount`

- `POST /api/evolution/checkpoint/save`
  - Saves a full evolution checkpoint to `data/checkpoints/`.
  - Body:
    - optional: `name`
  - Returns:
    - `id`
    - `path`

- `GET /api/evolution/checkpoint/list`
  - Lists available checkpoints.

- `POST /api/evolution/checkpoint/load`
  - Loads a checkpoint into the backend worker.
  - Body:
    - optional: `id` (if omitted, loads `latest.json`)
  - Returns:
    - `id`

- `POST /api/eval/generation`
  - Body:
    - `genomes` (array)
    - `seeds` (array)
    - optional: `durationSeconds`, `dt`
  - Returns per-genome aggregate fitness and descriptor summaries.

## WebSocket

- `GET /api/evolution/ws`
  - Streams:
    - `status`
    - `generation_summary`
    - `trial_started`
    - `snapshot`
    - `trial_complete`
    - `error`

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
  - First client message must be a JSON generation-eval request:
    - `genomes` (array)
    - `seeds` (array)
    - optional: `durationSeconds`, `dt`
  - Streams:
    - `generation_started`
    - `attempt_trial_started`
    - `attempt_complete`
    - `generation_complete`
