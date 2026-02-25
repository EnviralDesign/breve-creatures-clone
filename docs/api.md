# API and Streaming Contracts

The `breve-creatures` executable serves both UI assets and backend simulation APIs.

## HTTP

- `GET /health`
  - Returns `{ "status": "ok" }`.

- `GET /api/evolution/state`
  - Returns current backend-owned evolution status.
  - Includes:
    - `minPopulationSize`, `maxPopulationSize`, `defaultPopulationSize`
    - `defaultTrialCount`, `maxTrialCount`, `defaultGenerationSeconds`
    - `minRunSpeed`, `maxRunSpeed`
    - `generation`, `populationSize`, `pendingPopulationSize`
    - `currentAttemptIndex`, `currentTrialIndex`, `trialCount`
    - `bestEverScore`, `currentScore`, `paused`
    - `runSpeed`
    - `fastForwardRemaining`, `fastForwardActive`
    - `injectionQueueCount`
    - `morphologyMode`, `morphologyPreset`
    - `connectedSatellites`
    - startup-reject telemetry from the latest completed generation:
      - `latestInvalidStartupAttempts`, `latestInvalidStartupAttemptRate`
      - `latestInvalidStartupTrials`, `latestInvalidStartupTrialRate`
    - `currentGenome`, `bestGenome`

- `GET /api/evolution/history`
  - Returns backend-owned fitness history points.
  - Response shape: `{ history: GenerationFitnessSummary[] }`.
  - Each point includes:
    - `generation`
    - `bestFitness`
    - `attemptFitnesses` (all candidate fitness values for that generation)
    - startup-reject telemetry:
      - `invalidStartupAttempts`, `invalidStartupAttemptRate`
      - `invalidStartupTrials`, `invalidStartupTrialRate`
      - `totalTrials`

- `GET /api/evolution/performance`
  - Returns bounded structured evolution analytics focused on tuning signals.
  - Query params:
    - `windowGenerations` (optional, default `120`, max `400`)
    - `stride` (optional, default `1`, max `8`)
    - `includeParamStats` (optional, default `true`)
    - `includeDescriptors` (optional, default `true`)
    - `includeTopology` (optional, default `true`)
  - Includes:
    - `run`, `window`, `trends`
    - `run` includes `generation`, `populationSize`, `trialCount`, `runSpeed`, `paused`, `morphologyMode`, `morphologyPreset`
    - `generations` with per-generation fitness/selection/diversity/breeding stats
    - `generations[*].topology` with winner + best-N topology profiles and:
      - `distinctFingerprintCount`
      - `distinctCoarseFingerprintCount`
      - per-profile `coarseTopologyKey`
    - `learnedParams` with bounded parameter-distribution summaries

- `GET /api/evolution/performance/summary`
  - Returns compact summary fields suitable for AI control loops.
  - Includes:
    - `generation`, `bestEverFitness`, `recentBestFitness`
    - `morphologyMode`, `morphologyPreset`
    - `stagnationGenerations`, `diversityState`
    - `mutationPressure`
    - `convergence`
    - `signals`
    - `latestTopology`
    - `bestEverTopology`
    - `bestNTopologies` (deduped by topology fingerprint)

- `GET /api/evolution/performance/diagnose`
  - Returns server-side diagnosis of run health and likely next actions.
  - Includes:
    - `generation`, `timestampUnixMs`
    - `states` (`plateauState`, `volatilityState`, `noveltyState`, `trendState`)
    - `metrics` (recent-window best/novelty stats, slopes, mutation pressure)
    - `topology` (distinct + coarse fingerprint ratios, enabled-limb-count mix, representative topologies)
    - `findings` (coded diagnostics)
    - `recommendedActions`

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
      - `set_morphology_mode` (requires `morphologyMode`, optional `morphologyPreset`)
  - Returns updated evolution status.
  - Notes:
    - `set_population_size` is clamped to backend min/max (`minPopulationSize`/`maxPopulationSize`).
    - `set_run_speed` is clamped to `[0.5, 8.0]`.
    - `queue_fast_forward` adds to queue and caps each request at `50_000`.

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
  - Response shape: `{ checkpoints: CheckpointSummary[] }`.
  - Each checkpoint includes: `id`, `createdAtUnixMs`, `generation`, `bestEverScore`.

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
    - optional: `durationSeconds`, `dt`, `motorPowerScale`
  - Notes:
    - `dt` is currently ignored; backend runs fixed-step simulation at `1/120s`.
    - `motorPowerScale` is clamped to `[0.35, 1.5]`.
  - Returns per-genome aggregate fitness and descriptor summaries.
  - Response shape: `{ results: GenerationEvalResult[] }`.
  - Each result includes:
    - `fitness`, `descriptor`, `trialCount`
    - `medianProgress`, `medianUpright`, `medianStraightness`
    - startup-reject telemetry:
      - `invalidStartupTrials`, `invalidStartupTrialRate`
      - `allTrialsInvalidStartup`

## WebSocket

- `GET /api/evolution/ws`
  - Streams:
    - `status`
    - `generation_summary`
    - `trial_started`
    - `snapshot`
    - `trial_complete`
    - `error`
  - `generation_summary.summary` matches `/api/evolution/history` point shape.
  - `trial_complete.result.metrics` includes:
    - `quality`, `progress`, `uprightAvg`, `avgHeight`
    - `instabilityNorm`, `energyNorm`, `fallenRatio`
    - `straightness`, `netDistance`
    - `invalidStartup`

- `GET /api/trial/ws`
  - First client message must be a JSON trial request:
    - `genome`
    - `seed`
    - optional: `durationSeconds`, `dt`, `snapshotHz`, `motorPowerScale`
  - Notes:
    - `dt` is currently ignored; backend runs fixed-step simulation at `1/120s`.
    - `motorPowerScale` is clamped to `[0.35, 1.5]`.
  - Streams:
    - `trial_started`
    - `snapshot`
    - `trial_complete`
    - `error`

- `GET /api/eval/ws`
  - First client message must be a JSON generation-eval request:
    - `genomes` (array)
    - `seeds` (array)
    - optional: `durationSeconds`, `dt`, `motorPowerScale`
  - Notes:
    - `dt` is currently ignored; backend runs fixed-step simulation at `1/120s`.
    - `motorPowerScale` is clamped to `[0.35, 1.5]`.
  - Streams:
    - `generation_started`
    - `attempt_trial_started`
    - `attempt_complete`
    - `generation_complete`
    - `error`
  - `attempt_complete.result` and `generation_complete.results[*]` match `POST /api/eval/generation` result shape.

- `GET /api/satellite/ws`
  - Internal distributed worker socket used by satellite clients.
  - Message envelope uses tagged JSON with `type` (`snake_case`).
  - Server -> satellite:
    - `welcome { id }`
    - `run_trial { trialId, genome, seed, durationSeconds, dt, motorPowerScale, fixedStartup }`
    - `ping`
  - Satellite -> server:
    - `ready { slots }`
    - `trial_result { trialId, fitness, metrics, descriptor }`
    - `trial_error { trialId, message }`
    - `pong`
