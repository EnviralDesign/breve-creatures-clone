---
name: evolution-performance-api
description: Consume structured evolution-run analytics from the breve-creatures backend without reading source code. Use when an AI needs current standing and historical performance signals to tune evolution behavior, detect stagnation/diversity collapse, inspect learned parameter ranges, or decide evolution control actions.
---

# Evolution Performance API

## Use This Skill

Use these HTTP routes:
- `GET /api/evolution/performance/summary`
- `GET /api/evolution/performance`
- `GET /api/evolution/performance/diagnose`

Call summary first, then request detailed history only when needed.

## Route Contracts

### `GET /api/evolution/performance/summary`

Compact control-loop payload:
- `generation`
- `bestEverFitness`
- `recentBestFitness`
- `stagnationGenerations`
- `diversityState` (`low` | `medium` | `high` | `unknown`)
- `mutationPressure`:
  - `currentRate`
  - `atLowerClamp`
  - `atUpperClamp`
- `convergence`: array of `{ name, state, std }`
- `signals`: string tags such as `fitness_plateau`, `novelty_declining`, `mutation_pressure_high`
- `latestTopology`: most recent winner topology profile
- `bestEverTopology`: best-fitness topology profile seen so far
- `bestNTopologies`: top topology profiles, deduped by `topologyFingerprint`

Use this route for frequent polling.

### `GET /api/evolution/performance/diagnose`

Server-side interpretation payload:
- `states`: `plateauState`, `volatilityState`, `noveltyState`, `trendState`
- `metrics`: recent-window best and novelty dispersion, slopes, mutation clamp state
- `topology`: `distinctFingerprintRatio`, `distinctCoarseFingerprintRatio`, `topEnabledLimbCounts`, `representativeTopologies`
- `findings`: `{ code, severity, message }`
- `recommendedActions`: short action list

Use this route when you want diagnosis directly from the backend instead of deriving it client-side.

### `GET /api/evolution/performance`

Bounded structured payload for deeper analysis.

Query params:
- `windowGenerations` (default `120`, max `400`)
- `stride` (default `1`, max `8`)
- `includeParamStats` (default `true`)
- `includeDescriptors` (default `true`)
- `includeTopology` (default `true`)

Response:
- `run`: `{ generation, populationSize, trialCount, runSpeed, paused }`
- `window`: `{ fromGeneration, toGeneration, count, stride }`
- `trends`:
  - `bestFitnessSlope`
  - `medianFitnessSlope`
  - `stagnationGenerations`
- `generations[]` per generation:
  - `fitness`: `{ best, p50, p90, std }`
  - `selection`: `{ mean, p90 }`
  - `diversity`: `{ noveltyMean, noveltyP90, localCompetitionMean }`
  - `descriptor` (optional): `{ centroid[5], spread[5] }`
  - `topology` (optional):
    - `winner`
    - `bestN` (bounded; at most 3 profiles)
    - `distinctFingerprintCount`
    - `distinctCoarseFingerprintCount`
    - `winner.coarseTopologyKey` and `bestN[*].coarseTopologyKey`
  - `breeding`: `{ mutationRate, randomInjectChance, injectedGenomes, eliteKept }`
- `learnedParams[]` (if enabled):
  - `{ name, bounds: [min,max], population: { min,p50,p90,max,std }, champion }`

## Learned Parameter Names

Current `learnedParams.name` values:
- `torso.w`
- `torso.h`
- `torso.d`
- `torso.mass`
- `mass_scale`
- `limb.enabled_ratio`
- `limb.segment_count_mean`
- `segment.length_mean`
- `segment.mass_mean`
- `control.amp_mean`
- `control.freq_mean`

## Practical AI Workflow

1. Read `/summary`.
2. Read `/diagnose` for server-side interpretation.
3. If needed, request `/performance` with:
   - `windowGenerations=120`
   - `stride=2` (or `4` for low bandwidth)
4. Check:
   - `trends.bestFitnessSlope` and `trends.medianFitnessSlope`
   - latest `generations[-1].diversity.noveltyMean`
   - latest `generations[-1].breeding.mutationRate`
   - `latestTopology` and `bestNTopologies` for structural convergence/diversity
   - narrowing/widening in `learnedParams[*].population.std`
5. Decide control action via `/api/evolution/control` only after these checks.

## Data Volume Guardrails

- Prefer `/summary` for high-frequency checks.
- Keep `windowGenerations` small unless debugging long regressions.
- Raise `stride` when generation count is large.
- Disable heavy fields when not needed:
  - `includeParamStats=false`
  - `includeDescriptors=false`
  - `includeTopology=false`
