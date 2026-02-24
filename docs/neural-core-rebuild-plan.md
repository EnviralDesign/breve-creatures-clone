# Neural Core Rebuild Plan (No Legacy-First Iteration)

This plan assumes a hard pivot: rebuild the evolutionary backend around graph morphology + neural control, then wire every dependent surface to that core.

## Stage 1 - Core Representation Swap (Completed)

- Replace flat-limb-first control path with `Genome.version=2` graph representation.
- Add directed morphology graph primitives (nodes/edges, recursion/instancing controls).
- Add local and global recurrent neural gene structures with effectors.
- Keep only a projection layer to populate legacy fields for UI/API compatibility.

## Stage 2 - Simulation Kernel Rebuild (Completed)

- Build physical bodies/joints from graph expansion, not legacy limb arrays.
- Drive joints from neural effectors + sensor vectors (contact, proprioception, hierarchy).
- Run brain updates in-step with physics and apply motor targets per axis.
- Switch fitness to simple progress-quality objective from the audit guidance.

## Stage 3 - Evolutionary System Rebuild (Completed)

- Increase exploration capacity (`population=160`, `trials=3`, max population bump).
- Replace reproduction flow with Sims-style mix:
  - 40% mutation-only
  - 30% crossover+mutation
  - 30% graft+mutation
- Add graph-aware crossover, mutation, and grafting operators.
- Randomize trial seeds per generation to reduce overfitting.

## Stage 4 - Platform Wiring (Completed)

- Keep API and frontend consuming backend genomes/states with v2 fields.
- Keep morphology mode/preset constraints operating on graph topology.
- Update topology diagnostics/features to read graph traits.
- Keep viewer/backend path active (no local-fallback downgrade on backend disconnect).

## Stage 5 - Throughput + Benchmarking (In Progress, Working)

- Add fast-forward preemption of in-flight real-time trials.
- Add wall-time cap for fast eval trials to avoid pathological stalls.
- Add benchmark harness script with optional benchmark population reset/restart.
- Verified smoke runs:
  - `data/benchmarks/neural-v2-benchmark-smoke.csv`
  - `data/benchmarks/neural-v2-benchmark-24pop-3gen.csv`
  - Observed ~2.0 minutes/generation at `population=24` for the latest 3-gen run.

## Stage 6 - Core Pillar Completion for Open-Ended Evolution (Next)

This stage is the first major remaining pillar if we want sustained complexity growth beyond gait tuning:

- Morphological innovation protection/grace-window selection handling.
- Optional curriculum/task progression hooks (ESP-style pressure, behavior preservation hooks).
- Longer benchmark matrix (`population=64/96/160`) with holdout checks and topology-diversity tracking.

Notes:
- Directed graph morphology + neural control + structural reproduction are treated as non-negotiable core pillars.
- Stage 6 mechanisms are not mandatory for basic Sims-style capability, but are likely required to avoid re-plateauing once locomotion quality stabilizes.
