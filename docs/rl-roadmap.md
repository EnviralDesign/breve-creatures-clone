# RL / ML Investigation Roadmap

This document proposes a practical path from the current evolutionary controller to a hybrid morphology + policy learning system.

## Why This Path

The current system is strong at exploring body plans but can still exploit reward loopholes (collapse/flail behaviors).  
RL-style policy learning is better at time-dependent control, while evolution is good for topology search.  
The roadmap below combines both without rewriting physics.

## Principles

- Keep changes staged and measurable.
- Separate control-learning risk from morphology-search risk.
- Use robust evaluation (multi-seed, perturbations), not single-run scores.
- Keep distributed workers useful for both evolution and RL rollouts.

## Phase 0: Baselines and Instrumentation

Objective: establish reliable baselines before adding ML.

1. Lock one morphology and run controller optimization only.
2. Keep one fully-evolutionary baseline run in parallel for comparison.
3. Record:
   - best fitness, median fitness, p90
   - stagnation generations
   - run-to-run variance across seeds
   - sustained-motion diagnostics (late-phase progress)

Success criterion:
- stable, reproducible baseline curves for at least 100 generations-equivalent runtime.

## Phase 1: RL Environment Contract

Objective: expose the simulator as an RL-compatible environment.

Required API contract:
- `reset(genome, seed) -> obs`
- `step(action) -> obs, reward, done, info`

Observation candidates:
- torso orientation/velocity
- joint angles/velocities
- contact flags
- previous action
- normalized phase/time

Action candidates:
- joint target angles or target velocities
- optional torque scale

Reward:
- start with current anti-flail locomotion reward
- include sustained progress and robustness to perturbations

Success criterion:
- deterministic stepping and stable rollout generation for fixed policy + fixed seed.

## Phase 2: Fixed-Morphology RL Policy

Objective: prove learned control works before co-optimizing morphology.

1. Fix morphology.
2. Train policy with PPO (recommended first).
3. Compare against evolutionary oscillator controller on same morphology.
4. Stress test with domain randomization:
   - spawn orientation jitter
   - friction/mass perturbations
   - seed perturbations

Success criterion:
- policy beats oscillator baseline on robust median score (not just peak).

## Phase 3: Morphology + Policy Co-Optimization

Objective: combine strengths of both optimizers.

Two-timescale loop:
1. Outer loop proposes morphologies (evolutionary search or learned proposer).
2. Inner loop trains/evaluates policy for each proposed morphology.
3. Morphology score is robust policy performance.

Recommended constraints:
- warm-start policy from nearby morphology when possible
- budgeted inner training steps per candidate
- early stop poor candidates quickly

Success criterion:
- better robust fitness than Phase 2 fixed-morphology policy and pure evolution baseline.

## Phase 4: Scale and Productization

Objective: operational efficiency and continuous experimentation.

- Use satellites as rollout workers.
- Central learner updates weights and broadcasts checkpoints.
- Add automated experiment registry:
  - config hash
  - reward version
  - morphology ID
  - policy checkpoint

Success criterion:
- repeatable large-batch experiments with clear lineage and rollback.

## Immediate Experiments (Low Risk, High Value)

These can run in the current framework today.

1. Fixed morphology, optimize controller only
- Freeze one morphology:
  - best-ever genome
  - one random morphology
  - one hand-defined locomotion-friendly morphology
- Run evolution/optimization on control params only.
- Compare sustained progress and stability.

2. Random morphology bank test
- Sample N random morphologies.
- Evaluate each under same control-optimization budget.
- Rank by robust median score and variance.

3. Anti-collapse reward audit
- For top candidates, inspect:
  - early-phase distance vs late-phase gain
  - instability-per-progress
  - energy-per-progress
- Reject candidates that only win via early collapse displacement.

## Why Fixed Morphology Testing Is Valuable

Yes, it is very valuable.

- It isolates whether stalling comes from:
  - bad control search dynamics
  - or bad morphology search dynamics.
- It removes a large confounder and speeds up iteration.
- It gives a clean benchmark for future RL controller work.

If fixed morphology still stalls, reward/control is the bottleneck.  
If fixed morphology improves clearly, morphology search pressure/objective is likely the bottleneck.

## Suggested Next Step

Start with Phase 0 + the fixed morphology experiments above before full RL integration.  
This gives immediate signal with minimal architecture change and directly informs RL design choices.
