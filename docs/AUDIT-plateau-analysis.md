# Breve Creatures Clone: Comprehensive Audit & Plateau Analysis

**Date:** February 24, 2026
**Scope:** Full codebase audit, comparison with Karl Sims (1994), breve/spiderland, and the broader evolved virtual creatures literature.
**Problem Statement:** Creatures plateau in fitness. The spider preset evolves some non-morphological traits but overall progress stalls. This document identifies root causes and proposes solutions grounded in the original research.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What the Originals Actually Did](#2-what-the-originals-actually-did)
3. [Gap Analysis: Our System vs. the Originals](#3-gap-analysis-our-system-vs-the-originals)
4. [Deep Dive: The Controller Problem](#4-deep-dive-the-controller-problem)
5. [Deep Dive: The Fitness Function Problem](#5-deep-dive-the-fitness-function-problem)
6. [Deep Dive: The EA & Population Problem](#6-deep-dive-the-ea--population-problem)
7. [Deep Dive: The Genome Representation Problem](#7-deep-dive-the-genome-representation-problem)
8. [Recommendations (Prioritized)](#8-recommendations-prioritized)
9. [Reference Materials](#9-reference-materials)

---

## 1. Executive Summary

The creatures plateau because **the controller architecture has a hard expressiveness ceiling**. The control system is a pure open-loop sinusoidal oscillator: `f(time) -> motor_target`. Once evolution finds the best oscillator parameters for a given body plan, there is literally nowhere further to go. This is the single most important structural difference from every successful implementation of evolved virtual creatures.

Secondary causes compound the problem:
- The fitness function has ~15 interacting penalty/bonus terms that create a rugged landscape with many local optima and narrow corridors
- Population size (40) is too small for effective novelty search
- Mutation rate annealing drives premature convergence
- Blended interpolation crossover destroys diversity (no grafting/structural crossover)
- The genome is a flat fixed-size list, not a directed graph that can grow in complexity

All of these are solvable. The recommendations section is ordered by impact.

---

## 2. What the Originals Actually Did

### 2.1 Karl Sims, 1994 (The Foundation)

Karl Sims published two papers in 1994 that defined this entire field:
- "Evolving Virtual Creatures" (SIGGRAPH '94) - see `.tmp/references/karl-sims-1994-siggraph-evolving-virtual-creatures.txt`
- "Evolving 3D Morphology and Behavior by Competition" (Artificial Life IV) - see `.tmp/references/karl-sims-1994-alife-coevolution-competition.txt`

**Key architecture decisions in Sims' system:**

#### Genome: Directed Graph

The genotype is a **directed graph** of nodes and connections. Each node describes:
- A rigid body part (dimensions, joint type, joint limits)
- A set of **local neurons** (sensors, neural nodes, effectors)

Each connection describes:
- Placement of child part relative to parent (position, orientation, scale, reflection)
- A **recursive limit** (how many times to instantiate in recursive cycles)
- A **terminal-only flag** (for end-of-chain parts like hands or tails)

The graph can be **recurrent** -- nodes can connect to themselves or in cycles, creating recursive/fractal body structures. A single genotype node can produce multiple phenotype parts. This means:
- Complexity is **unbounded** and can grow as evolution proceeds
- Component reuse is automatic (define a leg segment once, reuse everywhere)
- New parameters and dimensions are added to the search space dynamically

#### Control: Sensor-Neuron-Effector Networks

**Three types of sensors:**
1. **Joint angle sensors** -- current angle for each DOF of each joint (proprioception)
2. **Contact sensors** -- activate (1.0) on contact, negatively activate (-1.0) if no contact. One per face of each body part.
3. **Photosensors** -- 3D light direction relative to part orientation (for following behaviors)

**23+ neural node function types:**
`sum, product, divide, sum-threshold, greater-than, sign-of, min, max, abs, if, interpolate, sin, cos, atan, log, expt, sigmoid, integrate, differentiate, smooth, memory, oscillate-wave, oscillate-saw`

Note: oscillators are **one option among many**, not the only control mechanism. The neural graph could combine oscillators with sensor feedback, memory, differentiation, etc.

**The neural graph is recurrent** -- feedback loops and cycles are allowed. This gives creatures internal state and history-dependent behavior.

**Local neural circuits are replicated with body parts.** When a limb segment is instanced from the genome, its neurons come with it. Adjacent parts can communicate (parent<->child signals). Plus, a set of global "unassociated" neurons exists for centralized coordination.

**Effectors** control joint DOFs. Max force is proportional to cross-sectional area of the joined parts (strength scales with area, mass with volume -- same as nature).

#### Evolution: Three Reproduction Methods

- **40% asexual** (mutation only)
- **30% crossover** (node alignment + crossover points between two parents)
- **30% grafting** (splice a subtree from one parent onto another)

Population: **300 individuals**, survival ratio **1/5** (60 survive per generation).

Mutation adds new random nodes, mutates parameters, adds/removes connections, then garbage-collects disconnected elements.

#### Fitness: Simple

Walking fitness: **speed** (distance traveled per unit time), with vertical velocity component ignored. A short initial no-friction, no-effector simulation lets the creature settle to prevent "falling over = high velocity" exploits. Continued movement is rewarded over initial push by weighting final-phase velocity higher.

That's essentially it. No explicit uprightness bonus, no straightness gate, no energy penalty, no instability penalty, no sustain factor.

### 2.2 Breve / Spiderland (Jon Klein)

Breve is the open-source platform from which "breve creatures" gets its name. See `.tmp/references/github-repos-reference.txt` for repository links.

- Written in C++, scripts in Python or "steve" language
- Physics: ODE (Open Dynamics Engine)
- Used by multiple research groups for evolved creature experiments
- Jon Klein's platform; last actively maintained ~2009
- Available: https://github.com/jonklein/breve

The "breve creatures" demos that became famous (YouTube videos of evolved walkers) used neural network controllers, not oscillators. The platform supports arbitrary agent behaviors defined in script.

### 2.3 BREVE Monsters (UT Austin, Jacob Schrum)

An extension built on breve that used **NSGA-II multiobjective neuroevolution**. Key innovations:
- **Multimodal behavior** -- creatures could exhibit multiple distinct behaviors, not just one gait
- Module Mutation for evolving neural network topology
- Behavioral Diversity objectives to maintain population variety

See `.tmp/references/github-repos-reference.txt` for download link.

### 2.4 Keith Wiley's Creature Evolver

A personal implementation of Sims' system using breve. See `.tmp/references/keith-wiley-creature-evolver-notes.txt`.

Notable: Even with neural controllers and directed graph genomes, Wiley reported "most runs do not produce very interesting forms of movement." This confirms that the directed graph + neural network approach is **necessary but not sufficient** -- the problem is genuinely hard. But without those components, it's essentially impossible to get past basic oscillatory gaits.

### 2.5 The 20-Year Plateau (ESP Paper, 2015)

A critical paper by Lessin et al. (2015) documents that from 1994 to 2015, **there was no clear increase in behavioral complexity** for evolved virtual creatures beyond what Sims demonstrated. Even with neural networks! See `.tmp/references/esp-method-lessin-2015-abstract.txt`.

Their solution (ESP) involved:
- **Encapsulation**: Preserving learned behaviors so they aren't lost during further evolution
- **Syllabus**: A sequence of intermediate learning tasks of increasing difficulty
- **Pandemonium**: Multiple behavior modules competing for control

This doubled the state of the art. The implication for us: even after adding neural networks, we may want mechanisms for incremental task complexity and behavior preservation.

---

## 3. Gap Analysis: Our System vs. the Originals

| Aspect | Karl Sims (1994) | Our System | Impact |
|--------|-----------------|------------|--------|
| **Controller** | Sensor-neuron-effector network, 23+ function types, recurrent, reactive | Pure sinusoidal oscillators, open-loop, `f(time)` only | **CRITICAL** -- hard ceiling on behavioral expressiveness |
| **Sensors** | Joint angles, contact, photosensors | None | **CRITICAL** -- creatures are blind, no proprioception |
| **Genome** | Directed graph, recursive, growable complexity | Flat list of 6 limb slots x 5 segments | **HIGH** -- bounded complexity, no component reuse |
| **Fitness** | Distance traveled (simple) | ~15 components with gates, penalties, bonuses | **HIGH** -- rugged landscape, narrow corridors |
| **Population** | 300 | 40 (default) | **MEDIUM** -- insufficient for novelty search |
| **Crossover** | Crossover + Grafting + Asexual (30/30/40) | Blended interpolation only | **MEDIUM** -- drives toward mean, kills diversity |
| **Reproduction** | Offspring proportional to fitness | Tournament (size 4) + elite 2 | LOW -- tournament is fine, but small pop hurts |
| **Mutation** | Fixed rate, scales inversely with graph size | Adaptive + annealing (0.72 -> 0.42 over 140 gen) | **MEDIUM** -- premature convergence |
| **Simulation** | Custom physics on CM-5 | Rapier3D (modern, good) | Fine -- Rapier is excellent |
| **Joint types** | Rigid, revolute, twist, universal, bend-twist, twist-bend, spherical | Hinge, Ball | Fine -- sufficient variety |

---

## 4. Deep Dive: The Controller Problem

This is the core issue. Here is our entire control system:

**File:** `src/main.rs`, lines 154-198
```rust
struct ControlGene {
    amp: f32,       // amplitude
    freq: f32,      // frequency
    phase: f32,     // phase offset
    bias: f32,      // DC offset
    harm2_amp: f32, // second harmonic amplitude
    harm2_phase: f32,
    // ... Y and Z axis variants for ball joints
}

impl ControlGene {
    fn signal_x(&self, sim_time: f32) -> f32 {
        let theta = self.freq * sim_time + self.phase;
        self.bias
            + self.amp * theta.sin()
            + self.harm2_amp * (2.0 * theta + self.harm2_phase).sin()
    }
}
```

The signal is purely `f(time)`. It has **zero knowledge of what the body is actually doing**. The creature is blind and proprioceptively deaf. It plays a pre-recorded motor tape.

**Where this is used:** `src/main.rs`, lines 1040-1101. Each simulation step:
1. Compute `signal_x(sim_time)` for each controller
2. Map signal to target joint angle: `target = signal / MAX_MOTOR_SPEED * limit`
3. Set position-based motor with stiffness and max force
4. Repeat for Y and Z axes on ball joints

There is no sensor reading, no conditional logic, no feedback of any kind.

### Why This Causes Plateaus

With pure oscillators, the **entire behavioral repertoire** is: sinusoidal movements at fixed frequencies, phases, and amplitudes, plus a single second harmonic. Per joint, there are ~14 evolvable parameters (amp, freq, phase, bias, harm2_amp, harm2_phase for X, plus amp, freq, phase, bias for Y and Z).

For a spider4x2 (4 legs x 2 segments), that's roughly **8 joints x 14 params = ~112 real-valued parameters**. This is a standard numerical optimization problem, and tournament selection + mutation will find a good local optimum within a few dozen generations. After that, there is nowhere to go because:

1. **No reactive behavior** -- the creature can't lift a leg higher when it detects ground contact, shift weight when tipping, coordinate push-off based on actual joint angles, or adjust stride based on whether it's actually moving.

2. **No sensory coupling** -- in real animals (and Sims' creatures), limbs coordinate partly through mechanical coupling (ground reaction forces) AND partly through neural feedback. Our creatures only get the mechanical coupling, which limits coordination patterns.

3. **No internal state** -- the oscillator has no memory, no integration, no conditional logic. Every cycle is identical regardless of what happened before.

4. **Frequency locking** -- the independent frequencies for each joint can only produce useful gaits when they happen to be in simple rational relationships (1:1, 2:1, etc.). The search space is mostly filled with incommensurate frequency combinations that produce chaotic-looking movement.

### What Sims' Neural Nodes Could Do That Oscillators Can't

Consider a simple Sims creature brain for one leg:
```
joint_angle_sensor -> [greater-than threshold] -> [if: oscillator_fast, oscillator_slow] -> effector
```
This reads as: "If joint angle exceeds threshold, use fast oscillation; otherwise use slow oscillation." This simple 4-node circuit creates a **ground-contact-adaptive gait** -- the leg swings fast in the air and pushes slowly on the ground. No oscillator-only system can express this.

More complex examples from Sims' results:
- Memory nodes that track recent contact patterns
- Differentiator nodes that detect velocity changes
- Sigmoid nodes that create smooth transitions between behaviors
- Sum nodes that combine multiple sensor inputs for coordinated responses

---

## 5. Deep Dive: The Fitness Function Problem

**File:** `src/main.rs`, lines 529-612

Our fitness has approximately 15 interacting components:

```
quality = progress                              (gated by straightness^2, fallen_gate, sustain_factor)
        + upright_avg * 0.95                    (upright bonus)
        + straightness * 1.5                    (straightness bonus)
        + clamp(avg_height/3, 0, 1) * 0.6      (height bonus)
        - energy_norm * 0.8                     (energy penalty)
        - instability_norm * 1.25               (instability penalty)
        - energy_per_progress * 0.55            (thrashing energy ratio penalty)
        - instability_per_progress * 0.65       (thrashing instability ratio penalty)
        * (1 - fallen_ratio^1.5 * 0.6)         (fallen penalty multiplier)
        * upright_scaling_if_below_0.5          (upright penalty multiplier)
```

Additionally, `progress` itself is:
```
raw_progress = net_distance * 0.95 + peak_distance * 0.05
progress = raw_progress * straightness_gate * fallen_gate * sustain_factor
```

Where:
- `straightness_gate = straightness^2.0` (squared -- very aggressive)
- `fallen_gate = (1 - fallen_ratio)^1.3`
- `sustain_factor` requires measurable progress at 50% and 85% time marks

### Problems With This

**1. The straightness gate (squared) kills exploration.** A creature that discovers a curved locomotion strategy -- which might be a stepping stone to a better straight strategy -- gets heavily penalized. The gate is applied multiplicatively to ALL progress, so a creature that travels far but curves gets less credit than one that barely moves but goes straight.

**2. The sustain factor is prescriptive.** It requires progress at specific time checkpoints. A creature that takes time to "warm up" or uses a start-stop strategy gets penalized. Many real gaits have asymmetric acceleration profiles.

**3. Three overlapping penalties for "wild movement."** Energy penalty, instability penalty, AND thrashing ratios (energy/progress, instability/progress) all punish similar behaviors. This creates a steep penalty gradient around any strategy that involves vigorous movement, which is exactly what exploration requires.

**4. Multiple terms pushing toward the same goal.** The fallen penalty, upright bonus, AND height bonus all push toward "stay upright." This creates a very peaked landscape -- creatures strongly converge on upright postures very early and then can't explore alternatives.

**5. Conflicting gradients.** Moving fast (high progress) often requires high energy and some instability. But the penalties directly oppose this. The creature is trying to maximize progress while minimizing the things that cause progress.

**Karl Sims' fitness for walking was essentially:**
```
fitness = distance_traveled (ignoring vertical velocity)
        + late_phase_velocity_weight (reward continued movement)
```

With a simple anti-exploit: run a short initial simulation with no friction and no effectors to establish a stable resting height, preventing "fall over = high velocity" cheats.

That's it. The simplicity allowed radically different strategies to emerge -- shuffling, hopping, inchworming, crawling, slithering -- because the fitness function didn't prescribe HOW to move, only that movement should occur.

---

## 6. Deep Dive: The EA & Population Problem

### 6.1 Population Size

**Our system:** 40 (default), max 128
**Sims:** 300

With 40 individuals and tournament size 4, the effective selection pressure is very high. The novelty search system (k=8 neighbors, archive max 320) was designed for a larger flow of candidates. With only 40 per generation, the archive fills slowly with similar entries and novelty scores become unreliable.

### 6.2 Crossover

**Our system:** Blended interpolation only (lines 6347-6432)
```rust
// Torso: blend factor 0.35-0.65
// Limbs: per-limb blend 0.20-0.80
// Segments: random blend per segment
// Controls: random blend per control gene
```

This **averages** parent parameters. Over many generations, this drives the population toward the mean and destroys diversity. It cannot create genuinely novel combinations.

**Sims used three methods:**
- **40% Asexual** -- mutation only (preserves good solutions while exploring nearby)
- **30% Crossover** -- node alignment with crossover points (creates new combinations at graph structure level)
- **30% Grafting** -- splices a subtree from one parent onto another (creates radically new morphologies by combining body parts from different creatures)

Grafting is particularly important because it creates **structural novelty** -- not just parameter interpolation, but entirely new body configurations.

### 6.3 Mutation Rate Annealing

**File:** `src/main.rs`, lines 3202-3228

```
anneal_factor = annealing_progress(generation) * elite_consistency * (1 - 0.35 * stagnation_pressure)
min_mutation_rate = lerp(0.18, 0.06, anneal_factor)   // shrinks over time
max_mutation_rate = lerp(0.72, 0.42, anneal_factor)    // shrinks over time
```

With `ANNEALING_TIME_CONSTANT_GENERATIONS = 140`, by generation ~100 the system has significantly narrowed its mutation range. The stagnation pressure feedback (+0.08) is far too weak to counteract the annealing when already deep in a local optimum.

The effect: **the system locks in early winners**. Combined with 40 individuals and blended crossover, this guarantees premature convergence.

### 6.4 Random Injection Rate

```
random_inject_chance = (0.04 + (1 - mean_novelty) * 0.04) * lerp(1.0, 0.35, anneal_factor)
                     + stagnation_pressure * 0.015
                     + holdout_gap * 0.01
```

At best this is ~8% early on, annealing down to ~3%. With 40 individuals, that's 1-3 random creatures per generation. This is too few to meaningfully escape a local optimum, especially since random creatures rarely survive tournament selection against an optimized population.

### 6.5 Fixed Trial Seeds

```rust
fn build_trial_seed_set(generation_index: usize, count: usize) -> Vec<u64> {
    let _ = generation_index;  // generation_index is IGNORED
    build_seed_bank(TRAIN_TRIAL_SEED_BANK_TAG, count)
}
```

The same 5 seeds are used every generation. Creatures are always tested on the exact same initial conditions. This enables overfitting to those specific scenarios. The holdout system (5 different seeds) partially mitigates this, but the training fitness that drives selection is still based on the fixed seeds.

---

## 7. Deep Dive: The Genome Representation Problem

### 7.1 Our Flat Genome

```rust
struct Genome {
    torso: TorsoGene,          // 4 params (w, h, d, mass)
    limbs: Vec<LimbGene>,      // fixed 6 slots
    hue: f32,
    mass_scale: f32,
}

struct LimbGene {
    enabled: bool,
    segment_count: u32,        // 1-5
    anchor_x/y/z: f32,
    axis_y/z: f32,
    dir_x/y/z: f32,
    segments: Vec<SegmentGene>,
    controls: Vec<ControlGene>,
}
```

Maximum complexity: 6 limbs x 5 segments = 30 segments. This is the ceiling, set at compile time. The genome cannot grow.

### 7.2 Sims' Directed Graph Genome

Sims' genotype was a directed graph where:
- Nodes could reference themselves (recursion)
- Nodes could reference cycles (fractal structures)
- A single node instantiated multiple phenotype parts
- New nodes could be added during mutation
- Disconnected nodes were garbage collected
- Neural circuitry was embedded inside morphological nodes

This means:
- A simple 3-node genotype could produce a 20-part creature (through recursion)
- New body parts and neural circuits could emerge during evolution
- The search space dimensionality was **not fixed** -- it could grow
- Symmetry was natural (same node for left and right legs)
- Local neural circuits were automatically replicated with body parts

### 7.3 Impact on Our Spider Preset

When using `spider4x2` with `fixed_preset` morphology mode:
- Topology is locked (4 legs, 2 segments each)
- Joint types may be locked
- What evolves: oscillator parameters only

This reduces the entire evolution to a ~112-dimensional real-valued optimization problem. Tournament selection + Gaussian mutation will find a good local optimum quickly (perhaps 30-60 generations), and then the system plateaus because the search space is exhausted.

Even with morphology mode `random`, the flat genome limits what can emerge. You can't evolve a creature with 10 similar legs generated from a single genetic template, or recursive branching structures, or modular appendages with shared neural circuits.

---

## 8. Recommendations (Prioritized)

### Priority 1: CRITICAL -- Addresses the Core Plateau

#### 8.1 Add Sensory Feedback to the Controller

This is the highest-impact change. Even a minimal reactive layer on top of the existing oscillators would help enormously.

**Simplest viable approach -- Sensor-Modulated Oscillators:**

```
// Pseudo-code for new control signal
joint_angle = read_joint_angle_sensor(joint_id)
ground_contact = read_contact_sensor(part_id)   // 1.0 or -1.0

base_signal = oscillator(time)   // existing ControlGene.signal_x()
modulated = base_signal
          * (1.0 + w_angle * joint_angle)       // proprioceptive modulation
          + w_contact * ground_contact           // contact-reactive offset
          + w_bias                               // evolved bias

// w_angle, w_contact, w_bias are new evolvable parameters per joint
```

This requires:
- Reading joint angles from Rapier (already available via `impulse_joints`)
- Detecting ground contact (can use Rapier contact manifolds or collision events)
- Adding ~3 new evolvable parameters per joint to `ControlGene`
- Minimal changes to the simulation loop

**More expressive approach -- Small Neural Network:**

Add a per-joint micro-network with ~2-4 hidden nodes:
```
inputs: [sin(freq*t), cos(freq*t), joint_angle, contact, parent_joint_angle]
hidden: 2-4 nodes with tanh activation
output: motor_target
```

Weights are evolvable. This is still very lightweight but vastly more expressive than pure oscillators. It can represent:
- Phase-dependent contact responses
- Joint-angle-dependent stiffness
- Coordinated multi-joint patterns
- Any oscillatory pattern the current system can produce (as a subset)

**Even more expressive -- CPPN controller:**

Use Compositional Pattern-Producing Networks (CPPNs), which combine diverse activation functions (sin, Gaussian, sigmoid, linear, abs) evolved via NEAT-style topology evolution. These are specifically designed for evolving patterns including periodic locomotion patterns, and they naturally produce symmetry and repetition.

See: Stanley, K.O. "Compositional pattern producing networks: A novel abstraction of development" (2007). The Wikipedia article at https://en.wikipedia.org/wiki/Compositional_pattern-producing_network is a good overview.

#### 8.2 Simplify the Fitness Function

Start with something close to Sims' original:

```
fitness = net_distance * (1.0 - 0.3 * fallen_ratio)
```

That's it. No straightness gate, no energy penalty, no instability penalty, no sustain factor, no thrashing ratio. Let evolution discover what "good locomotion" looks like.

If creatures exploit falling-over-fast, add Sims' settle mechanic: run a short sim with no friction and no effectors to establish resting height, then penalize creatures that don't exceed this height during active simulation.

Gradually add complexity back only when specific exploits emerge:
- If creatures spin in circles: add a mild straightness weight (NOT a gate -- additive, not multiplicative)
- If creatures waste energy: add a mild efficiency bonus (NOT a penalty)
- If creatures flip and slide: strengthen the fallen penalty

The principle: **reward what you want, don't penalize everything you don't want**.

### Priority 2: HIGH -- Improves Exploration Capacity

#### 8.3 Increase Population Size

Minimum 100, ideally 200. Sims used 300 on 1994 hardware. We have far more compute available.

If simulation time is a concern, consider:
- Reducing `TRIALS_PER_CANDIDATE` from 5 to 3 (less overfitting protection but faster)
- Reducing `DEFAULT_GENERATION_SECONDS` from 18 to 12 for early generations
- Using the satellite pool for parallel evaluation

#### 8.4 Remove or Drastically Slow Annealing

Set `ANNEALING_TIME_CONSTANT_GENERATIONS` to 500+ or remove annealing entirely. Let the adaptive mechanisms (stagnation pressure, novelty response) handle mutation rate dynamics.

Alternatively, implement a "restart" mechanism: when stagnation exceeds N generations, reset mutation rate to maximum and inject a burst of random individuals (20-30% of population).

#### 8.5 Add Structural Crossover (Grafting)

Implement Sims' grafting operation: pick a limb from parent A, replace a limb slot in parent B. This creates genuinely novel combinations at the structural level.

For the current flat genome, this could be:
```
// Grafting for flat genome
child = clone(parent_a)
limb_index = random(0..6)
child.limbs[limb_index] = clone(parent_b.limbs[random(0..6)])
// Optionally also copy the anchor point from parent_b
```

Also implement Sims' reproduction ratios: 40% asexual (mutation only), 30% crossover (existing blended), 30% grafting (new structural).

### Priority 3: MEDIUM -- Longer-Term Improvements

#### 8.6 Randomize Trial Seeds Across Generations

Change the seed bank to vary per generation:
```rust
fn build_trial_seed_set(generation_index: usize, count: usize) -> Vec<u64> {
    build_seed_bank(
        TRAIN_TRIAL_SEED_BANK_TAG ^ (generation_index as u32),
        count,
    )
}
```

This prevents overfitting to specific initial conditions while still ensuring all candidates within a generation face the same conditions (fairness).

#### 8.7 Consider a Directed Graph Genome (Long-Term)

This is a larger architectural change but would unlock:
- Recursive body structures
- Automatic symmetry through shared nodes
- Growing complexity over evolutionary time
- Local neural circuits replicated with body parts

The 3DVCE project (see `.tmp/references/github-repos-reference.txt`) has an open-source implementation in C++ that could serve as architectural reference.

#### 8.8 Morphological Innovation Protection

When a creature's morphology changes significantly (e.g., limb added/removed, segment count changed), temporarily reduce selection pressure. One approach: give morphologically-changed offspring a "grace period" of 1-2 generations where their fitness floor is set to the population median, preventing immediate elimination before the control system can adapt.

This is based on research by Cheney et al. (2018) "Scalable co-optimization of morphology and control in embodied machines" which found that the tight coupling between morphology and control causes most morphological innovations to be immediately lethal because the existing control policy can't handle the new body.

---

## 9. Reference Materials

All reference files are in `.tmp/references/` (gitignored).

### Papers (Full Text in .tmp/references/)

| File | Description |
|------|-------------|
| `karl-sims-1994-siggraph-evolving-virtual-creatures.txt` | **THE foundational paper.** Full text of Sims' SIGGRAPH '94 paper. Read sections 2 (Morphology), 3 (Control), 5 (Behavior Selection), 6 (Evolution). The control section (3.1-3.4) is the most critical for understanding what we're missing. |
| `karl-sims-1994-alife-coevolution-competition.txt` | Sims' co-evolution paper. Relevant for understanding competitive fitness and how dynamic fitness landscapes prevent stagnation. |
| `esp-method-lessin-2015-abstract.txt` | ESP method for breaking through the 20-year behavioral complexity plateau. Relevant for understanding that even with neural networks, additional mechanisms (task syllabus, behavior encapsulation) may be needed for open-ended evolution. |
| `keith-wiley-creature-evolver-notes.txt` | Notes from Keith Wiley's personal Sims-style implementation using breve. Confirms that directed graph + neural net is standard, and that even with both, results are hard-won. |
| `github-repos-reference.txt` | Annotated list of all open-source evolved virtual creatures implementations with links and notes on relevance. |

### Key URLs

| Resource | URL |
|----------|-----|
| Karl Sims' main page | https://karlsims.com/evolved-virtual-creatures.html |
| Sims SIGGRAPH '94 PDF | https://karlsims.com/papers/siggraph94.pdf |
| Sims ALife '94 PDF | https://karlsims.com/papers/alife94.pdf |
| Sims "bloopers" video (exploit examples) | https://youtu.be/pxgLHuWfMS8 |
| Breve source (GitHub) | https://github.com/jonklein/breve |
| BREVE Monsters (UT Austin) | https://nn.cs.utexas.edu/?brevemonsters= |
| 3DVCE (Sims-style with neural nets) | https://github.com/222464/EvolvedVirtualCreaturesRepo |
| Minemonics (gait evolution) | https://github.com/benelot/minemonics |
| ESP Paper (arXiv) | https://arxiv.org/abs/1510.07957 |
| CPPN Wikipedia | https://en.wikipedia.org/wiki/Compositional_pattern-producing_network |
| POET (open-ended coevolution) | https://dl.acm.org/doi/10.1145/3321707.3321799 |

### Codebase Locations

Key areas of `src/main.rs` relevant to this audit:

| Lines | Content |
|-------|---------|
| 31-116 | All constants (population, fitness weights, mutation rates, physics) |
| 118-227 | Genome structs (TorsoGene, SegmentGene, ControlGene, LimbGene, Genome) |
| 154-198 | **ControlGene and signal functions** -- the core controller limitation |
| 529-612 | **Fitness computation** (compute_metrics) |
| 638-647 | Physics setup (gravity, timestep, solver iterations) |
| 734-935 | Body construction from genome |
| 1040-1101 | **Motor control loop** -- where oscillator signals become joint targets |
| 3140-3288 | **Evolution loop** -- selection, breeding, mutation, annealing |
| 3875-3888 | Stagnation detection |
| 5658-5676 | Tournament selection |
| 6025-6217 | Spider4x2 preset definition |
| 6347-6432 | Crossover (blended interpolation) |
| 6434-6559 | Mutation |
| 7418-7422 | Annealing progress function |

---

## Appendix A: Quick Comparison of Controller Expressiveness

| Approach | Parameters per Joint | Can React to Body State | Can Have Memory | Can Coordinate Across Joints | Complexity Growth |
|----------|---------------------|------------------------|----------------|-----------------------------|--------------------|
| Our sinusoidal oscillator | ~14 | No | No | No (independent) | Fixed |
| Sensor-modulated oscillator | ~17 | Yes (limited) | No | Limited | Fixed |
| Small fixed-topology NN | ~20-40 | Yes | With recurrence | Yes | Fixed |
| CPPN (NEAT-evolved topology) | Variable | Yes | With recurrence | Yes | **Grows** |
| Sims' neural graph | Variable | Yes | Yes (memory, integrate) | Yes (parent-child signals) | **Grows** |

## Appendix B: Fitness Function Comparison

| Component | Our System | Karl Sims | Recommendation |
|-----------|-----------|-----------|----------------|
| Distance | net_distance * 0.95 + peak * 0.05, gated | distance / time | Simplify to net_distance |
| Straightness | Gate (straightness^2) + Bonus (1.5x) | Not used | Remove gate, optional mild bonus |
| Upright | Bonus (0.95) + Penalty below 0.5 | Not used | Remove or reduce to mild bonus |
| Height | Bonus (0.6x) | Not used | Remove |
| Energy | Penalty (0.8x) + Ratio penalty (0.55x) | Not used | Remove entirely |
| Instability | Penalty (1.25x) + Ratio penalty (0.65x) | Not used | Remove entirely |
| Fallen | Gate (1-ratio)^1.3 + Multiplier (0.6x) | Simple height check | Simplify to (1 - fallen_ratio) |
| Sustain | Mid + Late checkpoint factors | Late-phase velocity weight | Simplify or remove |
| Anti-exploit | None | No-friction settle, then check | Add settle check |
