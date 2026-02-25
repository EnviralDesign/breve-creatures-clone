use std::collections::{HashMap, VecDeque};
use std::f32::consts::PI;
use std::fs;
use std::net::SocketAddr;
use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc as std_mpsc;
use std::time::Duration;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, Query, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::{SinkExt, StreamExt};
use include_dir::{Dir, include_dir};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rapier3d::na::{Isometry3, Translation3, UnitQuaternion, Vector3, point, vector};
use rapier3d::prelude::*;
use serde::{Deserialize, Serialize};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, broadcast, mpsc};
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info, warn};

const MAX_LIMBS: usize = 6;
const MAX_SEGMENTS_PER_LIMB: usize = 5;
const MAX_GRAPH_NODES: usize = 24;
const MAX_GRAPH_EDGES_PER_NODE: usize = 4;
const MAX_GRAPH_PARTS: usize = 72;
const MIN_LOCAL_NEURONS: usize = 2;
const MAX_LOCAL_NEURONS: usize = 4;
const MIN_GLOBAL_NEURONS: usize = 2;
const MAX_GLOBAL_NEURONS: usize = 4;
const LOCAL_SENSOR_DIM: usize = 14;
const GLOBAL_SENSOR_DIM: usize = 8;
const BRAIN_SUBSTEPS_PER_PHYSICS_STEP: usize = 2;
const FIXED_SIM_DT: f32 = 1.0 / 120.0;
const MASS_DENSITY_MULTIPLIER: f32 = 1.4;
const MAX_MOTOR_SPEED: f32 = 6.8;
const MAX_BODY_ANGULAR_SPEED: f32 = 15.0;
const MAX_BODY_LINEAR_SPEED: f32 = 22.0;
const EMERGENCY_MAX_BODY_ANGULAR_SPEED: f32 = 20.0;
const EMERGENCY_MAX_BODY_LINEAR_SPEED: f32 = 30.0;
const BODY_LINEAR_DAMPING: f32 = 0.19;
const BODY_ANGULAR_DAMPING: f32 = 25.0;
const QUADRATIC_ANGULAR_DRAG_COEFF: f32 = 0.22;
const QUADRATIC_LINEAR_DRAG_COEFF: f32 = 0.08;
const MOTOR_TORQUE_HIP: f32 = 85.0;
const BALL_AXIS_TORQUE_SCALE_Y: f32 = 0.7;
const BALL_AXIS_TORQUE_SCALE_Z: f32 = 0.7;
const BALL_AXIS_STIFFNESS_SCALE_Y: f32 = 0.75;
const BALL_AXIS_STIFFNESS_SCALE_Z: f32 = 0.75;
const JOINT_MOTOR_RESPONSE: f32 = 12.0;
const JOINT_MOTOR_FORCE_MULTIPLIER: f32 = 1.0;
const AXIS_TILT_GAIN: f32 = 1.9;
const FALLEN_HEIGHT_THRESHOLD: f32 = 0.35;
const FITNESS_PROGRESS_MIDPOINT_FRACTION: f32 = 0.5;
const FITNESS_PROGRESS_LATE_FRACTION: f32 = 0.85;
const FITNESS_PROGRESS_MID_WEIGHT: f32 = 1.35;
const FITNESS_PROGRESS_LATE_WEIGHT: f32 = 2.0;
const FITNESS_GROUNDED_BONUS_WEIGHT: f32 = 1.0;
const SETTLE_SECONDS: f32 = 1.5;
const PASSIVE_SETTLE_FRICTION: f32 = 0.0;
const ACTIVE_SURFACE_FRICTION: f32 = 1.08;
const JOINT_AREA_STRENGTH_MIN_SCALE: f32 = 0.12;
const JOINT_AREA_STRENGTH_MAX_SCALE: f32 = 5.5;
const GROUND_COLLISION_GROUP: Group = Group::GROUP_1;
const CREATURE_COLLISION_GROUP: Group = Group::GROUP_2;
const DEFAULT_BIND_HOST: &str = "0.0.0.0";
const DEFAULT_BIND_PORT: u16 = 8787;
const PORT_FALLBACK_ATTEMPTS: u16 = 32;
const SATELLITE_TRIAL_TIMEOUT: Duration = Duration::from_secs(10);
const SATELLITE_RECONNECT_DELAY: Duration = Duration::from_secs(3);
const SATELLITE_DISPATCH_RETRY_LIMIT: usize = 8;
const SATELLITE_CAPACITY_ERROR: &str = "satellite at capacity";
const FAST_EVAL_TRIAL_WALLTIME_LIMIT: Duration = Duration::from_secs(20);
const MAX_FITNESS_HISTORY_POINTS: usize = 4096;
const MAX_PERFORMANCE_HISTORY_POINTS: usize = 4096;
const DEFAULT_POPULATION_SIZE: usize = 160;
const MIN_POPULATION_SIZE: usize = 1;
const MAX_POPULATION_SIZE: usize = 384;
const ELITE_COUNT: usize = 2;
const TRIALS_PER_CANDIDATE: usize = 3;
const DEFAULT_GENERATION_SECONDS: f32 = 16.0;
const EVOLUTION_VIEW_FRAME_LIMIT: usize = 900;
const CHECKPOINT_DIR: &str = "data/checkpoints";
const AUTOSAVE_EVERY_GENERATIONS: usize = 5;
const DEFAULT_PERFORMANCE_WINDOW_GENERATIONS: usize = 120;
const MAX_PERFORMANCE_WINDOW_GENERATIONS: usize = 400;
const DEFAULT_PERFORMANCE_STRIDE: usize = 1;
const MAX_PERFORMANCE_STRIDE: usize = 8;
const MIN_BREEDING_MUTATION_RATE: f32 = 0.22;
const MAX_BREEDING_MUTATION_RATE: f32 = 0.78;
const FITNESS_STAGNATION_EPSILON: f32 = 1e-4;
const MAX_GENERATION_TOPOLOGY_CANDIDATES: usize = 3;
const SUMMARY_BEST_TOPOLOGY_COUNT: usize = 5;
const DIAG_RECENT_WINDOW: usize = 20;
const DIAG_PLATEAU_STAGNATION_GENERATIONS: usize = 20;
const DIAG_WATCH_STAGNATION_GENERATIONS: usize = 10;
const FIXED_PRESET_SPAWN_HEIGHT: f32 = 0.58;
const RANDOM_SPAWN_EXTRA_HEIGHT_MIN: f32 = 1.2;
const RANDOM_SPAWN_EXTRA_HEIGHT_MAX: f32 = 2.6;
const FIXED_PRESET_SETTLE_MIN_STABLE_SECONDS: f32 = 0.45;
const FIXED_PRESET_SETTLE_MAX_EXTRA_SECONDS: f32 = 1.8;
const FIXED_PRESET_SETTLE_LINEAR_SPEED_MAX: f32 = 0.65;
const FIXED_PRESET_SETTLE_ANGULAR_SPEED_MAX: f32 = 1.45;
const STARTUP_INVALID_LINEAR_SPEED: f32 = 20.0;
const STARTUP_INVALID_ANGULAR_SPEED: f32 = 24.0;
const TRAIN_TRIAL_SEED_BANK_TAG: u32 = 0x9e37_79b9;
const HOLDOUT_TRIAL_SEED_BANK_TAG: u32 = 0x7f4a_7c15;
const HOLDOUT_TRIALS_PER_CANDIDATE: usize = 5;
const TRIAL_DIVERGENCE_PENALTY_WEIGHT: f32 = 0.22;
const TRIAL_DIVERGENCE_PENALTY_FLOOR: f32 = 0.68;
const ENV_EVOLUTION_MORPHOLOGY_MODE: &str = "EVOLUTION_MORPHOLOGY_MODE";
const ENV_EVOLUTION_MORPHOLOGY_PRESET: &str = "EVOLUTION_MORPHOLOGY_PRESET";
static FRONTEND_ASSETS: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/ui");

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct TorsoGene {
    w: f32,
    h: f32,
    d: f32,
    mass: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct SegmentGene {
    length: f32,
    thickness: f32,
    mass: f32,
    #[serde(default = "default_joint_limit_x")]
    limit_x: f32,
    #[serde(default = "default_joint_limit_y")]
    limit_y: f32,
    #[serde(default = "default_joint_limit_z")]
    limit_z: f32,
    #[serde(default = "default_joint_type")]
    joint_type: JointTypeGene,
    #[serde(default = "default_motor_strength")]
    motor_strength: f32,
    #[serde(default = "default_joint_stiffness")]
    joint_stiffness: f32,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum JointTypeGene {
    Hinge,
    Ball,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum NeuralActivationGene {
    Tanh,
    Sigmoid,
    Sin,
    Cos,
    Identity,
    Relu,
    Softsign,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct NeuralUnitGene {
    #[serde(default = "default_neural_activation")]
    activation: NeuralActivationGene,
    #[serde(default)]
    input_weights: Vec<f32>,
    #[serde(default)]
    recurrent_weights: Vec<f32>,
    #[serde(default)]
    global_weights: Vec<f32>,
    #[serde(default)]
    bias: f32,
    #[serde(default = "default_neural_leak")]
    leak: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct JointEffectorGene {
    #[serde(default)]
    local_weights: Vec<f32>,
    #[serde(default)]
    global_weights: Vec<f32>,
    #[serde(default)]
    bias: f32,
    #[serde(default = "default_effector_gain")]
    gain: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct LocalBrainGene {
    #[serde(default)]
    neurons: Vec<NeuralUnitGene>,
    #[serde(default = "default_joint_effector_gene")]
    effector_x: JointEffectorGene,
    #[serde(default = "default_joint_effector_gene")]
    effector_y: JointEffectorGene,
    #[serde(default = "default_joint_effector_gene")]
    effector_z: JointEffectorGene,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct GlobalBrainGene {
    #[serde(default)]
    neurons: Vec<NeuralUnitGene>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct GraphPartGene {
    w: f32,
    h: f32,
    d: f32,
    mass: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct MorphEdgeGene {
    to: usize,
    anchor_x: f32,
    anchor_y: f32,
    anchor_z: f32,
    axis_y: f32,
    axis_z: f32,
    dir_x: f32,
    dir_y: f32,
    dir_z: f32,
    scale: f32,
    reflect_x: bool,
    recursive_limit: u32,
    terminal_only: bool,
    joint_type: JointTypeGene,
    limit_x: f32,
    limit_y: f32,
    limit_z: f32,
    motor_strength: f32,
    joint_stiffness: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct MorphNodeGene {
    part: GraphPartGene,
    #[serde(default)]
    edges: Vec<MorphEdgeGene>,
    #[serde(default = "default_local_brain_gene")]
    brain: LocalBrainGene,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct GraphGene {
    #[serde(default)]
    root: usize,
    #[serde(default)]
    nodes: Vec<MorphNodeGene>,
    #[serde(default = "default_global_brain_gene")]
    global_brain: GlobalBrainGene,
    #[serde(default = "default_graph_max_parts")]
    max_parts: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct ControlGene {
    amp: f32,
    freq: f32,
    phase: f32,
    bias: f32,
    #[serde(default = "default_second_harmonic_amp")]
    harm2_amp: f32,
    #[serde(default = "default_second_harmonic_phase")]
    harm2_phase: f32,
    #[serde(default = "default_secondary_control_amp")]
    amp_y: f32,
    #[serde(default = "default_secondary_control_freq")]
    freq_y: f32,
    #[serde(default = "default_secondary_control_phase")]
    phase_y: f32,
    #[serde(default = "default_secondary_control_bias")]
    bias_y: f32,
    #[serde(default = "default_secondary_control_amp")]
    amp_z: f32,
    #[serde(default = "default_secondary_control_freq")]
    freq_z: f32,
    #[serde(default = "default_secondary_control_phase")]
    phase_z: f32,
    #[serde(default = "default_secondary_control_bias")]
    bias_z: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct LimbGene {
    enabled: bool,
    segment_count: u32,
    anchor_x: f32,
    anchor_y: f32,
    anchor_z: f32,
    axis_y: f32,
    axis_z: f32,
    #[serde(default = "default_limb_dir_x")]
    dir_x: f32,
    #[serde(default = "default_limb_dir_y")]
    dir_y: f32,
    #[serde(default = "default_limb_dir_z")]
    dir_z: f32,
    segments: Vec<SegmentGene>,
    controls: Vec<ControlGene>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct Genome {
    #[serde(default = "default_genome_version")]
    version: u32,
    #[serde(default = "default_graph_gene")]
    graph: GraphGene,
    torso: TorsoGene,
    limbs: Vec<LimbGene>,
    hue: f32,
    mass_scale: f32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TrialRunRequest {
    genome: Genome,
    seed: u64,
    duration_seconds: Option<f32>,
    #[serde(default, rename = "dt")]
    _dt: Option<f32>,
    snapshot_hz: Option<f32>,
    motor_power_scale: Option<f32>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerationEvalRequest {
    genomes: Vec<Genome>,
    seeds: Vec<u64>,
    duration_seconds: Option<f32>,
    #[serde(default, rename = "dt")]
    _dt: Option<f32>,
    motor_power_scale: Option<f32>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerationEvalResponse {
    results: Vec<GenerationEvalResult>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerationEvalResult {
    fitness: f32,
    descriptor: [f32; 5],
    trial_count: usize,
    median_progress: f32,
    median_upright: f32,
    median_straightness: f32,
    invalid_startup_trials: usize,
    invalid_startup_trial_rate: f32,
    all_trials_invalid_startup: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BodyPoseSnapshot {
    p: [f32; 3],
    q: [f32; 4],
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SnapshotFrame {
    time: f32,
    score: f32,
    bodies: Vec<BodyPoseSnapshot>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TrialMetrics {
    quality: f32,
    progress: f32,
    upright_avg: f32,
    avg_height: f32,
    instability_norm: f32,
    energy_norm: f32,
    fallen_ratio: f32,
    straightness: f32,
    net_distance: f32,
    #[serde(default)]
    invalid_startup: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TrialResult {
    fitness: f32,
    metrics: TrialMetrics,
    descriptor: [f32; 5],
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamEvent {
    TrialStarted { part_sizes: Vec<[f32; 3]> },
    Snapshot { frame: SnapshotFrame },
    TrialComplete { result: TrialResult },
    Error { message: String },
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum GenerationStreamEvent {
    GenerationStarted {
        attempt_count: usize,
        trial_count: usize,
    },
    AttemptTrialStarted {
        attempt_index: usize,
        trial_index: usize,
        trial_count: usize,
    },
    AttemptComplete {
        attempt_index: usize,
        result: GenerationEvalResult,
    },
    GenerationComplete {
        results: Vec<GenerationEvalResult>,
    },
    Error {
        message: String,
    },
}

#[derive(Clone, Debug)]
struct TrialConfig {
    duration_seconds: f32,
    dt: f32,
    snapshot_hz: f32,
    motor_power_scale: f32,
    fixed_startup: bool,
}

impl TrialConfig {
    fn from_trial_request(request: &TrialRunRequest) -> Self {
        Self {
            duration_seconds: request.duration_seconds.unwrap_or(18.0).clamp(1.0, 120.0),
            dt: FIXED_SIM_DT,
            snapshot_hz: request.snapshot_hz.unwrap_or(30.0).clamp(1.0, 120.0),
            motor_power_scale: request.motor_power_scale.unwrap_or(1.0).clamp(0.35, 1.5),
            fixed_startup: false,
        }
    }

    fn from_generation_request(request: &GenerationEvalRequest) -> Self {
        Self {
            duration_seconds: request.duration_seconds.unwrap_or(18.0).clamp(1.0, 120.0),
            dt: FIXED_SIM_DT,
            snapshot_hz: 30.0,
            motor_power_scale: request.motor_power_scale.unwrap_or(1.0).clamp(0.35, 1.5),
            fixed_startup: false,
        }
    }
}

struct SimPart {
    body: RigidBodyHandle,
    size: [f32; 3],
    node_index: usize,
    parent_part: Option<usize>,
    child_parts: Vec<usize>,
}

struct SimController {
    joint: ImpulseJointHandle,
    joint_type: JointTypeGene,
    parent_part_index: usize,
    child_part_index: usize,
    node_index: usize,
    torque_x: f32,
    stiffness_x: f32,
    limit_x: f32,
    torque_y: f32,
    stiffness_y: f32,
    limit_y: f32,
    torque_z: f32,
    stiffness_z: f32,
    limit_z: f32,
}

#[derive(Clone, Debug)]
struct LocalBrainRuntime {
    outputs_prev: Vec<f32>,
    outputs_next: Vec<f32>,
}

#[derive(Clone, Debug)]
struct GlobalBrainRuntime {
    outputs_prev: Vec<f32>,
    outputs_next: Vec<f32>,
}

#[derive(Clone, Debug)]
struct ExpandedGraphPart {
    node_index: usize,
    parent_index: Option<usize>,
    incoming_edge: Option<MorphEdgeGene>,
}

struct TrialAccumulator {
    spawn: Vector3<f32>,
    path_length: f32,
    distance_at_mid_phase: Option<f32>,
    distance_at_late_phase: Option<f32>,
    sampled_time: f32,
    upright_integral: f32,
    height_integral: f32,
    instability_integral: f32,
    energy_integral: f32,
    fallen_time: f32,
    net_dx: f32,
    net_dz: f32,
    last_torso_pos: Option<Vector3<f32>>,
    live_score: f32,
    active_limb_count: usize,
    mean_segment_count: f32,
}

impl TrialAccumulator {
    fn new(spawn: Vector3<f32>, genome: &Genome) -> Self {
        let edge_total = genome
            .graph
            .nodes
            .iter()
            .map(|node| node.edges.len())
            .sum::<usize>();
        let active_limb_count = edge_total.clamp(1, MAX_GRAPH_PARTS);
        let mean_segment_count = if active_limb_count > 0 {
            (genome.graph.max_parts as f32 / active_limb_count as f32)
                .clamp(1.0, MAX_SEGMENTS_PER_LIMB as f32 * 2.0)
        } else {
            1.0
        };

        Self {
            spawn,
            path_length: 0.0,
            distance_at_mid_phase: None,
            distance_at_late_phase: None,
            sampled_time: 0.0,
            upright_integral: 0.0,
            height_integral: 0.0,
            instability_integral: 0.0,
            energy_integral: 0.0,
            fallen_time: 0.0,
            net_dx: 0.0,
            net_dz: 0.0,
            last_torso_pos: None,
            live_score: 0.0,
            active_limb_count,
            mean_segment_count,
        }
    }

    fn add_energy(&mut self, value: f32) {
        self.energy_integral += value.max(0.0);
    }

    fn update(
        &mut self,
        torso_pos: Vector3<f32>,
        torso_rot: UnitQuaternion<f32>,
        torso_angvel: Vector3<f32>,
        dt: f32,
        duration: f32,
    ) {
        if self.sampled_time < SETTLE_SECONDS {
            self.sampled_time += dt;
            self.last_torso_pos = Some(torso_pos);
            return;
        }

        let (raw_dx, raw_dz) = if let Some(last) = self.last_torso_pos {
            (torso_pos.x - last.x, torso_pos.z - last.z)
        } else {
            (torso_pos.x - self.spawn.x, torso_pos.z - self.spawn.z)
        };
        let raw_step = (raw_dx * raw_dx + raw_dz * raw_dz).sqrt();
        if raw_step.is_finite() && raw_step > 1e-6 {
            self.net_dx += raw_dx;
            self.net_dz += raw_dz;
            self.path_length += raw_step;
        }

        let current_distance = (self.net_dx * self.net_dx + self.net_dz * self.net_dz).sqrt();
        let effective_duration = (duration - SETTLE_SECONDS).max(1e-6);
        let eval_time = (self.sampled_time - SETTLE_SECONDS).max(0.0);
        if self.distance_at_mid_phase.is_none()
            && eval_time >= effective_duration * FITNESS_PROGRESS_MIDPOINT_FRACTION
        {
            self.distance_at_mid_phase = Some(current_distance);
        }
        if self.distance_at_late_phase.is_none()
            && eval_time >= effective_duration * FITNESS_PROGRESS_LATE_FRACTION
        {
            self.distance_at_late_phase = Some(current_distance);
        }

        self.last_torso_pos = Some(torso_pos);

        let up = torso_rot * vector![0.0, 1.0, 0.0];
        let upright = clamp((up.y + 1.0) * 0.5, 0.0, 1.0);
        self.upright_integral += upright * dt;
        self.height_integral += torso_pos.y * dt;

        let instability = torso_angvel.norm();
        self.instability_integral += instability * dt;

        if torso_pos.y < FALLEN_HEIGHT_THRESHOLD {
            self.fallen_time += dt;
        }

        self.sampled_time += dt;
        self.live_score = self.compute_metrics(duration).quality;
    }

    fn descriptor(&self, metrics: &TrialMetrics) -> [f32; 5] {
        [
            clamp(metrics.progress / 28.0, 0.0, 1.0),
            clamp(metrics.upright_avg, 0.0, 1.0),
            clamp(metrics.straightness, 0.0, 1.0),
            clamp(self.active_limb_count as f32 / MAX_LIMBS as f32, 0.0, 1.0),
            clamp(
                self.mean_segment_count / MAX_SEGMENTS_PER_LIMB as f32,
                0.0,
                1.0,
            ),
        ]
    }

    fn compute_metrics(&self, duration: f32) -> TrialMetrics {
        let effective_duration = (duration - SETTLE_SECONDS).max(1e-6);
        let sample_time = (self.sampled_time - SETTLE_SECONDS).max(1e-6);
        let upright_avg = self.upright_integral / sample_time;
        let avg_height = self.height_integral / sample_time;
        let instability_norm = clamp(
            (self.instability_integral / sample_time) / MAX_BODY_ANGULAR_SPEED,
            0.0,
            2.5,
        );
        let active_segment_estimate =
            (self.active_limb_count as f32 * self.mean_segment_count).max(1.0);
        let actuator_scale = active_segment_estimate * MOTOR_TORQUE_HIP * MAX_MOTOR_SPEED;
        let energy_norm = clamp(
            (self.energy_integral / sample_time) / actuator_scale.max(1.0),
            0.0,
            3.0,
        );
        let fallen_ratio = clamp(self.fallen_time / effective_duration, 0.0, 1.0);
        let net_distance = (self.net_dx * self.net_dx + self.net_dz * self.net_dz).sqrt();
        let straightness = if self.path_length > 1e-4 {
            clamp(net_distance / self.path_length, 0.0, 1.0)
        } else {
            0.0
        };
        let mid_distance = self
            .distance_at_mid_phase
            .unwrap_or(net_distance)
            .clamp(0.0, net_distance);
        let late_distance = self
            .distance_at_late_phase
            .unwrap_or(net_distance)
            .clamp(mid_distance, net_distance);
        let early_gain = mid_distance;
        let mid_gain = (late_distance - mid_distance).max(0.0);
        let late_gain = (net_distance - late_distance).max(0.0);
        let sustained_progress =
            early_gain + mid_gain * FITNESS_PROGRESS_MID_WEIGHT + late_gain * FITNESS_PROGRESS_LATE_WEIGHT;
        let progress = net_distance;
        let grounded_bonus = FITNESS_GROUNDED_BONUS_WEIGHT * (1.0 - fallen_ratio);
        let quality = (sustained_progress * (1.0 - 0.3 * fallen_ratio) + grounded_bonus).max(0.0);

        TrialMetrics {
            quality,
            progress,
            upright_avg,
            avg_height,
            instability_norm,
            energy_norm,
            fallen_ratio,
            straightness,
            net_distance,
            invalid_startup: false,
        }
    }
}

struct TrialSimulator {
    pipeline: PhysicsPipeline,
    gravity: Vector3<f32>,
    integration_parameters: IntegrationParameters,
    island_manager: IslandManager,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd_solver: CCDSolver,
    parts: Vec<SimPart>,
    controllers: Vec<SimController>,
    child_controller_index: Vec<Option<usize>>,
    local_brains: Vec<LocalBrainRuntime>,
    global_brain: GlobalBrainRuntime,
    ground_collider: ColliderHandle,
    torso_handle: RigidBodyHandle,
    genome: Genome,
    metrics: TrialAccumulator,
    elapsed: f32,
    duration: f32,
    require_settled_before_actuation: bool,
    settled_time_before_actuation: f32,
    actuation_started: bool,
    surface_friction_is_passive: bool,
    startup_invalid: bool,
}

impl TrialSimulator {
    fn new(genome: &Genome, seed: u64, config: &TrialConfig) -> Result<Self, String> {
        let mut pipeline = PhysicsPipeline::new();
        let gravity = vector![0.0, -15.5, 0.0];
        let mut integration_parameters = IntegrationParameters::default();
        integration_parameters.dt = config.dt;
        integration_parameters.max_ccd_substeps = 4;
        integration_parameters.num_solver_iterations = 16;
        integration_parameters.num_internal_pgs_iterations = 4;
        integration_parameters.num_internal_stabilization_iterations = 4;

        let mut island_manager = IslandManager::new();
        let mut broad_phase = BroadPhaseBvh::new();
        let mut narrow_phase = NarrowPhase::new();
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let mut impulse_joints = ImpulseJointSet::new();
        let multibody_joints = MultibodyJointSet::new();
        let ccd_solver = CCDSolver::new();

        let mut rng = SmallRng::seed_from_u64(seed);

        let ground_handle = bodies.insert(RigidBodyBuilder::fixed().build());
        let ground_collider = ColliderBuilder::cuboid(420.0, 5.0, 420.0)
            .translation(vector![0.0, -5.0, 0.0])
            .friction(ACTIVE_SURFACE_FRICTION)
            .restitution(0.015)
            .collision_groups(InteractionGroups::new(
                GROUND_COLLISION_GROUP,
                CREATURE_COLLISION_GROUP,
                InteractionTestMode::And,
            ))
            .build();
        let ground_collider =
            colliders.insert_with_parent(ground_collider, ground_handle, &mut bodies);

        let use_fixed_preset_startup = config.fixed_startup;
        let spawn = if use_fixed_preset_startup {
            vector![0.0, FIXED_PRESET_SPAWN_HEIGHT, 0.0]
        } else {
            vector![0.0, 0.05, 0.0]
        };
        let root_index = genome
            .graph
            .root
            .min(genome.graph.nodes.len().saturating_sub(1));
        let root_node = genome
            .graph
            .nodes
            .get(root_index)
            .cloned()
            .unwrap_or_else(default_graph_node_gene);
        let torso_dims = root_node.part;
        let torso_mass = (torso_dims.w
            * torso_dims.h
            * torso_dims.d
            * torso_dims.mass
            * genome.mass_scale
            * MASS_DENSITY_MULTIPLIER)
            .max(0.7);

        let drop_start = if use_fixed_preset_startup {
            spawn
        } else {
            vector![
                spawn.x,
                spawn.y
                    + rng_range(
                        &mut rng,
                        RANDOM_SPAWN_EXTRA_HEIGHT_MIN,
                        RANDOM_SPAWN_EXTRA_HEIGHT_MAX,
                    ),
                spawn.z
            ]
        };
        let torso_handle = insert_box_body(
            &mut bodies,
            &mut colliders,
            [torso_dims.w, torso_dims.h, torso_dims.d],
            torso_mass,
            drop_start,
        );
        {
            let torso_body = bodies
                .get_mut(torso_handle)
                .ok_or_else(|| "torso body missing".to_string())?;
            let rot = if use_fixed_preset_startup {
                UnitQuaternion::from_euler_angles(0.0, rng_range(&mut rng, 0.0, PI * 2.0), 0.0)
            } else {
                UnitQuaternion::from_euler_angles(
                    rng_range(&mut rng, -0.36, 0.36),
                    rng_range(&mut rng, 0.0, PI * 2.0),
                    rng_range(&mut rng, -0.28, 0.28),
                )
            };
            torso_body.set_rotation(rot, true);
            torso_body.set_linvel(vector![0.0, 0.0, 0.0], true);
            torso_body.set_angvel(vector![0.0, 0.0, 0.0], true);
        }

        let expanded_graph = Self::expand_graph(&genome.graph);
        let mut parts = vec![SimPart {
            body: torso_handle,
            size: [torso_dims.w, torso_dims.h, torso_dims.d],
            node_index: root_index,
            parent_part: None,
            child_parts: Vec::new(),
        }];
        let mut controllers = Vec::new();

        for (expanded_index, expanded_part) in expanded_graph.iter().enumerate().skip(1) {
            let Some(parent_index) = expanded_part.parent_index else {
                continue;
            };
            if parent_index >= parts.len() {
                continue;
            }
            let Some(edge) = expanded_part.incoming_edge.as_ref() else {
                continue;
            };
            let node = genome
                .graph
                .nodes
                .get(expanded_part.node_index)
                .cloned()
                .unwrap_or_else(default_graph_node_gene);
            let Some(parent_body) = bodies.get(parts[parent_index].body) else {
                continue;
            };
            let parent_translation = *parent_body.translation();
            let parent_rotation = *parent_body.rotation();
            let parent_size = parts[parent_index].size;
            let reflect_sign = if edge.reflect_x { -1.0 } else { 1.0 };
            let pivot_from_parent = vector![
                edge.anchor_x * parent_size[0] * 0.5,
                edge.anchor_y * parent_size[1] * 0.5,
                edge.anchor_z * parent_size[2] * 0.5
            ];
            let axis_local = normalized_axis(edge.axis_y, edge.axis_z);
            let local_growth = normalized_dir(
                edge.dir_x * reflect_sign,
                edge.dir_y,
                edge.dir_z * reflect_sign,
            );
            let anchor_world = parent_translation + parent_rotation * pivot_from_parent;
            let part_w = (node.part.w * edge.scale).abs().clamp(0.14, 2.8);
            let part_h = (node.part.h * edge.scale).abs().clamp(0.22, 3.4);
            let part_d = (node.part.d * edge.scale).abs().clamp(0.14, 2.8);
            let growth_world = parent_rotation * local_growth;
            let center = anchor_world + growth_world * (part_h * 0.5);
            let seg_local_rot =
                UnitQuaternion::rotation_between(&vector![0.0, -1.0, 0.0], &local_growth)
                    .unwrap_or_else(UnitQuaternion::identity);
            let child_rotation = parent_rotation * seg_local_rot;
            let part_mass = (part_w
                * part_h
                * part_d
                * node.part.mass
                * genome.mass_scale
                * MASS_DENSITY_MULTIPLIER)
                .max(0.08);
            let child = insert_box_body(
                &mut bodies,
                &mut colliders,
                [part_w, part_h, part_d],
                part_mass,
                center,
            );
            if let Some(child_body) = bodies.get_mut(child) {
                child_body.set_rotation(child_rotation, true);
            }
            let local_anchor2 = point![0.0, part_h * 0.5, 0.0];
            let limit_x = clamp(edge.limit_x, 0.12, PI * 0.95);
            let limit_y = clamp(edge.limit_y, 0.10, PI * 0.75);
            let limit_z = clamp(edge.limit_z, 0.10, PI * 0.75);
            let parent_joint_area = joint_area_strength_scale(parent_size, [part_w, part_h, part_d]);
            let torque_x = MOTOR_TORQUE_HIP
                * edge.motor_strength
                * JOINT_MOTOR_FORCE_MULTIPLIER
                * config.motor_power_scale
                * parent_joint_area;
            let stiffness_x = edge.joint_stiffness * edge.motor_strength * parent_joint_area;
            let torque_y = torque_x * BALL_AXIS_TORQUE_SCALE_Y;
            let torque_z = torque_x * BALL_AXIS_TORQUE_SCALE_Z;
            let stiffness_y = stiffness_x * BALL_AXIS_STIFFNESS_SCALE_Y;
            let stiffness_z = stiffness_x * BALL_AXIS_STIFFNESS_SCALE_Z;

            let joint_handle = match edge.joint_type {
                JointTypeGene::Hinge => {
                    let hinge_axis = UnitVector::new_normalize(axis_local);
                    let hinge_frame_parent_rot =
                        UnitQuaternion::rotation_between(&vector![1.0, 0.0, 0.0], &axis_local)
                            .unwrap_or_else(UnitQuaternion::identity);
                    let hinge_frame_child_rot = seg_local_rot.inverse() * hinge_frame_parent_rot;
                    let mut joint = RevoluteJointBuilder::new(hinge_axis)
                        .contacts_enabled(false)
                        .limits([-limit_x, limit_x])
                        .build();
                    let local_frame1 = Isometry3::from_parts(
                        Translation3::from(vector![
                            pivot_from_parent.x,
                            pivot_from_parent.y,
                            pivot_from_parent.z
                        ]),
                        hinge_frame_parent_rot,
                    );
                    let local_frame2 = Isometry3::from_parts(
                        Translation3::from(vector![
                            local_anchor2.x,
                            local_anchor2.y,
                            local_anchor2.z
                        ]),
                        hinge_frame_child_rot,
                    );
                    joint.data.set_local_frame1(local_frame1);
                    joint.data.set_local_frame2(local_frame2);
                    let handle =
                        impulse_joints.insert(parts[parent_index].body, child, joint, true);
                    if let Some(joint_ref) = impulse_joints.get_mut(handle, false) {
                        joint_ref
                            .data
                            .set_motor_model(JointAxis::AngX, MotorModel::ForceBased);
                        joint_ref.data.set_motor_position(
                            JointAxis::AngX,
                            0.0,
                            stiffness_x,
                            JOINT_MOTOR_RESPONSE,
                        );
                        joint_ref
                            .data
                            .set_motor_max_force(JointAxis::AngX, 0.0);
                    }
                    handle
                }
                JointTypeGene::Ball => {
                    let local_frame1 = Isometry3::from_parts(
                        Translation3::from(vector![
                            pivot_from_parent.x,
                            pivot_from_parent.y,
                            pivot_from_parent.z
                        ]),
                        seg_local_rot,
                    );
                    let local_frame2 = Isometry3::from_parts(
                        Translation3::from(vector![
                            local_anchor2.x,
                            local_anchor2.y,
                            local_anchor2.z
                        ]),
                        UnitQuaternion::identity(),
                    );
                    let joint = SphericalJointBuilder::new()
                        .local_frame1(local_frame1)
                        .local_frame2(local_frame2)
                        .contacts_enabled(false)
                        .limits(JointAxis::AngX, [-limit_x, limit_x])
                        .limits(JointAxis::AngY, [-limit_y, limit_y])
                        .limits(JointAxis::AngZ, [-limit_z, limit_z]);
                    let handle =
                        impulse_joints.insert(parts[parent_index].body, child, joint, true);
                    if let Some(joint_ref) = impulse_joints.get_mut(handle, false) {
                        for (axis, stiffness, _torque) in [
                            (JointAxis::AngX, stiffness_x, torque_x),
                            (JointAxis::AngY, stiffness_y, torque_y),
                            (JointAxis::AngZ, stiffness_z, torque_z),
                        ] {
                            joint_ref.data.set_motor_model(axis, MotorModel::ForceBased);
                            joint_ref.data.set_motor_position(
                                axis,
                                0.0,
                                stiffness,
                                JOINT_MOTOR_RESPONSE,
                            );
                            joint_ref.data.set_motor_max_force(axis, 0.0);
                        }
                    }
                    handle
                }
            };

            parts[parent_index].child_parts.push(expanded_index);
            parts.push(SimPart {
                body: child,
                size: [part_w, part_h, part_d],
                node_index: expanded_part.node_index,
                parent_part: Some(parent_index),
                child_parts: Vec::new(),
            });
            controllers.push(SimController {
                joint: joint_handle,
                joint_type: edge.joint_type,
                parent_part_index: parent_index,
                child_part_index: expanded_index,
                node_index: expanded_part.node_index,
                torque_x,
                stiffness_x,
                limit_x,
                torque_y,
                stiffness_y,
                limit_y,
                torque_z,
                stiffness_z,
                limit_z,
            });
        }

        let mut child_controller_index = vec![None; parts.len()];
        for (controller_index, controller) in controllers.iter().enumerate() {
            if controller.child_part_index < child_controller_index.len() {
                child_controller_index[controller.child_part_index] = Some(controller_index);
            }
        }

        let mut local_brains = Vec::with_capacity(parts.len());
        for part in &parts {
            let local_count = genome
                .graph
                .nodes
                .get(part.node_index)
                .map(|node| node.brain.neurons.len())
                .unwrap_or(MIN_LOCAL_NEURONS)
                .clamp(MIN_LOCAL_NEURONS, MAX_LOCAL_NEURONS);
            local_brains.push(LocalBrainRuntime {
                outputs_prev: vec![0.0; local_count],
                outputs_next: vec![0.0; local_count],
            });
        }
        let global_count = genome
            .graph
            .global_brain
            .neurons
            .len()
            .clamp(MIN_GLOBAL_NEURONS, MAX_GLOBAL_NEURONS);
        let global_brain = GlobalBrainRuntime {
            outputs_prev: vec![0.0; global_count],
            outputs_next: vec![0.0; global_count],
        };

        // prevent dead-code warning on queue index generation variable
        let _ = expanded_graph;

        // Advance one step so colliders settle into broadphase structures.
        pipeline.step(
            &gravity,
            &integration_parameters,
            &mut island_manager,
            &mut broad_phase,
            &mut narrow_phase,
            &mut bodies,
            &mut colliders,
            &mut impulse_joints,
            &mut MultibodyJointSet::new(),
            &mut CCDSolver::new(),
            &(),
            &(),
        );

        let mut sim = Self {
            pipeline,
            gravity,
            integration_parameters,
            island_manager,
            broad_phase,
            narrow_phase,
            bodies,
            colliders,
            impulse_joints,
            multibody_joints,
            ccd_solver,
            parts,
            controllers,
            child_controller_index,
            local_brains,
            global_brain,
            ground_collider,
            torso_handle,
            genome: genome.clone(),
            metrics: TrialAccumulator::new(spawn, genome),
            elapsed: 0.0,
            duration: config.duration_seconds,
            require_settled_before_actuation: true,
            settled_time_before_actuation: 0.0,
            actuation_started: false,
            surface_friction_is_passive: true,
            startup_invalid: false,
        };
        sim.set_surface_friction(PASSIVE_SETTLE_FRICTION);
        Ok(sim)
    }

    fn expand_graph(graph: &GraphGene) -> Vec<ExpandedGraphPart> {
        if graph.nodes.is_empty() {
            return vec![ExpandedGraphPart {
                node_index: 0,
                parent_index: None,
                incoming_edge: None,
            }];
        }
        let root = graph.root.min(graph.nodes.len().saturating_sub(1));
        let max_parts = graph.max_parts.clamp(1, MAX_GRAPH_PARTS);
        let mut expanded = vec![ExpandedGraphPart {
            node_index: root,
            parent_index: None,
            incoming_edge: None,
        }];
        let mut queue: VecDeque<(usize, usize, Vec<usize>)> = VecDeque::new();
        queue.push_back((0, root, vec![root]));
        while let Some((expanded_index, node_index, ancestry)) = queue.pop_front() {
            if expanded.len() >= max_parts {
                break;
            }
            let Some(node) = graph.nodes.get(node_index) else {
                continue;
            };
            for edge in node.edges.iter().take(MAX_GRAPH_EDGES_PER_NODE) {
                if expanded.len() >= max_parts {
                    break;
                }
                if edge.to >= graph.nodes.len() {
                    continue;
                }
                let recursive_limit = edge.recursive_limit.max(1) as usize;
                let occurrences = ancestry.iter().filter(|value| **value == edge.to).count();
                let is_recursive = occurrences > 0;
                if is_recursive && occurrences >= recursive_limit {
                    continue;
                }
                if edge.terminal_only && (!is_recursive || occurrences + 1 < recursive_limit) {
                    continue;
                }
                let mut child_ancestry = ancestry.clone();
                child_ancestry.push(edge.to);
                let child_index = expanded.len();
                expanded.push(ExpandedGraphPart {
                    node_index: edge.to,
                    parent_index: Some(expanded_index),
                    incoming_edge: Some(edge.clone()),
                });
                queue.push_back((child_index, edge.to, child_ancestry));
            }
        }
        if expanded.len() == 1 {
            let node = &graph.nodes[root];
            if let Some(edge) = node.edges.first() {
                let to = edge.to.min(graph.nodes.len().saturating_sub(1));
                expanded.push(ExpandedGraphPart {
                    node_index: to,
                    parent_index: Some(0),
                    incoming_edge: Some(edge.clone()),
                });
            }
        }
        expanded
    }

    fn part_sizes(&self) -> Vec<[f32; 3]> {
        self.parts.iter().map(|part| part.size).collect()
    }

    fn current_frame(&self) -> SnapshotFrame {
        let mut bodies = Vec::with_capacity(self.parts.len());
        for part in &self.parts {
            if let Some(body) = self.bodies.get(part.body) {
                let p = body.translation();
                let q = body.rotation();
                bodies.push(BodyPoseSnapshot {
                    p: [p.x, p.y, p.z],
                    q: [q.i, q.j, q.k, q.w],
                });
            }
        }
        SnapshotFrame {
            time: self.elapsed,
            score: self.metrics.live_score,
            bodies,
        }
    }

    fn final_result(&self) -> TrialResult {
        let mut metrics = self.metrics.compute_metrics(self.duration);
        if self.startup_invalid {
            metrics.quality = 0.0;
            metrics.progress = 0.0;
            metrics.net_distance = 0.0;
            metrics.invalid_startup = true;
        }
        let descriptor = self.metrics.descriptor(&metrics);
        TrialResult {
            fitness: metrics.quality,
            metrics,
            descriptor,
        }
    }

    fn part_contact_value(&self, part_index: usize) -> f32 {
        let Some(part) = self.parts.get(part_index) else {
            return -1.0;
        };
        let Some(body) = self.bodies.get(part.body) else {
            return -1.0;
        };
        for collider in body.colliders() {
            if let Some(pair) = self
                .narrow_phase
                .contact_pair(*collider, self.ground_collider)
            {
                if pair.has_any_active_contact {
                    return 1.0;
                }
            }
        }
        -1.0
    }

    fn joint_state_for_child(&self, child_part_index: usize) -> ([f32; 3], [f32; 3]) {
        let Some(controller_index) = self
            .child_controller_index
            .get(child_part_index)
            .and_then(|value| *value)
        else {
            return ([0.0; 3], [0.0; 3]);
        };
        let Some(controller) = self.controllers.get(controller_index) else {
            return ([0.0; 3], [0.0; 3]);
        };
        let Some(parent_body) = self
            .bodies
            .get(self.parts[controller.parent_part_index].body)
        else {
            return ([0.0; 3], [0.0; 3]);
        };
        let Some(child_body) = self
            .bodies
            .get(self.parts[controller.child_part_index].body)
        else {
            return ([0.0; 3], [0.0; 3]);
        };
        let rel_rot = parent_body.rotation().inverse() * child_body.rotation();
        let (rx, ry, rz) = rel_rot.euler_angles();
        let rel_angvel_world = child_body.angvel() - parent_body.angvel();
        let rel_angvel_local = parent_body
            .rotation()
            .inverse_transform_vector(&rel_angvel_world);
        let angle_x = clamp(rx / controller.limit_x.max(0.05), -1.0, 1.0);
        let angle_y = clamp(ry / controller.limit_y.max(0.05), -1.0, 1.0);
        let angle_z = clamp(rz / controller.limit_z.max(0.05), -1.0, 1.0);
        let vel_x = clamp(rel_angvel_local.x / MAX_MOTOR_SPEED, -1.0, 1.0);
        let vel_y = clamp(rel_angvel_local.y / MAX_MOTOR_SPEED, -1.0, 1.0);
        let vel_z = clamp(rel_angvel_local.z / MAX_MOTOR_SPEED, -1.0, 1.0);
        ([angle_x, angle_y, angle_z], [vel_x, vel_y, vel_z])
    }

    fn local_sensor_vector(&self, part_index: usize, sim_time: f32) -> [f32; LOCAL_SENSOR_DIM] {
        let mut sensors = [0.0f32; LOCAL_SENSOR_DIM];
        sensors[0] = sim_time.sin();
        sensors[1] = sim_time.cos();
        sensors[2] = self.part_contact_value(part_index);
        let (joint_angles, joint_vels) = self.joint_state_for_child(part_index);
        sensors[3] = joint_angles[0];
        sensors[4] = joint_angles[1];
        sensors[5] = joint_angles[2];
        sensors[6] = joint_vels[0];
        sensors[7] = joint_vels[1];
        sensors[8] = joint_vels[2];
        if let Some(torso) = self.bodies.get(self.torso_handle) {
            let up = torso.rotation() * vector![0.0, 1.0, 0.0];
            sensors[9] = clamp(up.y, -1.0, 1.0);
            sensors[10] = clamp(torso.linvel().x / MAX_BODY_LINEAR_SPEED, -1.0, 1.0);
            sensors[11] = clamp(torso.linvel().z / MAX_BODY_LINEAR_SPEED, -1.0, 1.0);
        }
        let parent_mean = self
            .parts
            .get(part_index)
            .and_then(|part| part.parent_part)
            .and_then(|parent_index| self.local_brains.get(parent_index))
            .map(|brain| mean(&brain.outputs_prev))
            .unwrap_or(0.0);
        let child_mean = self
            .parts
            .get(part_index)
            .map(|part| {
                if part.child_parts.is_empty() {
                    0.0
                } else {
                    let mut values = Vec::with_capacity(part.child_parts.len());
                    for child_index in &part.child_parts {
                        if let Some(brain) = self.local_brains.get(*child_index) {
                            values.push(mean(&brain.outputs_prev));
                        }
                    }
                    mean(&values)
                }
            })
            .unwrap_or(0.0);
        sensors[12] = parent_mean;
        sensors[13] = child_mean;
        sensors
    }

    fn global_sensor_vector(&self, sim_time: f32) -> [f32; GLOBAL_SENSOR_DIM] {
        let mut sensors = [0.0f32; GLOBAL_SENSOR_DIM];
        sensors[0] = sim_time.sin();
        sensors[1] = sim_time.cos();
        let mut contact_samples = Vec::with_capacity(self.parts.len());
        for part_index in 0..self.parts.len() {
            contact_samples.push((self.part_contact_value(part_index) + 1.0) * 0.5);
        }
        sensors[2] = mean(&contact_samples);
        if let Some(torso) = self.bodies.get(self.torso_handle) {
            let up = torso.rotation() * vector![0.0, 1.0, 0.0];
            sensors[3] = clamp(up.y, -1.0, 1.0);
            sensors[4] = clamp(torso.linvel().x / MAX_BODY_LINEAR_SPEED, -1.0, 1.0);
            sensors[5] = clamp(torso.linvel().z / MAX_BODY_LINEAR_SPEED, -1.0, 1.0);
            sensors[6] = clamp(torso.angvel().norm() / MAX_BODY_ANGULAR_SPEED, 0.0, 1.0);
            sensors[7] = clamp(torso.linvel().norm() / MAX_BODY_LINEAR_SPEED, 0.0, 1.0);
        }
        sensors
    }

    fn update_brains(&mut self, sim_time: f32, dt: f32) {
        for _ in 0..BRAIN_SUBSTEPS_PER_PHYSICS_STEP {
            let global_sensors = self.global_sensor_vector(sim_time);
            for global_index in 0..self.global_brain.outputs_prev.len() {
                let Some(gene) = self
                    .genome
                    .graph
                    .global_brain
                    .neurons
                    .get(global_index)
                    .cloned()
                else {
                    self.global_brain.outputs_next[global_index] = 0.0;
                    continue;
                };
                let mut signal = gene.bias;
                signal += dot_weights(&gene.input_weights, &global_sensors);
                signal += dot_weights(&gene.recurrent_weights, &self.global_brain.outputs_prev);
                let activated = apply_neural_activation(gene.activation, signal);
                let leak = clamp(gene.leak, 0.05, 1.0);
                self.global_brain.outputs_next[global_index] = lerp(
                    self.global_brain.outputs_prev[global_index],
                    activated,
                    leak * dt * 30.0,
                );
            }
            std::mem::swap(
                &mut self.global_brain.outputs_prev,
                &mut self.global_brain.outputs_next,
            );

            for part_index in 0..self.parts.len() {
                let node_index = self.parts[part_index].node_index;
                let brain_gene = self
                    .genome
                    .graph
                    .nodes
                    .get(node_index)
                    .map(|node| node.brain.clone())
                    .unwrap_or_else(default_local_brain_gene);
                let sensors = self.local_sensor_vector(part_index, sim_time);
                if part_index >= self.local_brains.len() {
                    continue;
                }
                let prev_outputs = self.local_brains[part_index].outputs_prev.clone();
                let next_len = self.local_brains[part_index].outputs_next.len();
                for neuron_index in 0..next_len {
                    let Some(neuron_gene) = brain_gene.neurons.get(neuron_index).cloned() else {
                        self.local_brains[part_index].outputs_next[neuron_index] = 0.0;
                        continue;
                    };
                    let mut signal = neuron_gene.bias;
                    signal += dot_weights(&neuron_gene.input_weights, &sensors);
                    signal += dot_weights(&neuron_gene.recurrent_weights, &prev_outputs);
                    signal +=
                        dot_weights(&neuron_gene.global_weights, &self.global_brain.outputs_prev);
                    let activated = apply_neural_activation(neuron_gene.activation, signal);
                    let leak = clamp(neuron_gene.leak, 0.05, 1.0);
                    self.local_brains[part_index].outputs_next[neuron_index] =
                        lerp(prev_outputs[neuron_index], activated, leak * dt * 30.0);
                }
            }
            for part_index in 0..self.local_brains.len() {
                if let Some(brain) = self.local_brains.get_mut(part_index) {
                    std::mem::swap(&mut brain.outputs_prev, &mut brain.outputs_next);
                }
            }
        }
    }

    fn set_surface_friction(&mut self, friction: f32) {
        if let Some(ground) = self.colliders.get_mut(self.ground_collider) {
            ground.set_friction(friction);
        }
        for part in &self.parts {
            let Some(body) = self.bodies.get(part.body) else {
                continue;
            };
            for collider_handle in body.colliders() {
                if let Some(collider) = self.colliders.get_mut(*collider_handle) {
                    collider.set_friction(friction);
                }
            }
        }
    }

    fn joint_axis_signal(
        &self,
        effector: &JointEffectorGene,
        local_outputs: &[f32],
        global_outputs: &[f32],
    ) -> f32 {
        let signal = effector.bias
            + dot_weights(&effector.local_weights, local_outputs)
            + dot_weights(&effector.global_weights, global_outputs);
        clamp(signal.tanh() * effector.gain, -1.2, 1.2)
    }

    fn step(&mut self) -> Result<(), String> {
        let dt = self.integration_parameters.dt;
        let sim_time = self.elapsed;
        if self.startup_invalid {
            self.elapsed += dt;
            return Ok(());
        }

        let mut can_actuate = sim_time >= SETTLE_SECONDS;
        if can_actuate
            && self.require_settled_before_actuation
            && !self.actuation_started
            && let Some(torso) = self.bodies.get(self.torso_handle)
        {
            let linear_speed = torso.linvel().norm();
            let angular_speed = torso.angvel().norm();
            if linear_speed > STARTUP_INVALID_LINEAR_SPEED
                || angular_speed > STARTUP_INVALID_ANGULAR_SPEED
            {
                self.startup_invalid = true;
            }
            if linear_speed <= FIXED_PRESET_SETTLE_LINEAR_SPEED_MAX
                && angular_speed <= FIXED_PRESET_SETTLE_ANGULAR_SPEED_MAX
            {
                self.settled_time_before_actuation += dt;
            } else {
                self.settled_time_before_actuation = 0.0;
            }

            let max_wait_elapsed =
                sim_time >= SETTLE_SECONDS + FIXED_PRESET_SETTLE_MAX_EXTRA_SECONDS;
            if self.settled_time_before_actuation >= FIXED_PRESET_SETTLE_MIN_STABLE_SECONDS
                || max_wait_elapsed
            {
                self.actuation_started = true;
            }
            can_actuate = self.actuation_started;
        }
        if self.startup_invalid {
            self.elapsed += dt;
            return Ok(());
        }
        if can_actuate && self.surface_friction_is_passive {
            self.set_surface_friction(ACTIVE_SURFACE_FRICTION);
            self.surface_friction_is_passive = false;
        }

        if can_actuate {
            self.update_brains(sim_time, dt);
            let mut energy_step = 0.0;
            for controller in &self.controllers {
                let brain_gene = self
                    .genome
                    .graph
                    .nodes
                    .get(controller.node_index)
                    .map(|node| node.brain.clone())
                    .unwrap_or_else(default_local_brain_gene);
                let local_outputs = self
                    .local_brains
                    .get(controller.child_part_index)
                    .map(|brain| brain.outputs_prev.as_slice())
                    .unwrap_or(&[]);
                let signal_x = self.joint_axis_signal(
                    &brain_gene.effector_x,
                    local_outputs,
                    &self.global_brain.outputs_prev,
                );
                let (signal_y, signal_z) = if matches!(controller.joint_type, JointTypeGene::Ball) {
                    (
                        self.joint_axis_signal(
                            &brain_gene.effector_y,
                            local_outputs,
                            &self.global_brain.outputs_prev,
                        ),
                        self.joint_axis_signal(
                            &brain_gene.effector_z,
                            local_outputs,
                            &self.global_brain.outputs_prev,
                        ),
                    )
                } else {
                    (0.0, 0.0)
                };
                let target_x = clamp(
                    signal_x * controller.limit_x,
                    -controller.limit_x,
                    controller.limit_x,
                );
                if let Some(joint) = self.impulse_joints.get_mut(controller.joint, true) {
                    joint.data.set_motor_position(
                        JointAxis::AngX,
                        target_x,
                        controller.stiffness_x,
                        JOINT_MOTOR_RESPONSE,
                    );
                    joint
                        .data
                        .set_motor_max_force(JointAxis::AngX, controller.torque_x);

                    energy_step +=
                        (target_x.abs() * controller.stiffness_x).min(controller.torque_x) * dt;
                    if matches!(controller.joint_type, JointTypeGene::Ball) {
                        let target_y = clamp(
                            signal_y * controller.limit_y,
                            -controller.limit_y,
                            controller.limit_y,
                        );
                        let target_z = clamp(
                            signal_z * controller.limit_z,
                            -controller.limit_z,
                            controller.limit_z,
                        );
                        joint.data.set_motor_position(
                            JointAxis::AngY,
                            target_y,
                            controller.stiffness_y,
                            JOINT_MOTOR_RESPONSE,
                        );
                        joint
                            .data
                            .set_motor_max_force(JointAxis::AngY, controller.torque_y);
                        joint.data.set_motor_position(
                            JointAxis::AngZ,
                            target_z,
                            controller.stiffness_z,
                            JOINT_MOTOR_RESPONSE,
                        );
                        joint
                            .data
                            .set_motor_max_force(JointAxis::AngZ, controller.torque_z);
                        energy_step +=
                            (target_y.abs() * controller.stiffness_y).min(controller.torque_y) * dt;
                        energy_step +=
                            (target_z.abs() * controller.stiffness_z).min(controller.torque_z) * dt;
                    }
                }
            }
            self.metrics.add_energy(energy_step);
        }

        self.pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            &(),
            &(),
        );

        let mut passive_instability = false;
        for part in &self.parts {
            if let Some(body) = self.bodies.get_mut(part.body) {
                let av = *body.angvel();
                let av_len = av.norm();
                let lv = *body.linvel();
                let lv_len = lv.norm();
                if !can_actuate
                    && (!av_len.is_finite()
                        || !lv_len.is_finite()
                        || av_len > STARTUP_INVALID_ANGULAR_SPEED
                        || lv_len > STARTUP_INVALID_LINEAR_SPEED)
                {
                    passive_instability = true;
                }
                if !av_len.is_finite() || !lv_len.is_finite() {
                    body.set_angvel(vector![0.0, 0.0, 0.0], true);
                    body.set_linvel(vector![0.0, 0.0, 0.0], true);
                    continue;
                }

                // Apply continuous quadratic drag so high-speed spikes decay smoothly
                // instead of only being chopped by the emergency clamp.
                let angular_drag =
                    1.0 / (1.0 + QUADRATIC_ANGULAR_DRAG_COEFF * av_len * av_len * dt);
                let linear_drag = 1.0 / (1.0 + QUADRATIC_LINEAR_DRAG_COEFF * lv_len * lv_len * dt);
                let mut next_av = av * angular_drag;
                let mut next_lv = lv * linear_drag;

                let next_av_len = next_av.norm();
                if next_av_len > EMERGENCY_MAX_BODY_ANGULAR_SPEED {
                    next_av *= EMERGENCY_MAX_BODY_ANGULAR_SPEED / next_av_len;
                }
                let next_lv_len = next_lv.norm();
                if next_lv_len > EMERGENCY_MAX_BODY_LINEAR_SPEED {
                    next_lv *= EMERGENCY_MAX_BODY_LINEAR_SPEED / next_lv_len;
                }
                body.set_angvel(next_av, true);
                body.set_linvel(next_lv, true);
            }
        }
        if !can_actuate && passive_instability {
            self.startup_invalid = true;
            self.elapsed += dt;
            return Ok(());
        }

        let torso = self
            .bodies
            .get(self.torso_handle)
            .ok_or_else(|| "missing torso during simulation step".to_string())?;
        self.metrics.update(
            *torso.translation(),
            *torso.rotation(),
            *torso.angvel(),
            dt,
            self.duration,
        );
        self.elapsed += dt;
        Ok(())
    }
}

#[derive(Clone)]
struct AppState {
    sim_slots: Arc<Semaphore>,
    sim_worker_limit: usize,
    evolution: Arc<EvolutionController>,
    satellite_pool: Arc<SatellitePool>,
}

impl AppState {
    fn new() -> Self {
        let sim_worker_limit = resolve_sim_worker_limit();
        let evolution = EvolutionController::new();
        let satellite_pool = Arc::new(SatellitePool::new());
        start_evolution_worker(evolution.clone(), satellite_pool.clone(), sim_worker_limit);
        Self {
            sim_slots: Arc::new(Semaphore::new(sim_worker_limit)),
            sim_worker_limit,
            evolution,
            satellite_pool,
        }
    }

    async fn acquire_sim_slot(&self) -> Result<OwnedSemaphorePermit, String> {
        self.sim_slots
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| "simulation worker queue closed".to_string())
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EvolutionControlRequest {
    action: String,
    population_size: Option<usize>,
    fast_forward_generations: Option<usize>,
    run_speed: Option<f32>,
    morphology_mode: Option<EvolutionMorphologyMode>,
    morphology_preset: Option<MorphologyPreset>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum EvolutionMorphologyMode {
    Random,
    FixedPreset,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum MorphologyPreset {
    Spider4x2,
}

#[derive(Clone, Copy, Debug)]
struct PresetConstraintProfile {
    lock_topology: bool,
    lock_joint_types: bool,
    lock_joint_limits: bool,
    lock_segment_dynamics: bool,
    lock_controls: bool,
    lock_visual_hue: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum InjectMutationMode {
    None,
    Light,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EvolutionGenomeImportRequest {
    genome: Option<Genome>,
    genomes: Option<Vec<Genome>>,
    #[serde(default)]
    mutation_mode: Option<InjectMutationMode>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EvolutionGenomeImportResponse {
    added_count: usize,
    queued_count: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CheckpointSaveRequest {
    name: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CheckpointLoadRequest {
    id: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CheckpointSaveResponse {
    id: String,
    path: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CheckpointLoadResponse {
    id: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CheckpointSummary {
    id: String,
    created_at_unix_ms: u128,
    generation: usize,
    best_ever_score: f32,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CheckpointListResponse {
    checkpoints: Vec<CheckpointSummary>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerationFitnessSummary {
    generation: usize,
    best_fitness: f32,
    attempt_fitnesses: Vec<f32>,
    #[serde(default)]
    invalid_startup_attempts: usize,
    #[serde(default)]
    invalid_startup_attempt_rate: f32,
    #[serde(default)]
    invalid_startup_trials: usize,
    #[serde(default)]
    invalid_startup_trial_rate: f32,
    #[serde(default)]
    total_trials: usize,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EvolutionFitnessHistoryResponse {
    history: Vec<GenerationFitnessSummary>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerationDistributionStats {
    min: f32,
    p50: f32,
    p90: f32,
    max: f32,
    std: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerationFitnessStats {
    best: f32,
    p50: f32,
    p90: f32,
    std: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerationSelectionStats {
    mean: f32,
    p90: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerationDiversityStats {
    novelty_mean: f32,
    novelty_p90: f32,
    local_competition_mean: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerationDescriptorStats {
    centroid: [f32; 5],
    spread: [f32; 5],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerationBreedingStats {
    mutation_rate: f32,
    random_inject_chance: f32,
    injected_genomes: usize,
    elite_kept: usize,
    #[serde(default)]
    holdout_best_fitness: f32,
    #[serde(default)]
    holdout_gap: f32,
    #[serde(default)]
    anneal_factor: f32,
    #[serde(default)]
    elite_consistency: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TopologyProfile {
    generation: usize,
    attempt: usize,
    fitness: f32,
    selection_score: f32,
    enabled_limb_count: usize,
    segment_count_histogram: [usize; MAX_SEGMENTS_PER_LIMB],
    mean_segment_length: f32,
    mean_segment_mass: f32,
    mean_control_amp: f32,
    mean_control_freq: f32,
    #[serde(default)]
    ball_joint_ratio: f32,
    descriptor: [f32; 5],
    topology_fingerprint: String,
    #[serde(default)]
    coarse_topology_key: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerationTopologyDiagnostics {
    winner: TopologyProfile,
    best_n: Vec<TopologyProfile>,
    distinct_fingerprint_count: usize,
    #[serde(default)]
    distinct_coarse_fingerprint_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerationPerformanceSummary {
    generation: usize,
    fitness: GenerationFitnessStats,
    selection: GenerationSelectionStats,
    diversity: GenerationDiversityStats,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    descriptor: Option<GenerationDescriptorStats>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    topology: Option<GenerationTopologyDiagnostics>,
    breeding: GenerationBreedingStats,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EvolutionPerformanceQuery {
    window_generations: Option<usize>,
    stride: Option<usize>,
    include_param_stats: Option<bool>,
    include_descriptors: Option<bool>,
    include_topology: Option<bool>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EvolutionPerformanceRun {
    generation: usize,
    population_size: usize,
    trial_count: usize,
    run_speed: f32,
    paused: bool,
    morphology_mode: EvolutionMorphologyMode,
    morphology_preset: MorphologyPreset,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EvolutionPerformanceWindow {
    from_generation: usize,
    to_generation: usize,
    count: usize,
    stride: usize,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EvolutionPerformanceTrends {
    best_fitness_slope: f32,
    median_fitness_slope: f32,
    stagnation_generations: usize,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct LearnedParameterSummary {
    name: String,
    bounds: [f32; 2],
    population: GenerationDistributionStats,
    champion: f32,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EvolutionPerformanceResponse {
    run: EvolutionPerformanceRun,
    window: EvolutionPerformanceWindow,
    trends: EvolutionPerformanceTrends,
    generations: Vec<GenerationPerformanceSummary>,
    learned_params: Vec<LearnedParameterSummary>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EvolutionConvergenceSignal {
    name: String,
    state: String,
    std: f32,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EvolutionMutationPressure {
    current_rate: f32,
    at_lower_clamp: bool,
    at_upper_clamp: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EvolutionPerformanceSummaryResponse {
    generation: usize,
    best_ever_fitness: f32,
    recent_best_fitness: f32,
    stagnation_generations: usize,
    morphology_mode: EvolutionMorphologyMode,
    morphology_preset: MorphologyPreset,
    diversity_state: String,
    mutation_pressure: EvolutionMutationPressure,
    convergence: Vec<EvolutionConvergenceSignal>,
    signals: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    latest_topology: Option<TopologyProfile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    best_ever_topology: Option<TopologyProfile>,
    best_n_topologies: Vec<TopologyProfile>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EvolutionPerformanceDiagnoseResponse {
    generation: usize,
    timestamp_unix_ms: u128,
    states: EvolutionDiagnosisStates,
    metrics: EvolutionDiagnosisMetrics,
    topology: EvolutionDiagnosisTopology,
    findings: Vec<EvolutionDiagnosisFinding>,
    recommended_actions: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EvolutionDiagnosisStates {
    plateau_state: String,
    volatility_state: String,
    novelty_state: String,
    trend_state: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EvolutionDiagnosisMetrics {
    best_ever_fitness: f32,
    recent_best_fitness: f32,
    stagnation_generations: usize,
    best_fitness_slope: f32,
    median_fitness_slope: f32,
    last_recent_best_std: f32,
    last_recent_best_min: f32,
    last_recent_best_max: f32,
    last_recent_novelty_mean: f32,
    last_recent_novelty_min: f32,
    last_recent_novelty_max: f32,
    current_mutation_rate: f32,
    mutation_at_lower_clamp: bool,
    mutation_at_upper_clamp: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EvolutionDiagnosisTopology {
    latest_distinct_fingerprint_count: usize,
    latest_population_size: usize,
    distinct_fingerprint_ratio: f32,
    latest_distinct_coarse_fingerprint_count: usize,
    distinct_coarse_fingerprint_ratio: f32,
    top_enabled_limb_counts: Vec<TopologyLimbCountStat>,
    representative_topologies: Vec<TopologyProfile>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct TopologyLimbCountStat {
    enabled_limb_count: usize,
    count: usize,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EvolutionDiagnosisFinding {
    code: String,
    severity: String,
    message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EvolutionStatus {
    #[serde(default = "default_min_population_size")]
    min_population_size: usize,
    #[serde(default = "default_max_population_size")]
    max_population_size: usize,
    #[serde(default = "default_population_size")]
    default_population_size: usize,
    #[serde(default = "default_trials_per_candidate")]
    default_trial_count: usize,
    #[serde(default = "default_max_trial_count")]
    max_trial_count: usize,
    #[serde(default = "default_generation_seconds")]
    default_generation_seconds: f32,
    #[serde(default = "default_min_run_speed")]
    min_run_speed: f32,
    #[serde(default = "default_max_run_speed")]
    max_run_speed: f32,
    generation: usize,
    population_size: usize,
    pending_population_size: usize,
    current_attempt_index: usize,
    current_trial_index: usize,
    trial_count: usize,
    best_ever_score: f32,
    current_score: f32,
    paused: bool,
    #[serde(default = "default_run_speed")]
    run_speed: f32,
    fast_forward_remaining: usize,
    fast_forward_active: bool,
    injection_queue_count: usize,
    #[serde(default = "default_morphology_mode")]
    morphology_mode: EvolutionMorphologyMode,
    #[serde(default = "default_morphology_preset")]
    morphology_preset: MorphologyPreset,
    #[serde(default)]
    connected_satellites: Vec<String>,
    #[serde(default)]
    latest_invalid_startup_attempts: usize,
    #[serde(default)]
    latest_invalid_startup_attempt_rate: f32,
    #[serde(default)]
    latest_invalid_startup_trials: usize,
    #[serde(default)]
    latest_invalid_startup_trial_rate: f32,
    current_genome: Option<Genome>,
    best_genome: Option<Genome>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EvolutionInjection {
    genome: Genome,
    mutation_mode: InjectMutationMode,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EvolutionRuntimeSnapshot {
    status: EvolutionStatus,
    batch_genomes: Vec<Genome>,
    batch_results: Vec<EvolutionCandidate>,
    attempt_trials: Vec<Vec<TrialResult>>,
    trial_seeds: Vec<u64>,
    current_attempt_index: usize,
    current_trial_index: usize,
    novelty_archive: Vec<NoveltyEntry>,
    injection_queue: Vec<EvolutionInjection>,
    #[serde(default)]
    fitness_history: Vec<GenerationFitnessSummary>,
    #[serde(default)]
    performance_history: Vec<GenerationPerformanceSummary>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EvolutionCheckpointFile {
    version: u32,
    id: String,
    created_at_unix_ms: u128,
    snapshot: EvolutionRuntimeSnapshot,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum EvolutionStreamEvent {
    Status {
        status: EvolutionStatus,
    },
    GenerationSummary {
        summary: GenerationFitnessSummary,
    },
    TrialStarted {
        genome: Genome,
        part_sizes: Vec<[f32; 3]>,
    },
    Snapshot {
        frame: SnapshotFrame,
    },
    TrialComplete {
        result: TrialResult,
    },
    Error {
        message: String,
    },
}

#[derive(Clone, Debug)]
struct EvolutionCommandState {
    paused: bool,
    restart_requested: bool,
    pending_population_size: usize,
    run_speed: f32,
    fast_forward_remaining: usize,
    morphology_mode: EvolutionMorphologyMode,
    morphology_preset: MorphologyPreset,
    injection_queue: VecDeque<EvolutionInjection>,
    pending_loaded_checkpoint: Option<EvolutionRuntimeSnapshot>,
}

#[derive(Default, Clone, Debug)]
struct EvolutionViewState {
    genome: Option<Genome>,
    part_sizes: Vec<[f32; 3]>,
    frames: VecDeque<SnapshotFrame>,
    trial_result: Option<TrialResult>,
}

#[derive(Clone, Debug)]
struct EvolutionSharedState {
    status: EvolutionStatus,
    view: EvolutionViewState,
    runtime_snapshot: Option<EvolutionRuntimeSnapshot>,
    fitness_history: Vec<GenerationFitnessSummary>,
    performance_history: Vec<GenerationPerformanceSummary>,
}

#[derive(Clone)]
struct EvolutionController {
    commands: Arc<Mutex<EvolutionCommandState>>,
    shared: Arc<Mutex<EvolutionSharedState>>,
    events: broadcast::Sender<EvolutionStreamEvent>,
}

impl EvolutionController {
    fn new() -> Arc<Self> {
        let initial_population_size = DEFAULT_POPULATION_SIZE;
        let (initial_morphology_mode, initial_morphology_preset) =
            resolve_initial_morphology_override();
        let initial_status = EvolutionStatus {
            min_population_size: MIN_POPULATION_SIZE,
            max_population_size: MAX_POPULATION_SIZE,
            default_population_size: DEFAULT_POPULATION_SIZE,
            default_trial_count: TRIALS_PER_CANDIDATE,
            max_trial_count: TRIALS_PER_CANDIDATE,
            default_generation_seconds: DEFAULT_GENERATION_SECONDS,
            min_run_speed: default_min_run_speed(),
            max_run_speed: default_max_run_speed(),
            generation: 1,
            population_size: initial_population_size,
            pending_population_size: initial_population_size,
            current_attempt_index: 0,
            current_trial_index: 0,
            trial_count: TRIALS_PER_CANDIDATE,
            best_ever_score: 0.0,
            current_score: 0.0,
            paused: false,
            run_speed: 1.0,
            fast_forward_remaining: 0,
            fast_forward_active: false,
            injection_queue_count: 0,
            morphology_mode: initial_morphology_mode,
            morphology_preset: initial_morphology_preset,
            connected_satellites: Vec::new(),
            latest_invalid_startup_attempts: 0,
            latest_invalid_startup_attempt_rate: 0.0,
            latest_invalid_startup_trials: 0,
            latest_invalid_startup_trial_rate: 0.0,
            current_genome: None,
            best_genome: None,
        };
        let (events, _) = broadcast::channel(256);
        Arc::new(Self {
            commands: Arc::new(Mutex::new(EvolutionCommandState {
                paused: false,
                restart_requested: false,
                pending_population_size: initial_population_size,
                run_speed: 1.0,
                fast_forward_remaining: 0,
                morphology_mode: initial_morphology_mode,
                morphology_preset: initial_morphology_preset,
                injection_queue: VecDeque::new(),
                pending_loaded_checkpoint: None,
            })),
            shared: Arc::new(Mutex::new(EvolutionSharedState {
                status: initial_status,
                view: EvolutionViewState::default(),
                runtime_snapshot: None,
                fitness_history: Vec::new(),
                performance_history: Vec::new(),
            })),
            events,
        })
    }

    fn subscribe(&self) -> broadcast::Receiver<EvolutionStreamEvent> {
        self.events.subscribe()
    }

    fn command_snapshot(
        &self,
    ) -> (
        bool,
        bool,
        usize,
        f32,
        usize,
        EvolutionMorphologyMode,
        MorphologyPreset,
    ) {
        let mut commands = self
            .commands
            .lock()
            .expect("evolution command mutex poisoned");
        let restart = commands.restart_requested;
        commands.restart_requested = false;
        (
            commands.paused,
            restart,
            commands.pending_population_size,
            commands.run_speed,
            commands.fast_forward_remaining,
            commands.morphology_mode,
            commands.morphology_preset,
        )
    }

    fn take_pending_loaded_checkpoint(&self) -> Option<EvolutionRuntimeSnapshot> {
        let mut commands = self
            .commands
            .lock()
            .expect("evolution command mutex poisoned");
        commands.pending_loaded_checkpoint.take()
    }

    fn force_restart(&self) {
        let mut commands = self
            .commands
            .lock()
            .expect("evolution command mutex poisoned");
        commands.restart_requested = true;
    }

    fn queue_injections(
        &self,
        genomes: Vec<Genome>,
        mutation_mode: InjectMutationMode,
    ) -> EvolutionGenomeImportResponse {
        let mut commands = self
            .commands
            .lock()
            .expect("evolution command mutex poisoned");
        let before = commands.injection_queue.len();
        for genome in genomes {
            commands.injection_queue.push_back(EvolutionInjection {
                genome,
                mutation_mode: mutation_mode.clone(),
            });
        }
        let queued_count = commands.injection_queue.len();
        let response = EvolutionGenomeImportResponse {
            added_count: queued_count.saturating_sub(before),
            queued_count,
        };
        drop(commands);
        let status = {
            let mut shared = self.shared.lock().expect("evolution shared mutex poisoned");
            shared.status.injection_queue_count = queued_count;
            shared.status.clone()
        };
        self.broadcast_status(status);
        response
    }

    fn take_injections(&self, max_count: usize) -> Vec<EvolutionInjection> {
        let mut commands = self
            .commands
            .lock()
            .expect("evolution command mutex poisoned");
        let mut items = Vec::new();
        for _ in 0..max_count {
            let Some(item) = commands.injection_queue.pop_front() else {
                break;
            };
            items.push(item);
        }
        let remaining = commands.injection_queue.len();
        drop(commands);
        self.update_status(|status| {
            status.injection_queue_count = remaining;
        });
        items
    }

    fn set_pending_loaded_checkpoint(&self, snapshot: EvolutionRuntimeSnapshot) {
        let mut commands = self
            .commands
            .lock()
            .expect("evolution command mutex poisoned");
        commands.pending_loaded_checkpoint = Some(snapshot);
        commands.restart_requested = false;
    }

    fn runtime_snapshot(&self) -> Option<EvolutionRuntimeSnapshot> {
        self.shared
            .lock()
            .expect("evolution shared mutex poisoned")
            .runtime_snapshot
            .clone()
    }

    fn update_runtime_snapshot(&self, snapshot: EvolutionRuntimeSnapshot) {
        let mut shared = self.shared.lock().expect("evolution shared mutex poisoned");
        shared.fitness_history = snapshot.fitness_history.clone();
        shared.performance_history = snapshot.performance_history.clone();
        shared.runtime_snapshot = Some(snapshot);
    }

    fn snapshot_fitness_history(&self) -> Vec<GenerationFitnessSummary> {
        self.shared
            .lock()
            .expect("evolution shared mutex poisoned")
            .fitness_history
            .clone()
    }

    fn set_fitness_history(&self, history: Vec<GenerationFitnessSummary>) {
        let mut shared = self.shared.lock().expect("evolution shared mutex poisoned");
        shared.fitness_history = history;
    }

    fn clear_fitness_history(&self) {
        self.set_fitness_history(Vec::new());
    }

    fn snapshot_performance_history(&self) -> Vec<GenerationPerformanceSummary> {
        self.shared
            .lock()
            .expect("evolution shared mutex poisoned")
            .performance_history
            .clone()
    }

    fn set_performance_history(&self, history: Vec<GenerationPerformanceSummary>) {
        let mut shared = self.shared.lock().expect("evolution shared mutex poisoned");
        shared.performance_history = history;
    }

    fn clear_performance_history(&self) {
        self.set_performance_history(Vec::new());
    }

    fn current_best_genome(&self) -> Option<Genome> {
        self.shared
            .lock()
            .expect("evolution shared mutex poisoned")
            .status
            .best_genome
            .clone()
    }

    fn current_genome(&self) -> Option<Genome> {
        self.shared
            .lock()
            .expect("evolution shared mutex poisoned")
            .status
            .current_genome
            .clone()
    }

    fn injection_queue_snapshot(&self) -> Vec<EvolutionInjection> {
        let commands = self
            .commands
            .lock()
            .expect("evolution command mutex poisoned");
        commands.injection_queue.iter().cloned().collect()
    }

    fn consume_fast_forward_generation(&self) -> usize {
        let remaining = {
            let mut commands = self
                .commands
                .lock()
                .expect("evolution command mutex poisoned");
            if commands.fast_forward_remaining > 0 {
                commands.fast_forward_remaining -= 1;
            }
            commands.fast_forward_remaining
        };
        let status = {
            let mut shared = self.shared.lock().expect("evolution shared mutex poisoned");
            shared.status.fast_forward_remaining = remaining;
            shared.status.fast_forward_active = remaining > 0;
            shared.status.clone()
        };
        self.broadcast_status(status);
        remaining
    }

    fn apply_control(&self, request: EvolutionControlRequest) -> Result<EvolutionStatus, String> {
        let mut commands = self
            .commands
            .lock()
            .expect("evolution command mutex poisoned");
        match request.action.as_str() {
            "pause" => commands.paused = true,
            "resume" => commands.paused = false,
            "toggle_pause" => commands.paused = !commands.paused,
            "restart" => {
                commands.restart_requested = true;
                commands.fast_forward_remaining = 0;
                commands.injection_queue.clear();
            }
            "set_population_size" => {
                let Some(requested) = request.population_size else {
                    return Err("populationSize is required for set_population_size".to_string());
                };
                commands.pending_population_size =
                    requested.clamp(MIN_POPULATION_SIZE, MAX_POPULATION_SIZE);
            }
            "set_run_speed" => {
                let Some(requested) = request.run_speed else {
                    return Err("runSpeed is required for set_run_speed".to_string());
                };
                commands.run_speed = requested.clamp(0.5, 8.0);
            }
            "queue_fast_forward" => {
                let Some(requested) = request.fast_forward_generations else {
                    return Err(
                        "fastForwardGenerations is required for queue_fast_forward".to_string()
                    );
                };
                let capped = requested.min(50_000);
                let before = commands.fast_forward_remaining;
                commands.fast_forward_remaining =
                    commands.fast_forward_remaining.saturating_add(capped);
                commands.paused = false;
                info!(
                    "control queue_fast_forward: requested={}, before={}, after={}",
                    requested, before, commands.fast_forward_remaining
                );
            }
            "stop_fast_forward" => {
                let before = commands.fast_forward_remaining;
                commands.fast_forward_remaining = 0;
                info!("control stop_fast_forward: before={}, after=0", before);
            }
            "set_morphology_mode" => {
                let Some(mode) = request.morphology_mode else {
                    return Err("morphologyMode is required for set_morphology_mode".to_string());
                };
                commands.morphology_mode = mode;
                if let Some(preset) = request.morphology_preset {
                    commands.morphology_preset = preset;
                } else if mode == EvolutionMorphologyMode::FixedPreset {
                    commands.morphology_preset = default_morphology_preset();
                }
                commands.restart_requested = true;
                commands.fast_forward_remaining = 0;
                commands.injection_queue.clear();
            }
            other => return Err(format!("unsupported action '{other}'")),
        }
        let paused = commands.paused;
        let pending_population_size = commands.pending_population_size;
        let run_speed = commands.run_speed;
        let fast_forward_remaining = commands.fast_forward_remaining;
        let morphology_mode = commands.morphology_mode;
        let morphology_preset = commands.morphology_preset;
        let injection_queue_count = commands.injection_queue.len();
        drop(commands);

        let status = {
            let mut shared = self.shared.lock().expect("evolution shared mutex poisoned");
            shared.status.paused = paused;
            shared.status.run_speed = run_speed;
            shared.status.pending_population_size = pending_population_size;
            shared.status.fast_forward_remaining = fast_forward_remaining;
            shared.status.fast_forward_active = fast_forward_remaining > 0;
            shared.status.injection_queue_count = injection_queue_count;
            shared.status.morphology_mode = morphology_mode;
            shared.status.morphology_preset = morphology_preset;
            shared.status.clone()
        };
        self.broadcast_status(status.clone());
        Ok(status)
    }

    fn snapshot_status(&self) -> EvolutionStatus {
        self.shared
            .lock()
            .expect("evolution shared mutex poisoned")
            .status
            .clone()
    }

    fn snapshot_view(&self) -> EvolutionViewState {
        self.shared
            .lock()
            .expect("evolution shared mutex poisoned")
            .view
            .clone()
    }

    fn update_status<F>(&self, mutator: F)
    where
        F: FnOnce(&mut EvolutionStatus),
    {
        let status = {
            let mut shared = self.shared.lock().expect("evolution shared mutex poisoned");
            mutator(&mut shared.status);
            shared.status.clone()
        };
        self.broadcast_status(status);
    }

    fn broadcast_status(&self, status: EvolutionStatus) {
        let _ = self.events.send(EvolutionStreamEvent::Status { status });
    }

    fn emit_generation_summary(&self, summary: GenerationFitnessSummary) {
        let status = {
            let mut shared = self.shared.lock().expect("evolution shared mutex poisoned");
            let append = match shared.fitness_history.last() {
                None => true,
                Some(last) if summary.generation > last.generation => true,
                Some(last) if summary.generation == last.generation => false,
                Some(_) => {
                    shared.fitness_history.clear();
                    true
                }
            };
            if append {
                shared.fitness_history.push(summary.clone());
            } else if let Some(last) = shared.fitness_history.last_mut() {
                *last = summary.clone();
            }
            if shared.fitness_history.len() > MAX_FITNESS_HISTORY_POINTS {
                let drop_count = shared.fitness_history.len() - MAX_FITNESS_HISTORY_POINTS;
                shared.fitness_history.drain(0..drop_count);
            }
            shared.status.latest_invalid_startup_attempts = summary.invalid_startup_attempts;
            shared.status.latest_invalid_startup_attempt_rate = summary.invalid_startup_attempt_rate;
            shared.status.latest_invalid_startup_trials = summary.invalid_startup_trials;
            shared.status.latest_invalid_startup_trial_rate = summary.invalid_startup_trial_rate;
            shared.status.clone()
        };
        self.broadcast_status(status);
        info!(
            "generation={} startup-invalid rejects: attempts={}/{} ({:.1}%), trials={}/{} ({:.1}%)",
            summary.generation,
            summary.invalid_startup_attempts,
            summary.attempt_fitnesses.len(),
            summary.invalid_startup_attempt_rate * 100.0,
            summary.invalid_startup_trials,
            summary.total_trials,
            summary.invalid_startup_trial_rate * 100.0
        );
        let _ = self
            .events
            .send(EvolutionStreamEvent::GenerationSummary { summary });
    }

    fn emit_generation_performance(&self, summary: GenerationPerformanceSummary) {
        let mut shared = self.shared.lock().expect("evolution shared mutex poisoned");
        let append = match shared.performance_history.last() {
            None => true,
            Some(last) if summary.generation > last.generation => true,
            Some(last) if summary.generation == last.generation => false,
            Some(_) => {
                shared.performance_history.clear();
                true
            }
        };
        if append {
            shared.performance_history.push(summary);
        } else if let Some(last) = shared.performance_history.last_mut() {
            *last = summary;
        }
        if shared.performance_history.len() > MAX_PERFORMANCE_HISTORY_POINTS {
            let drop_count = shared.performance_history.len() - MAX_PERFORMANCE_HISTORY_POINTS;
            shared.performance_history.drain(0..drop_count);
        }
    }

    fn start_trial(&self, genome: Genome, part_sizes: Vec<[f32; 3]>) {
        {
            let mut shared = self.shared.lock().expect("evolution shared mutex poisoned");
            shared.view.genome = Some(genome.clone());
            shared.view.part_sizes = part_sizes.clone();
            shared.view.frames.clear();
            shared.view.trial_result = None;
            shared.status.current_genome = Some(genome.clone());
            shared.status.current_score = 0.0;
        }
        let _ = self
            .events
            .send(EvolutionStreamEvent::TrialStarted { genome, part_sizes });
    }

    fn push_snapshot(&self, frame: SnapshotFrame) {
        {
            let mut shared = self.shared.lock().expect("evolution shared mutex poisoned");
            shared.view.frames.push_back(frame.clone());
            while shared.view.frames.len() > EVOLUTION_VIEW_FRAME_LIMIT {
                shared.view.frames.pop_front();
            }
            shared.status.current_score = frame.score.max(shared.status.current_score);
        }
        let _ = self.events.send(EvolutionStreamEvent::Snapshot { frame });
    }

    fn complete_trial(&self, result: TrialResult) {
        {
            let mut shared = self.shared.lock().expect("evolution shared mutex poisoned");
            shared.view.trial_result = Some(result.clone());
            shared.status.current_score = result.fitness.max(shared.status.current_score);
            if result.fitness > shared.status.best_ever_score {
                shared.status.best_ever_score = result.fitness;
                if let Some(genome) = shared.status.current_genome.clone() {
                    shared.status.best_genome = Some(genome);
                }
            }
        }
        let _ = self
            .events
            .send(EvolutionStreamEvent::TrialComplete { result });
    }

    fn emit_error(&self, message: String) {
        let _ = self.events.send(EvolutionStreamEvent::Error { message });
    }

    fn set_connected_satellites(&self, satellites: Vec<String>) {
        let status = {
            let mut shared = self.shared.lock().expect("evolution shared mutex poisoned");
            if shared.status.connected_satellites == satellites {
                None
            } else {
                shared.status.connected_satellites = satellites;
                Some(shared.status.clone())
            }
        };
        if let Some(status) = status {
            self.broadcast_status(status);
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct EvolutionCandidate {
    genome: Genome,
    fitness: f32,
    selection_score: f32,
    descriptor: [f32; 5],
    novelty: f32,
    novelty_norm: f32,
    quality_norm: f32,
    local_competition: f32,
    attempt: usize,
    #[serde(default = "default_trials_per_candidate")]
    trial_count: usize,
    #[serde(default)]
    invalid_startup_trials: usize,
    #[serde(default)]
    all_trials_invalid_startup: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NoveltyEntry {
    descriptor: [f32; 5],
    fitness: f32,
}

fn start_evolution_worker(
    controller: Arc<EvolutionController>,
    satellite_pool: Arc<SatellitePool>,
    sim_worker_limit: usize,
) {
    std::thread::spawn(move || {
        let mut rng = SmallRng::seed_from_u64(rand::random::<u64>());
        let mut config = TrialConfig {
            duration_seconds: DEFAULT_GENERATION_SECONDS,
            dt: FIXED_SIM_DT,
            snapshot_hz: 30.0,
            motor_power_scale: 1.0,
            fixed_startup: false,
        };

        let mut generation = 1usize;
        let mut population_size = DEFAULT_POPULATION_SIZE;
        let mut best_ever_score = 0.0f32;
        let mut best_genome: Option<Genome> = None;
        let mut novelty_archive: Vec<NoveltyEntry> = Vec::new();

        let mut batch_genomes: Vec<Genome> = Vec::new();
        let mut batch_results: Vec<EvolutionCandidate> = Vec::new();
        let mut attempt_trials: Vec<Vec<TrialResult>> = Vec::new();
        let mut trial_seeds: Vec<u64> = Vec::new();
        let holdout_trial_seeds = build_holdout_trial_seed_set(HOLDOUT_TRIALS_PER_CANDIDATE);
        let mut current_attempt_index = 0usize;
        let mut current_trial_index = 0usize;

        controller.force_restart();
        loop {
            controller.set_connected_satellites(satellite_pool.connected_ids());
            if let Some(loaded) = controller.take_pending_loaded_checkpoint() {
                let loaded_status = loaded.status.clone();
                generation = loaded_status.generation.max(1);
                population_size = loaded_status
                    .population_size
                    .clamp(MIN_POPULATION_SIZE, MAX_POPULATION_SIZE);
                best_ever_score = loaded_status.best_ever_score;
                best_genome = loaded_status.best_genome.clone();
                novelty_archive = loaded.novelty_archive.clone();
                controller.set_fitness_history(loaded.fitness_history.clone());
                controller.set_performance_history(loaded.performance_history.clone());
                batch_genomes = loaded.batch_genomes.clone();
                batch_results = loaded.batch_results.clone();
                attempt_trials = loaded.attempt_trials.clone();
                trial_seeds = loaded.trial_seeds.clone();
                if trial_seeds.len() != TRIALS_PER_CANDIDATE {
                    trial_seeds = build_trial_seed_set(generation, TRIALS_PER_CANDIDATE);
                }
                current_attempt_index = loaded.current_attempt_index.min(batch_genomes.len());
                current_trial_index = loaded
                    .current_trial_index
                    .min(TRIALS_PER_CANDIDATE.saturating_sub(1));
                {
                    let mut commands = controller
                        .commands
                        .lock()
                        .expect("evolution command mutex poisoned");
                    commands.paused = loaded_status.paused;
                    commands.pending_population_size = loaded_status.pending_population_size;
                    commands.run_speed = loaded_status.run_speed.clamp(0.5, 8.0);
                    commands.fast_forward_remaining = loaded_status.fast_forward_remaining;
                    commands.morphology_mode = loaded_status.morphology_mode;
                    commands.morphology_preset = loaded_status.morphology_preset;
                    commands.injection_queue = VecDeque::from(loaded.injection_queue.clone());
                }
                controller.update_status(|status| {
                    *status = loaded_status.clone();
                    status.run_speed = loaded_status.run_speed.clamp(0.5, 8.0);
                    status.current_attempt_index =
                        current_attempt_index.min(population_size.saturating_sub(1));
                    status.current_trial_index = current_trial_index;
                    status.injection_queue_count = loaded.injection_queue.len();
                });
                publish_runtime_snapshot(
                    &controller,
                    &batch_genomes,
                    &batch_results,
                    &attempt_trials,
                    &trial_seeds,
                    current_attempt_index,
                    current_trial_index,
                    &novelty_archive,
                );
                autosave_runtime_snapshot_if_due(&controller);
                info!(
                    "checkpoint loaded into worker: generation={}, population_size={}, fast_forward_remaining={}",
                    generation, population_size, loaded_status.fast_forward_remaining
                );
                continue;
            }

            let (
                paused,
                restart_requested,
                pending_population_size,
                run_speed,
                fast_forward_remaining,
                morphology_mode,
                morphology_preset,
            ) = controller.command_snapshot();
            config.fixed_startup = matches!(morphology_mode, EvolutionMorphologyMode::FixedPreset);
            if restart_requested {
                info!(
                    "evolution restart requested; resetting to generation=1, population_size={}",
                    pending_population_size
                );
                generation = 1;
                best_ever_score = 0.0;
                best_genome = None;
                novelty_archive.clear();
                controller.clear_fitness_history();
                controller.clear_performance_history();
                reset_evolution_batch(
                    &mut rng,
                    pending_population_size,
                    generation,
                    morphology_mode,
                    morphology_preset,
                    &mut population_size,
                    &mut batch_genomes,
                    &mut batch_results,
                    &mut attempt_trials,
                    &mut trial_seeds,
                    &mut current_attempt_index,
                    &mut current_trial_index,
                );
                controller.update_status(|status| {
                    status.generation = generation;
                    status.population_size = population_size;
                    status.current_attempt_index = 0;
                    status.current_trial_index = 0;
                    status.current_score = 0.0;
                    status.trial_count = TRIALS_PER_CANDIDATE;
                    status.current_genome = batch_genomes.first().cloned();
                    status.best_ever_score = 0.0;
                    status.best_genome = None;
                    status.fast_forward_remaining = 0;
                    status.fast_forward_active = false;
                    status.injection_queue_count = 0;
                    status.latest_invalid_startup_attempts = 0;
                    status.latest_invalid_startup_attempt_rate = 0.0;
                    status.latest_invalid_startup_trials = 0;
                    status.latest_invalid_startup_trial_rate = 0.0;
                });
                publish_runtime_snapshot(
                    &controller,
                    &batch_genomes,
                    &batch_results,
                    &attempt_trials,
                    &trial_seeds,
                    current_attempt_index,
                    current_trial_index,
                    &novelty_archive,
                );
                continue;
            }

            if fast_forward_remaining > 0 {
                controller.update_status(|status| {
                    status.fast_forward_remaining = fast_forward_remaining;
                    status.fast_forward_active = true;
                });
            }

            if paused && fast_forward_remaining == 0 {
                std::thread::sleep(Duration::from_millis(50));
                continue;
            }

            if batch_genomes.is_empty() {
                reset_evolution_batch(
                    &mut rng,
                    pending_population_size,
                    generation,
                    morphology_mode,
                    morphology_preset,
                    &mut population_size,
                    &mut batch_genomes,
                    &mut batch_results,
                    &mut attempt_trials,
                    &mut trial_seeds,
                    &mut current_attempt_index,
                    &mut current_trial_index,
                );
                controller.update_status(|status| {
                    status.generation = generation;
                    status.population_size = population_size;
                    status.current_attempt_index = 0;
                    status.current_trial_index = 0;
                    status.current_score = 0.0;
                    status.trial_count = TRIALS_PER_CANDIDATE;
                    status.current_genome = batch_genomes.first().cloned();
                });
                publish_runtime_snapshot(
                    &controller,
                    &batch_genomes,
                    &batch_results,
                    &attempt_trials,
                    &trial_seeds,
                    current_attempt_index,
                    current_trial_index,
                    &novelty_archive,
                );
                info!(
                    "evolution batch seeded: generation={}, population_size={}, trials_per_candidate={}",
                    generation, population_size, TRIALS_PER_CANDIDATE
                );
                continue;
            }

            if current_attempt_index >= batch_genomes.len() {
                if batch_results.is_empty() {
                    reset_evolution_batch(
                        &mut rng,
                        pending_population_size,
                        generation,
                        morphology_mode,
                        morphology_preset,
                        &mut population_size,
                        &mut batch_genomes,
                        &mut batch_results,
                        &mut attempt_trials,
                        &mut trial_seeds,
                        &mut current_attempt_index,
                        &mut current_trial_index,
                    );
                    controller.update_status(|status| {
                        status.generation = generation;
                        status.population_size = population_size;
                        status.current_attempt_index = 0;
                        status.current_trial_index = 0;
                        status.current_score = 0.0;
                        status.trial_count = TRIALS_PER_CANDIDATE;
                        status.current_genome = batch_genomes.first().cloned();
                    });
                    publish_runtime_snapshot(
                        &controller,
                        &batch_genomes,
                        &batch_results,
                        &attempt_trials,
                        &trial_seeds,
                        current_attempt_index,
                        current_trial_index,
                        &novelty_archive,
                    );
                    info!(
                        "evolution batch reseeded: generation={}, population_size={}, trials_per_candidate={}",
                        generation, population_size, TRIALS_PER_CANDIDATE
                    );
                    continue;
                }
                let injected_genomes = dequeue_injected_genomes(
                    &controller,
                    &mut rng,
                    1,
                    morphology_mode,
                    morphology_preset,
                );
                if let Some(summary) = build_generation_fitness_summary(generation, &batch_results)
                {
                    controller.emit_generation_summary(summary);
                }
                let performance_history = controller.snapshot_performance_history();
                let performance = finalize_generation(
                    &mut rng,
                    &mut batch_genomes,
                    &mut batch_results,
                    &mut attempt_trials,
                    &mut trial_seeds,
                    &mut novelty_archive,
                    &mut generation,
                    &mut population_size,
                    &mut current_attempt_index,
                    &mut current_trial_index,
                    &mut best_ever_score,
                    &mut best_genome,
                    pending_population_size,
                    injected_genomes,
                    &holdout_trial_seeds,
                    &config,
                    &performance_history,
                    morphology_mode,
                    morphology_preset,
                );
                if let Some(summary) = performance {
                    controller.emit_generation_performance(summary);
                }
                controller.update_status(|status| {
                    status.generation = generation;
                    status.population_size = population_size;
                    status.current_attempt_index = current_attempt_index;
                    status.current_trial_index = current_trial_index;
                    status.best_ever_score = best_ever_score;
                    status.best_genome = best_genome.clone();
                    status.current_score = 0.0;
                    status.current_genome = batch_genomes.first().cloned();
                    status.fast_forward_remaining = fast_forward_remaining;
                    status.fast_forward_active = fast_forward_remaining > 0;
                });
                publish_runtime_snapshot(
                    &controller,
                    &batch_genomes,
                    &batch_results,
                    &attempt_trials,
                    &trial_seeds,
                    current_attempt_index,
                    current_trial_index,
                    &novelty_archive,
                );
                info!(
                    "generation advanced: generation={}, population_size={}, best_ever_score={:.3}, fast_forward_remaining={}",
                    generation, population_size, best_ever_score, fast_forward_remaining
                );
                continue;
            }

            if fast_forward_remaining > 0 {
                info!(
                    "fast-forward processing generation={}, queued_remaining={}",
                    generation, fast_forward_remaining
                );
                let request = GenerationEvalRequest {
                    genomes: batch_genomes.clone(),
                    seeds: trial_seeds.clone(),
                    duration_seconds: Some(config.duration_seconds),
                    _dt: Some(config.dt),
                    motor_power_scale: Some(config.motor_power_scale),
                };
                let (tx_progress, mut rx_progress) = mpsc::channel::<GenerationStreamEvent>(256);
                let request_for_stream = request.clone();
                let config_for_stream = config.clone();
                let sat_pool_for_stream = satellite_pool.clone();
                let progress_worker = std::thread::spawn(move || {
                    run_generation_stream(
                        request_for_stream,
                        config_for_stream,
                        tx_progress,
                        sim_worker_limit,
                        sat_pool_for_stream,
                    )
                });
                let mut streamed_results: Option<Vec<GenerationEvalResult>> = None;
                let mut stream_error: Option<String> = None;
                while let Some(event) = rx_progress.blocking_recv() {
                    match event {
                        GenerationStreamEvent::GenerationStarted { trial_count, .. } => {
                            controller.update_status(|status| {
                                status.trial_count = trial_count;
                                status.current_attempt_index = 0;
                                status.current_trial_index = 0;
                            });
                        }
                        GenerationStreamEvent::AttemptTrialStarted {
                            attempt_index,
                            trial_index,
                            trial_count,
                        } => {
                            controller.update_status(|status| {
                                status.current_attempt_index =
                                    attempt_index.min(population_size.saturating_sub(1));
                                status.current_trial_index =
                                    trial_index.min(TRIALS_PER_CANDIDATE.saturating_sub(1));
                                status.trial_count = trial_count.max(1);
                            });
                        }
                        GenerationStreamEvent::AttemptComplete { attempt_index, .. } => {
                            controller.update_status(|status| {
                                status.current_attempt_index =
                                    attempt_index.min(population_size.saturating_sub(1));
                                status.current_trial_index = TRIALS_PER_CANDIDATE.saturating_sub(1);
                            });
                        }
                        GenerationStreamEvent::GenerationComplete { results } => {
                            streamed_results = Some(results);
                        }
                        GenerationStreamEvent::Error { message } => {
                            stream_error = Some(message);
                        }
                    }
                }
                let worker_result = progress_worker
                    .join()
                    .map_err(|_| "fast-forward generation worker panicked".to_string())
                    .and_then(|result| result);
                let final_results = if let Some(message) = stream_error {
                    Err(message)
                } else {
                    worker_result.and_then(|_| {
                        streamed_results.ok_or_else(|| {
                            "fast-forward generation produced no results".to_string()
                        })
                    })
                };
                match final_results {
                    Ok(results) => {
                        if results.len() != batch_genomes.len() {
                            controller.emit_error(
                                "fast-forward generation returned unexpected result count"
                                    .to_string(),
                            );
                            std::thread::sleep(Duration::from_millis(100));
                            continue;
                        }
                        batch_results = results
                            .into_iter()
                            .enumerate()
                            .map(|(attempt, result)| EvolutionCandidate {
                                genome: batch_genomes[attempt].clone(),
                                fitness: result.fitness,
                                selection_score: result.fitness,
                                descriptor: result.descriptor,
                                novelty: 0.0,
                                novelty_norm: 0.0,
                                quality_norm: 0.0,
                                local_competition: 0.0,
                                attempt,
                                trial_count: result.trial_count,
                                invalid_startup_trials: result.invalid_startup_trials,
                                all_trials_invalid_startup: result.all_trials_invalid_startup,
                            })
                            .collect();
                        if let Some(summary) =
                            build_generation_fitness_summary(generation, &batch_results)
                        {
                            controller.emit_generation_summary(summary);
                        }
                        let injected_genomes = dequeue_injected_genomes(
                            &controller,
                            &mut rng,
                            1,
                            morphology_mode,
                            morphology_preset,
                        );
                        let performance_history = controller.snapshot_performance_history();
                        let performance = finalize_generation(
                            &mut rng,
                            &mut batch_genomes,
                            &mut batch_results,
                            &mut attempt_trials,
                            &mut trial_seeds,
                            &mut novelty_archive,
                            &mut generation,
                            &mut population_size,
                            &mut current_attempt_index,
                            &mut current_trial_index,
                            &mut best_ever_score,
                            &mut best_genome,
                            pending_population_size,
                            injected_genomes,
                            &holdout_trial_seeds,
                            &config,
                            &performance_history,
                            morphology_mode,
                            morphology_preset,
                        );
                        if let Some(summary) = performance {
                            controller.emit_generation_performance(summary);
                        }
                        let remaining = controller.consume_fast_forward_generation();
                        controller.update_status(|status| {
                            status.generation = generation;
                            status.population_size = population_size;
                            status.current_attempt_index = current_attempt_index;
                            status.current_trial_index = current_trial_index;
                            status.best_ever_score = best_ever_score;
                            status.best_genome = best_genome.clone();
                            status.current_score = 0.0;
                            status.current_genome = batch_genomes.first().cloned();
                            status.fast_forward_remaining = remaining;
                            status.fast_forward_active = remaining > 0;
                        });
                        publish_runtime_snapshot(
                            &controller,
                            &batch_genomes,
                            &batch_results,
                            &attempt_trials,
                            &trial_seeds,
                            current_attempt_index,
                            current_trial_index,
                            &novelty_archive,
                        );
                        autosave_runtime_snapshot_if_due(&controller);
                        info!(
                            "fast-forward generation complete: generation={}, best_ever_score={:.3}, queued_remaining={}",
                            generation, best_ever_score, remaining
                        );
                    }
                    Err(err) => {
                        controller.emit_error(format!("fast-forward generation failed: {err}"));
                        std::thread::sleep(Duration::from_millis(100));
                    }
                }
                continue;
            }

            let genome = match batch_genomes.get(current_attempt_index) {
                Some(genome) => genome.clone(),
                None => {
                    current_attempt_index = batch_genomes.len();
                    continue;
                }
            };
            let seed = trial_seeds
                .get(current_trial_index)
                .copied()
                .unwrap_or_else(|| {
                    hash_uint32(
                        TRAIN_TRIAL_SEED_BANK_TAG,
                        (current_trial_index + 1) as u32,
                        0x85eb_ca6b,
                    ) as u64
                });
            let mut sim = match TrialSimulator::new(&genome, seed, &config) {
                Ok(sim) => sim,
                Err(err) => {
                    controller.emit_error(format!("failed to start trial: {err}"));
                    std::thread::sleep(Duration::from_millis(100));
                    continue;
                }
            };

            controller.update_status(|status| {
                status.generation = generation;
                status.population_size = population_size;
                status.current_attempt_index = current_attempt_index;
                status.current_trial_index = current_trial_index;
                status.current_score = 0.0;
                status.current_genome = Some(genome.clone());
                status.best_genome = best_genome.clone();
                status.best_ever_score = best_ever_score;
                status.fast_forward_active = false;
            });
            info!(
                "trial started: generation={}, attempt={}/{}, trial={}/{}, seed={}",
                generation,
                current_attempt_index + 1,
                population_size,
                current_trial_index + 1,
                TRIALS_PER_CANDIDATE,
                seed
            );
            controller.start_trial(genome.clone(), sim.part_sizes());
            controller.push_snapshot(sim.current_frame());

            let snapshot_every = ((1.0 / config.dt) / config.snapshot_hz).round().max(1.0) as usize;
            let steps = (config.duration_seconds / config.dt).ceil() as usize;
            let mut aborted = false;
            let mut preempted_for_fast_forward = false;
            let wall_trial_start = Instant::now();

            for step in 0..steps {
                let (paused_now, restart_now, _, _, fast_forward_now, _, _) =
                    controller.command_snapshot();
                if restart_now {
                    controller.force_restart();
                    aborted = true;
                    break;
                }
                if fast_forward_now > 0 {
                    preempted_for_fast_forward = true;
                    aborted = true;
                    break;
                }
                if paused_now {
                    loop {
                        std::thread::sleep(Duration::from_millis(25));
                        let (paused_loop, restart_loop, _, _, fast_forward_loop, _, _) =
                            controller.command_snapshot();
                        if restart_loop {
                            controller.force_restart();
                            aborted = true;
                            break;
                        }
                        if fast_forward_loop > 0 {
                            preempted_for_fast_forward = true;
                            aborted = true;
                            break;
                        }
                        if !paused_loop {
                            break;
                        }
                    }
                    if aborted {
                        break;
                    }
                }
                if let Err(err) = sim.step() {
                    controller.emit_error(format!("trial step failed: {err}"));
                    aborted = true;
                    break;
                }
                if (step + 1) % snapshot_every == 0 || step + 1 == steps {
                    controller.push_snapshot(sim.current_frame());
                }
                let pacing_speed = run_speed.clamp(0.5, 8.0);
                let target_elapsed =
                    Duration::from_secs_f32((step + 1) as f32 * (config.dt / pacing_speed));
                let elapsed = wall_trial_start.elapsed();
                if target_elapsed > elapsed {
                    std::thread::sleep(target_elapsed - elapsed);
                }
            }

            if aborted {
                if preempted_for_fast_forward {
                    info!(
                        "trial preempted for fast-forward: generation={}, attempt={}/{}, trial={}/{}",
                        generation,
                        current_attempt_index + 1,
                        population_size,
                        current_trial_index + 1,
                        TRIALS_PER_CANDIDATE
                    );
                }
                continue;
            }

            let result = sim.final_result();
            controller.complete_trial(result.clone());
            info!(
                "trial complete: generation={}, attempt={}/{}, trial={}/{}, fitness={:.3}, progress={:.3}, upright={:.3}",
                generation,
                current_attempt_index + 1,
                population_size,
                current_trial_index + 1,
                TRIALS_PER_CANDIDATE,
                result.fitness,
                result.metrics.progress,
                result.metrics.upright_avg
            );
            if let Some(trials) = attempt_trials.get_mut(current_attempt_index) {
                trials.push(result);
            }

            if current_trial_index + 1 < TRIALS_PER_CANDIDATE {
                current_trial_index += 1;
                controller.update_status(|status| {
                    status.current_trial_index = current_trial_index;
                    status.current_score = 0.0;
                });
                publish_runtime_snapshot(
                    &controller,
                    &batch_genomes,
                    &batch_results,
                    &attempt_trials,
                    &trial_seeds,
                    current_attempt_index,
                    current_trial_index,
                    &novelty_archive,
                );
                continue;
            }

            let trials = attempt_trials
                .get(current_attempt_index)
                .cloned()
                .unwrap_or_default();
            let summary = summarize_trials(&genome, &trials);
            batch_results.push(EvolutionCandidate {
                genome: genome.clone(),
                fitness: summary.fitness,
                selection_score: summary.fitness,
                descriptor: summary.descriptor,
                novelty: 0.0,
                novelty_norm: 0.0,
                quality_norm: 0.0,
                local_competition: 0.0,
                attempt: current_attempt_index,
                trial_count: summary.trial_count,
                invalid_startup_trials: summary.invalid_startup_trials,
                all_trials_invalid_startup: summary.all_trials_invalid_startup,
            });
            if summary.fitness > best_ever_score {
                best_ever_score = summary.fitness;
                best_genome = Some(genome);
            }
            info!(
                "attempt complete: generation={}, attempt={}/{}, summary_fitness={:.3}, best_ever_score={:.3}",
                generation,
                current_attempt_index + 1,
                population_size,
                summary.fitness,
                best_ever_score
            );

            current_attempt_index += 1;
            current_trial_index = 0;
            controller.update_status(|status| {
                status.current_attempt_index =
                    current_attempt_index.min(population_size.saturating_sub(1));
                status.current_trial_index = 0;
                status.best_ever_score = best_ever_score;
                status.best_genome = best_genome.clone();
                status.current_score = 0.0;
                status.current_genome = batch_genomes.get(current_attempt_index).cloned();
            });
            publish_runtime_snapshot(
                &controller,
                &batch_genomes,
                &batch_results,
                &attempt_trials,
                &trial_seeds,
                current_attempt_index,
                current_trial_index,
                &novelty_archive,
            );
        }
    });
}

fn reset_evolution_batch(
    rng: &mut SmallRng,
    target_population_size: usize,
    generation_index: usize,
    morphology_mode: EvolutionMorphologyMode,
    morphology_preset: MorphologyPreset,
    population_size: &mut usize,
    batch_genomes: &mut Vec<Genome>,
    batch_results: &mut Vec<EvolutionCandidate>,
    attempt_trials: &mut Vec<Vec<TrialResult>>,
    trial_seeds: &mut Vec<u64>,
    current_attempt_index: &mut usize,
    current_trial_index: &mut usize,
) {
    let clamped_size = target_population_size.clamp(MIN_POPULATION_SIZE, MAX_POPULATION_SIZE);
    *population_size = clamped_size;
    *batch_genomes = (0..clamped_size)
        .map(|_| {
            let genome = random_genome(rng);
            apply_morphology_mode(genome, morphology_mode, morphology_preset, rng)
        })
        .collect();
    batch_results.clear();
    *attempt_trials = vec![Vec::new(); clamped_size];
    *trial_seeds = build_trial_seed_set(generation_index, TRIALS_PER_CANDIDATE);
    *current_attempt_index = 0;
    *current_trial_index = 0;
}

fn dequeue_injected_genomes(
    controller: &EvolutionController,
    rng: &mut SmallRng,
    max_count: usize,
    morphology_mode: EvolutionMorphologyMode,
    morphology_preset: MorphologyPreset,
) -> Vec<Genome> {
    controller
        .take_injections(max_count)
        .into_iter()
        .map(|injection| {
            let genome = match injection.mutation_mode {
                InjectMutationMode::None => injection.genome,
                InjectMutationMode::Light => mutate_genome(injection.genome, 0.12, false, rng),
            };
            apply_morphology_mode(genome, morphology_mode, morphology_preset, rng)
        })
        .collect()
}

fn publish_runtime_snapshot(
    controller: &EvolutionController,
    batch_genomes: &[Genome],
    batch_results: &[EvolutionCandidate],
    attempt_trials: &[Vec<TrialResult>],
    trial_seeds: &[u64],
    current_attempt_index: usize,
    current_trial_index: usize,
    novelty_archive: &[NoveltyEntry],
) {
    let snapshot = EvolutionRuntimeSnapshot {
        status: controller.snapshot_status(),
        batch_genomes: batch_genomes.to_vec(),
        batch_results: batch_results.to_vec(),
        attempt_trials: attempt_trials.to_vec(),
        trial_seeds: trial_seeds.to_vec(),
        current_attempt_index,
        current_trial_index,
        novelty_archive: novelty_archive.to_vec(),
        injection_queue: controller.injection_queue_snapshot(),
        fitness_history: controller.snapshot_fitness_history(),
        performance_history: controller.snapshot_performance_history(),
    };
    controller.update_runtime_snapshot(snapshot);
}

fn build_generation_fitness_summary(
    generation: usize,
    batch_results: &[EvolutionCandidate],
) -> Option<GenerationFitnessSummary> {
    if batch_results.is_empty() {
        return None;
    }
    let mut ranked_attempts: Vec<(usize, f32)> = batch_results
        .iter()
        .map(|item| {
            let fitness = if item.fitness.is_finite() {
                item.fitness
            } else {
                0.0
            };
            (item.attempt, fitness)
        })
        .collect();
    ranked_attempts.sort_unstable_by_key(|(attempt, _)| *attempt);
    let attempt_fitnesses: Vec<f32> = ranked_attempts
        .into_iter()
        .map(|(_, fitness)| fitness)
        .collect();
    let attempt_count = batch_results.len();
    let invalid_startup_attempts = batch_results
        .iter()
        .filter(|candidate| candidate.all_trials_invalid_startup)
        .count();
    let invalid_startup_trials = batch_results
        .iter()
        .map(|candidate| candidate.invalid_startup_trials.min(candidate.trial_count))
        .sum::<usize>();
    let total_trials = batch_results
        .iter()
        .map(|candidate| candidate.trial_count)
        .sum::<usize>();
    let best_fitness = attempt_fitnesses
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    Some(GenerationFitnessSummary {
        generation,
        best_fitness: if best_fitness.is_finite() {
            best_fitness
        } else {
            0.0
        },
        attempt_fitnesses,
        invalid_startup_attempts,
        invalid_startup_attempt_rate: invalid_startup_attempts as f32 / attempt_count.max(1) as f32,
        invalid_startup_trials,
        invalid_startup_trial_rate: invalid_startup_trials as f32 / total_trials.max(1) as f32,
        total_trials,
    })
}

fn build_generation_performance_summary(
    generation: usize,
    batch_results: &[EvolutionCandidate],
    mutation_rate: f32,
    random_inject_chance: f32,
    injected_genomes: usize,
    elite_kept: usize,
    holdout_best_fitness: f32,
    holdout_gap: f32,
    anneal_factor: f32,
    elite_consistency: f32,
) -> Option<GenerationPerformanceSummary> {
    if batch_results.is_empty() {
        return None;
    }
    let fitnesses: Vec<f32> = batch_results
        .iter()
        .map(|item| finite_or_zero(item.fitness))
        .collect();
    let selection_scores: Vec<f32> = batch_results
        .iter()
        .map(|item| finite_or_zero(item.selection_score))
        .collect();
    let novelties: Vec<f32> = batch_results
        .iter()
        .map(|item| finite_or_zero(item.novelty))
        .collect();
    let local_competitions: Vec<f32> = batch_results
        .iter()
        .map(|item| finite_or_zero(item.local_competition))
        .collect();
    let best = fitnesses.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut centroid = [0.0f32; 5];
    let mut spread = [0.0f32; 5];
    for axis in 0..5 {
        let axis_values: Vec<f32> = batch_results
            .iter()
            .map(|item| finite_or_zero(item.descriptor[axis]))
            .collect();
        centroid[axis] = mean(&axis_values);
        spread[axis] = std_dev(&axis_values);
    }
    let topology = build_generation_topology_diagnostics(generation, batch_results);
    Some(GenerationPerformanceSummary {
        generation,
        fitness: GenerationFitnessStats {
            best: finite_or_zero(best),
            p50: quantile(&fitnesses, 0.5),
            p90: quantile(&fitnesses, 0.9),
            std: std_dev(&fitnesses),
        },
        selection: GenerationSelectionStats {
            mean: mean(&selection_scores),
            p90: quantile(&selection_scores, 0.9),
        },
        diversity: GenerationDiversityStats {
            novelty_mean: mean(&novelties),
            novelty_p90: quantile(&novelties, 0.9),
            local_competition_mean: mean(&local_competitions),
        },
        descriptor: Some(GenerationDescriptorStats { centroid, spread }),
        topology,
        breeding: GenerationBreedingStats {
            mutation_rate,
            random_inject_chance,
            injected_genomes,
            elite_kept,
            holdout_best_fitness,
            holdout_gap,
            anneal_factor,
            elite_consistency,
        },
    })
}

fn build_generation_topology_diagnostics(
    generation: usize,
    batch_results: &[EvolutionCandidate],
) -> Option<GenerationTopologyDiagnostics> {
    if batch_results.is_empty() {
        return None;
    }
    let mut by_fitness = batch_results.to_vec();
    by_fitness.sort_by(|a, b| {
        b.fitness
            .partial_cmp(&a.fitness)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let winner = build_topology_profile(generation, &by_fitness[0]);
    let best_n = by_fitness
        .iter()
        .take(MAX_GENERATION_TOPOLOGY_CANDIDATES)
        .map(|candidate| build_topology_profile(generation, candidate))
        .collect::<Vec<_>>();
    let distinct_fingerprint_count = batch_results
        .iter()
        .map(|candidate| topology_fingerprint(&candidate.genome))
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let distinct_coarse_fingerprint_count = batch_results
        .iter()
        .map(|candidate| coarse_topology_key(&candidate.genome))
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    Some(GenerationTopologyDiagnostics {
        winner,
        best_n,
        distinct_fingerprint_count,
        distinct_coarse_fingerprint_count,
    })
}

fn finalize_generation(
    rng: &mut SmallRng,
    batch_genomes: &mut Vec<Genome>,
    batch_results: &mut Vec<EvolutionCandidate>,
    attempt_trials: &mut Vec<Vec<TrialResult>>,
    trial_seeds: &mut Vec<u64>,
    novelty_archive: &mut Vec<NoveltyEntry>,
    generation: &mut usize,
    population_size: &mut usize,
    current_attempt_index: &mut usize,
    current_trial_index: &mut usize,
    best_ever_score: &mut f32,
    best_genome: &mut Option<Genome>,
    pending_population_size: usize,
    injected_genomes: Vec<Genome>,
    holdout_trial_seeds: &[u64],
    config: &TrialConfig,
    performance_history: &[GenerationPerformanceSummary],
    morphology_mode: EvolutionMorphologyMode,
    morphology_preset: MorphologyPreset,
) -> Option<GenerationPerformanceSummary> {
    apply_diversity_scores(batch_results, novelty_archive);
    update_novelty_archive(batch_results, novelty_archive);

    let mut ranked_by_fitness = batch_results.clone();
    ranked_by_fitness.sort_by(|a, b| {
        b.fitness
            .partial_cmp(&a.fitness)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranked_for_breeding = batch_results.clone();
    ranked_for_breeding.sort_by(|a, b| {
        b.selection_score
            .partial_cmp(&a.selection_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let Some(top) = ranked_by_fitness.first() else {
        return None;
    };
    if top.fitness > *best_ever_score {
        *best_ever_score = top.fitness;
        *best_genome = Some(top.genome.clone());
    }
    let train_best_fitness = finite_or_zero(top.fitness);
    let holdout_best_fitness = if holdout_trial_seeds.is_empty() {
        train_best_fitness
    } else {
        match evaluate_generation_attempt(0, &top.genome, holdout_trial_seeds, config, |_| Ok(())) {
            Ok(result) => finite_or_zero(result.fitness),
            Err(err) => {
                warn!("holdout evaluation failed: {err}");
                train_best_fitness
            }
        }
    };
    let holdout_gap = (train_best_fitness - holdout_best_fitness).max(0.0);
    let holdout_gap_norm = holdout_gap / train_best_fitness.abs().max(1.0);

    let target_population_size =
        pending_population_size.clamp(MIN_POPULATION_SIZE, MAX_POPULATION_SIZE);
    let mut next_genomes: Vec<Genome> = Vec::with_capacity(target_population_size);
    let mut injected_used = 0usize;
    for genome in injected_genomes.into_iter().take(target_population_size) {
        next_genomes.push(genome);
        injected_used += 1;
    }
    let elite_count = ELITE_COUNT
        .min(ranked_by_fitness.len())
        .min(target_population_size.saturating_sub(next_genomes.len()));
    let mut elite_kept = 0usize;
    if elite_count > 0 && next_genomes.len() < target_population_size {
        next_genomes.push(ranked_by_fitness[0].genome.clone());
        elite_kept += 1;
    }
    if elite_count > 1 && next_genomes.len() < target_population_size {
        let diversity_elite = ranked_for_breeding.first();
        if let Some(candidate) = diversity_elite {
            if candidate.attempt != ranked_by_fitness[0].attempt {
                next_genomes.push(candidate.genome.clone());
                elite_kept += 1;
            } else if ranked_by_fitness.len() > 1 {
                next_genomes.push(ranked_by_fitness[1].genome.clone());
                elite_kept += 1;
            }
        }
    }
    if target_population_size == 1 && next_genomes.is_empty() && !ranked_by_fitness.is_empty() {
        next_genomes.clear();
        next_genomes.push(mutate_genome(
            ranked_by_fitness[0].genome.clone(),
            0.7,
            false,
            rng,
        ));
    }

    let mean_novelty_norm = if ranked_for_breeding.is_empty() {
        0.0
    } else {
        ranked_for_breeding
            .iter()
            .map(|candidate| candidate.novelty_norm)
            .sum::<f32>()
            / ranked_for_breeding.len() as f32
    };
    let elite_fitnesses: Vec<f32> = ranked_by_fitness
        .iter()
        .take(ranked_by_fitness.len().min(8))
        .map(|candidate| finite_or_zero(candidate.fitness))
        .collect();
    let elite_consistency = if elite_fitnesses.len() < 2 {
        1.0
    } else {
        let elite_std = std_dev(&elite_fitnesses);
        let elite_scale = quantile(&elite_fitnesses, 0.5).abs().max(1.0);
        clamp(1.0 - elite_std / elite_scale, 0.0, 1.0)
    };
    let stagnation_pressure = clamp(
        stagnation_generations(performance_history) as f32 / 24.0,
        0.0,
        1.0,
    );
    let base_mutation_rate = if ranked_for_breeding.len() == 1 {
        0.62
    } else {
        0.26
    };
    let raw_mutation_rate = base_mutation_rate
        + (1.0 - mean_novelty_norm) * 0.08
        + stagnation_pressure * 0.14
        + holdout_gap_norm * 0.08;
    let min_mutation_rate = MIN_BREEDING_MUTATION_RATE;
    let max_mutation_rate = MAX_BREEDING_MUTATION_RATE.max(min_mutation_rate + 0.01);
    let mutation_rate = clamp(raw_mutation_rate, min_mutation_rate, max_mutation_rate);
    let random_inject_chance = if ranked_for_breeding.len() > 1 {
        clamp(
            (0.035 + (1.0 - mean_novelty_norm) * 0.05)
                + stagnation_pressure * 0.08
                + holdout_gap_norm * 0.01,
            0.0,
            0.3,
        )
    } else {
        0.0
    };
    while next_genomes.len() < target_population_size {
        let tournament_size = 4usize.min(ranked_for_breeding.len().max(1));
        let parent_a = tournament_select(&ranked_for_breeding, tournament_size, rng)
            .genome
            .clone();
        let parent_b = if ranked_for_breeding.len() > 1 {
            tournament_select(&ranked_for_breeding, tournament_size, rng)
                .genome
                .clone()
        } else {
            parent_a.clone()
        };
        let operation_roll = rng.random::<f32>();
        let mut child = if operation_roll < 0.60 {
            mutate_genome(parent_a.clone(), mutation_rate, false, rng)
        } else if operation_roll < 0.80 {
            let crossed = crossover_genome(&parent_a, &parent_b, rng);
            mutate_genome(crossed, mutation_rate, false, rng)
        } else if operation_roll < 0.92 {
            mutate_genome(parent_a.clone(), mutation_rate, true, rng)
        } else {
            let grafted = graft_genome(&parent_a, &parent_b, rng);
            mutate_genome(grafted, mutation_rate, false, rng)
        };
        if ranked_for_breeding.len() > 1 && rng.random::<f32>() < random_inject_chance {
            child = random_genome(rng);
        }
        next_genomes.push(child);
    }
    for genome in &mut next_genomes {
        let constrained =
            apply_morphology_mode(genome.clone(), morphology_mode, morphology_preset, rng);
        *genome = constrained;
    }

    let performance_summary = build_generation_performance_summary(
        *generation,
        batch_results,
        mutation_rate,
        random_inject_chance,
        injected_used,
        elite_kept,
        holdout_best_fitness,
        holdout_gap,
        0.0,
        elite_consistency,
    );

    *generation += 1;
    *population_size = target_population_size;
    *batch_genomes = next_genomes;
    batch_results.clear();
    *attempt_trials = vec![Vec::new(); *population_size];
    *trial_seeds = build_trial_seed_set(*generation, TRIALS_PER_CANDIDATE);
    *current_attempt_index = 0;
    *current_trial_index = 0;
    performance_summary
}

#[derive(Clone, Copy)]
struct ParameterFeatureDefinition {
    name: &'static str,
    bounds: [f32; 2],
    extractor: fn(&Genome) -> f32,
}

const LEARNED_PARAMETER_FEATURES: [ParameterFeatureDefinition; 12] = [
    ParameterFeatureDefinition {
        name: "torso.w",
        bounds: [0.45, 3.1],
        extractor: feature_torso_w,
    },
    ParameterFeatureDefinition {
        name: "torso.h",
        bounds: [0.45, 3.1],
        extractor: feature_torso_h,
    },
    ParameterFeatureDefinition {
        name: "torso.d",
        bounds: [0.45, 3.1],
        extractor: feature_torso_d,
    },
    ParameterFeatureDefinition {
        name: "torso.mass",
        bounds: [0.2, 1.95],
        extractor: feature_torso_mass,
    },
    ParameterFeatureDefinition {
        name: "mass_scale",
        bounds: [0.7, 1.36],
        extractor: feature_mass_scale,
    },
    ParameterFeatureDefinition {
        name: "limb.enabled_ratio",
        bounds: [0.0, 1.0],
        extractor: feature_enabled_limb_ratio,
    },
    ParameterFeatureDefinition {
        name: "limb.segment_count_mean",
        bounds: [1.0, MAX_SEGMENTS_PER_LIMB as f32],
        extractor: feature_segment_count_mean,
    },
    ParameterFeatureDefinition {
        name: "segment.length_mean",
        bounds: [0.4, 2.6],
        extractor: feature_segment_length_mean,
    },
    ParameterFeatureDefinition {
        name: "segment.mass_mean",
        bounds: [0.1, 2.25],
        extractor: feature_segment_mass_mean,
    },
    ParameterFeatureDefinition {
        name: "control.amp_mean",
        bounds: [0.0, 11.6],
        extractor: feature_control_amp_mean,
    },
    ParameterFeatureDefinition {
        name: "control.freq_mean",
        bounds: [0.3, 4.9],
        extractor: feature_control_freq_mean,
    },
    ParameterFeatureDefinition {
        name: "joint.ball_ratio",
        bounds: [0.0, 1.0],
        extractor: feature_ball_joint_ratio,
    },
];

fn build_evolution_performance_response(
    controller: &EvolutionController,
    query: EvolutionPerformanceQuery,
) -> EvolutionPerformanceResponse {
    let status = controller.snapshot_status();
    let history = controller.snapshot_performance_history();
    let runtime_snapshot = controller.runtime_snapshot();
    let window_generations = query
        .window_generations
        .unwrap_or(DEFAULT_PERFORMANCE_WINDOW_GENERATIONS)
        .clamp(1, MAX_PERFORMANCE_WINDOW_GENERATIONS);
    let stride = query
        .stride
        .unwrap_or(DEFAULT_PERFORMANCE_STRIDE)
        .clamp(1, MAX_PERFORMANCE_STRIDE);
    let include_param_stats = query.include_param_stats.unwrap_or(true);
    let include_descriptors = query.include_descriptors.unwrap_or(true);
    let include_topology = query.include_topology.unwrap_or(true);

    let (from_generation, to_generation, recent_records) =
        select_performance_window(&history, status.generation, window_generations);
    let trends = EvolutionPerformanceTrends {
        best_fitness_slope: linear_regression_slope(
            &recent_records
                .iter()
                .map(|entry| (entry.generation as f32, entry.fitness.best))
                .collect::<Vec<_>>(),
        ),
        median_fitness_slope: linear_regression_slope(
            &recent_records
                .iter()
                .map(|entry| (entry.generation as f32, entry.fitness.p50))
                .collect::<Vec<_>>(),
        ),
        stagnation_generations: stagnation_generations(&history),
    };
    let mut generations = downsample_performance_history(&recent_records, stride);
    if !include_descriptors {
        for point in &mut generations {
            point.descriptor = None;
        }
    }
    if !include_topology {
        for point in &mut generations {
            point.topology = None;
        }
    }
    let learned_params = if include_param_stats {
        build_learned_parameter_summaries(runtime_snapshot.as_ref(), status.best_genome.as_ref())
    } else {
        Vec::new()
    };

    EvolutionPerformanceResponse {
        run: EvolutionPerformanceRun {
            generation: status.generation,
            population_size: status.population_size,
            trial_count: status.trial_count,
            run_speed: status.run_speed,
            paused: status.paused,
            morphology_mode: status.morphology_mode,
            morphology_preset: status.morphology_preset,
        },
        window: EvolutionPerformanceWindow {
            from_generation,
            to_generation,
            count: generations.len(),
            stride,
        },
        trends,
        generations,
        learned_params,
    }
}

fn build_evolution_performance_summary_response(
    controller: &EvolutionController,
) -> EvolutionPerformanceSummaryResponse {
    let status = controller.snapshot_status();
    let history = controller.snapshot_performance_history();
    let runtime_snapshot = controller.runtime_snapshot();
    let learned_params =
        build_learned_parameter_summaries(runtime_snapshot.as_ref(), status.best_genome.as_ref());
    let latest_topology = history.iter().rev().find_map(|entry| {
        entry
            .topology
            .as_ref()
            .map(|topology| topology.winner.clone())
    });
    let best_ever_topology = history
        .iter()
        .filter_map(|entry| {
            entry
                .topology
                .as_ref()
                .map(|topology| topology.winner.clone())
        })
        .max_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    let best_n_topologies = summarize_best_n_topologies(&history, SUMMARY_BEST_TOPOLOGY_COUNT);

    let recent_best_fitness = history
        .iter()
        .rev()
        .take(20)
        .map(|item| item.fitness.best)
        .fold(f32::NEG_INFINITY, f32::max);
    let recent_best_fitness = if recent_best_fitness.is_finite() {
        recent_best_fitness
    } else {
        0.0
    };
    let stagnation = stagnation_generations(&history);
    let latest = history.last();
    let mutation_rate = latest
        .map(|item| item.breeding.mutation_rate)
        .unwrap_or(0.0);
    let has_mutation_rate = latest.is_some();
    let novelty_slope = {
        let points: Vec<(f32, f32)> = history
            .iter()
            .rev()
            .take(32)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|item| (item.generation as f32, item.diversity.novelty_mean))
            .collect();
        linear_regression_slope(&points)
    };
    let diversity_state = match latest.map(|item| item.diversity.novelty_mean) {
        Some(novelty) if novelty < 0.22 => "low",
        Some(novelty) if novelty < 0.48 => "medium",
        Some(_) => "high",
        None => "unknown",
    };
    let mut convergence = Vec::new();
    for target in ["torso.mass", "control.amp_mean"] {
        if let Some(item) = learned_params.iter().find(|entry| entry.name == target) {
            convergence.push(EvolutionConvergenceSignal {
                name: item.name.clone(),
                state: convergence_state(item),
                std: item.population.std,
            });
        }
    }
    let mut signals = Vec::new();
    if stagnation >= 20 {
        signals.push("fitness_plateau".to_string());
    }
    if novelty_slope < -0.002 {
        signals.push("novelty_declining".to_string());
    }
    if mutation_rate > 0.55 {
        signals.push("mutation_pressure_high".to_string());
    }

    EvolutionPerformanceSummaryResponse {
        generation: status.generation,
        best_ever_fitness: status.best_ever_score,
        recent_best_fitness,
        stagnation_generations: stagnation,
        morphology_mode: status.morphology_mode,
        morphology_preset: status.morphology_preset,
        diversity_state: diversity_state.to_string(),
        mutation_pressure: EvolutionMutationPressure {
            current_rate: mutation_rate,
            at_lower_clamp: has_mutation_rate && mutation_rate <= MIN_BREEDING_MUTATION_RATE + 1e-4,
            at_upper_clamp: has_mutation_rate && mutation_rate >= MAX_BREEDING_MUTATION_RATE - 1e-4,
        },
        convergence,
        signals,
        latest_topology,
        best_ever_topology,
        best_n_topologies,
    }
}

fn build_evolution_performance_diagnose_response(
    controller: &EvolutionController,
) -> EvolutionPerformanceDiagnoseResponse {
    let status = controller.snapshot_status();
    let history = controller.snapshot_performance_history();
    let recent = recent_history_window(&history, DIAG_RECENT_WINDOW);
    let best_points = recent
        .iter()
        .map(|item| finite_or_zero(item.fitness.best))
        .collect::<Vec<_>>();
    let novelty_points = recent
        .iter()
        .map(|item| finite_or_zero(item.diversity.novelty_mean))
        .collect::<Vec<_>>();
    let recent_best_fitness = best_points
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let recent_best_fitness = finite_or_zero(recent_best_fitness);
    let best_fitness_slope = linear_regression_slope(
        &recent
            .iter()
            .map(|entry| (entry.generation as f32, entry.fitness.best))
            .collect::<Vec<_>>(),
    );
    let median_fitness_slope = linear_regression_slope(
        &recent
            .iter()
            .map(|entry| (entry.generation as f32, entry.fitness.p50))
            .collect::<Vec<_>>(),
    );
    let stagnation = stagnation_generations(&history);
    let novelty_mean = mean(&novelty_points);
    let novelty_min = novelty_points.iter().copied().fold(f32::INFINITY, f32::min);
    let novelty_max = novelty_points
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let best_min = best_points.iter().copied().fold(f32::INFINITY, f32::min);
    let best_max = best_points
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let best_std = std_dev(&best_points);
    let latest_mutation_rate = history
        .last()
        .map(|item| finite_or_zero(item.breeding.mutation_rate))
        .unwrap_or(0.0);
    let mutation_at_lower = history
        .last()
        .map(|_| latest_mutation_rate <= MIN_BREEDING_MUTATION_RATE + 1e-4)
        .unwrap_or(false);
    let mutation_at_upper = history
        .last()
        .map(|_| latest_mutation_rate >= MAX_BREEDING_MUTATION_RATE - 1e-4)
        .unwrap_or(false);

    let plateau_state = if stagnation >= DIAG_PLATEAU_STAGNATION_GENERATIONS {
        "plateau"
    } else if stagnation >= DIAG_WATCH_STAGNATION_GENERATIONS {
        "watch"
    } else {
        "active"
    };
    let volatility_state = if best_std >= 3.0 {
        "high"
    } else if best_std >= 1.5 {
        "medium"
    } else {
        "low"
    };
    let novelty_state = if novelty_mean < 0.22 {
        "low"
    } else if novelty_mean < 0.48 {
        "medium"
    } else {
        "high"
    };
    let trend_state = if best_fitness_slope > 0.08 {
        "improving"
    } else if best_fitness_slope < -0.03 {
        "declining"
    } else {
        "flat"
    };

    let representative_topologies = summarize_best_n_topologies(&history, 3);
    let mut limb_count_hist: HashMap<usize, usize> = HashMap::new();
    for profile in &representative_topologies {
        *limb_count_hist
            .entry(profile.enabled_limb_count)
            .or_insert(0) += 1;
    }
    let mut top_enabled_limb_counts = limb_count_hist
        .into_iter()
        .map(|(enabled_limb_count, count)| TopologyLimbCountStat {
            enabled_limb_count,
            count,
        })
        .collect::<Vec<_>>();
    top_enabled_limb_counts.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then_with(|| b.enabled_limb_count.cmp(&a.enabled_limb_count))
    });
    let latest_distinct_fingerprint_count = history
        .last()
        .and_then(|entry| entry.topology.as_ref())
        .map(|topology| topology.distinct_fingerprint_count)
        .unwrap_or(0);
    let latest_distinct_coarse_fingerprint_count = history
        .last()
        .and_then(|entry| entry.topology.as_ref())
        .map(|topology| topology.distinct_coarse_fingerprint_count)
        .unwrap_or(0);
    let latest_population_size = status.population_size;
    let distinct_fingerprint_ratio = if latest_population_size > 0 {
        latest_distinct_fingerprint_count as f32 / latest_population_size as f32
    } else {
        0.0
    };
    let distinct_coarse_fingerprint_ratio = if latest_population_size > 0 {
        latest_distinct_coarse_fingerprint_count as f32 / latest_population_size as f32
    } else {
        0.0
    };

    let mut findings = Vec::new();
    if plateau_state == "plateau" && novelty_state == "low" {
        findings.push(EvolutionDiagnosisFinding {
            code: "plateau_low_novelty".to_string(),
            severity: "warn".to_string(),
            message: "Best fitness has stalled while novelty remains low; search is likely exploiting a narrow behavior manifold.".to_string(),
        });
    }
    if volatility_state == "high" {
        findings.push(EvolutionDiagnosisFinding {
            code: "high_generation_variance".to_string(),
            severity: "info".to_string(),
            message: "Generation winners are volatile; robust quality is improving but outcomes remain noisy between generations.".to_string(),
        });
    }
    if distinct_fingerprint_ratio > 0.9 && novelty_state == "low" {
        findings.push(EvolutionDiagnosisFinding {
            code: "structural_diversity_behavioral_convergence".to_string(),
            severity: "info".to_string(),
            message: "Topology fingerprints are diverse, but behavior-level novelty is low; many structures map to similar gaits.".to_string(),
        });
    }
    if distinct_coarse_fingerprint_ratio < 0.25 {
        findings.push(EvolutionDiagnosisFinding {
            code: "coarse_topology_convergence".to_string(),
            severity: "info".to_string(),
            message: "Coarse topology classes are converging, even if fine-grained parameters still vary.".to_string(),
        });
    }
    if mutation_at_upper {
        findings.push(EvolutionDiagnosisFinding {
            code: "mutation_at_upper_clamp".to_string(),
            severity: "warn".to_string(),
            message: "Adaptive mutation is pinned near max clamp; this usually indicates aggressive exploration pressure.".to_string(),
        });
    }
    if mutation_at_lower {
        findings.push(EvolutionDiagnosisFinding {
            code: "mutation_at_lower_clamp".to_string(),
            severity: "info".to_string(),
            message: "Adaptive mutation is near minimum clamp; exploration may be underpowered."
                .to_string(),
        });
    }
    if findings.is_empty() {
        findings.push(EvolutionDiagnosisFinding {
            code: "no_critical_alerts".to_string(),
            severity: "info".to_string(),
            message: "No immediate diagnostic alerts; continue monitoring stagnation and novelty windows.".to_string(),
        });
    }

    let mut recommended_actions = Vec::new();
    if plateau_state == "plateau" {
        recommended_actions.push(
            "Increase diversity pressure for 10-20 generations (higher novelty weight or temporary mutation floor bump).".to_string(),
        );
    }
    if novelty_state == "low" && trend_state != "improving" {
        recommended_actions.push(
            "Inject topology alternates from distinct fingerprints in bestNTopologies to broaden behavior exploration.".to_string(),
        );
    }
    if volatility_state == "high" {
        recommended_actions.push(
            "Keep selection robust to variance (median/p25 emphasis) and avoid overreacting to one-generation spikes.".to_string(),
        );
    }
    if recommended_actions.is_empty() {
        recommended_actions
            .push("Continue current run and re-evaluate after 10-15 generations.".to_string());
    }

    EvolutionPerformanceDiagnoseResponse {
        generation: status.generation,
        timestamp_unix_ms: now_unix_ms(),
        states: EvolutionDiagnosisStates {
            plateau_state: plateau_state.to_string(),
            volatility_state: volatility_state.to_string(),
            novelty_state: novelty_state.to_string(),
            trend_state: trend_state.to_string(),
        },
        metrics: EvolutionDiagnosisMetrics {
            best_ever_fitness: finite_or_zero(status.best_ever_score),
            recent_best_fitness,
            stagnation_generations: stagnation,
            best_fitness_slope,
            median_fitness_slope,
            last_recent_best_std: best_std,
            last_recent_best_min: finite_or_zero(best_min),
            last_recent_best_max: finite_or_zero(best_max),
            last_recent_novelty_mean: finite_or_zero(novelty_mean),
            last_recent_novelty_min: finite_or_zero(novelty_min),
            last_recent_novelty_max: finite_or_zero(novelty_max),
            current_mutation_rate: latest_mutation_rate,
            mutation_at_lower_clamp: mutation_at_lower,
            mutation_at_upper_clamp: mutation_at_upper,
        },
        topology: EvolutionDiagnosisTopology {
            latest_distinct_fingerprint_count,
            latest_population_size,
            distinct_fingerprint_ratio,
            latest_distinct_coarse_fingerprint_count,
            distinct_coarse_fingerprint_ratio,
            top_enabled_limb_counts,
            representative_topologies,
        },
        findings,
        recommended_actions,
    }
}

fn recent_history_window(
    history: &[GenerationPerformanceSummary],
    count: usize,
) -> Vec<GenerationPerformanceSummary> {
    if count == 0 {
        return Vec::new();
    }
    let start = history.len().saturating_sub(count);
    history.iter().skip(start).cloned().collect()
}

fn summarize_best_n_topologies(
    history: &[GenerationPerformanceSummary],
    n: usize,
) -> Vec<TopologyProfile> {
    if n == 0 {
        return Vec::new();
    }
    let mut best_by_fingerprint: HashMap<String, TopologyProfile> = HashMap::new();
    for entry in history {
        let Some(topology) = &entry.topology else {
            continue;
        };
        for profile in &topology.best_n {
            best_by_fingerprint
                .entry(profile.topology_fingerprint.clone())
                .and_modify(|existing| {
                    if profile.fitness > existing.fitness {
                        *existing = profile.clone();
                    }
                })
                .or_insert_with(|| profile.clone());
        }
    }
    let mut values = best_by_fingerprint.into_values().collect::<Vec<_>>();
    values.sort_by(|a, b| {
        b.fitness
            .partial_cmp(&a.fitness)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.generation.cmp(&a.generation))
    });
    values.truncate(n);
    values
}

fn select_performance_window(
    history: &[GenerationPerformanceSummary],
    fallback_generation: usize,
    window_generations: usize,
) -> (usize, usize, Vec<GenerationPerformanceSummary>) {
    let to_generation = history
        .last()
        .map(|item| item.generation)
        .unwrap_or(fallback_generation.max(1));
    let from_generation = to_generation
        .saturating_sub(window_generations.saturating_sub(1))
        .max(1);
    let window = history
        .iter()
        .filter(|entry| entry.generation >= from_generation && entry.generation <= to_generation)
        .cloned()
        .collect::<Vec<_>>();
    (from_generation, to_generation, window)
}

fn downsample_performance_history(
    history: &[GenerationPerformanceSummary],
    stride: usize,
) -> Vec<GenerationPerformanceSummary> {
    if stride <= 1 || history.len() <= 1 {
        return history.to_vec();
    }
    let mut downsampled = history
        .iter()
        .enumerate()
        .filter_map(|(index, item)| {
            if index % stride == 0 {
                Some(item.clone())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if let Some(last) = history.last() {
        let needs_tail = downsampled
            .last()
            .map(|item| item.generation != last.generation)
            .unwrap_or(true);
        if needs_tail {
            downsampled.push(last.clone());
        }
    }
    downsampled
}

fn stagnation_generations(history: &[GenerationPerformanceSummary]) -> usize {
    let Some(last) = history.last() else {
        return 0;
    };
    let mut best = f32::NEG_INFINITY;
    let mut last_improved_generation = last.generation;
    for entry in history {
        if entry.fitness.best > best + FITNESS_STAGNATION_EPSILON {
            best = entry.fitness.best;
            last_improved_generation = entry.generation;
        }
    }
    last.generation.saturating_sub(last_improved_generation)
}

fn convergence_state(summary: &LearnedParameterSummary) -> String {
    let width = (summary.bounds[1] - summary.bounds[0]).max(1e-6);
    let ratio = summary.population.std / width;
    if ratio < 0.08 {
        "narrowing".to_string()
    } else if ratio < 0.2 {
        "stable".to_string()
    } else {
        "wide".to_string()
    }
}

fn build_learned_parameter_summaries(
    runtime_snapshot: Option<&EvolutionRuntimeSnapshot>,
    best_genome: Option<&Genome>,
) -> Vec<LearnedParameterSummary> {
    let Some(snapshot) = runtime_snapshot else {
        return Vec::new();
    };
    if snapshot.batch_genomes.is_empty() {
        return Vec::new();
    }
    LEARNED_PARAMETER_FEATURES
        .iter()
        .map(|feature| {
            let values: Vec<f32> = snapshot
                .batch_genomes
                .iter()
                .map(|genome| finite_or_zero((feature.extractor)(genome)))
                .collect();
            let population = summarize_distribution(&values);
            let champion = best_genome
                .map(|genome| finite_or_zero((feature.extractor)(genome)))
                .unwrap_or(population.p90);
            LearnedParameterSummary {
                name: feature.name.to_string(),
                bounds: feature.bounds,
                population,
                champion,
            }
        })
        .collect()
}

fn summarize_distribution(values: &[f32]) -> GenerationDistributionStats {
    if values.is_empty() {
        return GenerationDistributionStats {
            min: 0.0,
            p50: 0.0,
            p90: 0.0,
            max: 0.0,
            std: 0.0,
        };
    }
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    GenerationDistributionStats {
        min: finite_or_zero(min),
        p50: quantile(values, 0.5),
        p90: quantile(values, 0.9),
        max: finite_or_zero(max),
        std: std_dev(values),
    }
}

fn linear_regression_slope(points: &[(f32, f32)]) -> f32 {
    if points.len() < 2 {
        return 0.0;
    }
    let mean_x = points.iter().map(|(x, _)| *x).sum::<f32>() / points.len() as f32;
    let mean_y = points.iter().map(|(_, y)| *y).sum::<f32>() / points.len() as f32;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (x, y) in points {
        let dx = *x - mean_x;
        numerator += dx * (*y - mean_y);
        denominator += dx * dx;
    }
    if denominator.abs() < 1e-9 {
        0.0
    } else {
        numerator / denominator
    }
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

fn std_dev(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let avg = mean(values);
    let variance = values
        .iter()
        .map(|value| {
            let delta = *value - avg;
            delta * delta
        })
        .sum::<f32>()
        / values.len() as f32;
    variance.sqrt()
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

fn build_topology_profile(generation: usize, candidate: &EvolutionCandidate) -> TopologyProfile {
    let genome = &candidate.genome;
    let enabled_limb_count = enabled_limb_count(genome);
    let segment_count_histogram = segment_count_histogram(genome);
    TopologyProfile {
        generation,
        attempt: candidate.attempt,
        fitness: finite_or_zero(candidate.fitness),
        selection_score: finite_or_zero(candidate.selection_score),
        enabled_limb_count,
        segment_count_histogram,
        mean_segment_length: feature_segment_length_mean(genome),
        mean_segment_mass: feature_segment_mass_mean(genome),
        mean_control_amp: feature_control_amp_mean(genome),
        mean_control_freq: feature_control_freq_mean(genome),
        ball_joint_ratio: feature_ball_joint_ratio(genome),
        descriptor: [
            finite_or_zero(candidate.descriptor[0]),
            finite_or_zero(candidate.descriptor[1]),
            finite_or_zero(candidate.descriptor[2]),
            finite_or_zero(candidate.descriptor[3]),
            finite_or_zero(candidate.descriptor[4]),
        ],
        topology_fingerprint: topology_fingerprint(genome),
        coarse_topology_key: coarse_topology_key(genome),
    }
}

fn topology_fingerprint(genome: &Genome) -> String {
    let mut hash = 1469598103934665603u64;
    stable_hash_mix(&mut hash, genome.graph.nodes.len() as u64);
    stable_hash_mix(&mut hash, genome.graph.max_parts as u64);
    stable_hash_mix(&mut hash, genome.graph.root as u64);
    for node in &genome.graph.nodes {
        stable_hash_mix(&mut hash, quantize_to_bucket(node.part.w, 0.05) as u64);
        stable_hash_mix(&mut hash, quantize_to_bucket(node.part.h, 0.05) as u64);
        stable_hash_mix(&mut hash, quantize_to_bucket(node.part.d, 0.05) as u64);
        stable_hash_mix(&mut hash, quantize_to_bucket(node.part.mass, 0.05) as u64);
        stable_hash_mix(&mut hash, node.brain.neurons.len() as u64);
        stable_hash_mix(&mut hash, node.edges.len() as u64);
        for edge in &node.edges {
            stable_hash_mix(&mut hash, edge.to as u64);
            stable_hash_mix(&mut hash, quantize_to_bucket(edge.anchor_x, 0.05) as u64);
            stable_hash_mix(&mut hash, quantize_to_bucket(edge.anchor_y, 0.05) as u64);
            stable_hash_mix(&mut hash, quantize_to_bucket(edge.anchor_z, 0.05) as u64);
            stable_hash_mix(&mut hash, quantize_to_bucket(edge.scale, 0.05) as u64);
            stable_hash_mix(&mut hash, edge.recursive_limit as u64);
            stable_hash_mix(&mut hash, u64::from(edge.terminal_only));
            stable_hash_mix(
                &mut hash,
                match edge.joint_type {
                    JointTypeGene::Hinge => 0,
                    JointTypeGene::Ball => 1,
                },
            );
        }
    }
    format!("{hash:016x}")
}

fn coarse_topology_key(genome: &Genome) -> String {
    let enabled = enabled_limb_count(genome);
    let hist = segment_count_histogram(genome);
    let mean_segment_length = feature_segment_length_mean(genome);
    let mean_segment_mass = feature_segment_mass_mean(genome);
    let mean_control_amp = feature_control_amp_mean(genome);
    let mean_control_freq = feature_control_freq_mean(genome);
    let ball_ratio = feature_ball_joint_ratio(genome);
    let mean_segment_count = feature_segment_count_mean(genome);
    let root = genome
        .graph
        .nodes
        .get(
            genome
                .graph
                .root
                .min(genome.graph.nodes.len().saturating_sub(1)),
        )
        .map(|node| node.part.clone())
        .unwrap_or(GraphPartGene {
            w: 1.0,
            h: 1.0,
            d: 1.0,
            mass: 1.0,
        });
    let torso_wh = if root.h.abs() > 1e-5 {
        root.w / root.h
    } else {
        1.0
    };
    let torso_dh = if root.h.abs() > 1e-5 {
        root.d / root.h
    } else {
        1.0
    };
    format!(
        "e{}-h{}:{}:{}:{}:{}-sc{}-sl{}-sm{}-ca{}-cf{}-br{}-wh{}-dh{}",
        enabled,
        hist[0],
        hist[1],
        hist[2],
        hist[3],
        hist[4],
        quantize_to_bucket(mean_segment_count, 0.5),
        quantize_to_bucket(mean_segment_length, 0.25),
        quantize_to_bucket(mean_segment_mass, 0.2),
        quantize_to_bucket(mean_control_amp, 0.5),
        quantize_to_bucket(mean_control_freq, 0.25),
        quantize_to_bucket(ball_ratio, 0.1),
        quantize_to_bucket(torso_wh, 0.2),
        quantize_to_bucket(torso_dh, 0.2)
    )
}

fn stable_hash_mix(hash: &mut u64, value: u64) {
    const FNV_PRIME: u64 = 1099511628211;
    *hash ^= value;
    *hash = hash.wrapping_mul(FNV_PRIME);
}

fn quantize_to_bucket(value: f32, step: f32) -> i64 {
    if step <= 0.0 {
        return 0;
    }
    (finite_or_zero(value) / step).round() as i64
}

fn enabled_limb_count(genome: &Genome) -> usize {
    if genome.graph.nodes.is_empty() {
        return 0;
    }
    let root = genome
        .graph
        .root
        .min(genome.graph.nodes.len().saturating_sub(1));
    genome.graph.nodes[root].edges.len()
}

fn segment_count_histogram(genome: &Genome) -> [usize; MAX_SEGMENTS_PER_LIMB] {
    let mut histogram = [0usize; MAX_SEGMENTS_PER_LIMB];
    if genome.graph.nodes.is_empty() {
        return histogram;
    }
    let root = genome
        .graph
        .root
        .min(genome.graph.nodes.len().saturating_sub(1));
    for edge in &genome.graph.nodes[root].edges {
        let depth = graph_chain_length(&genome.graph, edge.to, MAX_SEGMENTS_PER_LIMB);
        let clamped = depth.clamp(1, MAX_SEGMENTS_PER_LIMB);
        histogram[clamped - 1] += 1;
    }
    histogram
}

fn graph_chain_length(graph: &GraphGene, start: usize, max_depth: usize) -> usize {
    if graph.nodes.is_empty() {
        return 0;
    }
    let mut depth = 0usize;
    let mut current = start.min(graph.nodes.len().saturating_sub(1));
    let mut seen = std::collections::BTreeSet::new();
    while depth < max_depth {
        depth += 1;
        if !seen.insert(current) {
            break;
        }
        let Some(next) = graph.nodes[current].edges.first() else {
            break;
        };
        current = next.to.min(graph.nodes.len().saturating_sub(1));
    }
    depth
}

fn feature_torso_w(genome: &Genome) -> f32 {
    genome
        .graph
        .nodes
        .get(
            genome
                .graph
                .root
                .min(genome.graph.nodes.len().saturating_sub(1)),
        )
        .map(|node| node.part.w)
        .unwrap_or(genome.torso.w)
}

fn feature_torso_h(genome: &Genome) -> f32 {
    genome
        .graph
        .nodes
        .get(
            genome
                .graph
                .root
                .min(genome.graph.nodes.len().saturating_sub(1)),
        )
        .map(|node| node.part.h)
        .unwrap_or(genome.torso.h)
}

fn feature_torso_d(genome: &Genome) -> f32 {
    genome
        .graph
        .nodes
        .get(
            genome
                .graph
                .root
                .min(genome.graph.nodes.len().saturating_sub(1)),
        )
        .map(|node| node.part.d)
        .unwrap_or(genome.torso.d)
}

fn feature_torso_mass(genome: &Genome) -> f32 {
    genome
        .graph
        .nodes
        .get(
            genome
                .graph
                .root
                .min(genome.graph.nodes.len().saturating_sub(1)),
        )
        .map(|node| node.part.mass)
        .unwrap_or(genome.torso.mass)
}

fn feature_mass_scale(genome: &Genome) -> f32 {
    genome.mass_scale
}

fn feature_enabled_limb_ratio(genome: &Genome) -> f32 {
    let enabled = enabled_limb_count(genome);
    enabled as f32 / MAX_LIMBS as f32
}

fn feature_segment_count_mean(genome: &Genome) -> f32 {
    let hist = segment_count_histogram(genome);
    let mut total = 0.0f32;
    let mut count = 0usize;
    for (index, bucket) in hist.iter().enumerate() {
        total += (index + 1) as f32 * *bucket as f32;
        count += *bucket;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

fn feature_segment_length_mean(genome: &Genome) -> f32 {
    let mut count = 0usize;
    let mut total = 0.0f32;
    for (index, node) in genome.graph.nodes.iter().enumerate() {
        if index
            == genome
                .graph
                .root
                .min(genome.graph.nodes.len().saturating_sub(1))
        {
            continue;
        }
        total += node.part.h;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

fn feature_segment_mass_mean(genome: &Genome) -> f32 {
    let mut count = 0usize;
    let mut total = 0.0f32;
    for (index, node) in genome.graph.nodes.iter().enumerate() {
        if index
            == genome
                .graph
                .root
                .min(genome.graph.nodes.len().saturating_sub(1))
        {
            continue;
        }
        total += node.part.mass;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

fn feature_control_amp_mean(genome: &Genome) -> f32 {
    let mut count = 0usize;
    let mut total = 0.0f32;
    for node in &genome.graph.nodes {
        for weight in &node.brain.effector_x.local_weights {
            total += weight.abs();
            count += 1;
        }
        for weight in &node.brain.effector_y.local_weights {
            total += weight.abs();
            count += 1;
        }
        for weight in &node.brain.effector_z.local_weights {
            total += weight.abs();
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

fn feature_control_freq_mean(genome: &Genome) -> f32 {
    let mut count = 0usize;
    let mut total = 0.0f32;
    for node in &genome.graph.nodes {
        for neuron in &node.brain.neurons {
            for value in &neuron.recurrent_weights {
                total += value.abs();
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

fn feature_ball_joint_ratio(genome: &Genome) -> f32 {
    let mut total = 0usize;
    let mut ball = 0usize;
    for node in &genome.graph.nodes {
        for edge in &node.edges {
            total += 1;
            if matches!(edge.joint_type, JointTypeGene::Ball) {
                ball += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        ball as f32 / total as f32
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .compact()
        .init();

    let args: Vec<String> = std::env::args().collect();
    if let Some(satellite_url) = parse_satellite_arg(&args) {
        run_satellite_client(satellite_url).await;
        return;
    }

    let state = AppState::new();
    info!(
        "simulation worker slots: {} (override with SIM_MAX_CONCURRENT_JOBS)",
        state.sim_worker_limit
    );
    info!(
        "satellite workers connected: {} (satellites connect to /api/satellite/ws)",
        state.satellite_pool.connected_count()
    );

    let app = Router::new()
        .route("/health", get(health))
        .route("/api/trial/ws", get(ws_trial_handler))
        .route("/api/eval/ws", get(ws_eval_handler))
        .route("/api/eval/generation", post(eval_generation_handler))
        .route("/api/evolution/state", get(evolution_state_handler))
        .route("/api/evolution/history", get(evolution_history_handler))
        .route(
            "/api/evolution/performance",
            get(evolution_performance_handler),
        )
        .route(
            "/api/evolution/performance/summary",
            get(evolution_performance_summary_handler),
        )
        .route(
            "/api/evolution/performance/diagnose",
            get(evolution_performance_diagnose_handler),
        )
        .route("/api/evolution/control", post(evolution_control_handler))
        .route(
            "/api/evolution/genome/current",
            get(evolution_current_genome_handler),
        )
        .route(
            "/api/evolution/genome/best",
            get(evolution_best_genome_handler),
        )
        .route(
            "/api/evolution/genome/import",
            post(evolution_import_genome_handler),
        )
        .route(
            "/api/evolution/checkpoint/save",
            post(evolution_checkpoint_save_handler),
        )
        .route(
            "/api/evolution/checkpoint/list",
            get(evolution_checkpoint_list_handler),
        )
        .route(
            "/api/evolution/checkpoint/load",
            post(evolution_checkpoint_load_handler),
        )
        .route("/api/evolution/ws", get(ws_evolution_handler))
        .route("/api/satellite/ws", get(ws_satellite_handler))
        .route("/", get(frontend_root))
        .route("/favicon.ico", get(frontend_favicon))
        .route("/{*path}", get(frontend_path))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state);
    let bind_host = DEFAULT_BIND_HOST;
    let bind_port = resolve_bind_port();
    let (listener, addr) = match bind_listener(bind_host, bind_port).await {
        Ok(bound) => bound,
        Err(message) => {
            error!("{message}");
            return;
        }
    };
    info!("breve-creatures listening on http://{addr}");
    info!(
        "frontend UI and satellite connections available on your LAN ip address at port {bind_port}"
    );
    if let Err(err) = axum::serve(listener, app).await {
        error!("server exited unexpectedly: {err}");
    }
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn frontend_root() -> Response {
    serve_frontend_asset("")
}

async fn frontend_path(Path(path): Path<String>) -> Response {
    serve_frontend_asset(&path)
}

async fn frontend_favicon() -> StatusCode {
    StatusCode::NO_CONTENT
}

fn serve_frontend_asset(path: &str) -> Response {
    let normalized = path.trim_start_matches('/');
    if normalized.starts_with("api/") || normalized == "health" {
        return StatusCode::NOT_FOUND.into_response();
    }

    let candidate = if normalized.is_empty() {
        "index.html"
    } else {
        normalized
    };

    if let Some(response) = frontend_asset_response(candidate) {
        return response;
    }

    if !candidate.contains('.')
        && let Some(response) = frontend_asset_response("index.html")
    {
        return response;
    }

    StatusCode::NOT_FOUND.into_response()
}

fn frontend_asset_response(path: &str) -> Option<Response> {
    let file = FRONTEND_ASSETS.get_file(path)?;
    let mime = mime_guess::from_path(path).first_or_octet_stream();
    let content_type = HeaderValue::from_str(mime.as_ref()).ok()?;
    let mut response = Response::new(Body::from(file.contents().to_vec()));
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, content_type);
    if path == "index.html" {
        response
            .headers_mut()
            .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    }
    Some(response)
}

async fn ws_trial_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_trial_socket(socket, state))
}

async fn ws_eval_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_eval_socket(socket, state))
}

async fn evolution_state_handler(State(state): State<AppState>) -> Json<EvolutionStatus> {
    Json(state.evolution.snapshot_status())
}

async fn evolution_history_handler(
    State(state): State<AppState>,
) -> Json<EvolutionFitnessHistoryResponse> {
    Json(EvolutionFitnessHistoryResponse {
        history: state.evolution.snapshot_fitness_history(),
    })
}

async fn evolution_performance_handler(
    State(state): State<AppState>,
    Query(query): Query<EvolutionPerformanceQuery>,
) -> Json<EvolutionPerformanceResponse> {
    Json(build_evolution_performance_response(
        &state.evolution,
        query,
    ))
}

async fn evolution_performance_summary_handler(
    State(state): State<AppState>,
) -> Json<EvolutionPerformanceSummaryResponse> {
    Json(build_evolution_performance_summary_response(
        &state.evolution,
    ))
}

async fn evolution_performance_diagnose_handler(
    State(state): State<AppState>,
) -> Json<EvolutionPerformanceDiagnoseResponse> {
    Json(build_evolution_performance_diagnose_response(
        &state.evolution,
    ))
}

async fn evolution_control_handler(
    State(state): State<AppState>,
    Json(request): Json<EvolutionControlRequest>,
) -> Result<Json<EvolutionStatus>, (StatusCode, String)> {
    let status = state
        .evolution
        .apply_control(request)
        .map_err(|message| (StatusCode::BAD_REQUEST, message))?;
    Ok(Json(status))
}

async fn evolution_current_genome_handler(
    State(state): State<AppState>,
) -> Result<Json<Genome>, (StatusCode, String)> {
    state.evolution.current_genome().map(Json).ok_or((
        StatusCode::NOT_FOUND,
        "no current genome available".to_string(),
    ))
}

async fn evolution_best_genome_handler(
    State(state): State<AppState>,
) -> Result<Json<Genome>, (StatusCode, String)> {
    state.evolution.current_best_genome().map(Json).ok_or((
        StatusCode::NOT_FOUND,
        "no best genome available".to_string(),
    ))
}

async fn evolution_import_genome_handler(
    State(state): State<AppState>,
    Json(request): Json<EvolutionGenomeImportRequest>,
) -> Result<Json<EvolutionGenomeImportResponse>, (StatusCode, String)> {
    let mut genomes = Vec::new();
    if let Some(genome) = request.genome {
        genomes.push(genome);
    }
    if let Some(items) = request.genomes {
        genomes.extend(items);
    }
    if genomes.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "request must include genome or genomes".to_string(),
        ));
    }
    let mutation_mode = request.mutation_mode.unwrap_or(InjectMutationMode::None);
    let response = state.evolution.queue_injections(genomes, mutation_mode);
    Ok(Json(response))
}

async fn evolution_checkpoint_save_handler(
    State(state): State<AppState>,
    Json(request): Json<CheckpointSaveRequest>,
) -> Result<Json<CheckpointSaveResponse>, (StatusCode, String)> {
    let snapshot = state.evolution.runtime_snapshot().ok_or((
        StatusCode::CONFLICT,
        "runtime snapshot not yet available".to_string(),
    ))?;
    let saved =
        save_checkpoint_snapshot(&snapshot, request.name.as_deref()).map_err(internal_err)?;
    Ok(Json(saved))
}

async fn evolution_checkpoint_list_handler(
    State(_state): State<AppState>,
) -> Result<Json<CheckpointListResponse>, (StatusCode, String)> {
    let checkpoints = list_checkpoint_summaries().map_err(internal_err)?;
    Ok(Json(CheckpointListResponse { checkpoints }))
}

async fn evolution_checkpoint_load_handler(
    State(state): State<AppState>,
    Json(request): Json<CheckpointLoadRequest>,
) -> Result<Json<CheckpointLoadResponse>, (StatusCode, String)> {
    let (id, mut snapshot) = load_checkpoint_snapshot(request.id.as_deref())
        .map_err(|message| (StatusCode::BAD_REQUEST, message))?;
    snapshot.status.fast_forward_remaining = 0;
    snapshot.status.fast_forward_active = false;
    state.evolution.set_pending_loaded_checkpoint(snapshot);
    Ok(Json(CheckpointLoadResponse { id }))
}

async fn ws_evolution_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_evolution_socket(socket, state))
}

async fn handle_trial_socket(mut socket: WebSocket, state: AppState) {
    let request = loop {
        match socket.next().await {
            Some(Ok(Message::Text(text))) => match serde_json::from_str::<TrialRunRequest>(&text) {
                Ok(request) => break request,
                Err(err) => {
                    let _ = send_stream_event(
                        &mut socket,
                        StreamEvent::Error {
                            message: format!("invalid trial request: {err}"),
                        },
                    )
                    .await;
                    return;
                }
            },
            Some(Ok(Message::Close(_))) | None => return,
            Some(Ok(_)) => continue,
            Some(Err(err)) => {
                error!("websocket receive error: {err}");
                return;
            }
        }
    };

    let config = TrialConfig::from_trial_request(&request);
    let permit = match state.acquire_sim_slot().await {
        Ok(permit) => permit,
        Err(err) => {
            let _ = send_stream_event(
                &mut socket,
                StreamEvent::Error {
                    message: format!("unable to schedule trial: {err}"),
                },
            )
            .await;
            return;
        }
    };
    let (tx, mut rx) = mpsc::channel::<StreamEvent>(128);
    let request_for_task = request.clone();
    let config_for_task = config.clone();
    let worker = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        if let Err(err) = run_trial_stream(request_for_task, config_for_task, tx) {
            error!("trial stream worker failed: {err}");
        }
    });

    while let Some(event) = rx.recv().await {
        if send_stream_event(&mut socket, event).await.is_err() {
            break;
        }
    }

    let _ = worker.await;
}

async fn handle_evolution_socket(mut socket: WebSocket, state: AppState) {
    let status = state.evolution.snapshot_status();
    if send_evolution_stream_event(&mut socket, EvolutionStreamEvent::Status { status })
        .await
        .is_err()
    {
        return;
    }

    let view = state.evolution.snapshot_view();
    if let Some(genome) = view.genome {
        if send_evolution_stream_event(
            &mut socket,
            EvolutionStreamEvent::TrialStarted {
                genome,
                part_sizes: view.part_sizes,
            },
        )
        .await
        .is_err()
        {
            return;
        }
        for frame in view.frames {
            if send_evolution_stream_event(&mut socket, EvolutionStreamEvent::Snapshot { frame })
                .await
                .is_err()
            {
                return;
            }
        }
        if let Some(result) = view.trial_result {
            if send_evolution_stream_event(
                &mut socket,
                EvolutionStreamEvent::TrialComplete { result },
            )
            .await
            .is_err()
            {
                return;
            }
        }
    }

    let mut rx = state.evolution.subscribe();
    loop {
        match rx.recv().await {
            Ok(event) => {
                if send_evolution_stream_event(&mut socket, event)
                    .await
                    .is_err()
                {
                    break;
                }
            }
            Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                warn!("evolution websocket lagged by {skipped} events");
                let status = state.evolution.snapshot_status();
                if send_evolution_stream_event(&mut socket, EvolutionStreamEvent::Status { status })
                    .await
                    .is_err()
                {
                    break;
                }
            }
            Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
        }
    }
}

async fn handle_eval_socket(mut socket: WebSocket, state: AppState) {
    let request = loop {
        match socket.next().await {
            Some(Ok(Message::Text(text))) => {
                match serde_json::from_str::<GenerationEvalRequest>(&text) {
                    Ok(request) => break request,
                    Err(err) => {
                        let _ = send_generation_stream_event(
                            &mut socket,
                            GenerationStreamEvent::Error {
                                message: format!("invalid generation request: {err}"),
                            },
                        )
                        .await;
                        return;
                    }
                }
            }
            Some(Ok(Message::Close(_))) | None => return,
            Some(Ok(_)) => continue,
            Some(Err(err)) => {
                error!("generation websocket receive error: {err}");
                return;
            }
        }
    };

    let config = TrialConfig::from_generation_request(&request);
    let permit = match state.acquire_sim_slot().await {
        Ok(permit) => permit,
        Err(err) => {
            let _ = send_generation_stream_event(
                &mut socket,
                GenerationStreamEvent::Error {
                    message: format!("unable to schedule generation: {err}"),
                },
            )
            .await;
            return;
        }
    };
    let (tx, mut rx) = mpsc::channel::<GenerationStreamEvent>(128);
    let request_for_task = request.clone();
    let config_for_task = config.clone();
    let parallel_workers = state.sim_worker_limit;
    let worker = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        if let Err(err) = run_generation_stream(
            request_for_task,
            config_for_task,
            tx,
            parallel_workers,
            state.satellite_pool.clone(),
        ) {
            error!("generation stream worker failed: {err}");
        }
    });

    while let Some(event) = rx.recv().await {
        if send_generation_stream_event(&mut socket, event)
            .await
            .is_err()
        {
            break;
        }
    }

    let _ = worker.await;
}

async fn send_stream_event(socket: &mut WebSocket, event: StreamEvent) -> Result<(), ()> {
    let text = match serde_json::to_string(&event) {
        Ok(value) => value,
        Err(err) => {
            error!("failed to serialize stream event: {err}");
            return Err(());
        }
    };
    socket
        .send(Message::Text(text.into()))
        .await
        .map_err(|err| {
            error!("failed to send stream event: {err}");
        })
}

async fn send_generation_stream_event(
    socket: &mut WebSocket,
    event: GenerationStreamEvent,
) -> Result<(), ()> {
    let text = match serde_json::to_string(&event) {
        Ok(value) => value,
        Err(err) => {
            error!("failed to serialize generation stream event: {err}");
            return Err(());
        }
    };
    socket
        .send(Message::Text(text.into()))
        .await
        .map_err(|err| {
            error!("failed to send generation stream event: {err}");
        })
}

async fn send_evolution_stream_event(
    socket: &mut WebSocket,
    event: EvolutionStreamEvent,
) -> Result<(), ()> {
    let text = match serde_json::to_string(&event) {
        Ok(value) => value,
        Err(err) => {
            error!("failed to serialize evolution stream event: {err}");
            return Err(());
        }
    };
    socket
        .send(Message::Text(text.into()))
        .await
        .map_err(|err| {
            let msg = err.to_string();
            if msg.contains("os error 10053")
                || msg.contains("connection closed")
                || msg.contains("Connection reset")
            {
                info!("evolution websocket disconnected while sending event");
            } else {
                warn!("failed to send evolution stream event: {err}");
            }
        })
}

fn run_trial_stream(
    request: TrialRunRequest,
    config: TrialConfig,
    tx: mpsc::Sender<StreamEvent>,
) -> Result<(), String> {
    let mut sim = TrialSimulator::new(&request.genome, request.seed, &config)?;
    let snapshot_every = ((1.0 / config.dt) / config.snapshot_hz).round().max(1.0) as usize;
    let steps = (config.duration_seconds / config.dt).ceil() as usize;

    tx.blocking_send(StreamEvent::TrialStarted {
        part_sizes: sim.part_sizes(),
    })
    .map_err(|err| format!("failed sending trial start: {err}"))?;
    tx.blocking_send(StreamEvent::Snapshot {
        frame: sim.current_frame(),
    })
    .map_err(|err| format!("failed sending initial snapshot: {err}"))?;

    for step in 0..steps {
        sim.step()?;
        if (step + 1) % snapshot_every == 0 || step + 1 == steps {
            tx.blocking_send(StreamEvent::Snapshot {
                frame: sim.current_frame(),
            })
            .map_err(|err| format!("failed sending snapshot: {err}"))?;
        }
    }

    tx.blocking_send(StreamEvent::TrialComplete {
        result: sim.final_result(),
    })
    .map_err(|err| format!("failed sending trial complete: {err}"))?;
    Ok(())
}

fn run_generation_stream(
    request: GenerationEvalRequest,
    config: TrialConfig,
    tx: mpsc::Sender<GenerationStreamEvent>,
    max_workers: usize,
    satellite_pool: Arc<SatellitePool>,
) -> Result<(), String> {
    if request.genomes.is_empty() {
        tx.blocking_send(GenerationStreamEvent::Error {
            message: "genomes must not be empty".to_string(),
        })
        .ok();
        return Err("genomes must not be empty".to_string());
    }
    if request.seeds.is_empty() {
        tx.blocking_send(GenerationStreamEvent::Error {
            message: "seeds must not be empty".to_string(),
        })
        .ok();
        return Err("seeds must not be empty".to_string());
    }

    tx.blocking_send(GenerationStreamEvent::GenerationStarted {
        attempt_count: request.genomes.len(),
        trial_count: request.seeds.len(),
    })
    .map_err(|err| format!("failed sending generation start: {err}"))?;

    let attempt_count = request.genomes.len();
    let trial_count = request.seeds.len();
    let worker_count = resolve_generation_worker_count(max_workers, attempt_count);
    let pending_jobs = Arc::new(Mutex::new(Vec::new()));
    for a in (0..attempt_count).rev() {
        for t in (0..trial_count).rev() {
            pending_jobs.lock().unwrap().push((a, t));
        }
    }

    let (result_tx, result_rx) = std_mpsc::channel::<Result<(usize, usize, TrialResult), String>>();
    let done = Arc::new(AtomicBool::new(false));

    std::thread::scope(|scope| {
        for _ in 0..worker_count {
            let pending = pending_jobs.clone();
            let res_tx = result_tx.clone();
            let genomes = &request.genomes;
            let seeds = &request.seeds;
            let config_ref = &config;
            let done_ref = &done;
            scope.spawn(move || {
                loop {
                    if done_ref.load(Ordering::Relaxed) {
                        break;
                    }
                    let job = pending.lock().unwrap().pop();
                    if let Some((a, t)) = job {
                        let outcome = run_trial_unpaced_with_wall_limit(
                            &genomes[a],
                            seeds[t],
                            config_ref,
                            FAST_EVAL_TRIAL_WALLTIME_LIMIT,
                        );
                        match outcome {
                            Ok(res) => {
                                if res_tx.send(Ok((a, t, res))).is_err() {
                                    break;
                                }
                            }
                            Err(e) => {
                                if res_tx.send(Err(e)).is_err() {
                                    break;
                                }
                            }
                        }
                    } else {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                }
            });
        }

        let mut working_satellites: HashMap<u64, (usize, usize)> = HashMap::new();
        let mut completed_trials = vec![vec![None; trial_count]; attempt_count];
        let mut satellite_retry_counts: HashMap<(usize, usize), usize> = HashMap::new();
        let mut finished_trials = 0;
        let total_trials = attempt_count * trial_count;
        let mut first_error: Option<String> = None;
        let mut pending_report = vec![false; attempt_count];

        loop {
            while let Ok(outcome) = result_rx.try_recv() {
                match outcome {
                    Ok((a, t, res)) => {
                        completed_trials[a][t] = Some(res);
                        satellite_retry_counts.remove(&(a, t));
                        finished_trials += 1;
                        if !pending_report[a] {
                            let _ = tx.blocking_send(GenerationStreamEvent::AttemptTrialStarted {
                                attempt_index: a,
                                trial_index: t,
                                trial_count,
                            });
                            pending_report[a] = true;
                        }
                    }
                    Err(e) => {
                        if first_error.is_none() {
                            first_error = Some(e);
                        }
                    }
                }
            }

            if first_error.is_some() {
                break;
            }

            let active_trial_ids: Vec<u64> = working_satellites.keys().copied().collect();
            for tid in active_trial_ids {
                if let Some(res) = satellite_pool.take_result(tid) {
                    if let Some((a, t)) = working_satellites.remove(&tid) {
                        completed_trials[a][t] = Some(res);
                        satellite_retry_counts.remove(&(a, t));
                        finished_trials += 1;
                        if !pending_report[a] {
                            let _ = tx.blocking_send(GenerationStreamEvent::AttemptTrialStarted {
                                attempt_index: a,
                                trial_index: t,
                                trial_count,
                            });
                            pending_report[a] = true;
                        }
                    }
                }
                if let Some(message) = satellite_pool.take_failure(tid) {
                    if let Some((a, t)) = working_satellites.remove(&tid) {
                        if message == SATELLITE_CAPACITY_ERROR {
                            let retries = satellite_retry_counts.entry((a, t)).or_insert(0);
                            *retries += 1;
                            if *retries > SATELLITE_DISPATCH_RETRY_LIMIT {
                                if first_error.is_none() {
                                    first_error = Some(format!(
                                        "satellite repeatedly rejected trial {tid} as at-capacity (retries exceeded)"
                                    ));
                                }
                            } else {
                                pending_jobs.lock().unwrap().push((a, t));
                            }
                        } else if first_error.is_none() {
                            first_error = Some(format!("satellite trial {tid} failed: {message}"));
                        }
                    }
                }
            }

            if first_error.is_some() {
                break;
            }

            let orphaned = satellite_pool.take_orphaned();
            if !orphaned.is_empty() {
                let mut p = pending_jobs.lock().unwrap();
                for tid in orphaned {
                    if let Some((a, t)) = working_satellites.remove(&tid) {
                        p.push((a, t));
                    }
                }
            }

            let timeouts = satellite_pool.reap_timeouts();
            if !timeouts.is_empty() {
                let mut p = pending_jobs.lock().unwrap();
                for tid in timeouts {
                    if let Some((a, t)) = working_satellites.remove(&tid) {
                        p.push((a, t));
                    }
                }
            }

            while satellite_pool.available_slot_count() > 0 {
                let mut p = pending_jobs.lock().unwrap();
                if let Some((a, t)) = p.pop() {
                    let genome = &request.genomes[a];
                    let seed = request.seeds[t];
                    if let Some(tid) = satellite_pool.try_dispatch(genome, seed, &config) {
                        working_satellites.insert(tid, (a, t));
                    } else {
                        p.push((a, t));
                        break;
                    }
                } else {
                    break;
                }
            }

            if finished_trials == total_trials {
                break;
            }

            std::thread::sleep(Duration::from_millis(5));
        }

        done.store(true, Ordering::Relaxed);

        if let Some(err) = first_error {
            let _ = tx.blocking_send(GenerationStreamEvent::Error {
                message: err.clone(),
            });
            return Err(err);
        }

        let mut results = Vec::with_capacity(attempt_count);
        for (a, block) in completed_trials.into_iter().enumerate() {
            let attempt_t = block.into_iter().map(|o| o.unwrap()).collect::<Vec<_>>();
            let res = summarize_trials(&request.genomes[a], &attempt_t);
            let _ = tx.blocking_send(GenerationStreamEvent::AttemptComplete {
                attempt_index: a,
                result: res.clone(),
            });
            results.push(res);
        }

        let _ = tx.blocking_send(GenerationStreamEvent::GenerationComplete { results });
        Ok(())
    })
}

async fn eval_generation_handler(
    State(state): State<AppState>,
    Json(request): Json<GenerationEvalRequest>,
) -> Result<Json<GenerationEvalResponse>, (StatusCode, String)> {
    if request.genomes.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "genomes must not be empty".to_string(),
        ));
    }
    if request.seeds.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "seeds must not be empty".to_string(),
        ));
    }

    let config = TrialConfig::from_generation_request(&request);
    let parallel_workers = state.sim_worker_limit;
    let permit = state
        .acquire_sim_slot()
        .await
        .map_err(service_unavailable_err)?;
    let response = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        run_generation_eval(request, config, parallel_workers)
    })
    .await
    .map_err(|err| internal_err(format!("generation eval worker join error: {err}")))?
    .map_err(internal_err)?;
    Ok(Json(response))
}

fn run_generation_eval(
    request: GenerationEvalRequest,
    config: TrialConfig,
    max_workers: usize,
) -> Result<GenerationEvalResponse, String> {
    let attempt_count = request.genomes.len();
    let worker_count = resolve_generation_worker_count(max_workers, attempt_count);
    let next_attempt = AtomicUsize::new(0);
    let cancelled = AtomicBool::new(false);
    let (result_tx, result_rx) =
        std_mpsc::channel::<Result<(usize, GenerationEvalResult), String>>();

    std::thread::scope(|scope| {
        for _ in 0..worker_count {
            let result_tx = result_tx.clone();
            let genomes = &request.genomes;
            let seeds = &request.seeds;
            let config_ref = &config;
            let next_attempt_ref = &next_attempt;
            let cancelled_ref = &cancelled;
            scope.spawn(move || {
                loop {
                    if cancelled_ref.load(Ordering::Relaxed) {
                        break;
                    }
                    let attempt_index = next_attempt_ref.fetch_add(1, Ordering::Relaxed);
                    if attempt_index >= genomes.len() {
                        break;
                    }
                    let outcome = evaluate_generation_attempt(
                        attempt_index,
                        &genomes[attempt_index],
                        seeds,
                        config_ref,
                        |_| Ok(()),
                    )
                    .map(|result| (attempt_index, result));
                    if outcome.is_err() {
                        cancelled_ref.store(true, Ordering::Relaxed);
                    }
                    if result_tx.send(outcome).is_err() {
                        break;
                    }
                }
            });
        }
    });
    drop(result_tx);

    let mut results: Vec<Option<GenerationEvalResult>> = vec![None; attempt_count];
    let mut first_error: Option<String> = None;
    for outcome in result_rx {
        match outcome {
            Ok((attempt_index, result)) => {
                if attempt_index < results.len() {
                    results[attempt_index] = Some(result);
                }
            }
            Err(err) => {
                if first_error.is_none() {
                    first_error = Some(err);
                }
            }
        }
    }
    if let Some(err) = first_error {
        return Err(err);
    }

    let results: Vec<GenerationEvalResult> = results
        .into_iter()
        .enumerate()
        .map(|(attempt_index, item)| {
            item.ok_or_else(|| {
                format!("generation attempt {attempt_index} did not produce a result")
            })
        })
        .collect::<Result<_, _>>()?;

    Ok(GenerationEvalResponse { results })
}

fn resolve_generation_worker_count(max_workers: usize, attempt_count: usize) -> usize {
    max_workers.max(1).min(attempt_count.max(1))
}

fn run_trial_unpaced_with_wall_limit(
    genome: &Genome,
    seed: u64,
    config: &TrialConfig,
    walltime_limit: Duration,
) -> Result<TrialResult, String> {
    let mut sim = TrialSimulator::new(genome, seed, config)?;
    let steps = (config.duration_seconds / config.dt).ceil() as usize;
    let started_at = Instant::now();
    for _ in 0..steps {
        if started_at.elapsed() >= walltime_limit {
            break;
        }
        sim.step()?;
    }
    Ok(sim.final_result())
}

fn evaluate_generation_attempt<F>(
    _attempt_index: usize,
    genome: &Genome,
    seeds: &[u64],
    config: &TrialConfig,
    mut on_trial_started: F,
) -> Result<GenerationEvalResult, String>
where
    F: FnMut(usize) -> Result<(), String>,
{
    let mut trials = Vec::with_capacity(seeds.len());
    for (trial_index, &seed) in seeds.iter().enumerate() {
        on_trial_started(trial_index)?;
        let result = run_trial_unpaced_with_wall_limit(
            genome,
            seed,
            config,
            FAST_EVAL_TRIAL_WALLTIME_LIMIT,
        )?;
        trials.push(result);
    }
    Ok(summarize_trials(genome, &trials))
}

fn summarize_trials(genome: &Genome, trials: &[TrialResult]) -> GenerationEvalResult {
    if trials.is_empty() {
        return GenerationEvalResult {
            fitness: 0.0,
            descriptor: [0.0; 5],
            trial_count: 0,
            median_progress: 0.0,
            median_upright: 0.0,
            median_straightness: 0.0,
            invalid_startup_trials: 0,
            invalid_startup_trial_rate: 0.0,
            all_trials_invalid_startup: false,
        };
    }

    let qualities: Vec<f32> = trials.iter().map(|trial| trial.metrics.quality).collect();
    let progresses: Vec<f32> = trials.iter().map(|trial| trial.metrics.progress).collect();
    let uprights: Vec<f32> = trials
        .iter()
        .map(|trial| trial.metrics.upright_avg)
        .collect();
    let straightnesses: Vec<f32> = trials
        .iter()
        .map(|trial| trial.metrics.straightness)
        .collect();
    let invalid_startup_trials = trials
        .iter()
        .filter(|trial| trial.metrics.invalid_startup)
        .count();
    let invalid_startup_trial_rate = invalid_startup_trials as f32 / trials.len().max(1) as f32;
    let all_trials_invalid_startup = invalid_startup_trials == trials.len();

    let robust_quality = 0.7 * quantile(&qualities, 0.5) + 0.3 * quantile(&qualities, 0.25);
    let consistency_gate = if qualities.len() < 2 {
        1.0
    } else {
        let quality_std = std_dev(&qualities);
        let quality_scale = quantile(&qualities, 0.5).abs().max(1.0);
        let divergence_ratio = quality_std / quality_scale;
        clamp(
            1.0 - divergence_ratio * TRIAL_DIVERGENCE_PENALTY_WEIGHT,
            TRIAL_DIVERGENCE_PENALTY_FLOOR,
            1.0,
        )
    };
    let median_progress = quantile(&progresses, 0.5);
    let median_upright = quantile(&uprights, 0.5);
    let median_straightness = quantile(&straightnesses, 0.5);

    let active_limbs = enabled_limb_count(genome);
    let mean_segment_count = feature_segment_count_mean(genome);

    let descriptor = [
        clamp(median_progress / 28.0, 0.0, 1.0),
        clamp(median_upright, 0.0, 1.0),
        clamp(median_straightness, 0.0, 1.0),
        clamp(active_limbs as f32 / MAX_LIMBS.max(1) as f32, 0.0, 1.0),
        clamp(
            mean_segment_count / MAX_SEGMENTS_PER_LIMB.max(1) as f32,
            0.0,
            1.0,
        ),
    ];

    GenerationEvalResult {
        fitness: (robust_quality * consistency_gate).max(0.0),
        descriptor,
        trial_count: trials.len(),
        median_progress,
        median_upright,
        median_straightness,
        invalid_startup_trials,
        invalid_startup_trial_rate,
        all_trials_invalid_startup,
    }
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn checkpoint_dir() -> PathBuf {
    PathBuf::from(CHECKPOINT_DIR)
}

fn ensure_checkpoint_dir() -> Result<PathBuf, String> {
    let dir = checkpoint_dir();
    fs::create_dir_all(&dir).map_err(|err| {
        format!(
            "failed creating checkpoint directory '{}': {err}",
            dir.display()
        )
    })?;
    Ok(dir)
}

fn sanitize_checkpoint_name(name: &str) -> String {
    let mut cleaned = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            cleaned.push(ch);
        } else if ch.is_ascii_whitespace() {
            cleaned.push('-');
        }
    }
    cleaned.trim_matches('-').to_string()
}

fn checkpoint_file_from_id(id: &str) -> PathBuf {
    checkpoint_dir().join(format!("{id}.json"))
}

fn write_checkpoint_file(
    path: &FsPath,
    checkpoint: &EvolutionCheckpointFile,
) -> Result<(), String> {
    let payload = serde_json::to_vec_pretty(checkpoint)
        .map_err(|err| format!("failed serializing checkpoint: {err}"))?;
    fs::write(path, payload)
        .map_err(|err| format!("failed writing checkpoint '{}': {err}", path.display()))
}

fn read_checkpoint_file(path: &FsPath) -> Result<EvolutionCheckpointFile, String> {
    let payload = fs::read(path)
        .map_err(|err| format!("failed reading checkpoint '{}': {err}", path.display()))?;
    serde_json::from_slice::<EvolutionCheckpointFile>(&payload)
        .map_err(|err| format!("failed parsing checkpoint '{}': {err}", path.display()))
}

fn save_checkpoint_snapshot(
    snapshot: &EvolutionRuntimeSnapshot,
    name: Option<&str>,
) -> Result<CheckpointSaveResponse, String> {
    let dir = ensure_checkpoint_dir()?;
    let timestamp = now_unix_ms();
    let name_suffix = name
        .map(sanitize_checkpoint_name)
        .filter(|value| !value.is_empty())
        .map(|value| format!("-{value}"))
        .unwrap_or_default();
    let id = format!("ckpt-{timestamp}{name_suffix}");
    let path = dir.join(format!("{id}.json"));
    let checkpoint = EvolutionCheckpointFile {
        version: 1,
        id: id.clone(),
        created_at_unix_ms: timestamp,
        snapshot: snapshot.clone(),
    };
    write_checkpoint_file(&path, &checkpoint)?;
    let latest_path = dir.join("latest.json");
    write_checkpoint_file(&latest_path, &checkpoint)?;
    Ok(CheckpointSaveResponse {
        id,
        path: path.to_string_lossy().to_string(),
    })
}

fn autosave_runtime_snapshot_if_due(controller: &EvolutionController) {
    let status = controller.snapshot_status();
    if status.generation == 0
        || status.generation % AUTOSAVE_EVERY_GENERATIONS != 0
        || status.current_attempt_index != 0
        || status.current_trial_index != 0
    {
        return;
    }
    let Some(snapshot) = controller.runtime_snapshot() else {
        return;
    };
    match save_checkpoint_snapshot(&snapshot, Some("autosave")) {
        Ok(saved) => info!(
            "autosaved checkpoint: id={}, generation={}",
            saved.id, status.generation
        ),
        Err(err) => warn!("autosave checkpoint failed: {err}"),
    }
}

fn list_checkpoint_summaries() -> Result<Vec<CheckpointSummary>, String> {
    let dir = ensure_checkpoint_dir()?;
    let entries = fs::read_dir(&dir)
        .map_err(|err| format!("failed listing checkpoints in '{}': {err}", dir.display()))?;
    let mut summaries = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|err| format!("failed reading checkpoint entry: {err}"))?;
        let path = entry.path();
        let is_json = path
            .extension()
            .and_then(|value| value.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("json"))
            .unwrap_or(false);
        if !is_json {
            continue;
        }
        if path.file_name().and_then(|v| v.to_str()) == Some("latest.json") {
            continue;
        }
        if let Ok(file) = read_checkpoint_file(&path) {
            summaries.push(CheckpointSummary {
                id: file.id,
                created_at_unix_ms: file.created_at_unix_ms,
                generation: file.snapshot.status.generation,
                best_ever_score: file.snapshot.status.best_ever_score,
            });
        }
    }
    summaries.sort_by(|a, b| b.created_at_unix_ms.cmp(&a.created_at_unix_ms));
    Ok(summaries)
}

fn load_checkpoint_snapshot(
    id: Option<&str>,
) -> Result<(String, EvolutionRuntimeSnapshot), String> {
    let dir = ensure_checkpoint_dir()?;
    let path = match id {
        Some(requested_id) => checkpoint_file_from_id(requested_id),
        None => dir.join("latest.json"),
    };
    if !path.exists() {
        return Err(format!("checkpoint file '{}' not found", path.display()));
    }
    let checkpoint = read_checkpoint_file(&path)?;
    Ok((checkpoint.id, checkpoint.snapshot))
}

fn apply_diversity_scores(results: &mut [EvolutionCandidate], archive: &[NoveltyEntry]) {
    if results.is_empty() {
        return;
    }
    for i in 0..results.len() {
        let descriptor = results[i].descriptor;
        let mut novelty_distances: Vec<f32> = Vec::new();
        let mut neighbor_distances: Vec<(f32, f32)> = Vec::new();
        for (j, other) in results.iter().enumerate() {
            if i == j {
                continue;
            }
            let distance = descriptor_distance(descriptor, other.descriptor);
            novelty_distances.push(distance);
            neighbor_distances.push((distance, other.fitness));
        }
        for entry in archive {
            novelty_distances.push(descriptor_distance(descriptor, entry.descriptor));
        }
        novelty_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let novelty_k = novelty_distances.len().min(8);
        results[i].novelty = if novelty_k > 0 {
            novelty_distances.iter().take(novelty_k).sum::<f32>() / novelty_k as f32
        } else {
            0.0
        };

        neighbor_distances
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let competition_k = neighbor_distances.len().min(6);
        if competition_k == 0 {
            results[i].local_competition = 1.0;
        } else {
            let wins = neighbor_distances
                .iter()
                .take(competition_k)
                .filter(|(_, fitness)| results[i].fitness >= *fitness)
                .count();
            results[i].local_competition = wins as f32 / competition_k as f32;
        }
    }

    normalize_candidate_field(
        results,
        |candidate| candidate.fitness,
        |candidate, value| {
            candidate.quality_norm = value;
        },
    );
    normalize_candidate_field(
        results,
        |candidate| candidate.novelty,
        |candidate, value| {
            candidate.novelty_norm = value;
        },
    );
    for result in results.iter_mut() {
        result.selection_score = 0.62 * result.quality_norm
            + 0.28 * result.novelty_norm
            + 0.1 * result.local_competition;
    }
}

fn normalize_candidate_field<FGet, FSet>(
    results: &mut [EvolutionCandidate],
    get: FGet,
    mut set: FSet,
) where
    FGet: Fn(&EvolutionCandidate) -> f32,
    FSet: FnMut(&mut EvolutionCandidate, f32),
{
    let mut min_value = f32::INFINITY;
    let mut max_value = f32::NEG_INFINITY;
    for result in results.iter() {
        let value = get(result);
        min_value = min_value.min(value);
        max_value = max_value.max(value);
    }
    if !min_value.is_finite() || !max_value.is_finite() {
        for result in results.iter_mut() {
            set(result, 0.0);
        }
        return;
    }
    let span = max_value - min_value;
    if span < 1e-9 {
        for result in results.iter_mut() {
            set(result, 1.0);
        }
        return;
    }
    for result in results.iter_mut() {
        set(result, (get(result) - min_value) / span);
    }
}

fn update_novelty_archive(results: &[EvolutionCandidate], archive: &mut Vec<NoveltyEntry>) {
    if results.is_empty() {
        return;
    }
    let mut by_novelty = results.to_vec();
    by_novelty.sort_by(|a, b| {
        b.novelty
            .partial_cmp(&a.novelty)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for candidate in by_novelty.into_iter().take(3) {
        archive.push(NoveltyEntry {
            descriptor: candidate.descriptor,
            fitness: candidate.fitness,
        });
    }
    if let Some(best_fitness) = results.iter().max_by(|a, b| {
        a.fitness
            .partial_cmp(&b.fitness)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        archive.push(NoveltyEntry {
            descriptor: best_fitness.descriptor,
            fitness: best_fitness.fitness,
        });
    }
    archive.sort_by(|a, b| {
        b.fitness
            .partial_cmp(&a.fitness)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if archive.len() > 320 {
        archive.truncate(320);
    }
}

fn descriptor_distance(a: [f32; 5], b: [f32; 5]) -> f32 {
    let mut sum = 0.0;
    for i in 0..5 {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum.sqrt()
}

fn tournament_select<'a>(
    ranked: &'a [EvolutionCandidate],
    size: usize,
    rng: &mut SmallRng,
) -> &'a EvolutionCandidate {
    assert!(
        !ranked.is_empty(),
        "tournament_select requires non-empty input"
    );
    let len = ranked.len();
    let mut best = &ranked[rng.random_range(0..len)];
    for _ in 1..size.max(1) {
        let candidate = &ranked[rng.random_range(0..len)];
        if candidate.selection_score > best.selection_score {
            best = candidate;
        }
    }
    best
}

fn build_trial_seed_set(generation_index: usize, count: usize) -> Vec<u64> {
    let generation_tag = hash_uint32(
        TRAIN_TRIAL_SEED_BANK_TAG,
        generation_index.max(1) as u32,
        0xa511_e9b3,
    );
    build_seed_bank(generation_tag, count)
}

fn build_holdout_trial_seed_set(count: usize) -> Vec<u64> {
    build_seed_bank(HOLDOUT_TRIAL_SEED_BANK_TAG, count)
}

fn build_seed_bank(tag: u32, count: usize) -> Vec<u64> {
    (0..count)
        .map(|i| hash_uint32(tag, (i + 1) as u32, 0x85eb_ca6b) as u64)
        .collect()
}

fn hash_uint32(a: u32, b: u32, c: u32) -> u32 {
    let mut h = 2_166_136_261u32;
    h ^= a;
    h = h.wrapping_mul(16_777_619);
    h ^= b;
    h = h.wrapping_mul(16_777_619);
    h ^= c;
    h = h.wrapping_mul(16_777_619);
    h = h.wrapping_add(h << 13);
    h ^= h >> 7;
    h = h.wrapping_add(h << 3);
    h ^= h >> 17;
    h = h.wrapping_add(h << 5);
    h
}

fn default_morphology_mode() -> EvolutionMorphologyMode {
    EvolutionMorphologyMode::Random
}

fn default_morphology_preset() -> MorphologyPreset {
    MorphologyPreset::Spider4x2
}

fn parse_morphology_mode_token(token: &str) -> Option<EvolutionMorphologyMode> {
    let normalized = token.trim().to_ascii_lowercase().replace('-', "_");
    match normalized.as_str() {
        "random" => Some(EvolutionMorphologyMode::Random),
        "fixed" | "fixed_preset" | "preset" => Some(EvolutionMorphologyMode::FixedPreset),
        _ => None,
    }
}

fn parse_morphology_preset_token(token: &str) -> Option<MorphologyPreset> {
    let normalized = token.trim().to_ascii_lowercase().replace('-', "_");
    match normalized.as_str() {
        "spider4x2" | "spider_4x2" | "spider" => Some(MorphologyPreset::Spider4x2),
        _ => None,
    }
}

fn resolve_initial_morphology_override() -> (EvolutionMorphologyMode, MorphologyPreset) {
    let mut mode = default_morphology_mode();
    let mut preset = default_morphology_preset();
    let mut mode_explicitly_set = false;

    if let Ok(raw_mode) = std::env::var(ENV_EVOLUTION_MORPHOLOGY_MODE) {
        if let Some(parsed) = parse_morphology_mode_token(&raw_mode) {
            mode = parsed;
            mode_explicitly_set = true;
        } else {
            warn!(
                "{} has invalid value '{}'; expected 'random' or 'fixed_preset'",
                ENV_EVOLUTION_MORPHOLOGY_MODE, raw_mode
            );
        }
    }

    if let Ok(raw_preset) = std::env::var(ENV_EVOLUTION_MORPHOLOGY_PRESET) {
        if let Some(parsed) = parse_morphology_preset_token(&raw_preset) {
            preset = parsed;
            if !mode_explicitly_set {
                mode = EvolutionMorphologyMode::FixedPreset;
            }
        } else {
            warn!(
                "{} has invalid value '{}'; expected 'spider4x2'",
                ENV_EVOLUTION_MORPHOLOGY_PRESET, raw_preset
            );
        }
    }

    info!(
        "evolution morphology startup: mode={}, preset={} (override with {} and {})",
        morphology_mode_label(mode),
        morphology_preset_label(preset),
        ENV_EVOLUTION_MORPHOLOGY_MODE,
        ENV_EVOLUTION_MORPHOLOGY_PRESET
    );
    (mode, preset)
}

fn morphology_mode_label(mode: EvolutionMorphologyMode) -> &'static str {
    match mode {
        EvolutionMorphologyMode::Random => "random",
        EvolutionMorphologyMode::FixedPreset => "fixed_preset",
    }
}

fn morphology_preset_label(preset: MorphologyPreset) -> &'static str {
    match preset {
        MorphologyPreset::Spider4x2 => "spider4x2",
    }
}

fn apply_morphology_mode(
    genome: Genome,
    mode: EvolutionMorphologyMode,
    preset: MorphologyPreset,
    rng: &mut SmallRng,
) -> Genome {
    match mode {
        EvolutionMorphologyMode::Random => {
            let mut adjusted = genome;
            ensure_active_body_plan(&mut adjusted, rng);
            adjusted
        }
        EvolutionMorphologyMode::FixedPreset => {
            constrain_genome_to_morphology_preset(genome, preset, rng)
        }
    }
}

fn constrain_genome_to_morphology_preset(
    source: Genome,
    preset: MorphologyPreset,
    rng: &mut SmallRng,
) -> Genome {
    let mut constrained = morphology_preset_template_genome(preset);
    let profile = morphology_preset_constraint_profile(preset);

    if !profile.lock_topology {
        copy_topology_genes(&source, &mut constrained);
    }
    if !profile.lock_joint_types {
        copy_joint_type_genes(&source, &mut constrained);
    }
    if !profile.lock_joint_limits {
        copy_joint_limit_genes(&source, &mut constrained);
    }
    if !profile.lock_segment_dynamics {
        copy_segment_dynamics_genes(&source, &mut constrained);
    }
    if !profile.lock_controls {
        copy_control_genes(&source, &mut constrained);
    }
    if !profile.lock_visual_hue {
        constrained.hue = source.hue;
    }
    constrained.version = default_genome_version();
    constrained.graph = if profile.lock_topology {
        let mut graph = morphology_preset_template_genome(preset).graph;
        if !profile.lock_controls {
            graph.global_brain = source.graph.global_brain.clone();
            let copy_count = graph.nodes.len().min(source.graph.nodes.len());
            for index in 0..copy_count {
                graph.nodes[index].brain = source.graph.nodes[index].brain.clone();
            }
        }
        graph.max_parts = graph.max_parts.clamp(6, MAX_GRAPH_PARTS);
        graph
    } else {
        source.graph.clone()
    };
    ensure_graph_valid(&mut constrained.graph, rng);
    project_graph_to_legacy(&mut constrained);
    ensure_active_body_plan(&mut constrained, rng);
    constrained
}

fn morphology_preset_template_genome(preset: MorphologyPreset) -> Genome {
    match preset {
        MorphologyPreset::Spider4x2 => spider4x2_template_genome(),
    }
}

fn morphology_preset_constraint_profile(preset: MorphologyPreset) -> PresetConstraintProfile {
    match preset {
        MorphologyPreset::Spider4x2 => PresetConstraintProfile {
            lock_topology: true,
            lock_joint_types: true,
            lock_joint_limits: false,
            lock_segment_dynamics: true,
            lock_controls: false,
            lock_visual_hue: false,
        },
    }
}

fn copy_topology_genes(source: &Genome, target: &mut Genome) {
    target.torso = source.torso.clone();
    target.mass_scale = source.mass_scale;
    for (limb_index, target_limb) in target.limbs.iter_mut().enumerate() {
        let Some(source_limb) = source.limbs.get(limb_index) else {
            continue;
        };
        target_limb.enabled = source_limb.enabled;
        target_limb.segment_count = source_limb.segment_count;
        target_limb.anchor_x = source_limb.anchor_x;
        target_limb.anchor_y = source_limb.anchor_y;
        target_limb.anchor_z = source_limb.anchor_z;
        target_limb.axis_y = source_limb.axis_y;
        target_limb.axis_z = source_limb.axis_z;
        target_limb.dir_x = source_limb.dir_x;
        target_limb.dir_y = source_limb.dir_y;
        target_limb.dir_z = source_limb.dir_z;
        for (segment_index, target_segment) in target_limb.segments.iter_mut().enumerate() {
            let Some(source_segment) = source_limb.segments.get(segment_index) else {
                continue;
            };
            target_segment.length = source_segment.length;
            target_segment.thickness = source_segment.thickness;
            target_segment.mass = source_segment.mass;
        }
    }
}

fn copy_joint_type_genes(source: &Genome, target: &mut Genome) {
    for (limb_index, target_limb) in target.limbs.iter_mut().enumerate() {
        let Some(source_limb) = source.limbs.get(limb_index) else {
            continue;
        };
        for (segment_index, target_segment) in target_limb.segments.iter_mut().enumerate() {
            let Some(source_segment) = source_limb.segments.get(segment_index) else {
                continue;
            };
            target_segment.joint_type = source_segment.joint_type;
        }
    }
}

fn copy_joint_limit_genes(source: &Genome, target: &mut Genome) {
    for (limb_index, target_limb) in target.limbs.iter_mut().enumerate() {
        let Some(source_limb) = source.limbs.get(limb_index) else {
            continue;
        };
        for (segment_index, target_segment) in target_limb.segments.iter_mut().enumerate() {
            let Some(source_segment) = source_limb.segments.get(segment_index) else {
                continue;
            };
            target_segment.limit_x = source_segment.limit_x;
            target_segment.limit_y = source_segment.limit_y;
            target_segment.limit_z = source_segment.limit_z;
        }
    }
}

fn copy_segment_dynamics_genes(source: &Genome, target: &mut Genome) {
    for (limb_index, target_limb) in target.limbs.iter_mut().enumerate() {
        let Some(source_limb) = source.limbs.get(limb_index) else {
            continue;
        };
        for (segment_index, target_segment) in target_limb.segments.iter_mut().enumerate() {
            let Some(source_segment) = source_limb.segments.get(segment_index) else {
                continue;
            };
            target_segment.motor_strength = source_segment.motor_strength;
            target_segment.joint_stiffness = source_segment.joint_stiffness;
        }
    }
}

fn copy_control_genes(source: &Genome, target: &mut Genome) {
    for (limb_index, target_limb) in target.limbs.iter_mut().enumerate() {
        let Some(source_limb) = source.limbs.get(limb_index) else {
            continue;
        };
        for (control_index, target_control) in target_limb.controls.iter_mut().enumerate() {
            let Some(source_control) = source_limb.controls.get(control_index) else {
                continue;
            };
            *target_control = source_control.clone();
        }
    }
}

fn default_control_gene() -> ControlGene {
    ControlGene {
        amp: 1.0,
        freq: 1.0,
        phase: 0.0,
        bias: 0.0,
        harm2_amp: default_second_harmonic_amp(),
        harm2_phase: default_second_harmonic_phase(),
        amp_y: default_secondary_control_amp(),
        freq_y: default_secondary_control_freq(),
        phase_y: default_secondary_control_phase(),
        bias_y: default_secondary_control_bias(),
        amp_z: default_secondary_control_amp(),
        freq_z: default_secondary_control_freq(),
        phase_z: default_secondary_control_phase(),
        bias_z: default_secondary_control_bias(),
    }
}

fn random_neural_activation(rng: &mut SmallRng) -> NeuralActivationGene {
    match rng.random_range(0..7) {
        0 => NeuralActivationGene::Tanh,
        1 => NeuralActivationGene::Sigmoid,
        2 => NeuralActivationGene::Sin,
        3 => NeuralActivationGene::Cos,
        4 => NeuralActivationGene::Identity,
        5 => NeuralActivationGene::Relu,
        _ => NeuralActivationGene::Softsign,
    }
}

fn random_neural_unit_gene(
    rng: &mut SmallRng,
    input_dim: usize,
    recurrent_dim: usize,
    global_dim: usize,
) -> NeuralUnitGene {
    NeuralUnitGene {
        activation: random_neural_activation(rng),
        input_weights: (0..input_dim).map(|_| rng_range(rng, -1.0, 1.0)).collect(),
        recurrent_weights: (0..recurrent_dim)
            .map(|_| rng_range(rng, -0.7, 0.7))
            .collect(),
        global_weights: (0..global_dim).map(|_| rng_range(rng, -0.9, 0.9)).collect(),
        bias: rng_range(rng, -0.8, 0.8),
        leak: rng_range(rng, 0.15, 0.95),
    }
}

fn random_effector_gene(
    rng: &mut SmallRng,
    local_dim: usize,
    global_dim: usize,
) -> JointEffectorGene {
    JointEffectorGene {
        local_weights: (0..local_dim).map(|_| rng_range(rng, -1.4, 1.4)).collect(),
        global_weights: (0..global_dim).map(|_| rng_range(rng, -1.0, 1.0)).collect(),
        bias: rng_range(rng, -0.6, 0.6),
        gain: rng_range(rng, 0.55, 1.45),
    }
}

fn random_global_brain_gene(rng: &mut SmallRng) -> GlobalBrainGene {
    let count = rng.random_range(MIN_GLOBAL_NEURONS..=MAX_GLOBAL_NEURONS.min(9));
    let neurons = (0..count)
        .map(|_| random_neural_unit_gene(rng, GLOBAL_SENSOR_DIM, count, 0))
        .collect::<Vec<_>>();
    GlobalBrainGene { neurons }
}

fn random_local_brain_gene(rng: &mut SmallRng, global_dim: usize) -> LocalBrainGene {
    let count = rng.random_range(MIN_LOCAL_NEURONS..=MAX_LOCAL_NEURONS.min(8));
    let neurons = (0..count)
        .map(|_| random_neural_unit_gene(rng, LOCAL_SENSOR_DIM, count, global_dim))
        .collect::<Vec<_>>();
    LocalBrainGene {
        neurons,
        effector_x: random_effector_gene(rng, count, global_dim),
        effector_y: random_effector_gene(rng, count, global_dim),
        effector_z: random_effector_gene(rng, count, global_dim),
    }
}

fn random_graph_edge_gene(rng: &mut SmallRng, to: usize) -> MorphEdgeGene {
    MorphEdgeGene {
        to,
        anchor_x: rng_range(rng, -0.95, 0.95),
        anchor_y: rng_range(rng, -0.95, 0.95),
        anchor_z: rng_range(rng, -0.95, 0.95),
        axis_y: rng_range(rng, -0.55, 0.55),
        axis_z: rng_range(rng, -0.55, 0.55),
        dir_x: rng_range(rng, -1.0, 1.0),
        dir_y: rng_range(rng, -1.0, 0.2),
        dir_z: rng_range(rng, -1.0, 1.0),
        scale: rng_range(rng, 0.65, 1.25),
        reflect_x: rng.random::<f32>() < 0.28,
        recursive_limit: rng.random_range(1..=3),
        terminal_only: rng.random::<f32>() < 0.2,
        joint_type: if rng.random::<f32>() < 0.35 {
            JointTypeGene::Ball
        } else {
            JointTypeGene::Hinge
        },
        limit_x: rng_range(rng, 0.4, 2.15),
        limit_y: rng_range(rng, 0.25, 1.35),
        limit_z: rng_range(rng, 0.25, 1.35),
        motor_strength: rng_range(rng, 0.65, 2.8),
        joint_stiffness: rng_range(rng, 24.0, 95.0),
    }
}

fn random_graph_gene(rng: &mut SmallRng) -> GraphGene {
    let global_brain = random_global_brain_gene(rng);
    let global_dim = global_brain.neurons.len();
    let node_count = rng.random_range(3..=7);
    let mut nodes = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        nodes.push(MorphNodeGene {
            part: GraphPartGene {
                w: rng_range(rng, 0.28, 1.45),
                h: rng_range(rng, 0.34, 1.9),
                d: rng_range(rng, 0.28, 1.45),
                mass: rng_range(rng, 0.32, 1.45),
            },
            edges: Vec::new(),
            brain: random_local_brain_gene(rng, global_dim),
        });
    }
    for index in 0..node_count.saturating_sub(1) {
        nodes[index]
            .edges
            .push(random_graph_edge_gene(rng, index + 1));
    }
    for index in 0..node_count {
        let extra_edges = rng.random_range(0..=MAX_GRAPH_EDGES_PER_NODE.min(3));
        for _ in 0..extra_edges {
            if nodes[index].edges.len() >= MAX_GRAPH_EDGES_PER_NODE {
                break;
            }
            let to = rng.random_range(0..node_count);
            nodes[index].edges.push(random_graph_edge_gene(rng, to));
        }
    }
    GraphGene {
        root: 0,
        nodes,
        global_brain,
        max_parts: rng.random_range(12..=MAX_GRAPH_PARTS.min(16)),
    }
}

fn spider_graph_gene_template() -> GraphGene {
    let global_brain = default_global_brain_gene();
    let global_dim = global_brain.neurons.len();
    let torso_brain = LocalBrainGene {
        neurons: vec![
            random_neural_unit_gene(
                &mut SmallRng::seed_from_u64(11),
                LOCAL_SENSOR_DIM,
                4,
                global_dim,
            ),
            random_neural_unit_gene(
                &mut SmallRng::seed_from_u64(12),
                LOCAL_SENSOR_DIM,
                4,
                global_dim,
            ),
            random_neural_unit_gene(
                &mut SmallRng::seed_from_u64(13),
                LOCAL_SENSOR_DIM,
                4,
                global_dim,
            ),
            random_neural_unit_gene(
                &mut SmallRng::seed_from_u64(14),
                LOCAL_SENSOR_DIM,
                4,
                global_dim,
            ),
        ],
        effector_x: default_joint_effector_gene(),
        effector_y: default_joint_effector_gene(),
        effector_z: default_joint_effector_gene(),
    };
    let leg_brain = LocalBrainGene {
        neurons: (0..5)
            .map(|offset| {
                random_neural_unit_gene(
                    &mut SmallRng::seed_from_u64(100 + offset as u64),
                    LOCAL_SENSOR_DIM,
                    5,
                    global_dim,
                )
            })
            .collect(),
        effector_x: JointEffectorGene {
            local_weights: vec![1.1, -0.6, 0.8, 0.0, 0.0],
            global_weights: vec![0.35; global_dim],
            bias: 0.0,
            gain: 1.25,
        },
        effector_y: JointEffectorGene {
            local_weights: vec![0.6, 0.3, -0.4, 0.2, 0.0],
            global_weights: vec![0.1; global_dim],
            bias: 0.0,
            gain: 0.85,
        },
        effector_z: JointEffectorGene {
            local_weights: vec![-0.5, 0.7, 0.2, -0.3, 0.0],
            global_weights: vec![0.08; global_dim],
            bias: 0.0,
            gain: 0.85,
        },
    };
    let tip_brain = LocalBrainGene {
        neurons: (0..4)
            .map(|offset| {
                random_neural_unit_gene(
                    &mut SmallRng::seed_from_u64(200 + offset as u64),
                    LOCAL_SENSOR_DIM,
                    4,
                    global_dim,
                )
            })
            .collect(),
        effector_x: JointEffectorGene {
            local_weights: vec![1.0, -0.2, 0.4, 0.3],
            global_weights: vec![0.1; global_dim],
            bias: -0.1,
            gain: 1.1,
        },
        effector_y: default_joint_effector_gene(),
        effector_z: default_joint_effector_gene(),
    };
    let root = MorphNodeGene {
        part: GraphPartGene {
            w: 1.68,
            h: 0.66,
            d: 1.22,
            mass: 1.08,
        },
        edges: vec![
            MorphEdgeGene {
                to: 1,
                anchor_x: 0.72,
                anchor_y: -0.18,
                anchor_z: 0.46,
                axis_y: 0.14,
                axis_z: 0.16,
                dir_x: 0.92,
                dir_y: 0.0,
                dir_z: 0.38,
                scale: 1.0,
                reflect_x: false,
                recursive_limit: 1,
                terminal_only: false,
                joint_type: JointTypeGene::Ball,
                limit_x: 1.57,
                limit_y: 1.08,
                limit_z: 1.08,
                motor_strength: 1.0,
                joint_stiffness: 48.0,
            },
            MorphEdgeGene {
                to: 1,
                anchor_x: -0.72,
                anchor_y: -0.18,
                anchor_z: 0.46,
                axis_y: 0.14,
                axis_z: -0.16,
                dir_x: -0.92,
                dir_y: 0.0,
                dir_z: 0.38,
                scale: 1.0,
                reflect_x: false,
                recursive_limit: 1,
                terminal_only: false,
                joint_type: JointTypeGene::Ball,
                limit_x: 1.57,
                limit_y: 1.08,
                limit_z: 1.08,
                motor_strength: 1.0,
                joint_stiffness: 48.0,
            },
            MorphEdgeGene {
                to: 1,
                anchor_x: 0.72,
                anchor_y: -0.18,
                anchor_z: -0.46,
                axis_y: -0.14,
                axis_z: 0.16,
                dir_x: 0.92,
                dir_y: 0.0,
                dir_z: -0.38,
                scale: 1.0,
                reflect_x: false,
                recursive_limit: 1,
                terminal_only: false,
                joint_type: JointTypeGene::Ball,
                limit_x: 1.57,
                limit_y: 1.08,
                limit_z: 1.08,
                motor_strength: 1.0,
                joint_stiffness: 48.0,
            },
            MorphEdgeGene {
                to: 1,
                anchor_x: -0.72,
                anchor_y: -0.18,
                anchor_z: -0.46,
                axis_y: -0.14,
                axis_z: -0.16,
                dir_x: -0.92,
                dir_y: 0.0,
                dir_z: -0.38,
                scale: 1.0,
                reflect_x: false,
                recursive_limit: 1,
                terminal_only: false,
                joint_type: JointTypeGene::Ball,
                limit_x: 1.57,
                limit_y: 1.08,
                limit_z: 1.08,
                motor_strength: 1.0,
                joint_stiffness: 48.0,
            },
        ],
        brain: torso_brain,
    };
    let leg = MorphNodeGene {
        part: GraphPartGene {
            w: 0.24,
            h: 1.33,
            d: 0.24,
            mass: 0.8,
        },
        edges: vec![MorphEdgeGene {
            to: 2,
            anchor_x: 0.0,
            anchor_y: -0.48,
            anchor_z: 0.0,
            axis_y: 0.0,
            axis_z: 0.0,
            dir_x: 0.0,
            dir_y: -1.0,
            dir_z: 0.0,
            scale: 1.0,
            reflect_x: false,
            recursive_limit: 1,
            terminal_only: false,
            joint_type: JointTypeGene::Hinge,
            limit_x: 2.09,
            limit_y: 1.22,
            limit_z: 1.22,
            motor_strength: 0.9,
            joint_stiffness: 44.0,
        }],
        brain: leg_brain,
    };
    let tip = MorphNodeGene {
        part: GraphPartGene {
            w: 0.21,
            h: 1.33,
            d: 0.21,
            mass: 0.58,
        },
        edges: Vec::new(),
        brain: tip_brain,
    };
    GraphGene {
        root: 0,
        nodes: vec![root, leg, tip],
        global_brain,
        max_parts: 16,
    }
}

fn spider4x2_template_genome() -> Genome {
    let torso = TorsoGene {
        w: 1.68,
        h: 0.66,
        d: 1.22,
        mass: 1.08,
    };
    let mut limbs = vec![disabled_limb_template(); MAX_LIMBS];
    let anchor_y = -0.18;
    let freq = 1.7;
    limbs[0] = spider_leg_template(
        0.72,
        anchor_y,
        0.46,
        0.14,
        0.16,
        [0.92, 0.0, 0.38],
        0.0,
        freq,
    );
    limbs[1] = spider_leg_template(
        -0.72,
        anchor_y,
        0.46,
        0.14,
        -0.16,
        [-0.92, 0.0, 0.38],
        PI,
        freq,
    );
    limbs[2] = spider_leg_template(
        0.72,
        anchor_y,
        -0.46,
        -0.14,
        0.16,
        [0.92, 0.0, -0.38],
        PI,
        freq,
    );
    limbs[3] = spider_leg_template(
        -0.72,
        anchor_y,
        -0.46,
        -0.14,
        -0.16,
        [-0.92, 0.0, -0.38],
        0.0,
        freq,
    );
    let graph = spider_graph_gene_template();
    Genome {
        version: default_genome_version(),
        graph,
        torso,
        limbs,
        hue: 0.0,
        mass_scale: 1.0,
    }
}

fn disabled_limb_template() -> LimbGene {
    LimbGene {
        enabled: false,
        segment_count: 1,
        anchor_x: 0.0,
        anchor_y: 0.0,
        anchor_z: 0.0,
        axis_y: 0.0,
        axis_z: 0.0,
        dir_x: 0.0,
        dir_y: -1.0,
        dir_z: 0.0,
        segments: (0..MAX_SEGMENTS_PER_LIMB)
            .map(|_| SegmentGene {
                length: 0.9,
                thickness: 0.2,
                mass: 0.45,
                limit_x: 1.57,
                limit_y: 1.08,
                limit_z: 1.08,
                joint_type: JointTypeGene::Hinge,
                motor_strength: 1.0,
                joint_stiffness: 45.0,
            })
            .collect(),
        controls: (0..MAX_SEGMENTS_PER_LIMB)
            .map(|_| default_control_gene())
            .collect(),
    }
}

fn spider_leg_template(
    anchor_x: f32,
    anchor_y: f32,
    anchor_z: f32,
    axis_y: f32,
    axis_z: f32,
    dir: [f32; 3],
    phase: f32,
    freq: f32,
) -> LimbGene {
    let mut segments = vec![
        SegmentGene {
            length: 1.33,
            thickness: 0.24,
            mass: 0.8,
            limit_x: 1.57,
            limit_y: 1.08,
            limit_z: 1.08,
            joint_type: JointTypeGene::Ball,
            motor_strength: 1.0,
            joint_stiffness: 48.0,
        },
        SegmentGene {
            length: 1.33,
            thickness: 0.21,
            mass: 0.58,
            limit_x: 2.09,
            limit_y: 1.22,
            limit_z: 1.22,
            joint_type: JointTypeGene::Hinge,
            motor_strength: 0.9,
            joint_stiffness: 44.0,
        },
    ];
    while segments.len() < MAX_SEGMENTS_PER_LIMB {
        segments.push(SegmentGene {
            length: 0.56,
            thickness: 0.20,
            mass: 0.52,
            limit_x: 2.09,
            limit_y: 1.22,
            limit_z: 1.22,
            joint_type: JointTypeGene::Hinge,
            motor_strength: 0.88,
            joint_stiffness: 40.0,
        });
    }

    let mut controls = vec![
        ControlGene {
            amp: 2.9,
            freq,
            phase,
            bias: 0.0,
            harm2_amp: 0.8,
            harm2_phase: wrap_phase(phase + PI * 0.5),
            amp_y: 2.1,
            freq_y: freq * 0.92,
            phase_y: wrap_phase(phase + PI * 0.28),
            bias_y: 0.0,
            amp_z: 1.8,
            freq_z: freq * 1.08,
            phase_z: wrap_phase(phase + PI * 0.66),
            bias_z: 0.0,
        },
        ControlGene {
            amp: 2.2,
            freq,
            phase: wrap_phase(phase + PI * 0.45),
            bias: -0.16,
            harm2_amp: 0.45,
            harm2_phase: wrap_phase(phase + PI * 0.8),
            amp_y: 0.0,
            freq_y: default_secondary_control_freq(),
            phase_y: default_secondary_control_phase(),
            bias_y: default_secondary_control_bias(),
            amp_z: 0.0,
            freq_z: default_secondary_control_freq(),
            phase_z: default_secondary_control_phase(),
            bias_z: default_secondary_control_bias(),
        },
    ];
    while controls.len() < MAX_SEGMENTS_PER_LIMB {
        let mut control = default_control_gene();
        control.freq = freq;
        control.phase = phase;
        controls.push(control);
    }

    LimbGene {
        enabled: true,
        segment_count: 2,
        anchor_x,
        anchor_y,
        anchor_z,
        axis_y,
        axis_z,
        dir_x: dir[0],
        dir_y: dir[1],
        dir_z: dir[2],
        segments,
        controls,
    }
}

fn random_genome(rng: &mut SmallRng) -> Genome {
    let torso = TorsoGene {
        w: clamp(
            rng_range(rng, 0.5, 2.75) * rng_range(rng, 0.86, 1.18),
            0.5,
            3.0,
        ),
        h: clamp(
            rng_range(rng, 0.5, 2.75) * rng_range(rng, 0.72, 1.15),
            0.5,
            3.0,
        ),
        d: clamp(
            rng_range(rng, 0.5, 2.75) * rng_range(rng, 0.86, 1.18),
            0.5,
            3.0,
        ),
        mass: rng_range(rng, 0.26, 1.7),
    };

    let limbs = (0..MAX_LIMBS)
        .map(|_| {
            let anchor = random_torso_surface_anchor(rng, &torso);
            let dir = random_unit_vec3(rng);
            let segments = (0..MAX_SEGMENTS_PER_LIMB)
                .map(|seg_index| {
                    let hierarchy_scale = rng_range(rng, 0.86, 1.16).powf(seg_index as f32);
                    SegmentGene {
                        length: clamp(rng_range(rng, 0.5, 2.25) * hierarchy_scale, 0.45, 2.5),
                        thickness: clamp(rng_range(rng, 0.16, 0.95) * hierarchy_scale, 0.14, 1.05),
                        mass: clamp(rng_range(rng, 0.24, 1.75) * hierarchy_scale, 0.14, 2.0),
                        limit_x: if seg_index == 0 {
                            rng_range(rng, 0.7, 1.85)
                        } else {
                            rng_range(rng, 1.0, 2.35)
                        },
                        limit_y: if seg_index == 0 {
                            rng_range(rng, 0.45, 1.25)
                        } else {
                            rng_range(rng, 0.35, 1.4)
                        },
                        limit_z: if seg_index == 0 {
                            rng_range(rng, 0.45, 1.25)
                        } else {
                            rng_range(rng, 0.35, 1.4)
                        },
                        joint_type: if seg_index == 0 {
                            if rng.random::<f32>() < 0.45 {
                                JointTypeGene::Ball
                            } else {
                                JointTypeGene::Hinge
                            }
                        } else if rng.random::<f32>() < 0.18 {
                            JointTypeGene::Ball
                        } else {
                            JointTypeGene::Hinge
                        },
                        motor_strength: rng_range(rng, 0.5, 5.0),
                        joint_stiffness: rng_range(rng, 15.0, 120.0),
                    }
                })
                .collect::<Vec<_>>();
            let controls = (0..MAX_SEGMENTS_PER_LIMB)
                .map(|seg_index| {
                    let joint_type = segments
                        .get(seg_index)
                        .map(|segment| segment.joint_type)
                        .unwrap_or(JointTypeGene::Hinge);
                    let mut control = ControlGene {
                        amp: if seg_index == 0 {
                            rng_range(rng, 1.7, 10.3)
                        } else {
                            rng_range(rng, 1.05, 9.1)
                        },
                        freq: rng_range(rng, 0.55, 4.4),
                        phase: rng_range(rng, 0.0, PI * 2.0),
                        bias: rng_range(rng, -1.6, 1.6),
                        harm2_amp: rng_range(rng, 0.0, 2.4),
                        harm2_phase: rng_range(rng, 0.0, PI * 2.0),
                        amp_y: default_secondary_control_amp(),
                        freq_y: default_secondary_control_freq(),
                        phase_y: default_secondary_control_phase(),
                        bias_y: default_secondary_control_bias(),
                        amp_z: default_secondary_control_amp(),
                        freq_z: default_secondary_control_freq(),
                        phase_z: default_secondary_control_phase(),
                        bias_z: default_secondary_control_bias(),
                    };
                    if matches!(joint_type, JointTypeGene::Ball) {
                        control.amp_y = rng_range(rng, 0.0, 4.4);
                        control.freq_y = rng_range(rng, 0.55, 4.4);
                        control.phase_y = rng_range(rng, 0.0, PI * 2.0);
                        control.bias_y = rng_range(rng, -1.6, 1.6);
                        control.amp_z = rng_range(rng, 0.0, 4.4);
                        control.freq_z = rng_range(rng, 0.55, 4.4);
                        control.phase_z = rng_range(rng, 0.0, PI * 2.0);
                        control.bias_z = rng_range(rng, -1.6, 1.6);
                    }
                    control
                })
                .collect::<Vec<_>>();
            LimbGene {
                enabled: rng.random::<f32>() < 0.78,
                segment_count: rng.random_range(1..=MAX_SEGMENTS_PER_LIMB as u32),
                anchor_x: anchor[0],
                anchor_y: anchor[1],
                anchor_z: anchor[2],
                axis_y: rng_range(rng, -0.35, 0.35),
                axis_z: rng_range(rng, -0.35, 0.35),
                dir_x: dir[0],
                dir_y: dir[1],
                dir_z: dir[2],
                segments,
                controls,
            }
        })
        .collect::<Vec<_>>();

    let mut genome = Genome {
        version: default_genome_version(),
        graph: random_graph_gene(rng),
        torso,
        limbs,
        hue: rng.random::<f32>(),
        mass_scale: rng_range(rng, 0.78, 1.3),
    };
    project_graph_to_legacy(&mut genome);
    ensure_active_body_plan(&mut genome, rng);
    genome
}

fn mutate_weight_vector(weights: &mut Vec<f32>, chance: f32, scale: f32, rng: &mut SmallRng) {
    for weight in weights {
        if rng.random::<f32>() < chance {
            *weight = clamp(*weight + rand_normal(rng) * scale, -3.5, 3.5);
        }
    }
}

fn blend_weight_vectors(a: &[f32], b: &[f32], rng: &mut SmallRng) -> Vec<f32> {
    let len = a.len().max(b.len());
    let mut result = Vec::with_capacity(len);
    for i in 0..len {
        let av = a.get(i).copied().unwrap_or(0.0);
        let bv = b.get(i).copied().unwrap_or(0.0);
        let t = rng_range(rng, 0.15, 0.85);
        result.push(lerp(av, bv, t));
    }
    result
}

fn resize_neural_gene(
    neuron: &mut NeuralUnitGene,
    input_dim: usize,
    recurrent_dim: usize,
    global_dim: usize,
) {
    neuron.input_weights.resize(input_dim, 0.0);
    neuron.recurrent_weights.resize(recurrent_dim, 0.0);
    neuron.global_weights.resize(global_dim, 0.0);
    neuron.leak = clamp(neuron.leak, 0.05, 1.0);
}

fn resize_global_brain(
    brain: &mut GlobalBrainGene,
    target_count: Option<usize>,
    rng: &mut SmallRng,
) {
    let requested = target_count.unwrap_or_else(|| brain.neurons.len().max(MIN_GLOBAL_NEURONS));
    let neuron_count = requested.clamp(MIN_GLOBAL_NEURONS, MAX_GLOBAL_NEURONS);
    brain.neurons.truncate(neuron_count);
    while brain.neurons.len() < neuron_count {
        brain.neurons.push(random_neural_unit_gene(
            rng,
            GLOBAL_SENSOR_DIM,
            neuron_count,
            0,
        ));
    }
    for neuron in &mut brain.neurons {
        resize_neural_gene(neuron, GLOBAL_SENSOR_DIM, neuron_count, 0);
    }
}

fn resize_local_brain(
    brain: &mut LocalBrainGene,
    global_dim: usize,
    target_count: Option<usize>,
    rng: &mut SmallRng,
) {
    let requested = target_count.unwrap_or_else(|| brain.neurons.len().max(MIN_LOCAL_NEURONS));
    let neuron_count = requested.clamp(MIN_LOCAL_NEURONS, MAX_LOCAL_NEURONS);
    brain.neurons.truncate(neuron_count);
    while brain.neurons.len() < neuron_count {
        brain
            .neurons
            .push(random_neural_unit_gene(rng, LOCAL_SENSOR_DIM, neuron_count, global_dim));
    }
    for neuron in &mut brain.neurons {
        resize_neural_gene(neuron, LOCAL_SENSOR_DIM, neuron_count, global_dim);
    }
    brain.effector_x.local_weights.resize(neuron_count, 0.0);
    brain.effector_y.local_weights.resize(neuron_count, 0.0);
    brain.effector_z.local_weights.resize(neuron_count, 0.0);
    brain.effector_x.global_weights.resize(global_dim, 0.0);
    brain.effector_y.global_weights.resize(global_dim, 0.0);
    brain.effector_z.global_weights.resize(global_dim, 0.0);
    brain.effector_x.gain = clamp(brain.effector_x.gain, 0.2, 2.0);
    brain.effector_y.gain = clamp(brain.effector_y.gain, 0.2, 2.0);
    brain.effector_z.gain = clamp(brain.effector_z.gain, 0.2, 2.0);
}

fn ensure_graph_valid(graph: &mut GraphGene, rng: &mut SmallRng) {
    if graph.nodes.is_empty() {
        *graph = random_graph_gene(rng);
        return;
    }
    graph.max_parts = graph.max_parts.clamp(6, MAX_GRAPH_PARTS);
    graph.root = graph.root.min(graph.nodes.len().saturating_sub(1));
    resize_global_brain(&mut graph.global_brain, None, rng);
    let global_count = graph.global_brain.neurons.len();
    let node_len = graph.nodes.len().max(1);
    for node in &mut graph.nodes {
        resize_local_brain(&mut node.brain, global_count, None, rng);
        node.part.w = clamp(node.part.w, 0.14, 2.8);
        node.part.h = clamp(node.part.h, 0.2, 3.4);
        node.part.d = clamp(node.part.d, 0.14, 2.8);
        node.part.mass = clamp(node.part.mass, 0.08, 3.4);
        node.edges.truncate(MAX_GRAPH_EDGES_PER_NODE);
        for edge in &mut node.edges {
            edge.to = edge.to.min(node_len.saturating_sub(1));
            edge.anchor_x = clamp(edge.anchor_x, -0.98, 0.98);
            edge.anchor_y = clamp(edge.anchor_y, -0.98, 0.98);
            edge.anchor_z = clamp(edge.anchor_z, -0.98, 0.98);
            edge.axis_y = clamp(edge.axis_y, -0.75, 0.75);
            edge.axis_z = clamp(edge.axis_z, -0.75, 0.75);
            edge.dir_x = clamp(edge.dir_x, -1.4, 1.4);
            edge.dir_y = clamp(edge.dir_y, -1.4, 1.0);
            edge.dir_z = clamp(edge.dir_z, -1.4, 1.4);
            edge.scale = clamp(edge.scale, 0.3, 1.8);
            edge.recursive_limit = edge.recursive_limit.max(1).min(5);
            edge.limit_x = clamp(edge.limit_x, 0.12, PI * 0.95);
            edge.limit_y = clamp(edge.limit_y, 0.10, PI * 0.75);
            edge.limit_z = clamp(edge.limit_z, 0.10, PI * 0.75);
            edge.motor_strength = clamp(edge.motor_strength, 0.3, 4.5);
            edge.joint_stiffness = clamp(edge.joint_stiffness, 12.0, 160.0);
        }
        if node.edges.is_empty() {
            let to = rng.random_range(0..node_len);
            node.edges.push(random_graph_edge_gene(rng, to));
        }
    }
}

fn crossover_graph_gene(a: &GraphGene, b: &GraphGene, rng: &mut SmallRng) -> GraphGene {
    let mut child = if rng.random::<f32>() < 0.5 {
        a.clone()
    } else {
        b.clone()
    };
    let node_count = a.nodes.len().max(b.nodes.len()).clamp(1, MAX_GRAPH_NODES);
    child.nodes.clear();
    for i in 0..node_count {
        let node = match (a.nodes.get(i), b.nodes.get(i)) {
            (Some(na), Some(nb)) => {
                let mut blended = if rng.random::<f32>() < 0.5 {
                    na.clone()
                } else {
                    nb.clone()
                };
                let t = rng_range(rng, 0.2, 0.8);
                blended.part.w = lerp(na.part.w, nb.part.w, t);
                blended.part.h = lerp(na.part.h, nb.part.h, t);
                blended.part.d = lerp(na.part.d, nb.part.d, t);
                blended.part.mass = lerp(na.part.mass, nb.part.mass, t);
                if rng.random::<f32>() < 0.5 {
                    blended.edges = na.edges.clone();
                } else {
                    blended.edges = nb.edges.clone();
                }
                let local_count = na
                    .brain
                    .neurons
                    .len()
                    .max(nb.brain.neurons.len())
                    .clamp(MIN_LOCAL_NEURONS, MAX_LOCAL_NEURONS);
                blended.brain.neurons.truncate(local_count);
                while blended.brain.neurons.len() < local_count {
                    blended.brain.neurons.push(random_neural_unit_gene(
                        rng,
                        LOCAL_SENSOR_DIM,
                        local_count,
                        child.global_brain.neurons.len(),
                    ));
                }
                for neuron_index in 0..local_count {
                    let an = na.brain.neurons.get(neuron_index);
                    let bn = nb.brain.neurons.get(neuron_index);
                    if let (Some(an), Some(bn), Some(target)) =
                        (an, bn, blended.brain.neurons.get_mut(neuron_index))
                    {
                        target.activation = if rng.random::<f32>() < 0.5 {
                            an.activation
                        } else {
                            bn.activation
                        };
                        target.input_weights =
                            blend_weight_vectors(&an.input_weights, &bn.input_weights, rng);
                        target.recurrent_weights =
                            blend_weight_vectors(&an.recurrent_weights, &bn.recurrent_weights, rng);
                        target.global_weights =
                            blend_weight_vectors(&an.global_weights, &bn.global_weights, rng);
                        target.bias = lerp(an.bias, bn.bias, rng.random::<f32>());
                        target.leak = lerp(an.leak, bn.leak, rng.random::<f32>());
                    }
                }
                blended.brain.effector_x.local_weights = blend_weight_vectors(
                    &na.brain.effector_x.local_weights,
                    &nb.brain.effector_x.local_weights,
                    rng,
                );
                blended.brain.effector_y.local_weights = blend_weight_vectors(
                    &na.brain.effector_y.local_weights,
                    &nb.brain.effector_y.local_weights,
                    rng,
                );
                blended.brain.effector_z.local_weights = blend_weight_vectors(
                    &na.brain.effector_z.local_weights,
                    &nb.brain.effector_z.local_weights,
                    rng,
                );
                blended
            }
            (Some(na), None) => na.clone(),
            (None, Some(nb)) => nb.clone(),
            (None, None) => random_graph_gene(rng).nodes[0].clone(),
        };
        child.nodes.push(node);
    }
    if !a.global_brain.neurons.is_empty() && !b.global_brain.neurons.is_empty() {
        let global_count = a
            .global_brain
            .neurons
            .len()
            .max(b.global_brain.neurons.len())
            .clamp(MIN_GLOBAL_NEURONS, MAX_GLOBAL_NEURONS);
        child.global_brain.neurons.truncate(global_count);
        while child.global_brain.neurons.len() < global_count {
            child.global_brain.neurons.push(random_neural_unit_gene(
                rng,
                GLOBAL_SENSOR_DIM,
                global_count,
                0,
            ));
        }
        for index in 0..global_count {
            if let (Some(ga), Some(gb), Some(target)) = (
                a.global_brain.neurons.get(index),
                b.global_brain.neurons.get(index),
                child.global_brain.neurons.get_mut(index),
            ) {
                target.activation = if rng.random::<f32>() < 0.5 {
                    ga.activation
                } else {
                    gb.activation
                };
                target.input_weights =
                    blend_weight_vectors(&ga.input_weights, &gb.input_weights, rng);
                target.recurrent_weights =
                    blend_weight_vectors(&ga.recurrent_weights, &gb.recurrent_weights, rng);
                target.bias = lerp(ga.bias, gb.bias, rng.random::<f32>());
                target.leak = lerp(ga.leak, gb.leak, rng.random::<f32>());
            }
        }
    }
    child.max_parts = ((a.max_parts + b.max_parts) / 2).clamp(6, MAX_GRAPH_PARTS);
    child.root = child.root.min(child.nodes.len().saturating_sub(1));
    ensure_graph_valid(&mut child, rng);
    child
}

fn mutate_graph_gene(graph: &mut GraphGene, chance: f32, structural: bool, rng: &mut SmallRng) {
    if graph.nodes.is_empty() {
        *graph = random_graph_gene(rng);
        return;
    }
    let mut local_chance = chance;
    if structural && rng.random::<f32>() < chance * 0.28 && graph.nodes.len() < MAX_GRAPH_NODES {
        let current_node_len = graph.nodes.len().max(1);
        let to = rng.random_range(0..current_node_len);
        graph.nodes.push(MorphNodeGene {
            part: GraphPartGene {
                w: rng_range(rng, 0.2, 1.4),
                h: rng_range(rng, 0.3, 1.8),
                d: rng_range(rng, 0.2, 1.4),
                mass: rng_range(rng, 0.22, 1.7),
            },
            edges: vec![random_graph_edge_gene(rng, to)],
            brain: random_local_brain_gene(rng, graph.global_brain.neurons.len()),
        });
        local_chance = (local_chance + 0.05).min(1.0);
    }
    if structural && rng.random::<f32>() < chance * 0.18 && graph.nodes.len() > 3 {
        let remove_index = rng.random_range(1..graph.nodes.len());
        graph.nodes.remove(remove_index);
        let fallback_to = graph.root.min(graph.nodes.len().saturating_sub(1));
        for node in &mut graph.nodes {
            for edge in &mut node.edges {
                if edge.to == remove_index {
                    edge.to = fallback_to;
                } else if edge.to > remove_index {
                    edge.to -= 1;
                }
            }
        }
    }
    if structural && rng.random::<f32>() < chance * 0.4 {
        graph.max_parts = (graph.max_parts as i32 + if rng.random::<f32>() < 0.5 { -3 } else { 3 })
            .clamp(6, MAX_GRAPH_PARTS as i32) as usize;
    }
    let mut global_target = graph
        .global_brain
        .neurons
        .len()
        .clamp(MIN_GLOBAL_NEURONS, MAX_GLOBAL_NEURONS);
    if structural && rng.random::<f32>() < chance * 0.22 {
        let delta = if rng.random::<f32>() < 0.5 { -1 } else { 1 };
        global_target = (global_target as i32 + delta)
            .clamp(MIN_GLOBAL_NEURONS as i32, MAX_GLOBAL_NEURONS as i32)
            as usize;
    }
    resize_global_brain(&mut graph.global_brain, Some(global_target), rng);
    for neuron in &mut graph.global_brain.neurons {
        mutate_weight_vector(&mut neuron.input_weights, local_chance * 0.7, 0.22, rng);
        mutate_weight_vector(&mut neuron.recurrent_weights, local_chance * 0.7, 0.18, rng);
        if rng.random::<f32>() < local_chance * 0.35 {
            neuron.bias = clamp(neuron.bias + rand_normal(rng) * 0.22, -2.0, 2.0);
        }
        if rng.random::<f32>() < local_chance * 0.25 {
            neuron.leak = clamp(neuron.leak + rand_normal(rng) * 0.14, 0.05, 1.0);
        }
        if rng.random::<f32>() < local_chance * 0.08 {
            neuron.activation = random_neural_activation(rng);
        }
    }
    let global_dim = graph.global_brain.neurons.len();
    let node_len = graph.nodes.len().max(1);
    for node in &mut graph.nodes {
        node.part.w = mutate_number(node.part.w, 0.14, 2.8, local_chance, 0.14, rng);
        node.part.h = mutate_number(node.part.h, 0.2, 3.4, local_chance, 0.14, rng);
        node.part.d = mutate_number(node.part.d, 0.14, 2.8, local_chance, 0.14, rng);
        node.part.mass = mutate_number(node.part.mass, 0.08, 3.4, local_chance, 0.18, rng);
        let mut local_target = node
            .brain
            .neurons
            .len()
            .clamp(MIN_LOCAL_NEURONS, MAX_LOCAL_NEURONS);
        if structural && rng.random::<f32>() < local_chance * 0.28 {
            let delta = if rng.random::<f32>() < 0.5 { -1 } else { 1 };
            local_target = (local_target as i32 + delta)
                .clamp(MIN_LOCAL_NEURONS as i32, MAX_LOCAL_NEURONS as i32)
                as usize;
        }
        resize_local_brain(&mut node.brain, global_dim, Some(local_target), rng);
        for neuron in &mut node.brain.neurons {
            mutate_weight_vector(&mut neuron.input_weights, local_chance * 0.75, 0.24, rng);
            mutate_weight_vector(
                &mut neuron.recurrent_weights,
                local_chance * 0.75,
                0.18,
                rng,
            );
            mutate_weight_vector(&mut neuron.global_weights, local_chance * 0.75, 0.18, rng);
            if rng.random::<f32>() < local_chance * 0.35 {
                neuron.bias = clamp(neuron.bias + rand_normal(rng) * 0.3, -2.5, 2.5);
            }
            if rng.random::<f32>() < local_chance * 0.25 {
                neuron.leak = clamp(neuron.leak + rand_normal(rng) * 0.16, 0.05, 1.0);
            }
            if rng.random::<f32>() < local_chance * 0.1 {
                neuron.activation = random_neural_activation(rng);
            }
        }
        mutate_weight_vector(
            &mut node.brain.effector_x.local_weights,
            local_chance * 0.7,
            0.28,
            rng,
        );
        mutate_weight_vector(
            &mut node.brain.effector_y.local_weights,
            local_chance * 0.7,
            0.28,
            rng,
        );
        mutate_weight_vector(
            &mut node.brain.effector_z.local_weights,
            local_chance * 0.7,
            0.28,
            rng,
        );
        mutate_weight_vector(
            &mut node.brain.effector_x.global_weights,
            local_chance * 0.7,
            0.24,
            rng,
        );
        mutate_weight_vector(
            &mut node.brain.effector_y.global_weights,
            local_chance * 0.7,
            0.24,
            rng,
        );
        mutate_weight_vector(
            &mut node.brain.effector_z.global_weights,
            local_chance * 0.7,
            0.24,
            rng,
        );
        if rng.random::<f32>() < local_chance * 0.25 {
            node.brain.effector_x.bias += rand_normal(rng) * 0.16;
            node.brain.effector_y.bias += rand_normal(rng) * 0.16;
            node.brain.effector_z.bias += rand_normal(rng) * 0.16;
        }
        if rng.random::<f32>() < local_chance * 0.25 {
            node.brain.effector_x.gain = clamp(
                node.brain.effector_x.gain + rand_normal(rng) * 0.12,
                0.2,
                2.0,
            );
            node.brain.effector_y.gain = clamp(
                node.brain.effector_y.gain + rand_normal(rng) * 0.12,
                0.2,
                2.0,
            );
            node.brain.effector_z.gain = clamp(
                node.brain.effector_z.gain + rand_normal(rng) * 0.12,
                0.2,
                2.0,
            );
        }
        for edge in &mut node.edges {
            edge.anchor_x = mutate_number(edge.anchor_x, -0.98, 0.98, local_chance, 0.18, rng);
            edge.anchor_y = mutate_number(edge.anchor_y, -0.98, 0.98, local_chance, 0.18, rng);
            edge.anchor_z = mutate_number(edge.anchor_z, -0.98, 0.98, local_chance, 0.18, rng);
            edge.axis_y = mutate_number(edge.axis_y, -0.75, 0.75, local_chance, 0.18, rng);
            edge.axis_z = mutate_number(edge.axis_z, -0.75, 0.75, local_chance, 0.18, rng);
            edge.dir_x = mutate_number(edge.dir_x, -1.4, 1.4, local_chance, 0.2, rng);
            edge.dir_y = mutate_number(edge.dir_y, -1.4, 1.0, local_chance, 0.2, rng);
            edge.dir_z = mutate_number(edge.dir_z, -1.4, 1.4, local_chance, 0.2, rng);
            edge.scale = mutate_number(edge.scale, 0.3, 1.8, local_chance, 0.16, rng);
            edge.limit_x = mutate_number(edge.limit_x, 0.12, PI * 0.95, local_chance, 0.14, rng);
            edge.limit_y = mutate_number(edge.limit_y, 0.10, PI * 0.75, local_chance, 0.14, rng);
            edge.limit_z = mutate_number(edge.limit_z, 0.10, PI * 0.75, local_chance, 0.14, rng);
            edge.motor_strength =
                mutate_number(edge.motor_strength, 0.3, 4.5, local_chance, 0.2, rng);
            edge.joint_stiffness =
                mutate_number(edge.joint_stiffness, 12.0, 160.0, local_chance, 0.16, rng);
            if structural && rng.random::<f32>() < local_chance * 0.18 {
                edge.reflect_x = !edge.reflect_x;
            }
            if structural && rng.random::<f32>() < local_chance * 0.12 {
                edge.terminal_only = !edge.terminal_only;
            }
            if structural && rng.random::<f32>() < local_chance * 0.16 {
                edge.joint_type = match edge.joint_type {
                    JointTypeGene::Hinge => JointTypeGene::Ball,
                    JointTypeGene::Ball => JointTypeGene::Hinge,
                };
            }
            if structural && rng.random::<f32>() < local_chance * 0.15 {
                let delta = if rng.random::<f32>() < 0.5 {
                    -1i32
                } else {
                    1i32
                };
                edge.recursive_limit = (edge.recursive_limit as i32 + delta).clamp(1, 5) as u32;
            }
            if structural && rng.random::<f32>() < local_chance * 0.2 {
                edge.to = rng.random_range(0..node_len);
            }
        }
        if structural
            && node.edges.len() < MAX_GRAPH_EDGES_PER_NODE
            && rng.random::<f32>() < local_chance * 0.32
        {
            let to = rng.random_range(0..node_len);
            node.edges.push(random_graph_edge_gene(rng, to));
        }
        if structural && node.edges.len() > 1 && rng.random::<f32>() < local_chance * 0.18 {
            let remove = rng.random_range(0..node.edges.len());
            node.edges.remove(remove);
        }
    }
    ensure_graph_valid(graph, rng);
}

fn collect_reachable_subgraph_indices(
    graph: &GraphGene,
    start_index: usize,
    max_nodes: usize,
) -> Vec<usize> {
    if graph.nodes.is_empty() || max_nodes == 0 {
        return Vec::new();
    }
    let start = start_index.min(graph.nodes.len().saturating_sub(1));
    let mut visited = vec![false; graph.nodes.len()];
    let mut queue = VecDeque::new();
    let mut ordered = Vec::with_capacity(max_nodes.min(graph.nodes.len()));
    visited[start] = true;
    queue.push_back(start);
    while let Some(index) = queue.pop_front() {
        ordered.push(index);
        if ordered.len() >= max_nodes {
            break;
        }
        if let Some(node) = graph.nodes.get(index) {
            for edge in node.edges.iter().take(MAX_GRAPH_EDGES_PER_NODE) {
                let to = edge.to.min(graph.nodes.len().saturating_sub(1));
                if !visited[to] {
                    visited[to] = true;
                    queue.push_back(to);
                }
            }
        }
    }
    ordered
}

fn graft_graph_gene(a: &GraphGene, b: &GraphGene, rng: &mut SmallRng) -> GraphGene {
    let mut child = a.clone();
    if child.nodes.is_empty() {
        return b.clone();
    }
    if b.nodes.is_empty() {
        return child;
    }
    let replace_index = rng.random_range(0..child.nodes.len());
    let donor_root = rng.random_range(0..b.nodes.len());
    let room_for_new_nodes = MAX_GRAPH_NODES.saturating_sub(child.nodes.len());
    let donor_indices =
        collect_reachable_subgraph_indices(b, donor_root, room_for_new_nodes.saturating_add(1));
    if donor_indices.is_empty() {
        ensure_graph_valid(&mut child, rng);
        return child;
    }
    let mut donor_to_child: HashMap<usize, usize> = HashMap::new();
    donor_to_child.insert(donor_root, replace_index);
    for donor_index in donor_indices {
        if donor_index == donor_root {
            continue;
        }
        if child.nodes.len() >= MAX_GRAPH_NODES {
            break;
        }
        let new_child_index = child.nodes.len();
        child.nodes.push(b.nodes[donor_index].clone());
        donor_to_child.insert(donor_index, new_child_index);
    }
    let mut mapped_indices = donor_to_child
        .iter()
        .map(|(&donor_index, &child_index)| (donor_index, child_index))
        .collect::<Vec<_>>();
    mapped_indices.sort_by_key(|(_, child_index)| *child_index);
    for (donor_index, child_index) in mapped_indices {
        let mut donor_node = b.nodes[donor_index].clone();
        donor_node.edges.truncate(MAX_GRAPH_EDGES_PER_NODE);
        for edge in &mut donor_node.edges {
            edge.to = if let Some(mapped) = donor_to_child.get(&edge.to) {
                *mapped
            } else {
                replace_index
            };
        }
        child.nodes[child_index] = donor_node;
    }
    if rng.random::<f32>() < 0.5 {
        child.global_brain = b.global_brain.clone();
    }
    child.max_parts = ((child.max_parts + b.max_parts) / 2).clamp(6, MAX_GRAPH_PARTS);
    ensure_graph_valid(&mut child, rng);
    child
}

fn project_graph_to_legacy(genome: &mut Genome) {
    if genome.graph.nodes.is_empty() {
        return;
    }
    let root_index = genome
        .graph
        .root
        .min(genome.graph.nodes.len().saturating_sub(1));
    let root = &genome.graph.nodes[root_index];
    genome.torso.w = root.part.w;
    genome.torso.h = root.part.h;
    genome.torso.d = root.part.d;
    genome.torso.mass = root.part.mass;
    let mut limbs = vec![disabled_limb_template(); MAX_LIMBS];
    for (limb_index, edge) in root.edges.iter().take(MAX_LIMBS).enumerate() {
        let node = genome
            .graph
            .nodes
            .get(edge.to.min(genome.graph.nodes.len().saturating_sub(1)))
            .cloned()
            .unwrap_or_else(default_graph_node_gene);
        let mut limb = disabled_limb_template();
        limb.enabled = true;
        limb.segment_count = 1;
        limb.anchor_x = edge.anchor_x * root.part.w * 0.5;
        limb.anchor_y = edge.anchor_y * root.part.h * 0.5;
        limb.anchor_z = edge.anchor_z * root.part.d * 0.5;
        limb.axis_y = edge.axis_y;
        limb.axis_z = edge.axis_z;
        limb.dir_x = edge.dir_x;
        limb.dir_y = edge.dir_y;
        limb.dir_z = edge.dir_z;
        if let Some(segment) = limb.segments.get_mut(0) {
            segment.length = node.part.h;
            segment.thickness = (node.part.w + node.part.d) * 0.25;
            segment.mass = node.part.mass;
            segment.limit_x = edge.limit_x;
            segment.limit_y = edge.limit_y;
            segment.limit_z = edge.limit_z;
            segment.joint_type = edge.joint_type;
            segment.motor_strength = edge.motor_strength;
            segment.joint_stiffness = edge.joint_stiffness;
        }
        limbs[limb_index] = limb;
    }
    genome.limbs = limbs;
}

fn graft_genome(a: &Genome, b: &Genome, rng: &mut SmallRng) -> Genome {
    let mut child = a.clone();
    child.version = default_genome_version();
    child.graph = graft_graph_gene(&a.graph, &b.graph, rng);
    child.mass_scale = lerp(a.mass_scale, b.mass_scale, rng.random::<f32>());
    child.hue = if rng.random::<f32>() < 0.5 {
        a.hue
    } else {
        b.hue
    };
    project_graph_to_legacy(&mut child);
    ensure_active_body_plan(&mut child, rng);
    child
}

fn crossover_genome(a: &Genome, b: &Genome, rng: &mut SmallRng) -> Genome {
    let mut child = a.clone();
    let torso_blend = rng_range(rng, 0.35, 0.65);
    child.torso.w = lerp(a.torso.w, b.torso.w, torso_blend);
    child.torso.h = lerp(a.torso.h, b.torso.h, torso_blend);
    child.torso.d = lerp(a.torso.d, b.torso.d, torso_blend);
    child.torso.mass = lerp(a.torso.mass, b.torso.mass, torso_blend);

    for i in 0..MAX_LIMBS {
        let (Some(la), Some(lb), Some(limb)) =
            (a.limbs.get(i), b.limbs.get(i), child.limbs.get_mut(i))
        else {
            continue;
        };
        let blend = rng_range(rng, 0.2, 0.8);
        limb.enabled = if rng.random::<f32>() < 0.5 {
            la.enabled
        } else {
            lb.enabled
        };
        limb.segment_count = if rng.random::<f32>() < 0.5 {
            la.segment_count
        } else {
            lb.segment_count
        };
        limb.anchor_x = lerp(la.anchor_x, lb.anchor_x, blend);
        limb.anchor_y = lerp(la.anchor_y, lb.anchor_y, blend);
        limb.anchor_z = lerp(la.anchor_z, lb.anchor_z, blend);
        limb.axis_y = lerp(la.axis_y, lb.axis_y, blend);
        limb.axis_z = lerp(la.axis_z, lb.axis_z, blend);
        limb.dir_x = lerp(la.dir_x, lb.dir_x, blend);
        limb.dir_y = lerp(la.dir_y, lb.dir_y, blend);
        limb.dir_z = lerp(la.dir_z, lb.dir_z, blend);

        for j in 0..MAX_SEGMENTS_PER_LIMB {
            let (Some(sa), Some(sb), Some(sg), Some(ca), Some(cb), Some(cg)) = (
                la.segments.get(j),
                lb.segments.get(j),
                limb.segments.get_mut(j),
                la.controls.get(j),
                lb.controls.get(j),
                limb.controls.get_mut(j),
            ) else {
                continue;
            };
            let seg_blend = rng.random::<f32>();
            sg.length = lerp(sa.length, sb.length, seg_blend);
            sg.thickness = lerp(sa.thickness, sb.thickness, seg_blend);
            sg.mass = lerp(sa.mass, sb.mass, seg_blend);
            sg.limit_x = lerp(sa.limit_x, sb.limit_x, seg_blend);
            sg.limit_y = lerp(sa.limit_y, sb.limit_y, seg_blend);
            sg.limit_z = lerp(sa.limit_z, sb.limit_z, seg_blend);
            sg.joint_type = if rng.random::<f32>() < 0.5 {
                sa.joint_type
            } else {
                sb.joint_type
            };
            sg.motor_strength = lerp(sa.motor_strength, sb.motor_strength, seg_blend);
            sg.joint_stiffness = lerp(sa.joint_stiffness, sb.joint_stiffness, seg_blend);

            let ctrl_blend = rng.random::<f32>();
            cg.amp = lerp(ca.amp, cb.amp, ctrl_blend);
            cg.freq = lerp(ca.freq, cb.freq, ctrl_blend);
            cg.phase = wrap_phase(lerp(ca.phase, cb.phase, ctrl_blend));
            cg.bias = lerp(ca.bias, cb.bias, ctrl_blend);
            cg.harm2_amp = lerp(ca.harm2_amp, cb.harm2_amp, ctrl_blend);
            cg.harm2_phase = wrap_phase(lerp(ca.harm2_phase, cb.harm2_phase, ctrl_blend));
            cg.amp_y = lerp(ca.amp_y, cb.amp_y, ctrl_blend);
            cg.freq_y = lerp(ca.freq_y, cb.freq_y, ctrl_blend);
            cg.phase_y = wrap_phase(lerp(ca.phase_y, cb.phase_y, ctrl_blend));
            cg.bias_y = lerp(ca.bias_y, cb.bias_y, ctrl_blend);
            cg.amp_z = lerp(ca.amp_z, cb.amp_z, ctrl_blend);
            cg.freq_z = lerp(ca.freq_z, cb.freq_z, ctrl_blend);
            cg.phase_z = wrap_phase(lerp(ca.phase_z, cb.phase_z, ctrl_blend));
            cg.bias_z = lerp(ca.bias_z, cb.bias_z, ctrl_blend);
        }
    }
    child.hue = if rng.random::<f32>() < 0.5 {
        a.hue
    } else {
        b.hue
    };
    child.mass_scale = lerp(a.mass_scale, b.mass_scale, rng.random::<f32>());
    child.version = default_genome_version();
    child.graph = crossover_graph_gene(&a.graph, &b.graph, rng);
    project_graph_to_legacy(&mut child);
    ensure_active_body_plan(&mut child, rng);
    child
}

fn mutate_genome(mut genome: Genome, chance: f32, structural: bool, rng: &mut SmallRng) -> Genome {
    genome.torso.w = mutate_number(genome.torso.w, 0.45, 3.1, chance, 0.2, rng);
    genome.torso.h = mutate_number(genome.torso.h, 0.45, 3.1, chance, 0.2, rng);
    genome.torso.d = mutate_number(genome.torso.d, 0.45, 3.1, chance, 0.2, rng);
    genome.torso.mass = mutate_number(genome.torso.mass, 0.2, 1.95, chance, 0.22, rng);

    let anchor_x_limit = genome.torso.w * 0.56;
    let anchor_y_limit = genome.torso.h * 0.56;
    let anchor_z_limit = genome.torso.d * 0.56;
    for limb in &mut genome.limbs {
        if rng.random::<f32>() < chance * 0.55 {
            limb.enabled = !limb.enabled;
        }
        if rng.random::<f32>() < chance * 0.65 {
            let delta = if rng.random::<f32>() < 0.5 {
                -1i32
            } else {
                1i32
            };
            limb.segment_count =
                (limb.segment_count as i32 + delta).clamp(1, MAX_SEGMENTS_PER_LIMB as i32) as u32;
        }

        limb.anchor_x = mutate_number(
            limb.anchor_x,
            -anchor_x_limit,
            anchor_x_limit,
            chance,
            0.17,
            rng,
        );
        limb.anchor_y = mutate_number(
            limb.anchor_y,
            -anchor_y_limit,
            anchor_y_limit,
            chance,
            0.18,
            rng,
        );
        limb.anchor_z = mutate_number(
            limb.anchor_z,
            -anchor_z_limit,
            anchor_z_limit,
            chance,
            0.17,
            rng,
        );
        limb.axis_y = mutate_number(limb.axis_y, -0.6, 0.6, chance, 0.2, rng);
        limb.axis_z = mutate_number(limb.axis_z, -0.6, 0.6, chance, 0.2, rng);
        limb.dir_x = mutate_number(limb.dir_x, -1.2, 1.2, chance, 0.32, rng);
        limb.dir_y = mutate_number(limb.dir_y, -1.2, 1.2, chance, 0.32, rng);
        limb.dir_z = mutate_number(limb.dir_z, -1.2, 1.2, chance, 0.32, rng);

        for segment in &mut limb.segments {
            segment.length = mutate_number(segment.length, 0.4, 2.6, chance, 0.21, rng);
            segment.thickness = mutate_number(segment.thickness, 0.12, 1.1, chance, 0.24, rng);
            segment.mass = mutate_number(segment.mass, 0.1, 2.25, chance, 0.24, rng);
            segment.limit_x = mutate_number(segment.limit_x, 0.12, PI * 0.95, chance, 0.14, rng);
            segment.limit_y = mutate_number(segment.limit_y, 0.10, PI * 0.75, chance, 0.16, rng);
            segment.limit_z = mutate_number(segment.limit_z, 0.10, PI * 0.75, chance, 0.16, rng);
            if rng.random::<f32>() < chance * 0.26 {
                segment.joint_type = match segment.joint_type {
                    JointTypeGene::Hinge => JointTypeGene::Ball,
                    JointTypeGene::Ball => JointTypeGene::Hinge,
                };
            }
            segment.motor_strength =
                mutate_number(segment.motor_strength, 0.5, 5.0, chance, 0.2, rng);
            segment.joint_stiffness =
                mutate_number(segment.joint_stiffness, 15.0, 120.0, chance, 0.2, rng);
        }
        for control in &mut limb.controls {
            control.amp = mutate_number(control.amp, 0.0, 11.6, chance, 0.18, rng);
            control.freq = mutate_number(control.freq, 0.3, 4.9, chance, 0.14, rng);
            control.phase = wrap_phase(mutate_number(
                control.phase,
                0.0,
                PI * 2.0,
                chance,
                0.28,
                rng,
            ));
            control.bias = mutate_number(control.bias, -2.35, 2.35, chance, 0.16, rng);
            control.harm2_amp = mutate_number(control.harm2_amp, 0.0, 6.5, chance, 0.18, rng);
            control.harm2_phase = wrap_phase(mutate_number(
                control.harm2_phase,
                0.0,
                PI * 2.0,
                chance,
                0.28,
                rng,
            ));
            control.amp_y = mutate_number(control.amp_y, 0.0, 11.6, chance, 0.18, rng);
            control.freq_y = mutate_number(control.freq_y, 0.3, 4.9, chance, 0.14, rng);
            control.phase_y = wrap_phase(mutate_number(
                control.phase_y,
                0.0,
                PI * 2.0,
                chance,
                0.28,
                rng,
            ));
            control.bias_y = mutate_number(control.bias_y, -2.35, 2.35, chance, 0.16, rng);
            control.amp_z = mutate_number(control.amp_z, 0.0, 11.6, chance, 0.18, rng);
            control.freq_z = mutate_number(control.freq_z, 0.3, 4.9, chance, 0.14, rng);
            control.phase_z = wrap_phase(mutate_number(
                control.phase_z,
                0.0,
                PI * 2.0,
                chance,
                0.28,
                rng,
            ));
            control.bias_z = mutate_number(control.bias_z, -2.35, 2.35, chance, 0.16, rng);
        }
    }

    if rng.random::<f32>() < chance {
        genome.hue = (genome.hue + rand_normal(rng) * 0.08 + 1.0) % 1.0;
    }
    if rng.random::<f32>() < chance {
        genome.mass_scale = mutate_number(genome.mass_scale, 0.7, 1.36, 1.0, 0.18, rng);
    }
    genome.version = default_genome_version();
    mutate_graph_gene(&mut genome.graph, chance, structural, rng);
    project_graph_to_legacy(&mut genome);
    ensure_active_body_plan(&mut genome, rng);
    genome
}

fn ensure_active_body_plan(genome: &mut Genome, rng: &mut SmallRng) {
    let anchor_x_limit = genome.torso.w * 0.56;
    let anchor_y_limit = genome.torso.h * 0.56;
    let anchor_z_limit = genome.torso.d * 0.56;
    for limb in &mut genome.limbs {
        limb.anchor_x = clamp(limb.anchor_x, -anchor_x_limit, anchor_x_limit);
        limb.anchor_y = clamp(limb.anchor_y, -anchor_y_limit, anchor_y_limit);
        limb.anchor_z = clamp(limb.anchor_z, -anchor_z_limit, anchor_z_limit);
        let [x, y, z] = normalize_vec3(limb.dir_x, limb.dir_y, limb.dir_z, [0.0, -1.0, 0.0]);
        limb.dir_x = x;
        limb.dir_y = y;
        limb.dir_z = z;
    }
    let mut enabled_count = genome.limbs.iter().filter(|limb| limb.enabled).count();
    if enabled_count >= 2 {
        return;
    }
    let mut attempts = 0;
    while enabled_count < 2 && attempts < 20 {
        attempts += 1;
        let index = rng.random_range(0..genome.limbs.len());
        if !genome.limbs[index].enabled {
            genome.limbs[index].enabled = true;
            genome.limbs[index].segment_count = genome.limbs[index].segment_count.max(1);
            enabled_count += 1;
        }
    }
}

fn normalize_vec3(x: f32, y: f32, z: f32, fallback: [f32; 3]) -> [f32; 3] {
    let len = (x * x + y * y + z * z).sqrt();
    if len < 1e-5 {
        fallback
    } else {
        [x / len, y / len, z / len]
    }
}

fn random_unit_vec3(rng: &mut SmallRng) -> [f32; 3] {
    let u = rng.random::<f32>();
    let v = rng.random::<f32>();
    let theta = u * PI * 2.0;
    let y = v * 2.0 - 1.0;
    let radial = (1.0 - y * y).max(0.0).sqrt();
    [theta.cos() * radial, y, theta.sin() * radial]
}

fn random_torso_surface_anchor(rng: &mut SmallRng, torso: &TorsoGene) -> [f32; 3] {
    let hx = torso.w * 0.5;
    let hy = torso.h * 0.5;
    let hz = torso.d * 0.5;
    let inset = 0.92;
    let r = rng.random_range(0..6);
    let x = rng_range(rng, -hx * inset, hx * inset);
    let y = rng_range(rng, -hy * inset, hy * inset);
    let z = rng_range(rng, -hz * inset, hz * inset);
    match r {
        0 => [hx * inset, y, z],
        1 => [-hx * inset, y, z],
        2 => [x, hy * inset, z],
        3 => [x, -hy * inset, z],
        4 => [x, y, hz * inset],
        _ => [x, y, -hz * inset],
    }
}

fn mutate_number(
    value: f32,
    min: f32,
    max: f32,
    chance: f32,
    variance: f32,
    rng: &mut SmallRng,
) -> f32 {
    if rng.random::<f32>() > chance {
        return value;
    }
    let scale = (max - min) * variance;
    clamp(value + rand_normal(rng) * scale, min, max)
}

fn rand_normal(rng: &mut SmallRng) -> f32 {
    let mut u = 0.0;
    let mut v = 0.0;
    while u == 0.0 {
        u = rng.random::<f32>();
    }
    while v == 0.0 {
        v = rng.random::<f32>();
    }
    (-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos()
}

fn wrap_phase(phase: f32) -> f32 {
    let two_pi = PI * 2.0;
    let mut p = phase % two_pi;
    if p < 0.0 {
        p += two_pi;
    }
    p
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn dot_weights(weights: &[f32], values: &[f32]) -> f32 {
    let mut sum = 0.0;
    for (w, v) in weights.iter().zip(values.iter()) {
        sum += *w * *v;
    }
    sum
}

fn apply_neural_activation(kind: NeuralActivationGene, value: f32) -> f32 {
    match kind {
        NeuralActivationGene::Tanh => value.tanh(),
        NeuralActivationGene::Sigmoid => 2.0 / (1.0 + (-value).exp()) - 1.0,
        NeuralActivationGene::Sin => value.sin(),
        NeuralActivationGene::Cos => value.cos(),
        NeuralActivationGene::Identity => value,
        NeuralActivationGene::Relu => value.max(0.0),
        NeuralActivationGene::Softsign => value / (1.0 + value.abs()),
    }
}

fn internal_err(message: String) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, message)
}

fn service_unavailable_err(message: String) -> (StatusCode, String) {
    (StatusCode::SERVICE_UNAVAILABLE, message)
}

fn resolve_configured_sim_worker_limit() -> Option<usize> {
    const ENV_VAR: &str = "SIM_MAX_CONCURRENT_JOBS";
    if let Ok(raw_value) = std::env::var(ENV_VAR) {
        match raw_value.parse::<usize>() {
            Ok(parsed) if parsed > 0 => return Some(parsed),
            _ => warn!("{ENV_VAR} must be a positive integer; got '{raw_value}'"),
        }
    }
    None
}

fn resolve_sim_worker_limit() -> usize {
    if let Some(limit) = resolve_configured_sim_worker_limit() {
        return limit;
    }

    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get().saturating_sub(1).max(1))
        .unwrap_or(1)
}

fn resolve_satellite_worker_limit() -> usize {
    if let Some(limit) = resolve_configured_sim_worker_limit() {
        return limit;
    }

    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get().max(1))
        .unwrap_or(1)
}

fn resolve_bind_port() -> u16 {
    const ENV_VAR: &str = "SIM_PORT";
    if let Ok(raw_value) = std::env::var(ENV_VAR) {
        match raw_value.parse::<u16>() {
            Ok(parsed) if parsed > 0 => return parsed,
            _ => warn!(
                "{ENV_VAR} must be an integer in range 1-65535; got '{raw_value}'. Using default {DEFAULT_BIND_PORT}"
            ),
        }
    }
    DEFAULT_BIND_PORT
}

async fn bind_listener(
    host: &str,
    desired_port: u16,
) -> Result<(tokio::net::TcpListener, SocketAddr), String> {
    let prefer_default_port = desired_port == DEFAULT_BIND_PORT;
    match tokio::net::TcpListener::bind((host, desired_port)).await {
        Ok(listener) => {
            let addr = listener
                .local_addr()
                .map_err(|err| format!("bound listener but failed reading local address: {err}"))?;
            return Ok((listener, addr));
        }
        Err(err) if err.kind() == std::io::ErrorKind::AddrInUse && prefer_default_port => {
            for offset in 1..=PORT_FALLBACK_ATTEMPTS {
                let Some(candidate_port) = desired_port.checked_add(offset) else {
                    break;
                };
                match tokio::net::TcpListener::bind((host, candidate_port)).await {
                    Ok(listener) => {
                        let addr = listener.local_addr().map_err(|bind_err| {
                            format!(
                                "bound listener on fallback port but failed reading local address: {bind_err}"
                            )
                        })?;
                        warn!(
                            "port {desired_port} is in use, falling back to http://{addr}; set SIM_PORT to choose a fixed port"
                        );
                        return Ok((listener, addr));
                    }
                    Err(bind_err) if bind_err.kind() == std::io::ErrorKind::AddrInUse => continue,
                    Err(bind_err) => {
                        return Err(format!(
                            "failed to bind fallback port {candidate_port} on {host}: {bind_err}"
                        ));
                    }
                }
            }
            Err(format!(
                "port {desired_port} is in use on {host}, and no free fallback port was found in range {}-{}; stop the existing process or set SIM_PORT",
                desired_port + 1,
                desired_port + PORT_FALLBACK_ATTEMPTS
            ))
        }
        Err(err) if err.kind() == std::io::ErrorKind::AddrInUse => Err(format!(
            "port {desired_port} is already in use on {host}; stop the existing process or choose another port via SIM_PORT"
        )),
        Err(err) => Err(format!(
            "failed to bind socket on {host}:{desired_port}: {err}"
        )),
    }
}

// ─── Satellite Distributed Computing ────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SatelliteMessage {
    Welcome {
        id: String,
    },
    Ready {
        #[serde(default = "default_satellite_ready_slots")]
        slots: usize,
    },
    RunTrial {
        trial_id: u64,
        genome: Genome,
        seed: u64,
        duration_seconds: f32,
        dt: f32,
        motor_power_scale: f32,
        fixed_startup: bool,
    },
    TrialResult {
        trial_id: u64,
        fitness: f32,
        metrics: TrialMetrics,
        descriptor: [f32; 5],
    },
    TrialError {
        trial_id: u64,
        message: String,
    },
    Ping,
    Pong,
}

fn default_satellite_ready_slots() -> usize {
    1
}

struct SatelliteConnection {
    tx: mpsc::UnboundedSender<SatelliteMessage>,
    available_slots: usize,
    capacity_slots: usize,
    in_flight: HashMap<u64, Instant>,
}

struct SatellitePool {
    inner: Mutex<SatellitePoolInner>,
    next_trial_id: AtomicUsize,
}

struct SatellitePoolInner {
    connections: HashMap<String, SatelliteConnection>,
    completed: HashMap<u64, TrialResult>,
    failed: HashMap<u64, String>,
    orphaned: Vec<u64>,
}

impl SatellitePool {
    fn new() -> Self {
        Self {
            inner: Mutex::new(SatellitePoolInner {
                connections: HashMap::new(),
                completed: HashMap::new(),
                failed: HashMap::new(),
                orphaned: Vec::new(),
            }),
            next_trial_id: AtomicUsize::new(1),
        }
    }

    fn connected_count(&self) -> usize {
        self.inner.lock().unwrap().connections.len()
    }

    fn connected_ids(&self) -> Vec<String> {
        let inner = self.inner.lock().unwrap();
        let mut ids: Vec<String> = inner.connections.keys().cloned().collect();
        ids.sort_unstable();
        ids
    }

    fn push_result(&self, trial_id: u64, result: TrialResult) {
        let mut inner = self.inner.lock().unwrap();
        inner.completed.insert(trial_id, result);
    }

    fn take_result(&self, trial_id: u64) -> Option<TrialResult> {
        let mut inner = self.inner.lock().unwrap();
        inner.completed.remove(&trial_id)
    }

    fn push_failure(&self, trial_id: u64, message: String) {
        let mut inner = self.inner.lock().unwrap();
        inner.failed.insert(trial_id, message);
    }

    fn take_failure(&self, trial_id: u64) -> Option<String> {
        let mut inner = self.inner.lock().unwrap();
        inner.failed.remove(&trial_id)
    }

    fn take_orphaned(&self) -> Vec<u64> {
        let mut inner = self.inner.lock().unwrap();
        std::mem::take(&mut inner.orphaned)
    }

    fn register(&self, id: String, tx: mpsc::UnboundedSender<SatelliteMessage>) {
        let mut inner = self.inner.lock().unwrap();
        inner.connections.insert(
            id,
            SatelliteConnection {
                tx,
                available_slots: 0,
                capacity_slots: 0,
                in_flight: HashMap::new(),
            },
        );
    }

    fn unregister(&self, id: &str) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(conn) = inner.connections.remove(id) {
            inner.orphaned.extend(conn.in_flight.into_keys());
        }
    }

    fn next_trial_id(&self) -> u64 {
        self.next_trial_id.fetch_add(1, Ordering::Relaxed) as u64
    }

    /// Try to dispatch a trial to a satellite with available slots. Returns trial_id if dispatched.
    fn try_dispatch(&self, genome: &Genome, seed: u64, config: &TrialConfig) -> Option<u64> {
        let trial_id = self.next_trial_id();
        let mut inner = self.inner.lock().unwrap();
        for conn in inner.connections.values_mut() {
            if conn.available_slots > 0 {
                let msg = SatelliteMessage::RunTrial {
                    trial_id,
                    genome: genome.clone(),
                    seed,
                    duration_seconds: config.duration_seconds,
                    dt: config.dt,
                    motor_power_scale: config.motor_power_scale,
                    fixed_startup: config.fixed_startup,
                };
                if conn.tx.send(msg).is_ok() {
                    conn.available_slots -= 1;
                    conn.in_flight.insert(trial_id, Instant::now());
                    return Some(trial_id);
                }
            }
        }
        None
    }

    fn mark_ready(&self, satellite_id: &str, slots: usize) {
        if slots == 0 {
            return;
        }
        let mut inner = self.inner.lock().unwrap();
        if let Some(conn) = inner.connections.get_mut(satellite_id) {
            conn.available_slots = conn.available_slots.saturating_add(slots);
            let observed_capacity = conn.available_slots.saturating_add(conn.in_flight.len());
            conn.capacity_slots = conn.capacity_slots.max(observed_capacity);
        }
    }

    fn mark_trial_complete(&self, satellite_id: &str, trial_id: u64) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(conn) = inner.connections.get_mut(satellite_id) {
            conn.in_flight.remove(&trial_id);
        }
    }

    fn available_slot_count(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner.connections.values().map(|c| c.available_slots).sum()
    }

    fn reap_timeouts(&self) -> Vec<u64> {
        let mut inner = self.inner.lock().unwrap();
        let now = Instant::now();
        let mut dead = Vec::new();
        for conn in inner.connections.values_mut() {
            let timed_out_ids: Vec<u64> = conn
                .in_flight
                .iter()
                .filter_map(|(trial_id, assigned_at)| {
                    (now.duration_since(*assigned_at) > SATELLITE_TRIAL_TIMEOUT)
                        .then_some(*trial_id)
                })
                .collect();
            if timed_out_ids.is_empty() {
                continue;
            }
            for trial_id in &timed_out_ids {
                conn.in_flight.remove(trial_id);
            }
            conn.available_slots = conn.available_slots.saturating_add(timed_out_ids.len());
            if conn.capacity_slots > 0 {
                let max_available = conn.capacity_slots.saturating_sub(conn.in_flight.len());
                conn.available_slots = conn.available_slots.min(max_available);
            }
            dead.extend(timed_out_ids);
        }
        dead
    }
}

// ─── Satellite WebSocket Server Handler (Primary Side) ──────────────────

async fn ws_satellite_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    ws.on_upgrade(move |socket| handle_satellite_socket(socket, state))
}

async fn handle_satellite_socket(socket: WebSocket, state: AppState) {
    let (mut ws_tx, mut ws_rx) = socket.split();
    let satellite_id = format!("sat-{:08x}", rand::random::<u32>());
    let (msg_tx, mut msg_rx) = mpsc::unbounded_channel::<SatelliteMessage>();

    // Send welcome
    let welcome = SatelliteMessage::Welcome {
        id: satellite_id.clone(),
    };
    if let Ok(json) = serde_json::to_string(&welcome) {
        let _ = ws_tx.send(Message::Text(json.into())).await;
    }

    state.satellite_pool.register(satellite_id.clone(), msg_tx);
    state
        .evolution
        .set_connected_satellites(state.satellite_pool.connected_ids());
    info!(
        "satellite connected: id={}, total={}",
        satellite_id,
        state.satellite_pool.connected_count()
    );

    // Spawn a task to forward outgoing messages to the WebSocket
    let sat_id_clone = satellite_id.clone();
    let send_task = tokio::spawn(async move {
        while let Some(msg) = msg_rx.recv().await {
            if let Ok(json) = serde_json::to_string(&msg) {
                if ws_tx.send(Message::Text(json.into())).await.is_err() {
                    break;
                }
            }
        }
        let _ = ws_tx.close().await;
        sat_id_clone
    });

    // Read incoming messages from the satellite
    let pool = state.satellite_pool.clone();
    let sat_id_for_rx = satellite_id.clone();
    let recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = ws_rx.next().await {
            match msg {
                Message::Text(text) => {
                    if let Ok(sat_msg) = serde_json::from_str::<SatelliteMessage>(&text) {
                        match sat_msg {
                            SatelliteMessage::Ready { slots } => {
                                pool.mark_ready(&sat_id_for_rx, slots.max(1));
                            }
                            SatelliteMessage::TrialResult {
                                trial_id,
                                fitness,
                                metrics,
                                descriptor,
                            } => {
                                info!(
                                    "satellite {} trial {} complete: fitness={:.3}",
                                    sat_id_for_rx, trial_id, fitness
                                );
                                pool.mark_trial_complete(&sat_id_for_rx, trial_id);
                                pool.push_result(
                                    trial_id,
                                    TrialResult {
                                        fitness,
                                        metrics,
                                        descriptor,
                                    },
                                );
                            }
                            SatelliteMessage::TrialError { trial_id, message } => {
                                warn!(
                                    "satellite {} trial {} failed: {}",
                                    sat_id_for_rx, trial_id, message
                                );
                                pool.push_failure(trial_id, message);
                                pool.mark_trial_complete(&sat_id_for_rx, trial_id);
                            }
                            SatelliteMessage::Pong => {}
                            _ => {}
                        }
                    }
                }
                Message::Close(_) => break,
                _ => {}
            }
        }
        sat_id_for_rx
    });

    // Wait for either task to finish
    tokio::select! {
        _ = send_task => {}
        _ = recv_task => {}
    }

    state.satellite_pool.unregister(&satellite_id);
    state
        .evolution
        .set_connected_satellites(state.satellite_pool.connected_ids());
    info!(
        "satellite disconnected: id={}, total={}",
        satellite_id,
        state.satellite_pool.connected_count()
    );
}

// ─── Satellite Client Mode ──────────────────────────────────────────────

fn parse_satellite_arg(args: &[String]) -> Option<String> {
    for i in 0..args.len() {
        if args[i] == "--satellite" {
            if let Some(url) = args.get(i + 1) {
                return Some(url.clone());
            }
        }
    }
    None
}

async fn run_satellite_client(primary_url: String) {
    let worker_limit = resolve_satellite_worker_limit();
    info!(
        "starting satellite mode: primary={}, worker_slots={}",
        primary_url, worker_limit
    );

    let ws_url = if primary_url.contains("/api/satellite/ws") {
        primary_url.clone()
    } else {
        let base = primary_url.trim_end_matches('/');
        format!("{}/api/satellite/ws", base)
    };

    loop {
        info!("connecting to primary: {}", ws_url);
        match tokio_tungstenite::connect_async(&ws_url).await {
            Ok((ws_stream, _response)) => {
                info!("connected to primary");
                run_satellite_session(ws_stream, worker_limit).await;
                warn!(
                    "disconnected from primary, reconnecting in {:?}",
                    SATELLITE_RECONNECT_DELAY
                );
            }
            Err(err) => {
                warn!(
                    "failed to connect to primary: {}, retrying in {:?}",
                    err, SATELLITE_RECONNECT_DELAY
                );
            }
        }
        tokio::time::sleep(SATELLITE_RECONNECT_DELAY).await;
    }
}

async fn run_satellite_session(
    ws_stream: tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
    worker_limit: usize,
) {
    use tokio_tungstenite::tungstenite::Message as TMessage;

    let (mut ws_tx, mut ws_rx) = ws_stream.split();
    let (result_tx, mut result_rx) = mpsc::channel::<SatelliteMessage>(64);
    let active_jobs = Arc::new(AtomicUsize::new(0));
    let satellite_id = Arc::new(Mutex::new(String::new()));

    // Advertise total available slots to primary.
    let ready = serde_json::to_string(&SatelliteMessage::Ready {
        slots: worker_limit.max(1),
    })
    .unwrap();
    if ws_tx.send(TMessage::Text(ready.into())).await.is_err() {
        return;
    }

    loop {
        tokio::select! {
            msg_opt = ws_rx.next() => {
                match msg_opt {
                    Some(Ok(TMessage::Text(text))) => {
                        match serde_json::from_str::<SatelliteMessage>(&text.to_string()) {
                            Ok(SatelliteMessage::Welcome { id }) => {
                                info!("registered with primary as satellite: {}", id);
                                *satellite_id.lock().unwrap() = id;
                            }
                            Ok(SatelliteMessage::RunTrial {
                                trial_id,
                                genome,
                                seed,
                                duration_seconds,
                                dt,
                                motor_power_scale,
                                fixed_startup,
                            }) => {
                                let jobs = active_jobs.load(Ordering::Relaxed);
                                if jobs >= worker_limit {
                                    warn!("satellite: at capacity ({}/{}), rejecting trial {}",
                                        jobs, worker_limit, trial_id);
                                    let err = SatelliteMessage::TrialError {
                                        trial_id,
                                        message: "satellite at capacity".to_string(),
                                    };
                                    let _ = result_tx.send(err).await;
                                    continue;
                                }
                                active_jobs.fetch_add(1, Ordering::Relaxed);
                                let result_tx = result_tx.clone();
                                let active_jobs = active_jobs.clone();
                                tokio::task::spawn_blocking(move || {
                                    let config = TrialConfig {
                                        duration_seconds: duration_seconds.clamp(1.0, 120.0),
                                        dt,
                                        snapshot_hz: 0.0, // no snapshots needed
                                        motor_power_scale: motor_power_scale.clamp(0.35, 1.5),
                                        fixed_startup,
                                    };
                                    let result = run_satellite_trial(trial_id, &genome, seed, &config);
                                    let _ = result_tx.blocking_send(result);
                                    active_jobs.fetch_sub(1, Ordering::Relaxed);
                                });
                            }
                            Ok(SatelliteMessage::Ping) => {
                                let pong = serde_json::to_string(&SatelliteMessage::Pong).unwrap();
                                let _ = ws_tx.send(TMessage::Text(pong.into())).await;
                            }
                            Ok(_) => {}
                            Err(err) => {
                                warn!("satellite: failed to parse message: {}", err);
                            }
                        }
                    }
                    Some(Ok(TMessage::Close(_))) | None => {
                        info!("satellite: connection closed by primary");
                        break;
                    }
                    Some(Ok(_)) => {}
                    Some(Err(err)) => {
                        warn!("satellite: websocket error: {}", err);
                        break;
                    }
                }
            }
            Some(result_msg) = result_rx.recv() => {
                if let Ok(json) = serde_json::to_string(&result_msg) {
                    if ws_tx.send(TMessage::Text(json.into())).await.is_err() {
                        break;
                    }
                }
                // Return one slot after each completed or failed trial.
                if matches!(
                    &result_msg,
                    SatelliteMessage::TrialResult { .. } | SatelliteMessage::TrialError { .. }
                ) {
                    let ready = serde_json::to_string(&SatelliteMessage::Ready { slots: 1 }).unwrap();
                    if ws_tx.send(TMessage::Text(ready.into())).await.is_err() {
                        break;
                    }
                }
            }
        }
    }
}

fn run_satellite_trial(
    trial_id: u64,
    genome: &Genome,
    seed: u64,
    config: &TrialConfig,
) -> SatelliteMessage {
    let start = Instant::now();
    match TrialSimulator::new(genome, seed, config) {
        Ok(mut sim) => {
            let steps = (config.duration_seconds / config.dt).ceil() as usize;
            for _ in 0..steps {
                if let Err(err) = sim.step() {
                    return SatelliteMessage::TrialError {
                        trial_id,
                        message: format!("sim step failed: {err}"),
                    };
                }
            }
            let result = sim.final_result();
            let elapsed = start.elapsed();
            info!(
                "satellite trial {} complete: fitness={:.3}, elapsed={:.2}s",
                trial_id,
                result.fitness,
                elapsed.as_secs_f32()
            );
            SatelliteMessage::TrialResult {
                trial_id,
                fitness: result.fitness,
                metrics: result.metrics,
                descriptor: result.descriptor,
            }
        }
        Err(err) => SatelliteMessage::TrialError {
            trial_id,
            message: format!("sim init failed: {err}"),
        },
    }
}

// ─── End Satellite ──────────────────────────────────────────────────────

fn insert_box_body(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    size: [f32; 3],
    mass: f32,
    center: Vector3<f32>,
) -> RigidBodyHandle {
    let body = RigidBodyBuilder::dynamic()
        .translation(center)
        .linear_damping(BODY_LINEAR_DAMPING)
        .angular_damping(BODY_ANGULAR_DAMPING)
        .ccd_enabled(true)
        .build();
    let handle = bodies.insert(body);
    let collider = ColliderBuilder::cuboid(size[0] * 0.5, size[1] * 0.5, size[2] * 0.5)
        .mass(mass)
        .friction(ACTIVE_SURFACE_FRICTION)
        .restitution(0.015)
        .collision_groups(InteractionGroups::new(
            CREATURE_COLLISION_GROUP,
            GROUND_COLLISION_GROUP,
            InteractionTestMode::And,
        ))
        .build();
    colliders.insert_with_parent(collider, handle, bodies);
    handle
}

fn normalized_axis(axis_y: f32, axis_z: f32) -> Vector3<f32> {
    // Keep joints mostly in a leg-like sagittal plane while still allowing
    // substantial variation per limb from the two axis genes.
    let axis = vector![1.0, axis_y * AXIS_TILT_GAIN, axis_z * AXIS_TILT_GAIN];
    axis.try_normalize(1e-6).unwrap_or(vector![1.0, 0.0, 0.0])
}

fn max_box_cross_section(size: [f32; 3]) -> f32 {
    let wxh = (size[0] * size[1]).abs();
    let wxd = (size[0] * size[2]).abs();
    let hxd = (size[1] * size[2]).abs();
    wxh.max(wxd).max(hxd).max(1e-4)
}

fn joint_area_strength_scale(parent_size: [f32; 3], child_size: [f32; 3]) -> f32 {
    let parent_area = max_box_cross_section(parent_size);
    let child_area = max_box_cross_section(child_size);
    clamp(
        parent_area.min(child_area),
        JOINT_AREA_STRENGTH_MIN_SCALE,
        JOINT_AREA_STRENGTH_MAX_SCALE,
    )
}

fn normalized_dir(x: f32, y: f32, z: f32) -> Vector3<f32> {
    let dir = vector![x, y, z];
    dir.try_normalize(1e-6).unwrap_or(vector![0.0, -1.0, 0.0])
}

fn default_genome_version() -> u32 {
    2
}

fn default_graph_max_parts() -> usize {
    32
}

fn default_neural_activation() -> NeuralActivationGene {
    NeuralActivationGene::Tanh
}

fn default_neural_leak() -> f32 {
    0.32
}

fn default_effector_gain() -> f32 {
    1.0
}

fn default_joint_effector_gene() -> JointEffectorGene {
    JointEffectorGene {
        local_weights: vec![0.0; MIN_LOCAL_NEURONS],
        global_weights: vec![0.0; MIN_GLOBAL_NEURONS],
        bias: 0.0,
        gain: 1.0,
    }
}

fn default_local_brain_gene() -> LocalBrainGene {
    let neurons = (0..MIN_LOCAL_NEURONS)
        .map(|_| NeuralUnitGene {
            activation: default_neural_activation(),
            input_weights: vec![0.0; LOCAL_SENSOR_DIM],
            recurrent_weights: vec![0.0; MIN_LOCAL_NEURONS],
            global_weights: vec![0.0; MIN_GLOBAL_NEURONS],
            bias: 0.0,
            leak: default_neural_leak(),
        })
        .collect::<Vec<_>>();
    LocalBrainGene {
        neurons,
        effector_x: default_joint_effector_gene(),
        effector_y: default_joint_effector_gene(),
        effector_z: default_joint_effector_gene(),
    }
}

fn default_global_brain_gene() -> GlobalBrainGene {
    let neurons = (0..MIN_GLOBAL_NEURONS)
        .map(|_| NeuralUnitGene {
            activation: default_neural_activation(),
            input_weights: vec![0.0; GLOBAL_SENSOR_DIM],
            recurrent_weights: vec![0.0; MIN_GLOBAL_NEURONS],
            global_weights: Vec::new(),
            bias: 0.0,
            leak: default_neural_leak(),
        })
        .collect::<Vec<_>>();
    GlobalBrainGene { neurons }
}

fn default_graph_node_gene() -> MorphNodeGene {
    MorphNodeGene {
        part: GraphPartGene {
            w: 0.7,
            h: 1.0,
            d: 0.7,
            mass: 0.8,
        },
        edges: Vec::new(),
        brain: default_local_brain_gene(),
    }
}

fn default_graph_gene() -> GraphGene {
    let mut node = default_graph_node_gene();
    node.edges.push(MorphEdgeGene {
        to: 0,
        anchor_x: 0.0,
        anchor_y: -0.48,
        anchor_z: 0.0,
        axis_y: 0.0,
        axis_z: 0.0,
        dir_x: 0.0,
        dir_y: -1.0,
        dir_z: 0.0,
        scale: 1.0,
        reflect_x: false,
        recursive_limit: 2,
        terminal_only: false,
        joint_type: JointTypeGene::Hinge,
        limit_x: 1.2,
        limit_y: 0.9,
        limit_z: 0.9,
        motor_strength: 1.0,
        joint_stiffness: 48.0,
    });
    GraphGene {
        root: 0,
        nodes: vec![node],
        global_brain: default_global_brain_gene(),
        max_parts: default_graph_max_parts(),
    }
}

fn default_limb_dir_x() -> f32 {
    0.0
}

fn default_limb_dir_y() -> f32 {
    -1.0
}

fn default_limb_dir_z() -> f32 {
    0.0
}

fn default_joint_type() -> JointTypeGene {
    JointTypeGene::Hinge
}

fn default_joint_limit_x() -> f32 {
    1.57
}

fn default_joint_limit_y() -> f32 {
    1.08
}

fn default_joint_limit_z() -> f32 {
    1.08
}

fn default_motor_strength() -> f32 {
    1.0
}

fn default_joint_stiffness() -> f32 {
    40.0
}

fn default_secondary_control_amp() -> f32 {
    0.0
}

fn default_secondary_control_freq() -> f32 {
    1.0
}

fn default_secondary_control_phase() -> f32 {
    0.0
}

fn default_secondary_control_bias() -> f32 {
    0.0
}

fn default_second_harmonic_amp() -> f32 {
    0.0
}

fn default_second_harmonic_phase() -> f32 {
    0.0
}

fn default_run_speed() -> f32 {
    1.0
}

fn default_min_population_size() -> usize {
    MIN_POPULATION_SIZE
}

fn default_max_population_size() -> usize {
    MAX_POPULATION_SIZE
}

fn default_population_size() -> usize {
    DEFAULT_POPULATION_SIZE
}

fn default_trials_per_candidate() -> usize {
    TRIALS_PER_CANDIDATE
}

fn default_max_trial_count() -> usize {
    TRIALS_PER_CANDIDATE
}

fn default_generation_seconds() -> f32 {
    DEFAULT_GENERATION_SECONDS
}

fn default_min_run_speed() -> f32 {
    0.5
}

fn default_max_run_speed() -> f32 {
    8.0
}

fn rng_range(rng: &mut SmallRng, min: f32, max: f32) -> f32 {
    min + rng.random::<f32>() * (max - min)
}

fn clamp(value: f32, min: f32, max: f32) -> f32 {
    value.max(min).min(max)
}

fn quantile(values: &[f32], q: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if sorted.len() == 1 {
        return sorted[0];
    }
    let t = clamp(q, 0.0, 1.0) * (sorted.len() - 1) as f32;
    let lower = t.floor() as usize;
    let upper = t.ceil() as usize;
    let frac = t - lower as f32;
    sorted[lower] + (sorted[upper] - sorted[lower]) * frac
}
