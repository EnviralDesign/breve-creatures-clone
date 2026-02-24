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
const FIXED_SIM_DT: f32 = 1.0 / 120.0;
const MASS_DENSITY_MULTIPLIER: f32 = 1.4;
const MAX_MOTOR_SPEED: f32 = 6.8;
const MAX_BODY_ANGULAR_SPEED: f32 = 15.0;
const MAX_BODY_LINEAR_SPEED: f32 = 22.0;
const MOTOR_TORQUE_HIP: f32 = 85.0;
const MOTOR_TORQUE_KNEE: f32 = 68.0;
const BALL_AXIS_TORQUE_SCALE_Y: f32 = 0.7;
const BALL_AXIS_TORQUE_SCALE_Z: f32 = 0.7;
const BALL_AXIS_STIFFNESS_SCALE_Y: f32 = 0.75;
const BALL_AXIS_STIFFNESS_SCALE_Z: f32 = 0.75;
const JOINT_MOTOR_RESPONSE: f32 = 12.0;
const JOINT_MOTOR_FORCE_MULTIPLIER: f32 = 1.0;
const AXIS_TILT_GAIN: f32 = 1.9;
const FALLEN_HEIGHT_THRESHOLD: f32 = 0.35;
const MAX_PLAUSIBLE_STEP_DISPLACEMENT: f32 = 1.5;
const FITNESS_UPRIGHT_BONUS: f32 = 0.95;
const FITNESS_STRAIGHTNESS_BONUS: f32 = 1.5;
const FITNESS_HEIGHT_BONUS: f32 = 0.6;
const FITNESS_ENERGY_PENALTY: f32 = 0.8;
const FITNESS_INSTABILITY_PENALTY: f32 = 1.25;
const FITNESS_NET_PROGRESS_WEIGHT: f32 = 0.95;
const FALLEN_PENALTY_STRENGTH: f32 = 0.6;
const UPRIGHT_FULL_SCORE_THRESHOLD: f32 = 0.5;
const UPRIGHT_PENALTY_FLOOR: f32 = 0.4;
const FITNESS_PROGRESS_STRAIGHTNESS_GATE_EXPONENT: f32 = 2.0;
const FITNESS_PROGRESS_FALLEN_GATE_EXPONENT: f32 = 1.3;
const FITNESS_PROGRESS_MIDPOINT_FRACTION: f32 = 0.5;
const FITNESS_PROGRESS_LATE_FRACTION: f32 = 0.85;
const FITNESS_SUSTAIN_BASE: f32 = 0.12;
const FITNESS_SUSTAIN_MID_GAIN_WEIGHT: f32 = 0.58;
const FITNESS_SUSTAIN_LATE_GAIN_WEIGHT: f32 = 0.30;
const FITNESS_THRASH_ENERGY_RATIO_PENALTY: f32 = 0.55;
const FITNESS_THRASH_INSTABILITY_RATIO_PENALTY: f32 = 0.65;
const FITNESS_THRASH_PROGRESS_EPS: f32 = 0.45;
const SETTLE_SECONDS: f32 = 2.25;
const GROUND_COLLISION_GROUP: Group = Group::GROUP_1;
const CREATURE_COLLISION_GROUP: Group = Group::GROUP_2;
const DEFAULT_BIND_HOST: &str = "0.0.0.0";
const DEFAULT_BIND_PORT: u16 = 8787;
const PORT_FALLBACK_ATTEMPTS: u16 = 32;
const SATELLITE_TRIAL_TIMEOUT: Duration = Duration::from_secs(10);
const SATELLITE_RECONNECT_DELAY: Duration = Duration::from_secs(3);
const SATELLITE_DISPATCH_RETRY_LIMIT: usize = 8;
const SATELLITE_CAPACITY_ERROR: &str = "satellite at capacity";
const MAX_FITNESS_HISTORY_POINTS: usize = 4096;
const MAX_PERFORMANCE_HISTORY_POINTS: usize = 4096;
const DEFAULT_POPULATION_SIZE: usize = 40;
const MIN_POPULATION_SIZE: usize = 1;
const MAX_POPULATION_SIZE: usize = 128;
const ELITE_COUNT: usize = 2;
const TRIALS_PER_CANDIDATE: usize = 5;
const DEFAULT_GENERATION_SECONDS: f32 = 18.0;
const EVOLUTION_VIEW_FRAME_LIMIT: usize = 900;
const CHECKPOINT_DIR: &str = "data/checkpoints";
const AUTOSAVE_EVERY_GENERATIONS: usize = 5;
const DEFAULT_PERFORMANCE_WINDOW_GENERATIONS: usize = 120;
const MAX_PERFORMANCE_WINDOW_GENERATIONS: usize = 400;
const DEFAULT_PERFORMANCE_STRIDE: usize = 1;
const MAX_PERFORMANCE_STRIDE: usize = 8;
const MIN_BREEDING_MUTATION_RATE: f32 = 0.18;
const MAX_BREEDING_MUTATION_RATE: f32 = 0.72;
const FITNESS_STAGNATION_EPSILON: f32 = 1e-4;
const MAX_GENERATION_TOPOLOGY_CANDIDATES: usize = 3;
const SUMMARY_BEST_TOPOLOGY_COUNT: usize = 5;
const DIAG_RECENT_WINDOW: usize = 20;
const DIAG_PLATEAU_STAGNATION_GENERATIONS: usize = 20;
const DIAG_WATCH_STAGNATION_GENERATIONS: usize = 10;
const FIXED_PRESET_SPAWN_HEIGHT: f32 = 0.58;
const FIXED_PRESET_SETTLE_MIN_STABLE_SECONDS: f32 = 0.45;
const FIXED_PRESET_SETTLE_MAX_EXTRA_SECONDS: f32 = 1.8;
const FIXED_PRESET_SETTLE_LINEAR_SPEED_MAX: f32 = 0.65;
const FIXED_PRESET_SETTLE_ANGULAR_SPEED_MAX: f32 = 1.45;
const TRAIN_TRIAL_SEED_BANK_TAG: u32 = 0x9e37_79b9;
const HOLDOUT_TRIAL_SEED_BANK_TAG: u32 = 0x7f4a_7c15;
const HOLDOUT_TRIALS_PER_CANDIDATE: usize = 5;
const TRIAL_DIVERGENCE_PENALTY_WEIGHT: f32 = 0.22;
const TRIAL_DIVERGENCE_PENALTY_FLOOR: f32 = 0.68;
const ANNEALING_TIME_CONSTANT_GENERATIONS: f32 = 140.0;
const MUTATION_RATE_FINAL_FLOOR: f32 = 0.06;
const MUTATION_RATE_FINAL_CEILING: f32 = 0.42;
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

impl ControlGene {
    fn signal_x(&self, sim_time: f32) -> f32 {
        let theta = self.freq * sim_time + self.phase;
        self.bias
            + self.amp * theta.sin()
            + self.harm2_amp * (2.0 * theta + self.harm2_phase).sin()
    }

    fn signal_y(&self, sim_time: f32) -> f32 {
        self.bias_y + self.amp_y * (self.freq_y * sim_time + self.phase_y).sin()
    }

    fn signal_z(&self, sim_time: f32) -> f32 {
        self.bias_z + self.amp_z * (self.freq_z * sim_time + self.phase_z).sin()
    }
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
}

impl TrialConfig {
    fn from_trial_request(request: &TrialRunRequest) -> Self {
        Self {
            duration_seconds: request.duration_seconds.unwrap_or(18.0).clamp(1.0, 120.0),
            dt: FIXED_SIM_DT,
            snapshot_hz: request.snapshot_hz.unwrap_or(30.0).clamp(1.0, 120.0),
            motor_power_scale: request.motor_power_scale.unwrap_or(1.0).clamp(0.35, 1.5),
        }
    }

    fn from_generation_request(request: &GenerationEvalRequest) -> Self {
        Self {
            duration_seconds: request.duration_seconds.unwrap_or(18.0).clamp(1.0, 120.0),
            dt: FIXED_SIM_DT,
            snapshot_hz: 30.0,
            motor_power_scale: request.motor_power_scale.unwrap_or(1.0).clamp(0.35, 1.5),
        }
    }
}

struct SimPart {
    body: RigidBodyHandle,
    size: [f32; 3],
}

struct SimController {
    joint: ImpulseJointHandle,
    joint_type: JointTypeGene,
    control: ControlGene,
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

struct TrialAccumulator {
    spawn: Vector3<f32>,
    best_distance: f32,
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
        let mut active_limb_count = 0usize;
        let mut segment_total = 0usize;
        for limb in &genome.limbs {
            if !limb.enabled {
                continue;
            }
            active_limb_count += 1;
            segment_total += limb.segment_count.clamp(1, MAX_SEGMENTS_PER_LIMB as u32) as usize;
        }
        let mean_segment_count = if active_limb_count > 0 {
            segment_total as f32 / active_limb_count as f32
        } else {
            0.0
        };

        Self {
            spawn,
            best_distance: 0.0,
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

        let step_displacement = if let Some(last) = self.last_torso_pos {
            (torso_pos - last).norm()
        } else {
            0.0
        };
        let plausible = step_displacement <= MAX_PLAUSIBLE_STEP_DISPLACEMENT;

        if plausible {
            self.net_dx = torso_pos.x - self.spawn.x;
            self.net_dz = torso_pos.z - self.spawn.z;
            let traveled = (self.net_dx * self.net_dx + self.net_dz * self.net_dz).sqrt();
            self.best_distance = self.best_distance.max(traveled);

            if let Some(last) = self.last_torso_pos {
                let dx = torso_pos.x - last.x;
                let dz = torso_pos.z - last.z;
                self.path_length += (dx * dx + dz * dz).sqrt();
            }
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
        let peak_distance = self.best_distance;
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
        let distance_at_mid = self.distance_at_mid_phase.unwrap_or(net_distance);
        let distance_at_late = self.distance_at_late_phase.unwrap_or(net_distance);
        let gain_after_mid_ratio = clamp(
            (net_distance - distance_at_mid).max(0.0)
                / net_distance.max(FITNESS_THRASH_PROGRESS_EPS),
            0.0,
            1.0,
        );
        let gain_after_late_ratio = clamp(
            (net_distance - distance_at_late).max(0.0)
                / net_distance.max(FITNESS_THRASH_PROGRESS_EPS),
            0.0,
            1.0,
        );
        let sustain_factor = clamp(
            FITNESS_SUSTAIN_BASE
                + gain_after_mid_ratio * FITNESS_SUSTAIN_MID_GAIN_WEIGHT
                + gain_after_late_ratio * FITNESS_SUSTAIN_LATE_GAIN_WEIGHT,
            0.0,
            1.0,
        );
        let raw_progress = net_distance * FITNESS_NET_PROGRESS_WEIGHT
            + peak_distance * (1.0 - FITNESS_NET_PROGRESS_WEIGHT);
        let straight_gate =
            clamp(straightness, 0.0, 1.0).powf(FITNESS_PROGRESS_STRAIGHTNESS_GATE_EXPONENT);
        let fallen_gate = (1.0 - fallen_ratio).powf(FITNESS_PROGRESS_FALLEN_GATE_EXPONENT);
        let progress = raw_progress * straight_gate * fallen_gate * sustain_factor;

        let mut quality = progress;
        quality += upright_avg * FITNESS_UPRIGHT_BONUS;
        quality += straightness * FITNESS_STRAIGHTNESS_BONUS;
        quality += clamp(avg_height / 3.0, 0.0, 1.0) * FITNESS_HEIGHT_BONUS;
        quality -= energy_norm * FITNESS_ENERGY_PENALTY;
        quality -= instability_norm * FITNESS_INSTABILITY_PENALTY;
        let progress_for_ratio = progress.max(FITNESS_THRASH_PROGRESS_EPS);
        let energy_per_progress = energy_norm / progress_for_ratio;
        let instability_per_progress = instability_norm / progress_for_ratio;
        quality -= energy_per_progress * FITNESS_THRASH_ENERGY_RATIO_PENALTY;
        quality -= instability_per_progress * FITNESS_THRASH_INSTABILITY_RATIO_PENALTY;
        quality *= 1.0 - fallen_ratio.powf(1.5) * FALLEN_PENALTY_STRENGTH;
        if upright_avg < UPRIGHT_FULL_SCORE_THRESHOLD {
            let upright_factor = clamp(upright_avg / UPRIGHT_FULL_SCORE_THRESHOLD, 0.0, 1.0);
            quality *= UPRIGHT_PENALTY_FLOOR + (1.0 - UPRIGHT_PENALTY_FLOOR) * upright_factor;
        }
        quality = quality.max(0.0);

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
    torso_handle: RigidBodyHandle,
    metrics: TrialAccumulator,
    elapsed: f32,
    duration: f32,
    require_settled_before_actuation: bool,
    settled_time_before_actuation: f32,
    actuation_started: bool,
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
            .friction(1.08)
            .restitution(0.015)
            .collision_groups(InteractionGroups::new(
                GROUND_COLLISION_GROUP,
                CREATURE_COLLISION_GROUP,
                InteractionTestMode::And,
            ))
            .build();
        colliders.insert_with_parent(ground_collider, ground_handle, &mut bodies);

        let spawn = vector![0.0, 0.05, 0.0];
        let torso_dims = &genome.torso;
        let torso_mass = (torso_dims.w
            * torso_dims.h
            * torso_dims.d
            * torso_dims.mass
            * genome.mass_scale
            * MASS_DENSITY_MULTIPLIER)
            .max(0.7);

        let fixed_preset = detect_morphology_preset(genome);
        let use_fixed_preset_startup = fixed_preset.is_some();
        let drop_start = if use_fixed_preset_startup {
            vector![spawn.x, FIXED_PRESET_SPAWN_HEIGHT, spawn.z]
        } else {
            vector![spawn.x, spawn.y + rng_range(&mut rng, 5.2, 7.7), spawn.z]
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
                UnitQuaternion::identity()
            } else {
                UnitQuaternion::from_euler_angles(
                    rng_range(&mut rng, -0.36, 0.36),
                    rng_range(&mut rng, 0.0, PI * 2.0),
                    rng_range(&mut rng, -0.28, 0.28),
                )
            };
            torso_body.set_rotation(rot, true);
            if use_fixed_preset_startup {
                torso_body.set_linvel(vector![0.0, 0.0, 0.0], true);
                torso_body.set_angvel(vector![0.0, 0.0, 0.0], true);
            } else {
                torso_body.set_linvel(
                    vector![
                        rng_range(&mut rng, -0.8, 0.8),
                        rng_range(&mut rng, -0.6, 0.3),
                        rng_range(&mut rng, -0.8, 0.8)
                    ],
                    true,
                );
                torso_body.set_angvel(
                    vector![
                        rng_range(&mut rng, -1.8, 1.8),
                        rng_range(&mut rng, -2.2, 2.2),
                        rng_range(&mut rng, -1.8, 1.8)
                    ],
                    true,
                );
            }
        }

        let mut parts = vec![SimPart {
            body: torso_handle,
            size: [torso_dims.w, torso_dims.h, torso_dims.d],
        }];
        let mut controllers = Vec::new();

        for limb_index in 0..MAX_LIMBS {
            let limb = match genome.limbs.get(limb_index) {
                Some(limb) => limb,
                None => continue,
            };
            if !limb.enabled {
                continue;
            }

            let segment_count = limb.segment_count.clamp(1, MAX_SEGMENTS_PER_LIMB as u32) as usize;
            let axis_local = normalized_axis(limb.axis_y, limb.axis_z);
            let first_growth_local = normalized_dir(limb.dir_x, limb.dir_y, limb.dir_z);
            let mut parent = torso_handle;
            let mut pivot_from_parent = vector![limb.anchor_x, limb.anchor_y, limb.anchor_z];

            for seg_index in 0..segment_count {
                let segment_gene = match limb.segments.get(seg_index) {
                    Some(segment) => segment,
                    None => break,
                };
                let control_gene = limb
                    .controls
                    .get(seg_index)
                    .cloned()
                    .unwrap_or_else(default_control_gene);

                let parent_body = bodies
                    .get(parent)
                    .ok_or_else(|| "parent body missing".to_string())?;
                let parent_translation = *parent_body.translation();
                let parent_rotation = *parent_body.rotation();

                let anchor_world = parent_translation + parent_rotation * pivot_from_parent;
                let local_growth = if seg_index == 0 {
                    first_growth_local
                } else {
                    vector![0.0, -1.0, 0.0]
                };
                let growth_world = parent_rotation * local_growth;
                let center = anchor_world + growth_world * (segment_gene.length * 0.5);
                let seg_local_rot =
                    UnitQuaternion::rotation_between(&vector![0.0, -1.0, 0.0], &local_growth)
                        .unwrap_or_else(UnitQuaternion::identity);
                let child_rotation = parent_rotation * seg_local_rot;
                let segment_mass = (segment_gene.thickness
                    * segment_gene.length
                    * segment_gene.thickness
                    * segment_gene.mass
                    * genome.mass_scale
                    * MASS_DENSITY_MULTIPLIER)
                    .max(0.08);

                let child = insert_box_body(
                    &mut bodies,
                    &mut colliders,
                    [
                        segment_gene.thickness,
                        segment_gene.length,
                        segment_gene.thickness,
                    ],
                    segment_mass,
                    center,
                );
                if let Some(child_body) = bodies.get_mut(child) {
                    child_body.set_rotation(child_rotation, true);
                }
                let local_anchor2 = point![0.0, segment_gene.length * 0.5, 0.0];

                let limit_x = clamp(segment_gene.limit_x, 0.12, PI * 0.95);
                let limit_y = clamp(segment_gene.limit_y, 0.10, PI * 0.75);
                let limit_z = clamp(segment_gene.limit_z, 0.10, PI * 0.75);
                let torque_x = if seg_index == 0 {
                    MOTOR_TORQUE_HIP
                } else {
                    MOTOR_TORQUE_KNEE
                } * segment_gene.motor_strength
                    * JOINT_MOTOR_FORCE_MULTIPLIER
                    * config.motor_power_scale;
                let stiffness_x = segment_gene.joint_stiffness * segment_gene.motor_strength;
                let torque_y = torque_x * BALL_AXIS_TORQUE_SCALE_Y;
                let torque_z = torque_x * BALL_AXIS_TORQUE_SCALE_Z;
                let stiffness_y = stiffness_x * BALL_AXIS_STIFFNESS_SCALE_Y;
                let stiffness_z = stiffness_x * BALL_AXIS_STIFFNESS_SCALE_Z;

                let joint_handle = match segment_gene.joint_type {
                    JointTypeGene::Hinge => {
                        let mut joint = RevoluteJointBuilder::new(UnitVector::new_normalize(
                            axis_local,
                        ))
                        .local_anchor1(point![
                            pivot_from_parent.x,
                            pivot_from_parent.y,
                            pivot_from_parent.z
                        ])
                        .local_anchor2(local_anchor2)
                        .contacts_enabled(false);
                        joint = joint.limits([-limit_x, limit_x]);
                        let handle = impulse_joints.insert(parent, child, joint, true);
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
                            joint_ref.data.set_motor_max_force(JointAxis::AngX, torque_x);
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
                        let handle = impulse_joints.insert(parent, child, joint, true);
                        if let Some(joint_ref) = impulse_joints.get_mut(handle, false) {
                            for (axis, stiffness, torque) in [
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
                                joint_ref.data.set_motor_max_force(axis, torque);
                            }
                        }
                        handle
                    }
                };

                if matches!(segment_gene.joint_type, JointTypeGene::Ball) && seg_index > 0 {
                    // Keep distal joints a bit tighter for stability when spherical.
                    if let Some(joint_ref) = impulse_joints.get_mut(joint_handle, false) {
                        joint_ref
                            .data
                            .set_limits(JointAxis::AngY, [-limit_y * 0.85, limit_y * 0.85]);
                        joint_ref
                            .data
                            .set_limits(JointAxis::AngZ, [-limit_z * 0.85, limit_z * 0.85]);
                    }
                }
                controllers.push(SimController {
                    joint: joint_handle,
                    joint_type: segment_gene.joint_type,
                    control: control_gene,
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

                parts.push(SimPart {
                    body: child,
                    size: [
                        segment_gene.thickness,
                        segment_gene.length,
                        segment_gene.thickness,
                    ],
                });

                parent = child;
                pivot_from_parent = vector![0.0, -segment_gene.length * 0.5, 0.0];
            }
        }

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

        Ok(Self {
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
            torso_handle,
            metrics: TrialAccumulator::new(spawn, genome),
            elapsed: 0.0,
            duration: config.duration_seconds,
            require_settled_before_actuation: use_fixed_preset_startup,
            settled_time_before_actuation: 0.0,
            actuation_started: false,
        })
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
        let metrics = self.metrics.compute_metrics(self.duration);
        let descriptor = self.metrics.descriptor(&metrics);
        TrialResult {
            fitness: metrics.quality,
            metrics,
            descriptor,
        }
    }

    fn step(&mut self) -> Result<(), String> {
        let dt = self.integration_parameters.dt;
        let sim_time = self.elapsed;

        let mut can_actuate = sim_time >= SETTLE_SECONDS;
        if can_actuate
            && self.require_settled_before_actuation
            && !self.actuation_started
            && let Some(torso) = self.bodies.get(self.torso_handle)
        {
            let linear_speed = torso.linvel().norm();
            let angular_speed = torso.angvel().norm();
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

        if can_actuate {
            let mut energy_step = 0.0;
            for controller in &self.controllers {
                let signal_x = clamp(controller.control.signal_x(sim_time), -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED);
                let target_x = clamp(
                    signal_x / MAX_MOTOR_SPEED * controller.limit_x,
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
                        let signal_y =
                            clamp(controller.control.signal_y(sim_time), -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED);
                        let signal_z =
                            clamp(controller.control.signal_z(sim_time), -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED);
                        let target_y = clamp(
                            signal_y / MAX_MOTOR_SPEED * controller.limit_y,
                            -controller.limit_y,
                            controller.limit_y,
                        );
                        let target_z = clamp(
                            signal_z / MAX_MOTOR_SPEED * controller.limit_z,
                            -controller.limit_z,
                            controller.limit_z,
                        );
                        joint.data.set_motor_position(
                            JointAxis::AngY,
                            target_y,
                            controller.stiffness_y,
                            JOINT_MOTOR_RESPONSE,
                        );
                        joint.data.set_motor_max_force(JointAxis::AngY, controller.torque_y);
                        joint.data.set_motor_position(
                            JointAxis::AngZ,
                            target_z,
                            controller.stiffness_z,
                            JOINT_MOTOR_RESPONSE,
                        );
                        joint.data.set_motor_max_force(JointAxis::AngZ, controller.torque_z);
                        energy_step +=
                            (target_y.abs() * controller.stiffness_y).min(controller.torque_y)
                                * dt;
                        energy_step +=
                            (target_z.abs() * controller.stiffness_z).min(controller.torque_z)
                                * dt;
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

        for part in &self.parts {
            if let Some(body) = self.bodies.get_mut(part.body) {
                let av = *body.angvel();
                let av_len = av.norm();
                if av_len > MAX_BODY_ANGULAR_SPEED {
                    body.set_angvel(av * (MAX_BODY_ANGULAR_SPEED / av_len), true);
                }
                let lv = *body.linvel();
                let lv_len = lv.norm();
                if lv_len > MAX_BODY_LINEAR_SPEED {
                    body.set_linvel(lv * (MAX_BODY_LINEAR_SPEED / lv_len), true);
                }
            }
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
                    return Err(
                        "morphologyMode is required for set_morphology_mode".to_string(),
                    );
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
        {
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
        }
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
        let config = TrialConfig {
            duration_seconds: DEFAULT_GENERATION_SECONDS,
            dt: FIXED_SIM_DT,
            snapshot_hz: 30.0,
            motor_power_scale: 1.0,
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
        let holdout_trial_seeds =
            build_holdout_trial_seed_set(HOLDOUT_TRIALS_PER_CANDIDATE);
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
            let wall_trial_start = Instant::now();

            for step in 0..steps {
                let (paused_now, restart_now, _, _, _, _, _) = controller.command_snapshot();
                if restart_now {
                    controller.force_restart();
                    aborted = true;
                    break;
                }
                if paused_now {
                    loop {
                        std::thread::sleep(Duration::from_millis(25));
                        let (paused_loop, restart_loop, _, _, _, _, _) =
                            controller.command_snapshot();
                        if restart_loop {
                            controller.force_restart();
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
                InjectMutationMode::Light => mutate_genome(injection.genome, 0.12, rng),
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
        match evaluate_generation_attempt(0, &top.genome, holdout_trial_seeds, config, |_| Ok(()))
        {
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
        next_genomes.push(mutate_genome(ranked_by_fitness[0].genome.clone(), 0.7, rng));
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
    let stagnation_pressure =
        clamp(stagnation_generations(performance_history) as f32 / 24.0, 0.0, 1.0);
    let anneal_factor = clamp(
        annealing_progress(*generation) * elite_consistency * (1.0 - 0.35 * stagnation_pressure),
        0.0,
        1.0,
    );
    let base_mutation_rate = if ranked_for_breeding.len() == 1 {
        0.65
    } else {
        0.24
    };
    let raw_mutation_rate = base_mutation_rate
        + (1.0 - mean_novelty_norm) * 0.08
        + stagnation_pressure * 0.08
        + holdout_gap_norm * 0.08;
    let min_mutation_rate = lerp(
        MIN_BREEDING_MUTATION_RATE,
        MUTATION_RATE_FINAL_FLOOR,
        anneal_factor,
    );
    let max_mutation_rate = lerp(
        MAX_BREEDING_MUTATION_RATE,
        MUTATION_RATE_FINAL_CEILING,
        anneal_factor,
    )
    .max(min_mutation_rate + 0.01);
    let mutation_rate = clamp(raw_mutation_rate, min_mutation_rate, max_mutation_rate);
    let random_inject_chance = if ranked_for_breeding.len() > 1 {
        clamp(
            (0.04 + (1.0 - mean_novelty_norm) * 0.04) * lerp(1.0, 0.35, anneal_factor)
                + stagnation_pressure * 0.015
                + holdout_gap_norm * 0.01,
            0.0,
            0.25,
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
        let mut child = crossover_genome(&parent_a, &parent_b, rng);
        child = mutate_genome(child, mutation_rate, rng);
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
        anneal_factor,
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
    stable_hash_mix(&mut hash, MAX_LIMBS as u64);
    stable_hash_mix(&mut hash, MAX_SEGMENTS_PER_LIMB as u64);
    stable_hash_mix(&mut hash, quantize_to_bucket(genome.torso.w, 0.05) as u64);
    stable_hash_mix(&mut hash, quantize_to_bucket(genome.torso.h, 0.05) as u64);
    stable_hash_mix(&mut hash, quantize_to_bucket(genome.torso.d, 0.05) as u64);
    for limb in &genome.limbs {
        stable_hash_mix(&mut hash, u64::from(limb.enabled));
        let count = active_segment_count(limb).clamp(1, MAX_SEGMENTS_PER_LIMB);
        stable_hash_mix(&mut hash, count as u64);
        if !limb.enabled {
            continue;
        }
        stable_hash_mix(&mut hash, quantize_to_bucket(limb.anchor_x, 0.05) as u64);
        stable_hash_mix(&mut hash, quantize_to_bucket(limb.anchor_y, 0.05) as u64);
        stable_hash_mix(&mut hash, quantize_to_bucket(limb.anchor_z, 0.05) as u64);
        stable_hash_mix(&mut hash, quantize_to_bucket(limb.axis_y, 0.05) as u64);
        stable_hash_mix(&mut hash, quantize_to_bucket(limb.axis_z, 0.05) as u64);
        stable_hash_mix(&mut hash, quantize_to_bucket(limb.dir_x, 0.05) as u64);
        stable_hash_mix(&mut hash, quantize_to_bucket(limb.dir_y, 0.05) as u64);
        stable_hash_mix(&mut hash, quantize_to_bucket(limb.dir_z, 0.05) as u64);
        for segment in limb.segments.iter().take(count.min(limb.segments.len())) {
            stable_hash_mix(&mut hash, quantize_to_bucket(segment.length, 0.05) as u64);
            stable_hash_mix(
                &mut hash,
                quantize_to_bucket(segment.thickness, 0.05) as u64,
            );
            stable_hash_mix(&mut hash, quantize_to_bucket(segment.mass, 0.05) as u64);
            stable_hash_mix(&mut hash, quantize_to_bucket(segment.limit_x, 0.05) as u64);
            stable_hash_mix(&mut hash, quantize_to_bucket(segment.limit_y, 0.05) as u64);
            stable_hash_mix(&mut hash, quantize_to_bucket(segment.limit_z, 0.05) as u64);
            stable_hash_mix(
                &mut hash,
                match segment.joint_type {
                    JointTypeGene::Hinge => 0,
                    JointTypeGene::Ball => 1,
                },
            );
        }
        for (index, control) in limb.controls.iter().take(count.min(limb.controls.len())).enumerate() {
            let joint_type = limb
                .segments
                .get(index)
                .map(|segment| segment.joint_type)
                .unwrap_or(JointTypeGene::Hinge);
            stable_hash_mix(&mut hash, quantize_to_bucket(control.amp, 0.1) as u64);
            stable_hash_mix(&mut hash, quantize_to_bucket(control.freq, 0.05) as u64);
            stable_hash_mix(&mut hash, quantize_to_bucket(control.phase, 0.1) as u64);
            stable_hash_mix(&mut hash, quantize_to_bucket(control.bias, 0.1) as u64);
            stable_hash_mix(&mut hash, quantize_to_bucket(control.harm2_amp, 0.1) as u64);
            stable_hash_mix(&mut hash, quantize_to_bucket(control.harm2_phase, 0.1) as u64);
            if matches!(joint_type, JointTypeGene::Ball) {
                stable_hash_mix(&mut hash, quantize_to_bucket(control.amp_y, 0.1) as u64);
                stable_hash_mix(&mut hash, quantize_to_bucket(control.freq_y, 0.05) as u64);
                stable_hash_mix(&mut hash, quantize_to_bucket(control.phase_y, 0.1) as u64);
                stable_hash_mix(&mut hash, quantize_to_bucket(control.bias_y, 0.1) as u64);
                stable_hash_mix(&mut hash, quantize_to_bucket(control.amp_z, 0.1) as u64);
                stable_hash_mix(&mut hash, quantize_to_bucket(control.freq_z, 0.05) as u64);
                stable_hash_mix(&mut hash, quantize_to_bucket(control.phase_z, 0.1) as u64);
                stable_hash_mix(&mut hash, quantize_to_bucket(control.bias_z, 0.1) as u64);
            }
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
    let torso_wh = if genome.torso.h.abs() > 1e-5 {
        genome.torso.w / genome.torso.h
    } else {
        1.0
    };
    let torso_dh = if genome.torso.h.abs() > 1e-5 {
        genome.torso.d / genome.torso.h
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
    genome.limbs.iter().filter(|limb| limb.enabled).count()
}

fn segment_count_histogram(genome: &Genome) -> [usize; MAX_SEGMENTS_PER_LIMB] {
    let mut histogram = [0usize; MAX_SEGMENTS_PER_LIMB];
    for limb in &genome.limbs {
        if !limb.enabled {
            continue;
        }
        let segment_count = active_segment_count(limb).clamp(1, MAX_SEGMENTS_PER_LIMB);
        histogram[segment_count - 1] += 1;
    }
    histogram
}

fn active_segment_count(limb: &LimbGene) -> usize {
    limb.segment_count.clamp(1, MAX_SEGMENTS_PER_LIMB as u32) as usize
}

fn for_each_active_segment<F>(genome: &Genome, mut f: F)
where
    F: FnMut(&SegmentGene),
{
    for limb in &genome.limbs {
        if !limb.enabled {
            continue;
        }
        let count = active_segment_count(limb).min(limb.segments.len());
        for segment in limb.segments.iter().take(count) {
            f(segment);
        }
    }
}

fn for_each_active_control<F>(genome: &Genome, mut f: F)
where
    F: FnMut(&ControlGene, JointTypeGene),
{
    for limb in &genome.limbs {
        if !limb.enabled {
            continue;
        }
        let count = active_segment_count(limb).min(limb.controls.len());
        for (index, control) in limb.controls.iter().take(count).enumerate() {
            let joint_type = limb
                .segments
                .get(index)
                .map(|segment| segment.joint_type)
                .unwrap_or(JointTypeGene::Hinge);
            f(control, joint_type);
        }
    }
}

fn feature_torso_w(genome: &Genome) -> f32 {
    genome.torso.w
}

fn feature_torso_h(genome: &Genome) -> f32 {
    genome.torso.h
}

fn feature_torso_d(genome: &Genome) -> f32 {
    genome.torso.d
}

fn feature_torso_mass(genome: &Genome) -> f32 {
    genome.torso.mass
}

fn feature_mass_scale(genome: &Genome) -> f32 {
    genome.mass_scale
}

fn feature_enabled_limb_ratio(genome: &Genome) -> f32 {
    let enabled = genome.limbs.iter().filter(|limb| limb.enabled).count();
    enabled as f32 / MAX_LIMBS as f32
}

fn feature_segment_count_mean(genome: &Genome) -> f32 {
    let mut count = 0usize;
    let mut total = 0.0f32;
    for limb in &genome.limbs {
        if !limb.enabled {
            continue;
        }
        total += active_segment_count(limb) as f32;
        count += 1;
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
    for_each_active_segment(genome, |segment| {
        total += segment.length;
        count += 1;
    });
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

fn feature_segment_mass_mean(genome: &Genome) -> f32 {
    let mut count = 0usize;
    let mut total = 0.0f32;
    for_each_active_segment(genome, |segment| {
        total += segment.mass;
        count += 1;
    });
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

fn feature_control_amp_mean(genome: &Genome) -> f32 {
    let mut count = 0usize;
    let mut total = 0.0f32;
    for_each_active_control(genome, |control, joint_type| {
        total += control.amp + control.harm2_amp;
        count += 2;
        if matches!(joint_type, JointTypeGene::Ball) {
            total += control.amp_y + control.amp_z;
            count += 2;
        }
    });
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

fn feature_control_freq_mean(genome: &Genome) -> f32 {
    let mut count = 0usize;
    let mut total = 0.0f32;
    for_each_active_control(genome, |control, joint_type| {
        total += control.freq;
        count += 1;
        if matches!(joint_type, JointTypeGene::Ball) {
            total += control.freq_y + control.freq_z;
            count += 2;
        }
    });
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

fn feature_ball_joint_ratio(genome: &Genome) -> f32 {
    let mut total = 0usize;
    let mut ball = 0usize;
    for limb in &genome.limbs {
        if !limb.enabled {
            continue;
        }
        let count = active_segment_count(limb).min(limb.segments.len());
        for segment in limb.segments.iter().take(count) {
            total += 1;
            if matches!(segment.joint_type, JointTypeGene::Ball) {
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
                        let outcome = TrialSimulator::new(&genomes[a], seeds[t], config_ref)
                            .and_then(|mut sim| {
                                let steps =
                                    (config_ref.duration_seconds / config_ref.dt).ceil() as usize;
                                for _ in 0..steps {
                                    sim.step()?;
                                }
                                Ok(sim.final_result())
                            });
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
    let steps = (config.duration_seconds / config.dt).ceil() as usize;
    for (trial_index, &seed) in seeds.iter().enumerate() {
        on_trial_started(trial_index)?;
        let mut sim = TrialSimulator::new(genome, seed, config)?;
        for _ in 0..steps {
            sim.step()?;
        }
        trials.push(sim.final_result());
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

    let mut active_limbs = 0usize;
    let mut segment_total = 0usize;
    for limb in &genome.limbs {
        if !limb.enabled {
            continue;
        }
        active_limbs += 1;
        segment_total += limb.segment_count.clamp(1, MAX_SEGMENTS_PER_LIMB as u32) as usize;
    }
    let mean_segment_count = if active_limbs > 0 {
        segment_total as f32 / active_limbs as f32
    } else {
        0.0
    };

    let descriptor = [
        clamp(median_progress / 28.0, 0.0, 1.0),
        clamp(median_upright, 0.0, 1.0),
        clamp(median_straightness, 0.0, 1.0),
        clamp(active_limbs as f32 / MAX_LIMBS as f32, 0.0, 1.0),
        clamp(mean_segment_count / MAX_SEGMENTS_PER_LIMB as f32, 0.0, 1.0),
    ];

    GenerationEvalResult {
        fitness: (robust_quality * consistency_gate).max(0.0),
        descriptor,
        trial_count: trials.len(),
        median_progress,
        median_upright,
        median_straightness,
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
    let _ = generation_index;
    build_seed_bank(TRAIN_TRIAL_SEED_BANK_TAG, count)
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

fn detect_morphology_preset(genome: &Genome) -> Option<MorphologyPreset> {
    if matches_spider4x2_preset(genome) {
        Some(MorphologyPreset::Spider4x2)
    } else {
        None
    }
}

fn approx_eq(a: f32, b: f32, tolerance: f32) -> bool {
    (a - b).abs() <= tolerance
}

fn matches_spider4x2_preset(genome: &Genome) -> bool {
    if genome.limbs.len() < MAX_LIMBS {
        return false;
    }
    if !approx_eq(genome.torso.w, 1.68, 0.02)
        || !approx_eq(genome.torso.h, 0.66, 0.02)
        || !approx_eq(genome.torso.d, 1.22, 0.02)
        || !approx_eq(genome.torso.mass, 1.08, 0.04)
        || !approx_eq(genome.mass_scale, 1.0, 0.05)
    {
        return false;
    }
    let expected_anchors = [
        (0.72, -0.18, 0.46),
        (-0.72, -0.18, 0.46),
        (0.72, -0.18, -0.46),
        (-0.72, -0.18, -0.46),
    ];
    for (limb_index, expected) in expected_anchors.iter().enumerate() {
        let limb = &genome.limbs[limb_index];
        if !limb.enabled || limb.segment_count != 2 {
            return false;
        }
        if !approx_eq(limb.anchor_x, expected.0, 0.05)
            || !approx_eq(limb.anchor_y, expected.1, 0.05)
            || !approx_eq(limb.anchor_z, expected.2, 0.05)
        {
            return false;
        }
        if let Some(shoulder) = limb.segments.first() {
            if !matches!(shoulder.joint_type, JointTypeGene::Ball) {
                return false;
            }
        } else {
            return false;
        }
        if let Some(distal) = limb.segments.get(1) {
            if !matches!(distal.joint_type, JointTypeGene::Hinge) {
                return false;
            }
        } else {
            return false;
        }
    }
    for limb in genome.limbs.iter().skip(4).take(MAX_LIMBS.saturating_sub(4)) {
        if limb.enabled {
            return false;
        }
    }
    true
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
    Genome {
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
        torso,
        limbs,
        hue: rng.random::<f32>(),
        mass_scale: rng_range(rng, 0.78, 1.3),
    };
    ensure_active_body_plan(&mut genome, rng);
    genome
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
    ensure_active_body_plan(&mut child, rng);
    child
}

fn mutate_genome(mut genome: Genome, chance: f32, rng: &mut SmallRng) -> Genome {
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
        .linear_damping(0.19)
        .angular_damping(0.31)
        .ccd_enabled(true)
        .build();
    let handle = bodies.insert(body);
    let collider = ColliderBuilder::cuboid(size[0] * 0.5, size[1] * 0.5, size[2] * 0.5)
        .mass(mass)
        .friction(1.08)
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

fn normalized_dir(x: f32, y: f32, z: f32) -> Vector3<f32> {
    let dir = vector![x, y, z];
    dir.try_normalize(1e-6).unwrap_or(vector![0.0, -1.0, 0.0])
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

fn annealing_progress(generation: usize) -> f32 {
    let generation_f = generation.max(1) as f32;
    clamp(
        1.0 - (-generation_f / ANNEALING_TIME_CONSTANT_GENERATIONS).exp(),
        0.0,
        1.0,
    )
}
