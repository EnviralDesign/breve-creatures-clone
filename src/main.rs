use std::f32::consts::PI;
use std::collections::VecDeque;
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
use axum::extract::{Path, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use include_dir::{Dir, include_dir};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rapier3d::na::{UnitQuaternion, Vector3, point, vector};
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
const JOINT_MOTOR_RESPONSE: f32 = 12.0;
const JOINT_MOTOR_FORCE_MULTIPLIER: f32 = 1.0;
const AXIS_TILT_GAIN: f32 = 1.9;
const FALLEN_HEIGHT_THRESHOLD: f32 = 0.35;
const MAX_PLAUSIBLE_STEP_DISPLACEMENT: f32 = 1.5;
const FITNESS_UPRIGHT_BONUS: f32 = 1.35;
const FITNESS_STRAIGHTNESS_BONUS: f32 = 1.5;
const FITNESS_HEIGHT_BONUS: f32 = 0.6;
const FITNESS_ENERGY_PENALTY: f32 = 1.2;
const FITNESS_INSTABILITY_PENALTY: f32 = 1.25;
const FITNESS_NET_PROGRESS_WEIGHT: f32 = 0.8;
const FALLEN_PENALTY_STRENGTH: f32 = 0.6;
const UPRIGHT_FULL_SCORE_THRESHOLD: f32 = 0.5;
const UPRIGHT_PENALTY_FLOOR: f32 = 0.4;
const SETTLE_SECONDS: f32 = 2.25;
const GROUND_COLLISION_GROUP: Group = Group::GROUP_1;
const CREATURE_COLLISION_GROUP: Group = Group::GROUP_2;
const DEFAULT_BIND_HOST: &str = "127.0.0.1";
const DEFAULT_BIND_PORT: u16 = 8787;
const PORT_FALLBACK_ATTEMPTS: u16 = 32;
const DEFAULT_POPULATION_SIZE: usize = 80;
const MIN_POPULATION_SIZE: usize = 1;
const MAX_POPULATION_SIZE: usize = 128;
const ELITE_COUNT: usize = 2;
const TRIALS_PER_CANDIDATE: usize = 3;
const DEFAULT_GENERATION_SECONDS: f32 = 18.0;
const EVOLUTION_VIEW_FRAME_LIMIT: usize = 900;
const CHECKPOINT_DIR: &str = "data/checkpoints";
const AUTOSAVE_EVERY_GENERATIONS: usize = 5;
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
    #[serde(default = "default_motor_strength")]
    motor_strength: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct ControlGene {
    amp: f32,
    freq: f32,
    phase: f32,
    bias: f32,
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
    control: ControlGene,
    torque: f32,
}

struct TrialAccumulator {
    spawn: Vector3<f32>,
    best_distance: f32,
    path_length: f32,
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
        let progress = net_distance * FITNESS_NET_PROGRESS_WEIGHT
            + peak_distance * (1.0 - FITNESS_NET_PROGRESS_WEIGHT);

        let mut quality = progress;
        quality += upright_avg * FITNESS_UPRIGHT_BONUS;
        quality += straightness * FITNESS_STRAIGHTNESS_BONUS;
        quality += clamp(avg_height / 3.0, 0.0, 1.0) * FITNESS_HEIGHT_BONUS;
        quality -= energy_norm * FITNESS_ENERGY_PENALTY;
        quality -= instability_norm * FITNESS_INSTABILITY_PENALTY;
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

        let drop_start = vector![spawn.x, spawn.y + rng_range(&mut rng, 5.2, 7.7), spawn.z];
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
            let rot = UnitQuaternion::from_euler_angles(
                rng_range(&mut rng, -0.36, 0.36),
                rng_range(&mut rng, 0.0, PI * 2.0),
                rng_range(&mut rng, -0.28, 0.28),
            );
            torso_body.set_rotation(rot, true);
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
                    .unwrap_or(ControlGene {
                        amp: 1.0,
                        freq: 1.0,
                        phase: 0.0,
                        bias: 0.0,
                    });

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
                    child_body.set_rotation(parent_rotation, true);
                }
                let local_anchor2 = point![
                    -local_growth.x * segment_gene.length * 0.5,
                    -local_growth.y * segment_gene.length * 0.5,
                    -local_growth.z * segment_gene.length * 0.5
                ];

                let mut joint = RevoluteJointBuilder::new(UnitVector::new_normalize(axis_local))
                    .local_anchor1(point![
                        pivot_from_parent.x,
                        pivot_from_parent.y,
                        pivot_from_parent.z
                    ])
                    .local_anchor2(local_anchor2)
                    .contacts_enabled(false);
                joint = if seg_index == 0 {
                    joint.limits([-1.57, 1.57])
                } else {
                    joint.limits([-2.09, 2.09])
                };
                let joint_handle = impulse_joints.insert(parent, child, joint, true);

                let torque = if seg_index == 0 {
                    MOTOR_TORQUE_HIP
                } else {
                    MOTOR_TORQUE_KNEE
                } * segment_gene.motor_strength
                    * JOINT_MOTOR_FORCE_MULTIPLIER
                    * config.motor_power_scale;
                if let Some(joint_ref) = impulse_joints.get_mut(joint_handle, false) {
                    joint_ref
                        .data
                        .set_motor_model(JointAxis::AngX, MotorModel::ForceBased);
                    joint_ref
                        .data
                        .set_motor_velocity(JointAxis::AngX, 0.0, JOINT_MOTOR_RESPONSE);
                    joint_ref.data.set_motor_max_force(JointAxis::AngX, torque);
                }
                controllers.push(SimController {
                    joint: joint_handle,
                    control: control_gene,
                    torque,
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

        if sim_time >= SETTLE_SECONDS {
            let mut energy_step = 0.0;
            for controller in &self.controllers {
                let speed = clamp(
                    controller.control.bias
                        + controller.control.amp
                            * (controller.control.freq * sim_time + controller.control.phase).sin(),
                    -MAX_MOTOR_SPEED,
                    MAX_MOTOR_SPEED,
                );
                if let Some(joint) = self.impulse_joints.get_mut(controller.joint, true) {
                    joint
                        .data
                        .set_motor_velocity(JointAxis::AngX, speed, JOINT_MOTOR_RESPONSE);
                    joint
                        .data
                        .set_motor_max_force(JointAxis::AngX, controller.torque);
                }
                energy_step += (speed * controller.torque).abs() * dt;
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
}

impl AppState {
    fn new() -> Self {
        let sim_worker_limit = resolve_sim_worker_limit();
        let evolution = EvolutionController::new();
        if let Ok((_id, snapshot)) = load_checkpoint_snapshot(None) {
            evolution.set_pending_loaded_checkpoint(snapshot);
        }
        start_evolution_worker(evolution.clone(), sim_worker_limit);
        Self {
            sim_slots: Arc::new(Semaphore::new(sim_worker_limit)),
            sim_worker_limit,
            evolution,
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
    Status { status: EvolutionStatus },
    TrialStarted {
        genome: Genome,
        part_sizes: Vec<[f32; 3]>,
    },
    Snapshot { frame: SnapshotFrame },
    TrialComplete { result: TrialResult },
    Error { message: String },
}

#[derive(Clone, Debug)]
struct EvolutionCommandState {
    paused: bool,
    restart_requested: bool,
    pending_population_size: usize,
    run_speed: f32,
    fast_forward_remaining: usize,
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
                injection_queue: VecDeque::new(),
                pending_loaded_checkpoint: None,
            })),
            shared: Arc::new(Mutex::new(EvolutionSharedState {
                status: initial_status,
                view: EvolutionViewState::default(),
                runtime_snapshot: None,
            })),
            events,
        })
    }

    fn subscribe(&self) -> broadcast::Receiver<EvolutionStreamEvent> {
        self.events.subscribe()
    }

    fn command_snapshot(&self) -> (bool, bool, usize, f32, usize) {
        let mut commands = self.commands.lock().expect("evolution command mutex poisoned");
        let restart = commands.restart_requested;
        commands.restart_requested = false;
        (
            commands.paused,
            restart,
            commands.pending_population_size,
            commands.run_speed,
            commands.fast_forward_remaining,
        )
    }

    fn take_pending_loaded_checkpoint(&self) -> Option<EvolutionRuntimeSnapshot> {
        let mut commands = self.commands.lock().expect("evolution command mutex poisoned");
        commands.pending_loaded_checkpoint.take()
    }

    fn force_restart(&self) {
        let mut commands = self.commands.lock().expect("evolution command mutex poisoned");
        commands.restart_requested = true;
    }

    fn queue_injections(
        &self,
        genomes: Vec<Genome>,
        mutation_mode: InjectMutationMode,
    ) -> EvolutionGenomeImportResponse {
        let mut commands = self.commands.lock().expect("evolution command mutex poisoned");
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
        let mut commands = self.commands.lock().expect("evolution command mutex poisoned");
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
        let mut commands = self.commands.lock().expect("evolution command mutex poisoned");
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
        shared.runtime_snapshot = Some(snapshot);
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
        let commands = self.commands.lock().expect("evolution command mutex poisoned");
        commands.injection_queue.iter().cloned().collect()
    }

    fn consume_fast_forward_generation(&self) -> usize {
        let remaining = {
            let mut commands = self.commands.lock().expect("evolution command mutex poisoned");
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
        let mut commands = self.commands.lock().expect("evolution command mutex poisoned");
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
                    requested,
                    before,
                    commands.fast_forward_remaining
                );
            }
            "stop_fast_forward" => {
                let before = commands.fast_forward_remaining;
                commands.fast_forward_remaining = 0;
                info!("control stop_fast_forward: before={}, after=0", before);
            }
            other => return Err(format!("unsupported action '{other}'")),
        }
        let paused = commands.paused;
        let pending_population_size = commands.pending_population_size;
        let run_speed = commands.run_speed;
        let fast_forward_remaining = commands.fast_forward_remaining;
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
        let _ = self.events.send(EvolutionStreamEvent::TrialStarted { genome, part_sizes });
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
        let _ = self.events.send(EvolutionStreamEvent::TrialComplete { result });
    }

    fn emit_error(&self, message: String) {
        let _ = self.events.send(EvolutionStreamEvent::Error { message });
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

fn start_evolution_worker(controller: Arc<EvolutionController>, sim_worker_limit: usize) {
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
        let mut current_attempt_index = 0usize;
        let mut current_trial_index = 0usize;

        controller.force_restart();
        loop {
            if let Some(loaded) = controller.take_pending_loaded_checkpoint() {
                let loaded_status = loaded.status.clone();
                generation = loaded_status.generation.max(1);
                population_size = loaded_status
                    .population_size
                    .clamp(MIN_POPULATION_SIZE, MAX_POPULATION_SIZE);
                best_ever_score = loaded_status.best_ever_score;
                best_genome = loaded_status.best_genome.clone();
                novelty_archive = loaded.novelty_archive.clone();
                batch_genomes = loaded.batch_genomes.clone();
                batch_results = loaded.batch_results.clone();
                attempt_trials = loaded.attempt_trials.clone();
                trial_seeds = loaded.trial_seeds.clone();
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
                    commands.injection_queue =
                        VecDeque::from(loaded.injection_queue.clone());
                }
                controller.update_status(|status| {
                    *status = loaded_status.clone();
                    status.run_speed = loaded_status.run_speed.clamp(0.5, 8.0);
                    status.current_attempt_index = current_attempt_index.min(population_size.saturating_sub(1));
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
                    generation,
                    population_size,
                    loaded_status.fast_forward_remaining
                );
                continue;
            }

            let (paused, restart_requested, pending_population_size, run_speed, fast_forward_remaining) =
                controller.command_snapshot();
            if restart_requested {
                info!(
                    "evolution restart requested; resetting to generation=1, population_size={}",
                    pending_population_size
                );
                generation = 1;
                best_ever_score = 0.0;
                best_genome = None;
                novelty_archive.clear();
                reset_evolution_batch(
                    &mut rng,
                    pending_population_size,
                    generation,
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
                    generation,
                    population_size,
                    TRIALS_PER_CANDIDATE
                );
                continue;
            }

            if current_attempt_index >= batch_genomes.len() {
                if batch_results.is_empty() {
                    reset_evolution_batch(
                        &mut rng,
                        pending_population_size,
                        generation,
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
                        generation,
                        population_size,
                        TRIALS_PER_CANDIDATE
                    );
                    continue;
                }
                let injected_genomes = dequeue_injected_genomes(&controller, &mut rng, 1);
                finalize_generation(
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
                );
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
                    generation,
                    population_size,
                    best_ever_score,
                    fast_forward_remaining
                );
                continue;
            }

            if fast_forward_remaining > 0 {
                info!(
                    "fast-forward processing generation={}, queued_remaining={}",
                    generation,
                    fast_forward_remaining
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
                let progress_worker = std::thread::spawn(move || {
                    run_generation_stream(
                        request_for_stream,
                        config_for_stream,
                        tx_progress,
                        sim_worker_limit,
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
                                status.current_trial_index =
                                    TRIALS_PER_CANDIDATE.saturating_sub(1);
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
                        streamed_results
                            .ok_or_else(|| "fast-forward generation produced no results".to_string())
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
                        let injected_genomes = dequeue_injected_genomes(&controller, &mut rng, 1);
                        finalize_generation(
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
                        );
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
                            generation,
                            best_ever_score,
                            remaining
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
                .unwrap_or_else(|| hash_uint32(generation as u32, current_trial_index as u32, 0x9e3779b9) as u64);
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
                let (paused_now, restart_now, _, _, _) = controller.command_snapshot();
                if restart_now {
                    controller.force_restart();
                    aborted = true;
                    break;
                }
                if paused_now {
                    loop {
                        std::thread::sleep(Duration::from_millis(25));
                        let (paused_loop, restart_loop, _, _, _) = controller.command_snapshot();
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
                status.current_attempt_index = current_attempt_index.min(population_size.saturating_sub(1));
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
    *batch_genomes = (0..clamped_size).map(|_| random_genome(rng)).collect();
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
) -> Vec<Genome> {
    controller
        .take_injections(max_count)
        .into_iter()
        .map(|injection| match injection.mutation_mode {
            InjectMutationMode::None => injection.genome,
            InjectMutationMode::Light => mutate_genome(injection.genome, 0.12, rng),
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
    };
    controller.update_runtime_snapshot(snapshot);
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
) {
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
        return;
    };
    if top.fitness > *best_ever_score {
        *best_ever_score = top.fitness;
        *best_genome = Some(top.genome.clone());
    }

    let target_population_size = pending_population_size.clamp(MIN_POPULATION_SIZE, MAX_POPULATION_SIZE);
    let mut next_genomes: Vec<Genome> = Vec::with_capacity(target_population_size);
    for genome in injected_genomes.into_iter().take(target_population_size) {
        next_genomes.push(genome);
    }
    let elite_count = ELITE_COUNT
        .min(ranked_by_fitness.len())
        .min(target_population_size.saturating_sub(next_genomes.len()));
    if elite_count > 0 && next_genomes.len() < target_population_size {
        next_genomes.push(ranked_by_fitness[0].genome.clone());
    }
    if elite_count > 1 && next_genomes.len() < target_population_size {
        let diversity_elite = ranked_for_breeding.first();
        if let Some(candidate) = diversity_elite {
            if candidate.attempt != ranked_by_fitness[0].attempt {
                next_genomes.push(candidate.genome.clone());
            } else if ranked_by_fitness.len() > 1 {
                next_genomes.push(ranked_by_fitness[1].genome.clone());
            }
        }
    }
    if target_population_size == 1 && next_genomes.is_empty() && !ranked_by_fitness.is_empty() {
        next_genomes.clear();
        next_genomes.push(mutate_genome(
            ranked_by_fitness[0].genome.clone(),
            0.7,
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
    while next_genomes.len() < target_population_size {
        let tournament_size = 4usize.min(ranked_for_breeding.len().max(1));
        let parent_a = tournament_select(&ranked_for_breeding, tournament_size, rng).genome.clone();
        let parent_b = if ranked_for_breeding.len() > 1 {
            tournament_select(&ranked_for_breeding, tournament_size, rng)
                .genome
                .clone()
        } else {
            parent_a.clone()
        };
        let mut child = crossover_genome(&parent_a, &parent_b, rng);
        let base_mutation_rate = if ranked_for_breeding.len() == 1 { 0.65 } else { 0.24 };
        let mutation_rate = clamp(base_mutation_rate + (1.0 - mean_novelty_norm) * 0.08, 0.18, 0.72);
        child = mutate_genome(child, mutation_rate, rng);
        let random_inject_chance = if ranked_for_breeding.len() > 1 {
            0.04 + (1.0 - mean_novelty_norm) * 0.04
        } else {
            0.0
        };
        if ranked_for_breeding.len() > 1 && rng.random::<f32>() < random_inject_chance {
            child = random_genome(rng);
        }
        next_genomes.push(child);
    }

    *generation += 1;
    *population_size = target_population_size;
    *batch_genomes = next_genomes;
    batch_results.clear();
    *attempt_trials = vec![Vec::new(); *population_size];
    *trial_seeds = build_trial_seed_set(*generation, TRIALS_PER_CANDIDATE);
    *current_attempt_index = 0;
    *current_trial_index = 0;
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .compact()
        .init();

    let state = AppState::new();
    info!(
        "simulation worker slots: {} (override with SIM_MAX_CONCURRENT_JOBS)",
        state.sim_worker_limit
    );

    let app = Router::new()
        .route("/health", get(health))
        .route("/api/trial/ws", get(ws_trial_handler))
        .route("/api/eval/ws", get(ws_eval_handler))
        .route("/api/eval/generation", post(eval_generation_handler))
        .route("/api/evolution/state", get(evolution_state_handler))
        .route("/api/evolution/control", post(evolution_control_handler))
        .route("/api/evolution/genome/current", get(evolution_current_genome_handler))
        .route("/api/evolution/genome/best", get(evolution_best_genome_handler))
        .route("/api/evolution/genome/import", post(evolution_import_genome_handler))
        .route("/api/evolution/checkpoint/save", post(evolution_checkpoint_save_handler))
        .route("/api/evolution/checkpoint/list", get(evolution_checkpoint_list_handler))
        .route("/api/evolution/checkpoint/load", post(evolution_checkpoint_load_handler))
        .route("/api/evolution/ws", get(ws_evolution_handler))
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

    let bind_port = resolve_bind_port();
    let (listener, addr) = match bind_listener(DEFAULT_BIND_HOST, bind_port).await {
        Ok(bound) => bound,
        Err(message) => {
            error!("{message}");
            return;
        }
    };
    info!("breve-creatures listening on http://{addr}");
    info!("frontend UI available at http://{addr}/");
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
    state
        .evolution
        .current_genome()
        .map(Json)
        .ok_or((StatusCode::NOT_FOUND, "no current genome available".to_string()))
}

async fn evolution_best_genome_handler(
    State(state): State<AppState>,
) -> Result<Json<Genome>, (StatusCode, String)> {
    state
        .evolution
        .current_best_genome()
        .map(Json)
        .ok_or((StatusCode::NOT_FOUND, "no best genome available".to_string()))
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
    let saved = save_checkpoint_snapshot(&snapshot, request.name.as_deref()).map_err(internal_err)?;
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
    let (id, snapshot) =
        load_checkpoint_snapshot(request.id.as_deref()).map_err(|message| (StatusCode::BAD_REQUEST, message))?;
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
                if send_evolution_stream_event(&mut socket, event).await.is_err() {
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
    let next_attempt = AtomicUsize::new(0);
    let cancelled = AtomicBool::new(false);
    let (result_tx, result_rx) =
        std_mpsc::channel::<Result<(usize, GenerationEvalResult), String>>();

    std::thread::scope(|scope| {
        for _ in 0..worker_count {
            let tx_progress = tx.clone();
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
                        |trial_index| {
                            tx_progress
                                .blocking_send(GenerationStreamEvent::AttemptTrialStarted {
                                    attempt_index,
                                    trial_index,
                                    trial_count,
                                })
                                .map_err(|err| format!("failed sending attempt progress: {err}"))
                        },
                    )
                    .and_then(|result| {
                        tx_progress
                            .blocking_send(GenerationStreamEvent::AttemptComplete {
                                attempt_index,
                                result: result.clone(),
                            })
                            .map_err(|err| format!("failed sending attempt complete: {err}"))?;
                        Ok((attempt_index, result))
                    });
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

    tx.blocking_send(GenerationStreamEvent::GenerationComplete { results })
        .map_err(|err| format!("failed sending generation complete: {err}"))?;
    Ok(())
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
        fitness: robust_quality.max(0.0),
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
    fs::create_dir_all(&dir)
        .map_err(|err| format!("failed creating checkpoint directory '{}': {err}", dir.display()))?;
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

fn write_checkpoint_file(path: &FsPath, checkpoint: &EvolutionCheckpointFile) -> Result<(), String> {
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

fn load_checkpoint_snapshot(id: Option<&str>) -> Result<(String, EvolutionRuntimeSnapshot), String> {
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

        neighbor_distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
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

    normalize_candidate_field(results, |candidate| candidate.fitness, |candidate, value| {
        candidate.quality_norm = value;
    });
    normalize_candidate_field(results, |candidate| candidate.novelty, |candidate, value| {
        candidate.novelty_norm = value;
    });
    for result in results.iter_mut() {
        result.selection_score =
            0.62 * result.quality_norm + 0.28 * result.novelty_norm + 0.1 * result.local_competition;
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
    assert!(!ranked.is_empty(), "tournament_select requires non-empty input");
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
    (0..count)
        .map(|i| hash_uint32(generation_index as u32, (i + 1) as u32, 0x9e3779b9) as u64)
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

fn random_genome(rng: &mut SmallRng) -> Genome {
    let torso = TorsoGene {
        w: clamp(rng_range(rng, 0.5, 2.75) * rng_range(rng, 0.86, 1.18), 0.5, 3.0),
        h: clamp(rng_range(rng, 0.5, 2.75) * rng_range(rng, 0.72, 1.15), 0.5, 3.0),
        d: clamp(rng_range(rng, 0.5, 2.75) * rng_range(rng, 0.86, 1.18), 0.5, 3.0),
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
                        thickness: clamp(
                            rng_range(rng, 0.16, 0.95) * hierarchy_scale,
                            0.14,
                            1.05,
                        ),
                        mass: clamp(rng_range(rng, 0.24, 1.75) * hierarchy_scale, 0.14, 2.0),
                        motor_strength: rng_range(rng, 0.5, 3.0),
                    }
                })
                .collect::<Vec<_>>();
            let controls = (0..MAX_SEGMENTS_PER_LIMB)
                .map(|seg_index| ControlGene {
                    amp: if seg_index == 0 {
                        rng_range(rng, 1.7, 10.3)
                    } else {
                        rng_range(rng, 1.05, 9.1)
                    },
                    freq: rng_range(rng, 0.55, 4.4),
                    phase: rng_range(rng, 0.0, PI * 2.0),
                    bias: rng_range(rng, -1.6, 1.6),
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
        limb.enabled = if rng.random::<f32>() < 0.5 { la.enabled } else { lb.enabled };
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
            sg.motor_strength = lerp(sa.motor_strength, sb.motor_strength, seg_blend);

            let ctrl_blend = rng.random::<f32>();
            cg.amp = lerp(ca.amp, cb.amp, ctrl_blend);
            cg.freq = lerp(ca.freq, cb.freq, ctrl_blend);
            cg.phase = wrap_phase(lerp(ca.phase, cb.phase, ctrl_blend));
            cg.bias = lerp(ca.bias, cb.bias, ctrl_blend);
        }
    }
    child.hue = if rng.random::<f32>() < 0.5 { a.hue } else { b.hue };
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
            let delta = if rng.random::<f32>() < 0.5 { -1i32 } else { 1i32 };
            limb.segment_count = (limb.segment_count as i32 + delta)
                .clamp(1, MAX_SEGMENTS_PER_LIMB as i32) as u32;
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
            segment.motor_strength = mutate_number(segment.motor_strength, 0.5, 3.0, chance, 0.2, rng);
        }
        for control in &mut limb.controls {
            control.amp = mutate_number(control.amp, 0.35, 11.6, chance, 0.18, rng);
            control.freq = mutate_number(control.freq, 0.3, 4.9, chance, 0.14, rng);
            control.phase = wrap_phase(mutate_number(control.phase, 0.0, PI * 2.0, chance, 0.28, rng));
            control.bias = mutate_number(control.bias, -2.35, 2.35, chance, 0.16, rng);
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

fn resolve_sim_worker_limit() -> usize {
    const ENV_VAR: &str = "SIM_MAX_CONCURRENT_JOBS";
    if let Ok(raw_value) = std::env::var(ENV_VAR) {
        match raw_value.parse::<usize>() {
            Ok(parsed) if parsed > 0 => return parsed,
            _ => warn!("{ENV_VAR} must be a positive integer; got '{raw_value}'"),
        }
    }

    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get().saturating_sub(1).max(1))
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

fn default_motor_strength() -> f32 {
    1.0
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
