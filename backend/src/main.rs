use std::f32::consts::PI;
use std::net::SocketAddr;

use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rapier3d::na::{UnitQuaternion, Vector3, point, vector};
use rapier3d::prelude::*;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info};

const MAX_LIMBS: usize = 6;
const MAX_SEGMENTS_PER_LIMB: usize = 3;
const FIXED_SIM_DT: f32 = 1.0 / 120.0;
const MASS_DENSITY_MULTIPLIER: f32 = 1.4;
const MAX_MOTOR_SPEED: f32 = 6.8;
const MAX_BODY_ANGULAR_SPEED: f32 = 15.0;
const MAX_BODY_LINEAR_SPEED: f32 = 22.0;
const MOTOR_TORQUE_HIP: f32 = 85.0;
const MOTOR_TORQUE_KNEE: f32 = 68.0;
const JOINT_MOTOR_RESPONSE: f32 = 24.0;
const JOINT_MOTOR_FORCE_MULTIPLIER: f32 = 1.8;
const AXIS_TILT_GAIN: f32 = 1.9;
const FALLEN_HEIGHT_THRESHOLD: f32 = 0.55;
const FITNESS_UPRIGHT_BONUS: f32 = 1.35;
const FITNESS_STRAIGHTNESS_BONUS: f32 = 0.8;
const FITNESS_HEIGHT_BONUS: f32 = 0.6;
const FITNESS_ENERGY_PENALTY: f32 = 3.6;
const FITNESS_INSTABILITY_PENALTY: f32 = 1.25;
const FITNESS_NET_PROGRESS_WEIGHT: f32 = 0.8;
const FALLEN_PENALTY_STRENGTH: f32 = 0.6;
const UPRIGHT_FULL_SCORE_THRESHOLD: f32 = 0.5;
const UPRIGHT_PENALTY_FLOOR: f32 = 0.4;
const SETTLE_SECONDS: f32 = 2.25;
const GROUND_COLLISION_GROUP: Group = Group::GROUP_1;
const CREATURE_COLLISION_GROUP: Group = Group::GROUP_2;

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
    #[serde(default)]
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
    #[serde(default)]
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

#[derive(Clone, Debug, Serialize)]
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

#[derive(Clone, Debug, Serialize)]
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

        self.net_dx = torso_pos.x - self.spawn.x;
        self.net_dz = torso_pos.z - self.spawn.z;
        let traveled = (self.net_dx * self.net_dx + self.net_dz * self.net_dz).sqrt();
        self.best_distance = self.best_distance.max(traveled);

        let up = torso_rot * vector![0.0, 1.0, 0.0];
        let upright = clamp((up.y + 1.0) * 0.5, 0.0, 1.0);
        self.upright_integral += upright * dt;
        self.height_integral += torso_pos.y * dt;

        let instability = torso_angvel.norm();
        self.instability_integral += instability * dt;

        if torso_pos.y < FALLEN_HEIGHT_THRESHOLD {
            self.fallen_time += dt;
        }

        if let Some(last) = self.last_torso_pos {
            let dx = torso_pos.x - last.x;
            let dz = torso_pos.z - last.z;
            self.path_length += (dx * dx + dz * dz).sqrt();
        }
        self.last_torso_pos = Some(torso_pos);

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
                    joint.limits([-1.05, 1.05])
                } else {
                    joint.limits([-1.35, 1.35])
                };
                let joint_handle = impulse_joints.insert(parent, child, joint, true);

                let torque = if seg_index == 0 {
                    MOTOR_TORQUE_HIP
                } else {
                    MOTOR_TORQUE_KNEE
                } * JOINT_MOTOR_FORCE_MULTIPLIER
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
struct AppState;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .compact()
        .init();

    let app = Router::new()
        .route("/health", get(health))
        .route("/api/trial/ws", get(ws_trial_handler))
        .route("/api/eval/ws", get(ws_eval_handler))
        .route("/api/eval/generation", post(eval_generation_handler))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(AppState);

    let addr = SocketAddr::from(([127, 0, 0, 1], 8787));
    info!("sim-backend listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind socket");
    axum::serve(listener, app)
        .await
        .expect("server exited unexpectedly");
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn ws_trial_handler(
    ws: WebSocketUpgrade,
    State(_state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(handle_trial_socket)
}

async fn ws_eval_handler(
    ws: WebSocketUpgrade,
    State(_state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(handle_eval_socket)
}

async fn handle_trial_socket(mut socket: WebSocket) {
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
    let (tx, mut rx) = mpsc::channel::<StreamEvent>(128);
    let request_for_task = request.clone();
    let config_for_task = config.clone();
    let worker = tokio::task::spawn_blocking(move || {
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

async fn handle_eval_socket(mut socket: WebSocket) {
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
    let (tx, mut rx) = mpsc::channel::<GenerationStreamEvent>(128);
    let request_for_task = request.clone();
    let config_for_task = config.clone();
    let worker = tokio::task::spawn_blocking(move || {
        if let Err(err) = run_generation_stream(request_for_task, config_for_task, tx) {
            error!("generation stream worker failed: {err}");
        }
    });

    while let Some(event) = rx.recv().await {
        if send_generation_stream_event(&mut socket, event).await.is_err() {
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

    let mut results = Vec::with_capacity(request.genomes.len());
    for (attempt_index, genome) in request.genomes.iter().enumerate() {
        let mut trials = Vec::with_capacity(request.seeds.len());
        for (trial_index, &seed) in request.seeds.iter().enumerate() {
            tx.blocking_send(GenerationStreamEvent::AttemptTrialStarted {
                attempt_index,
                trial_index,
                trial_count: request.seeds.len(),
            })
            .map_err(|err| format!("failed sending attempt progress: {err}"))?;

            let mut sim = TrialSimulator::new(genome, seed, &config)?;
            let steps = (config.duration_seconds / config.dt).ceil() as usize;
            for _ in 0..steps {
                sim.step()?;
            }
            trials.push(sim.final_result());
        }

        let result = summarize_trials(genome, &trials);
        results.push(result.clone());
        tx.blocking_send(GenerationStreamEvent::AttemptComplete {
            attempt_index,
            result,
        })
        .map_err(|err| format!("failed sending attempt complete: {err}"))?;
    }

    tx.blocking_send(GenerationStreamEvent::GenerationComplete { results })
        .map_err(|err| format!("failed sending generation complete: {err}"))?;
    Ok(())
}

async fn eval_generation_handler(
    State(_state): State<AppState>,
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
    let mut results = Vec::with_capacity(request.genomes.len());
    for genome in &request.genomes {
        let mut trials = Vec::with_capacity(request.seeds.len());
        for &seed in &request.seeds {
            let mut sim = TrialSimulator::new(genome, seed, &config).map_err(internal_err)?;
            let steps = (config.duration_seconds / config.dt).ceil() as usize;
            for _ in 0..steps {
                sim.step().map_err(internal_err)?;
            }
            trials.push(sim.final_result());
        }
        results.push(summarize_trials(genome, &trials));
    }
    Ok(Json(GenerationEvalResponse { results }))
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

fn internal_err(message: String) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, message)
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
