use std::hash::{Hash, Hasher};

use bevy::prelude::*;
use bevy_burn_human::{BurnHumanAssets, BurnHumanInput, BurnHumanMeshMode, BurnHumanPlugin};
use burn_human::data::reference::TensorData;
use burn_human::AnnyInput;
use noise::{NoiseFn, OpenSimplex};
use rand::Rng;

use crate::app::BevyZeroverseConfig;
use crate::camera::Playback;
use crate::scene::RegenerateSceneEvent;

pub struct ZeroverseBurnHumanPlugin;

impl Plugin for ZeroverseBurnHumanPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(
            BurnHumanPlugin::from_asset_path("burn_human/fullbody_default.meta.json")
                .with_render_mode(BurnHumanMeshMode::SkinnedMesh),
        );
        app.init_resource::<BurnHumanSettings>();
        app.init_resource::<BurnHumanPhenotypeSampler>();
        app.init_resource::<BurnHumanPoseNoiseSettings>();
        app.init_resource::<BurnHumanDescriptorPool>();

        app.register_type::<BurnHumanSettings>();
        app.register_type::<BurnHumanPhenotypeSampler>();
        app.register_type::<BurnHumanPoseNoiseSettings>();

        app.add_systems(Update, refresh_burn_human_pool);
        app.add_systems(
            PreUpdate,
            update_burn_human_inputs.run_if(resource_exists::<BurnHumanAssets>),
        );
    }
}

#[derive(Resource, Reflect, Debug, Clone)]
#[reflect(Resource)]
pub struct BurnHumanSettings {
    pub mesh_scale: f32,
    pub mesh_rotation: Quat,
    pub descriptor_pool_size: usize,
    pub compute_normals: bool,
}

impl Default for BurnHumanSettings {
    fn default() -> Self {
        Self {
            mesh_scale: 1.0,
            mesh_rotation: Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
            descriptor_pool_size: 32,
            compute_normals: true,
        }
    }
}

#[derive(Resource, Reflect, Debug, Clone)]
#[reflect(Resource)]
pub struct BurnHumanPhenotypeSampler {
    pub default_min: f64,
    pub default_max: f64,
    pub gender_min: f64,
    pub gender_max: f64,
    pub age_min: f64,
    pub age_max: f64,
    pub muscle_min: f64,
    pub muscle_max: f64,
    pub weight_min: f64,
    pub weight_max: f64,
    pub height_min: f64,
    pub height_max: f64,
    pub proportions_min: f64,
    pub proportions_max: f64,
}

impl Default for BurnHumanPhenotypeSampler {
    fn default() -> Self {
        Self {
            default_min: 0.15,
            default_max: 0.9,
            gender_min: 0.0,
            gender_max: 1.0,
            age_min: 0.3,
            age_max: 1.0,
            muscle_min: 0.2,
            muscle_max: 0.9,
            weight_min: 0.2,
            weight_max: 0.9,
            height_min: 0.2,
            height_max: 0.9,
            proportions_min: 0.2,
            proportions_max: 0.9,
        }
    }
}

impl BurnHumanPhenotypeSampler {
    fn range_for_label(&self, label: &str) -> (f64, f64) {
        match label {
            "gender" => (self.gender_min, self.gender_max),
            "age" => (self.age_min, self.age_max),
            "muscle" => (self.muscle_min, self.muscle_max),
            "weight" => (self.weight_min, self.weight_max),
            "height" => (self.height_min, self.height_max),
            "proportions" => (self.proportions_min, self.proportions_max),
            _ => (self.default_min, self.default_max),
        }
    }

    pub fn sample(&self, label: &str, rng: &mut impl Rng) -> f64 {
        let (mut min, mut max) = self.range_for_label(label);
        if min > max {
            std::mem::swap(&mut min, &mut max);
        }
        rng.random_range(min..=max)
    }
}

#[derive(Resource, Reflect, Debug, Clone)]
#[reflect(Resource)]
pub struct BurnHumanPoseNoiseSettings {
    pub noise_amp: f32,
    pub upper_leg_amp: f32,
    pub lower_leg_amp: f32,
    pub upper_arm_amp: f32,
    pub lower_arm_amp: f32,
    pub wrist_amp: f32,
    pub hand_amp: f32,
    pub spine_amp: f32,
    pub other_amp: f32,
    pub time_scale: f32,
}

impl Default for BurnHumanPoseNoiseSettings {
    fn default() -> Self {
        Self {
            noise_amp: 1.0,
            upper_leg_amp: 15.0,
            lower_leg_amp: 15.0,
            upper_arm_amp: 15.0,
            lower_arm_amp: 15.0,
            wrist_amp: 8.0,
            hand_amp: 5.0,
            spine_amp: 8.0,
            other_amp: 2.0,
            time_scale: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BurnHumanDescriptor {
    pub phenotype: Vec<f64>,
    pub pose_seed: u32,
}

#[derive(Resource, Debug, Default)]
pub struct BurnHumanDescriptorPool {
    pub pool: Vec<BurnHumanDescriptor>,
    pub next_seed: u32,
    pub regen_counter: u32,
}

#[derive(Component, Debug, Clone)]
pub struct BurnHumanInstance {
    pub phenotype: Vec<f64>,
    pub pose_seed: u32,
    pub base_pose: Vec<f64>,
    pub pose_baseline: Vec<[f32; 3]>,
    pub base_min: Vec3,
    pub base_max: Vec3,
    pub base_bounds_ready: bool,
    pub last_pose_key: u64,
}

#[derive(Component, Debug, Clone)]
pub struct BurnHumanPlacement {
    pub base_translation: Vec3,
    pub base_rotation: Quat,
    pub sampled_scale: Vec3,
    pub height_preserve_scale: bool,
    pub fit_to_sampled_box: bool,
}

impl BurnHumanInstance {
    pub fn new(descriptor: BurnHumanDescriptor, base_pose: Vec<f64>, bone_count: usize) -> Self {
        Self {
            phenotype: descriptor.phenotype,
            pose_seed: descriptor.pose_seed,
            base_pose,
            pose_baseline: vec![[0.0; 3]; bone_count],
            base_min: Vec3::ZERO,
            base_max: Vec3::ZERO,
            base_bounds_ready: false,
            last_pose_key: u64::MAX,
        }
    }
}

#[derive(Component, Debug, Default, Clone, Copy)]
pub struct BurnHumanPoseNoise;

pub fn sample_burn_human_descriptor(
    assets: &BurnHumanAssets,
    sampler: &BurnHumanPhenotypeSampler,
    settings: &BurnHumanSettings,
    pool: &mut BurnHumanDescriptorPool,
    rng: &mut impl Rng,
) -> BurnHumanDescriptor {
    ensure_descriptor_pool(assets, sampler, settings, pool, rng);
    if !pool.pool.is_empty() {
        let idx = rng.random_range(0..pool.pool.len());
        return pool.pool[idx].clone();
    }
    BurnHumanDescriptor {
        phenotype: sample_phenotype(assets, sampler, rng),
        pose_seed: next_seed(pool),
    }
}

fn ensure_descriptor_pool(
    assets: &BurnHumanAssets,
    sampler: &BurnHumanPhenotypeSampler,
    settings: &BurnHumanSettings,
    pool: &mut BurnHumanDescriptorPool,
    rng: &mut impl Rng,
) {
    if settings.descriptor_pool_size == 0 {
        return;
    }
    if !pool.pool.is_empty() {
        return;
    }
    for _ in 0..settings.descriptor_pool_size {
        let pose_seed = next_seed(pool);
        pool.pool.push(BurnHumanDescriptor {
            phenotype: sample_phenotype(assets, sampler, rng),
            pose_seed,
        });
    }
}

fn next_seed(pool: &mut BurnHumanDescriptorPool) -> u32 {
    let seed = pool.next_seed;
    pool.next_seed = pool.next_seed.wrapping_add(1);
    seed
}

fn sample_phenotype(
    assets: &BurnHumanAssets,
    sampler: &BurnHumanPhenotypeSampler,
    rng: &mut impl Rng,
) -> Vec<f64> {
    let labels = &assets.body.metadata().metadata.phenotype_labels;
    labels
        .iter()
        .map(|label| sampler.sample(label.as_str(), rng))
        .collect()
}

pub fn base_pose_from_assets(assets: &BurnHumanAssets) -> Vec<f64> {
    let bones = assets.body.metadata().metadata.bone_labels.len();
    let total = bones * 16;
    let mut pose = assets
        .body
        .metadata()
        .cases
        .iter()
        .find(|c| c.pose_parameters.shape[0] == 1)
        .map(|c| c.pose_parameters.data.clone())
        .unwrap_or_default();
    if pose.len() >= total {
        pose.truncate(total);
        return pose;
    }

    // Identity pose per bone.
    pose = vec![0.0; total];
    for bone in 0..bones {
        let base = bone * 16;
        pose[base] = 1.0;
        pose[base + 5] = 1.0;
        pose[base + 10] = 1.0;
        pose[base + 15] = 1.0;
    }
    pose
}

fn refresh_burn_human_pool(
    args: Res<BevyZeroverseConfig>,
    mut regen_events: MessageReader<RegenerateSceneEvent>,
    mut pool: ResMut<BurnHumanDescriptorPool>,
) {
    if args.regenerate_scene_mesh_shuffle_period == 0 {
        return;
    }

    for _ in regen_events.read() {
        pool.regen_counter = pool.regen_counter.saturating_add(1);
    }

    if pool.regen_counter >= args.regenerate_scene_mesh_shuffle_period {
        pool.regen_counter = 0;
        pool.pool.clear();
    }
}

fn update_burn_human_inputs(
    assets: Res<BurnHumanAssets>,
    playback: Res<Playback>,
    time: Res<Time>,
    noise_settings: Res<BurnHumanPoseNoiseSettings>,
    settings: Res<BurnHumanSettings>,
    mut regen_events: MessageReader<RegenerateSceneEvent>,
    mut humans: Query<(
        &mut BurnHumanInstance,
        &BurnHumanPlacement,
        &mut Transform,
        &mut BurnHumanInput,
        Option<&BurnHumanPoseNoise>,
    )>,
) {
    let mut predicted = *playback;
    if !regen_events.is_empty() {
        regen_events.clear();
        predicted.progress = 0.0;
    } else {
        predicted.step(&time);
    }
    let predicted_time = predicted.progress;

    for (mut instance, placement, mut transform, mut input, animated) in humans.iter_mut() {
        ensure_base_bounds(&assets, &settings, &mut instance);

        let time_key = if animated.is_some() {
            predicted_time * noise_settings.time_scale
        } else {
            0.0
        };

        let pose_key = compute_pose_key(&instance, time_key, &noise_settings);
        let phenotype_changed = input
            .phenotype_inputs
            .as_ref()
            .map(|values| values.as_slice() != instance.phenotype.as_slice())
            .unwrap_or(true);
        if phenotype_changed {
            input.phenotype_inputs = Some(instance.phenotype.clone());
        }
        if instance.last_pose_key != pose_key {
            let pose_parameters =
                build_pose_parameters(&assets, &instance, &noise_settings, time_key);
            input.pose_parameters = Some(pose_parameters);
            instance.last_pose_key = pose_key;
        }

        let final_scale = compute_final_scale(
            placement.sampled_scale,
            instance.base_min,
            instance.base_max,
            placement.height_preserve_scale,
            placement.fit_to_sampled_box,
        ) * settings.mesh_scale;
        let mut translation = placement.base_translation;
        let ground_min_y = instance.base_min.y;
        if ground_min_y != 0.0 {
            translation.y -= ground_min_y * final_scale.y;
        }
        if translation.y < 0.0 {
            translation.y = 0.0;
        }

        transform.translation = translation;
        transform.rotation = placement.base_rotation * settings.mesh_rotation;
        transform.scale = final_scale;
    }
}

fn compute_pose_key(
    instance: &BurnHumanInstance,
    time_key: f32,
    noise: &BurnHumanPoseNoiseSettings,
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    instance.pose_seed.hash(&mut hasher);
    time_key.to_bits().hash(&mut hasher);
    noise.noise_amp.to_bits().hash(&mut hasher);
    noise.upper_leg_amp.to_bits().hash(&mut hasher);
    noise.lower_leg_amp.to_bits().hash(&mut hasher);
    noise.upper_arm_amp.to_bits().hash(&mut hasher);
    noise.lower_arm_amp.to_bits().hash(&mut hasher);
    noise.wrist_amp.to_bits().hash(&mut hasher);
    noise.hand_amp.to_bits().hash(&mut hasher);
    noise.spine_amp.to_bits().hash(&mut hasher);
    noise.other_amp.to_bits().hash(&mut hasher);
    noise.time_scale.to_bits().hash(&mut hasher);
    hasher.finish()
}

fn build_pose_parameters(
    assets: &BurnHumanAssets,
    instance: &BurnHumanInstance,
    noise_settings: &BurnHumanPoseNoiseSettings,
    time_key: f32,
) -> Vec<f64> {
    let bone_labels = &assets.body.metadata().metadata.bone_labels;
    let bones = bone_labels.len();
    let mut pose = instance.base_pose.clone();
    if pose.len() < bones * 16 {
        pose.resize(bones * 16, 0.0);
    }

    let noise = OpenSimplex::new(instance.pose_seed);
    let t = time_key as f64;

    for (idx, label) in bone_labels.iter().enumerate() {
        if idx == 0 {
            continue;
        }
        let base = idx * 16;
        if base + 15 >= pose.len() {
            break;
        }
        let base_rot = [
            [pose[base], pose[base + 1], pose[base + 2]],
            [pose[base + 4], pose[base + 5], pose[base + 6]],
            [pose[base + 8], pose[base + 9], pose[base + 10]],
        ];
        let baseline = instance
            .pose_baseline
            .get(idx)
            .copied()
            .unwrap_or([0.0; 3]);
        let group_amp = pose_noise_scale_for_bone(label, noise_settings);
        let amp = group_amp * noise_settings.noise_amp;
        let rot_deg = if amp <= f32::EPSILON {
            baseline
        } else {
            let nx = noise.get([t * 0.35, idx as f64 * 0.17]);
            let ny = noise.get([t * 0.45 + 97.0, idx as f64 * 0.23]);
            let nz = noise.get([t * 0.55 + 197.0, idx as f64 * 0.13]);
            [
                (baseline[0] + (nx as f32) * amp).clamp(-90.0, 90.0),
                (baseline[1] + (ny as f32) * amp).clamp(-90.0, 90.0),
                (baseline[2] + (nz as f32) * amp).clamp(-90.0, 90.0),
            ]
        };
        let delta = euler_deg_to_mat3(rot_deg[0], rot_deg[1], rot_deg[2]);
        let rot = mat3_mul(delta, base_rot);
        pose[base] = rot[0][0];
        pose[base + 1] = rot[0][1];
        pose[base + 2] = rot[0][2];
        pose[base + 4] = rot[1][0];
        pose[base + 5] = rot[1][1];
        pose[base + 6] = rot[1][2];
        pose[base + 8] = rot[2][0];
        pose[base + 9] = rot[2][1];
        pose[base + 10] = rot[2][2];
        pose[base + 15] = 1.0;
    }

    pose
}

fn pose_noise_scale_for_bone(name: &str, scales: &BurnHumanPoseNoiseSettings) -> f32 {
    let lower = name.to_ascii_lowercase();
    if lower.contains("upperleg") || lower.contains("upper_leg") || lower.contains("thigh") {
        scales.upper_leg_amp
    } else if lower.contains("lowerleg")
        || lower.contains("lower_leg")
        || lower.contains("calf")
        || lower.contains("knee")
        || lower.contains("foot")
        || lower.contains("toe")
    {
        scales.lower_leg_amp
    } else if lower.contains("upper_arm")
        || lower.contains("shoulder")
        || lower.contains("clavicle")
        || lower.contains("upperarm")
    {
        scales.upper_arm_amp
    } else if lower.contains("lower_arm")
        || lower.contains("lowerarm")
        || lower.contains("elbow")
        || lower.contains("forearm")
    {
        scales.lower_arm_amp
    } else if lower.contains("wrist") {
        scales.wrist_amp
    } else if lower.contains("hand") || lower.contains("finger") || lower.contains("metacarpal") {
        scales.hand_amp
    } else if lower.contains("spine") || lower.contains("neck") || lower.contains("chest") {
        scales.spine_amp
    } else {
        scales.other_amp
    }
}

fn euler_deg_to_mat3(rx_deg: f32, ry_deg: f32, rz_deg: f32) -> [[f64; 3]; 3] {
    let (rx, ry, rz) = (
        rx_deg.to_radians() as f64,
        ry_deg.to_radians() as f64,
        rz_deg.to_radians() as f64,
    );
    let (sx, cx) = rx.sin_cos();
    let (sy, cy) = ry.sin_cos();
    let (sz, cz) = rz.sin_cos();
    [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]
}

fn mat3_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

fn apply_rotation(positions: &mut [Vec3], rotation: Quat) {
    if rotation == Quat::IDENTITY {
        return;
    }
    for position in positions.iter_mut() {
        *position = rotation * *position;
    }
}

fn bounds_from_positions(positions: &[Vec3]) -> Option<(Vec3, Vec3)> {
    if positions.is_empty() {
        return None;
    }

    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);

    for position in positions.iter() {
        min.x = min.x.min(position.x);
        min.y = min.y.min(position.y);
        min.z = min.z.min(position.z);

        max.x = max.x.max(position.x);
        max.y = max.y.max(position.y);
        max.z = max.z.max(position.z);
    }

    Some((min, max))
}

fn compute_final_scale(
    sampled_scale: Vec3,
    min: Vec3,
    max: Vec3,
    height_preserve_scale: bool,
    fit_to_sampled_box: bool,
) -> Vec3 {
    let mut final_scale = sampled_scale;
    let orig_size = max - min;
    let safe_size = Vec3::new(
        orig_size.x.max(f32::EPSILON),
        orig_size.y.max(f32::EPSILON),
        orig_size.z.max(f32::EPSILON),
    );
    if safe_size.max_element() <= f32::EPSILON {
        return final_scale;
    }

    if height_preserve_scale {
        // Preserve phenotype proportions by applying a uniform scale from target height.
        let desired_height = sampled_scale.y;
        let height_scale = desired_height / safe_size.y;
        final_scale = Vec3::splat(height_scale);
    } else if fit_to_sampled_box {
        final_scale = Vec3::new(
            sampled_scale.x / safe_size.x,
            sampled_scale.y / safe_size.y,
            sampled_scale.z / safe_size.z,
        );
    }

    final_scale
}

fn ensure_base_bounds(
    assets: &BurnHumanAssets,
    settings: &BurnHumanSettings,
    instance: &mut BurnHumanInstance,
) {
    if instance.base_bounds_ready {
        return;
    }

    let output = {
        let input = AnnyInput {
            case_name: None,
            phenotype_inputs: Some(&instance.phenotype),
            blendshape_weights: None,
            blendshape_delta: None,
            pose_parameters: Some(&instance.base_pose),
            pose_parameters_delta: None,
            root_translation_delta: None,
        };
        assets.body.forward(input).expect("burn_human forward")
    };

    let mut positions = tensor_to_vec3(&output.posed_vertices);
    apply_rotation(&mut positions, settings.mesh_rotation);
    let (min, max) = bounds_from_positions(&positions).unwrap_or((Vec3::ZERO, Vec3::ZERO));
    instance.base_min = min;
    instance.base_max = max;
    instance.base_bounds_ready = true;
}

fn tensor_to_vec3(data: &TensorData<f64>) -> Vec<Vec3> {
    match data.shape.as_slice() {
        [n, 3] => data
            .data
            .chunks_exact(3)
            .take(*n)
            .map(|c| Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32))
            .collect(),
        [b, n, 3] if *b >= 1 => data
            .data
            .chunks_exact(3)
            .take(*n)
            .map(|c| Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32))
            .collect(),
        other => panic!("expected [N,3] or [B,N,3] tensor, got shape {:?}", other),
    }
}
