//! Asset-free clothed adults for room-scale occupancy, with explicit anatomy,
//! support relationships and a stable skeleton. This is not a scanned-face model.
use std::{
    collections::BTreeMap,
    f32::consts::{PI, TAU},
};

use bevy::{camera::primitives::Aabb, prelude::*};
use rand::{seq::SliceRandom, Rng};
use serde::{Deserialize, Serialize};

use super::{
    geometry::Geometry,
    layout::{stream, IndoorManifest, ObjectKind, NEIGHBOR_DEPTH},
    materials::{IndoorMaterials, Surface},
};
use crate::{
    annotation::{
        obb::{ObbClass, ObbTracked},
        pose::HumanPose,
    },
    ovoxel::OvoxelTracked,
    render::semantic::SemanticLabel,
};

pub const HUMAN_BONE_NAMES: [&str; 21] = [
    "pelvis",
    "waist",
    "chest",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hand",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hand",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_toe",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_toe",
];
pub const HUMAN_BONE_PARENTS: [i64; 21] = [
    -1, 0, 1, 2, 3, 2, 5, 6, 7, 2, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanPoseKind {
    SeatedWorking,
    SeatedListening,
    StandingRelaxed,
    StandingPresenting,
    StandingConversation,
}
impl HumanPoseKind {
    pub fn seated(self) -> bool {
        matches!(self, Self::SeatedWorking | Self::SeatedListening)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanOutfit {
    Shirt,
    Knitwear,
    Blazer,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndoorHuman {
    pub id: usize,
    pub seed: u64,
    pub position: Vec3,
    pub yaw: f32,
    /// Nominal standing stature, metres; independent of seated pose height.
    pub stature: f32,
    pub build: f32,
    pub shoulder_width: f32,
    pub pose: HumanPoseKind,
    pub chair: Option<usize>,
    pub neighbor: bool,
    pub outfit: HumanOutfit,
    pub skin_tone: u8,
    pub top_color: u8,
    pub trouser_color: u8,
    pub hair_color: u8,
    pub hairstyle: u8,
    pub shoe_color: u8,
    pub glasses: bool,
    /// Skeleton joint centres in person-local coordinates, in HUMAN_BONE_NAMES order.
    pub joints: Vec<Vec3>,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
}

impl IndoorHuman {
    pub fn transform(&self) -> Transform {
        Transform::from_translation(self.position).with_rotation(Quat::from_rotation_y(self.yaw))
    }
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let tf = self.transform();
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for x in [self.bounds_min.x, self.bounds_max.x] {
            for y in [self.bounds_min.y, self.bounds_max.y] {
                for z in [self.bounds_min.z, self.bounds_max.z] {
                    let p = tf.transform_point(Vec3::new(x, y, z));
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
            }
        }
        (lo, hi)
    }
    pub fn material_color(&self, surface: HumanSurface) -> Color {
        let skin = [
            [0.91, 0.73, 0.60],
            [0.83, 0.61, 0.45],
            [0.73, 0.48, 0.31],
            [0.59, 0.36, 0.23],
            [0.46, 0.27, 0.17],
            [0.34, 0.18, 0.12],
            [0.76, 0.56, 0.43],
            [0.62, 0.43, 0.31],
        ][self.skin_tone as usize];
        let rgb = match surface {
            HumanSurface::Skin => skin,
            HumanSurface::Lip => [skin[0] * 0.79, skin[1] * 0.63, skin[2] * 0.66],
            HumanSurface::Top => [
                [0.18, 0.25, 0.34],
                [0.45, 0.54, 0.60],
                [0.73, 0.75, 0.70],
                [0.32, 0.37, 0.31],
                [0.48, 0.26, 0.23],
                [0.22, 0.21, 0.25],
                [0.60, 0.49, 0.37],
                [0.30, 0.41, 0.48],
                [0.64, 0.57, 0.56],
                [0.15, 0.29, 0.27],
                [0.51, 0.40, 0.51],
                [0.76, 0.72, 0.60],
            ][self.top_color as usize],
            HumanSurface::Trousers => [
                [0.12, 0.15, 0.20],
                [0.24, 0.26, 0.28],
                [0.41, 0.37, 0.29],
                [0.16, 0.18, 0.17],
                [0.32, 0.34, 0.36],
                [0.49, 0.46, 0.41],
                [0.20, 0.24, 0.31],
                [0.10, 0.10, 0.12],
            ][self.trouser_color as usize],
            HumanSurface::Hair => [
                [0.075, 0.055, 0.038],
                [0.21, 0.12, 0.065],
                [0.41, 0.27, 0.12],
                [0.63, 0.52, 0.34],
                [0.34, 0.16, 0.09],
                [0.52, 0.51, 0.48],
            ][self.hair_color as usize],
            HumanSurface::Shoes => [[0.07, 0.065, 0.06], [0.23, 0.13, 0.07], [0.67, 0.65, 0.60]]
                [self.shoe_color as usize],
            HumanSurface::Shirt => [0.84, 0.85, 0.81],
            HumanSurface::Eye => [0.79, 0.77, 0.71],
            HumanSurface::Detail => [0.042, 0.037, 0.032],
        };
        Color::srgb(rgb[0], rgb[1], rgb[2])
    }
    fn material_key(&self, surface: HumanSurface) -> u8 {
        match surface {
            HumanSurface::Skin | HumanSurface::Lip => self.skin_tone,
            HumanSurface::Top => self.top_color,
            HumanSurface::Trousers => self.trouser_color,
            HumanSurface::Hair => self.hair_color,
            HumanSurface::Shoes => self.shoe_color,
            _ => 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HumanSurface {
    Skin,
    Lip,
    Top,
    Trousers,
    Hair,
    Shoes,
    Shirt,
    Eye,
    Detail,
}

#[derive(Default)]
pub struct HumanAssembly {
    pub parts: BTreeMap<HumanSurface, Geometry>,
}
impl HumanAssembly {
    fn part(&mut self, surface: HumanSurface) -> &mut Geometry {
        self.parts.entry(surface).or_default()
    }
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for p in self.parts.values().flat_map(|mesh| &mesh.positions) {
            let p = Vec3::from_array(*p);
            lo = lo.min(p);
            hi = hi.max(p);
        }
        (lo, hi)
    }
}

#[derive(Component, Debug)]
pub struct IndoorHumanInstance {
    pub id: usize,
    pub local_joints: Vec<Vec3>,
}
#[derive(Component, Debug)]
pub struct IndoorHumanSurface(pub HumanSurface);

/// Material variants are deduplicated across every person in this scene and
/// reuse the same generated cloth maps. No image/mesh files are loaded.
pub fn spawn_people(
    scene: &IndoorManifest,
    parent: Entity,
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    indoor_materials: &IndoorMaterials,
) {
    let mut bank = BTreeMap::new();
    for person in &scene.humans {
        let assembly = build_human(person);
        let (lo, hi) = assembly.bounds();
        let root = commands
            .spawn((
                Name::new(format!("person_{}", person.id)),
                person.transform(),
                Visibility::default(),
                ChildOf(parent),
                IndoorHumanInstance {
                    id: person.id,
                    local_joints: person.joints.clone(),
                },
                ObbTracked,
                ObbClass("person".into()),
                Aabb::from_min_max(lo, hi),
            ))
            .id();
        for (surface, geometry) in assembly.parts {
            let handle = bank
                .entry((surface, person.material_key(surface)))
                .or_insert_with(|| {
                    let cloth = matches!(
                        surface,
                        HumanSurface::Top | HumanSurface::Trousers | HumanSurface::Shirt
                    );
                    let mut material = if cloth {
                        materials
                            .get(&indoor_materials.get(Surface::Fabric))
                            .unwrap()
                            .clone()
                    } else {
                        StandardMaterial::default()
                    };
                    material.base_color = person.material_color(surface);
                    material.perceptual_roughness = if cloth {
                        1.0
                    } else {
                        match surface {
                            HumanSurface::Skin | HumanSurface::Lip => 0.58,
                            HumanSurface::Eye => 0.25,
                            HumanSurface::Shoes => 0.52,
                            _ => 0.78,
                        }
                    };
                    material.double_sided = false;
                    material.cull_mode = Some(bevy::render::render_resource::Face::Back);
                    materials.add(material)
                })
                .clone();
            commands.spawn((
                Name::new(format!("person/{surface:?}")),
                Mesh3d(meshes.add(geometry.into_mesh())),
                MeshMaterial3d(handle),
                SemanticLabel::Person,
                IndoorHumanSurface(surface),
                OvoxelTracked,
                ChildOf(root),
            ));
        }
    }
}

#[allow(clippy::type_complexity)]
pub fn update_human_poses(
    mut commands: Commands,
    people: Query<
        (Entity, &IndoorHumanInstance, &GlobalTransform),
        Or<(Changed<GlobalTransform>, Added<IndoorHumanInstance>)>,
    >,
) {
    for (entity, person, global) in &people {
        let affine = global.affine();
        let positions: Vec<Vec3> = person
            .local_joints
            .iter()
            .map(|p| affine.transform_point3(*p))
            .collect();
        let rotations = positions
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let parent = HUMAN_BONE_PARENTS[i];
                if parent < 0 {
                    global.rotation()
                } else {
                    Quat::from_rotation_arc(
                        Vec3::Y,
                        (*p - positions[parent as usize]).normalize_or(Vec3::Y),
                    )
                }
            })
            .collect();
        commands.entity(entity).insert(HumanPose {
            bone_positions: positions,
            bone_rotations: rotations,
        });
    }
}

fn sample_person(
    seed: u64,
    id: usize,
    position: Vec3,
    yaw: f32,
    pose: HumanPoseKind,
    chair: Option<usize>,
    neighbor: bool,
) -> IndoorHuman {
    let mut rng = stream(seed, 40);
    let stature = rng.random_range(1.50..1.95);
    let build: f32 = rng.random_range(0.82..1.22);
    let shoulder_width = rng.random_range(0.36..0.47) * build.sqrt();
    let s = stature / 1.75;
    let pelvis = if pose.seated() {
        Vec3::new(0.0, 0.585, 0.015)
    } else {
        Vec3::new(0.0, 0.94 * s, 0.0)
    };
    let lean = if pose == HumanPoseKind::SeatedWorking {
        -0.075
    } else {
        -0.018
    };
    let waist = pelvis + Vec3::new(0.0, 0.15 * s, lean * 0.3);
    let chest = pelvis + Vec3::new(0.0, 0.43 * s, lean);
    let neck = pelvis + Vec3::new(0.0, 0.55 * s, lean - 0.005);
    let head = neck + Vec3::new(0.0, 0.13 * s, -0.006);
    let mut joints = vec![pelvis, waist, chest, neck, head];
    for side in [-1.0, 1.0] {
        let shoulder = chest + Vec3::new(side * shoulder_width * 0.5, -0.005, 0.0);
        let (elbow, wrist, hand) = if pose.seated() {
            let elbow = pelvis + Vec3::new(side * 0.225, 0.15 * s, -0.10);
            let wrist = pelvis + Vec3::new(side * 0.125, 0.15 * s, -0.285);
            (elbow, wrist, wrist + Vec3::new(0.0, -0.015, -0.065))
        } else if pose == HumanPoseKind::StandingPresenting && side > 0.0 {
            let elbow = chest + Vec3::new(0.44, -0.015, -0.06) * s;
            let wrist = chest + Vec3::new(0.62, 0.14, -0.14) * s;
            (elbow, wrist, wrist + Vec3::new(0.06, 0.01, -0.025) * s)
        } else if pose == HumanPoseKind::StandingConversation {
            let elbow = shoulder + Vec3::new(side * 0.025, -0.25, -0.06) * s;
            let wrist = elbow + Vec3::new(-side * 0.10, 0.06, -0.19) * s;
            (
                elbow,
                wrist,
                wrist + Vec3::new(-side * 0.02, 0.015, -0.065) * s,
            )
        } else {
            let elbow = shoulder + Vec3::new(side * 0.025, -0.27, 0.005) * s;
            let wrist = elbow + Vec3::new(side * 0.012, -0.245, -0.025) * s;
            (elbow, wrist, wrist + Vec3::new(0.0, -0.065, -0.005) * s)
        };
        joints.extend([shoulder, elbow, wrist, hand]);
    }
    for side in [-1.0, 1.0] {
        let hip = pelvis + Vec3::new(side * 0.092 * build, 0.0, 0.0);
        let (knee, ankle) = if pose.seated() {
            (
                Vec3::new(side * 0.12, 0.46, -0.34 * s),
                Vec3::new(side * 0.135, 0.105, -0.32 * s),
            )
        } else {
            (
                Vec3::new(side * 0.115, 0.51 * s, side * 0.015),
                Vec3::new(side * 0.13, 0.105, side * 0.025),
            )
        };
        joints.extend([
            hip,
            knee,
            ankle,
            Vec3::new(ankle.x, 0.045, ankle.z - 0.17 * s),
        ]);
    }
    let mut human = IndoorHuman {
        id,
        seed,
        position,
        yaw,
        stature,
        build,
        shoulder_width,
        pose,
        chair,
        neighbor,
        outfit: [
            HumanOutfit::Shirt,
            HumanOutfit::Knitwear,
            HumanOutfit::Blazer,
        ][rng.random_range(0..3)],
        skin_tone: rng.random_range(0..8),
        top_color: rng.random_range(0..12),
        trouser_color: rng.random_range(0..8),
        hair_color: rng.random_range(0..6),
        hairstyle: rng.random_range(0..6),
        shoe_color: rng.random_range(0..3),
        glasses: rng.random_bool(0.3),
        joints,
        bounds_min: Vec3::ZERO,
        bounds_max: Vec3::ZERO,
    };
    // Bounds enclose anatomy, clothing and fingers; ground contact is explicit.
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for (a, b, r) in collision_capsules(&human) {
        lo = lo.min(a.min(b) - Vec3::splat(r));
        hi = hi.max(a.max(b) + Vec3::splat(r));
    }
    human.bounds_min = lo.with_y(0.0) - Vec3::new(0.015, 0.0, 0.015);
    human.bounds_max = hi + Vec3::splat(0.015);
    human
}

pub fn populate(scene: &mut IndoorManifest, density: f32) {
    if density == 0.0 {
        return;
    }
    let mut rng = stream(scene.seed, 39);
    let mut chairs: Vec<_> = scene
        .objects
        .iter()
        .filter(|o| o.kind == ObjectKind::Chair)
        .map(|o| o.id)
        .collect();
    chairs.shuffle(&mut rng);
    for chair_id in chairs {
        if scene.humans.len() >= 16 || !rng.random_bool(density as f64) {
            continue;
        }
        let chair = &scene.objects[chair_id];
        let pose = if rng.random_bool(0.6) {
            HumanPoseKind::SeatedWorking
        } else {
            HumanPoseKind::SeatedListening
        };
        let person = sample_person(
            rng.random(),
            scene.objects.len() + scene.humans.len(),
            chair.position,
            chair.yaw,
            pose,
            Some(chair_id),
            chair.neighbor,
        );
        if placement_clear(scene, &person) {
            scene.humans.push(person);
        } else {
            scene.rejected_human_placements += 1;
        }
    }
    let standing = (density * 2.0).floor() as usize
        + usize::from(rng.random_bool((density * 2.0).fract() as f64));
    for index in 0..standing {
        for attempt in 0..96 {
            let pose = [
                HumanPoseKind::StandingRelaxed,
                HumanPoseKind::StandingPresenting,
                HumanPoseKind::StandingConversation,
            ][rng.random_range(0..3)];
            let p = if index == 0 && attempt < 4 && pose == HumanPoseKind::StandingPresenting {
                Vec3::new(scene.room_size.x * 0.18, 0.0, -scene.room_size.z * 0.34)
            } else {
                Vec3::new(
                    rng.random_range(-0.40..0.40) * scene.room_size.x,
                    0.0,
                    rng.random_range(-0.39..0.39) * scene.room_size.z,
                )
            };
            let yaw = if pose == HumanPoseKind::StandingPresenting {
                PI
            } else {
                rng.random_range(-PI..PI)
            };
            let person = sample_person(
                rng.random(),
                scene.objects.len() + scene.humans.len(),
                p,
                yaw,
                pose,
                None,
                false,
            );
            if placement_clear(scene, &person) {
                scene.humans.push(person);
                break;
            } else {
                scene.rejected_human_placements += 1;
            }
        }
    }
}

fn collision_capsules(human: &IndoorHuman) -> Vec<(Vec3, Vec3, f32)> {
    let p = &human.joints;
    let s = human.stature / 1.75;
    let mut result = vec![
        (p[0], p[1], 0.15 * human.build),
        (p[1], p[2], human.shoulder_width * 0.49),
        (p[3], p[4] + Vec3::Y * 0.03 * s, 0.155 * s),
    ];
    for base in [5, 9] {
        result.extend([
            (p[base], p[base + 1], 0.095 * human.build),
            (p[base + 1], p[base + 2], 0.065 * human.build),
            (
                p[base + 3],
                p[base + 3] + (p[base + 3] - p[base + 2]).normalize() * 0.075 * s,
                0.055 * s,
            ),
        ]);
    }
    for base in [13, 17] {
        result.extend([
            (p[base], p[base + 1], 0.094 * human.build),
            (p[base + 1], p[base + 2], 0.075 * human.build),
            (p[base + 2].with_y(0.05), p[base + 3], 0.07 * s),
        ]);
    }
    result
}

fn box_hit(a: Vec3, b: Vec3, r: f32, lo: Vec3, hi: Vec3) -> bool {
    super::layout::segment_hits_box(a, b, lo - Vec3::splat(r), hi + Vec3::splat(r))
}

pub fn placement_clear(scene: &IndoorManifest, person: &IndoorHuman) -> bool {
    let (lo, hi) = person.bounds();
    let half = scene.room_size * 0.5;
    let zmin = if person.neighbor {
        half.z + 0.30
    } else {
        -half.z + 0.30
    };
    let zmax = if person.neighbor {
        half.z + NEIGHBOR_DEPTH - 0.30
    } else {
        half.z - 0.30
    };
    if lo.x < -half.x + 0.30
        || hi.x > half.x - 0.30
        || lo.z < zmin
        || hi.z > zmax
        || hi.y > scene.room_size.y - 0.40
    {
        return false;
    }
    if hi.x > scene.door_x - 0.70
        && lo.x < scene.door_x + 0.70
        && (if person.neighbor {
            lo.z < half.z + 1.45
        } else {
            hi.z > half.z - 1.45
        })
    {
        return false;
    }
    if scene
        .humans
        .iter()
        .filter(|other| other.id != person.id)
        .any(|other| {
            let (a, b) = other.bounds();
            lo.cmplt(b).all() && hi.cmpgt(a).all()
        })
    {
        return false;
    }
    let tf = person.transform();
    for (a, b, r) in collision_capsules(person) {
        let a = tf.transform_point(a);
        let b = tf.transform_point(b);
        if scene
            .columns()
            .iter()
            .any(|(lo, hi)| box_hit(a, b, r, *lo, *hi))
        {
            return false;
        }
        for object in &scene.objects {
            if object.neighbor != person.neighbor
                || person.chair == Some(object.id)
                || object.kind == ObjectKind::Rug
            {
                continue;
            }
            let inverse = object.transform().compute_affine().inverse();
            let aa = inverse.transform_point3(a);
            let bb = inverse.transform_point3(b);
            let size = object.size;
            if matches!(
                object.kind,
                ObjectKind::Table | ObjectKind::Desk | ObjectKind::CoffeeTable
            ) {
                // Anatomy can occupy the actual free volume under a worktop.
                if box_hit(
                    aa,
                    bb,
                    r,
                    Vec3::new(-size.x * 0.5, size.y - 0.145, -size.z * 0.5),
                    Vec3::new(size.x * 0.5, size.y + 0.01, size.z * 0.5),
                ) {
                    return false;
                }
                for x in [-1.0, 1.0] {
                    for z in [-1.0, 1.0] {
                        let c = Vec3::new(
                            x * (size.x * 0.5 - 0.13),
                            size.y * 0.5,
                            z * (size.z * 0.5 - 0.13),
                        );
                        let h = Vec3::new(0.075, size.y * 0.5, 0.075);
                        if box_hit(aa, bb, r, c - h, c + h) {
                            return false;
                        }
                    }
                }
            } else if box_hit(
                aa,
                bb,
                r,
                Vec3::new(-size.x * 0.5, 0.0, -size.z * 0.5),
                Vec3::new(size.x * 0.5, size.y, size.z * 0.5),
            ) {
                return false;
            }
        }
    }
    true
}

pub fn validate(scene: &IndoorManifest) -> Result<(), String> {
    if !scene.human_density.is_finite() || !(0.0..=1.0).contains(&scene.human_density) {
        return Err("invalid human density".into());
    }
    for (index, human) in scene.humans.iter().enumerate() {
        if human.id != scene.objects.len() + index
            || !human.position.is_finite()
            || !human.yaw.is_finite()
            || !(0.82..=1.22).contains(&human.build)
            || !human.shoulder_width.is_finite()
            || !(0.30..=0.55).contains(&human.shoulder_width)
            || !human.bounds_min.is_finite()
            || !human.bounds_max.is_finite()
            || !human.bounds_min.cmplt(human.bounds_max).all()
            || human.joints.len() != HUMAN_BONE_NAMES.len()
            || human.joints.iter().any(|joint| !joint.is_finite())
            || !placement_clear(scene, human)
        {
            return Err(format!("invalid human {} placement or skeleton", human.id));
        }
        if human.pose.seated() {
            let chair = human
                .chair
                .and_then(|id| scene.objects.get(id))
                .ok_or("seated human missing chair")?;
            if chair.kind != ObjectKind::Chair
                || chair.neighbor != human.neighbor
                || chair.position.distance(human.position) > 0.001
                || (chair.yaw - human.yaw).abs() > 0.001
            {
                return Err("invalid seated support relationship".into());
            }
        } else if human.chair.is_some() {
            return Err("standing human has a chair support".into());
        }
        if !(1.50..=1.95).contains(&human.stature)
            || human.skin_tone >= 8
            || human.top_color >= 12
            || human.trouser_color >= 8
            || human.hair_color >= 6
            || human.hairstyle >= 6
            || human.shoe_color >= 3
        {
            return Err("invalid human morphology or material palette".into());
        }
    }
    Ok(())
}

// Geometry implementation follows below; each material retains its own mesh.

fn oval(g: &mut Geometry, radii: Vec3, position: Vec3, rotation: Quat) {
    g.mesh(
        Sphere::new(1.0).mesh().uv(16, 10),
        Transform::from_translation(position)
            .with_rotation(rotation)
            .with_scale(radii),
    );
}

/// Elliptical cross-sections with smooth normals, metre UVs and closed end caps.
/// Ring centres may bend to create chest lean and shaped rather than cylindrical limbs.
fn loft(g: &mut Geometry, rings: &[(Vec3, f32, f32)], segments: usize, tf: Transform, fold: f32) {
    let start = g.positions.len() as u32;
    let mut distances = vec![0.0; rings.len()];
    for i in 1..rings.len() {
        distances[i] = distances[i - 1] + rings[i].0.distance(rings[i - 1].0);
    }
    for (j, (centre, rx, rz)) in rings.iter().enumerate() {
        let before = rings[j.saturating_sub(1)];
        let after = rings[(j + 1).min(rings.len() - 1)];
        let dy = (after.0.y - before.0.y).abs().max(0.0001);
        for i in 0..=segments {
            let angle = i as f32 / segments as f32 * TAU;
            let wrinkle = 1.0 + fold * (angle * 5.0 + j as f32 * 1.8).sin();
            let p =
                *centre + Vec3::new(angle.sin() * rx * wrinkle, 0.0, angle.cos() * rz * wrinkle);
            let tangent = Vec3::new(angle.cos() * rx, 0.0, -angle.sin() * rz);
            let along = (after.0 - before.0) / dy
                + Vec3::new(
                    angle.sin() * (after.1 - before.1) / dy,
                    0.0,
                    angle.cos() * (after.2 - before.2) / dy,
                );
            let n = tangent.cross(along).normalize_or(Vec3::Y);
            g.positions.push(tf.transform_point(p).to_array());
            g.normals.push((tf.rotation * n).to_array());
            g.uvs.push([angle * (rx + rz) * 0.5, distances[j]]);
        }
    }
    for j in 0..rings.len() - 1 {
        for i in 0..segments {
            let a = start + (j * (segments + 1) + i) as u32;
            let b = a + (segments + 1) as u32;
            g.indices.extend([a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    for (index, up) in [(0, false), (rings.len() - 1, true)] {
        let (centre, rx, rz) = rings[index];
        let c = g.positions.len() as u32;
        let n = if up { Vec3::Y } else { -Vec3::Y };
        g.positions.push(tf.transform_point(centre).to_array());
        g.normals.push((tf.rotation * n).to_array());
        g.uvs.push([0.0, 0.0]);
        for i in 0..=segments {
            let angle = i as f32 / segments as f32 * TAU;
            let wrinkle = 1.0 + fold * (angle * 5.0 + index as f32 * 1.8).sin();
            let p = Vec3::new(angle.sin() * rx * wrinkle, 0.0, angle.cos() * rz * wrinkle);
            g.positions.push(tf.transform_point(centre + p).to_array());
            g.normals.push((tf.rotation * n).to_array());
            g.uvs.push([p.x, p.z]);
        }
        for i in 0..segments as u32 {
            if up {
                g.indices.extend([c, c + i + 1, c + i + 2]);
            } else {
                g.indices.extend([c, c + i + 2, c + i + 1]);
            }
        }
    }
}

fn limb(g: &mut Geometry, a: Vec3, b: Vec3, radii: &[f32], depth_ratio: f32, fold: f32) {
    let d = b - a;
    let length = d.length();
    let rings: Vec<_> = radii
        .iter()
        .enumerate()
        .map(|(i, r)| {
            (
                Vec3::Y * (i as f32 / (radii.len() - 1) as f32 - 0.5) * length,
                *r,
                *r * depth_ratio,
            )
        })
        .collect();
    loft(
        g,
        &rings,
        12,
        Transform::from_translation((a + b) * 0.5)
            .with_rotation(Quat::from_rotation_arc(Vec3::Y, d.normalize())),
        fold,
    );
}

fn hand(a: &mut HumanAssembly, wrist: Vec3, centre: Vec3, side: f32, s: f32) {
    let rotation = Quat::from_rotation_arc(Vec3::Y, (centre - wrist).normalize());
    let tf = Transform::from_translation(centre).with_rotation(rotation);
    oval(
        a.part(HumanSurface::Skin),
        Vec3::new(0.036, 0.046, 0.017) * s,
        centre,
        rotation,
    );
    for index in 0..4 {
        let x = (index as f32 - 1.5) * 0.017 * s;
        let length = [0.061, 0.075, 0.069, 0.054][index] * s;
        let p0 = tf.transform_point(Vec3::new(x, 0.025 * s, 0.0));
        let p1 = tf.transform_point(Vec3::new(x, length * 0.65, -0.007 * s));
        let p2 = tf.transform_point(Vec3::new(x, length, -0.016 * s));
        limb(
            a.part(HumanSurface::Skin),
            p0,
            p1,
            &[0.007 * s, 0.008 * s, 0.0068 * s],
            0.83,
            0.0,
        );
        limb(
            a.part(HumanSurface::Skin),
            p1,
            p2,
            &[0.0068 * s, 0.006 * s, 0.0045 * s],
            0.83,
            0.0,
        );
        oval(
            a.part(HumanSurface::Skin),
            Vec3::splat(0.0048 * s),
            p2,
            rotation,
        );
    }
    let p0 = tf.transform_point(Vec3::new(-side * 0.029 * s, -0.01 * s, 0.0));
    let p1 = tf.transform_point(Vec3::new(-side * 0.047 * s, 0.017 * s, -0.006 * s));
    let p2 = tf.transform_point(Vec3::new(-side * 0.047 * s, 0.040 * s, -0.017 * s));
    limb(
        a.part(HumanSurface::Skin),
        p0,
        p1,
        &[0.013 * s, 0.011 * s, 0.009 * s],
        0.85,
        0.0,
    );
    limb(
        a.part(HumanSurface::Skin),
        p1,
        p2,
        &[0.009 * s, 0.008 * s, 0.005 * s],
        0.85,
        0.0,
    );
}

fn hair_cap(g: &mut Geometry, head: Vec3, s: f32, style: u8) {
    if style == 5 {
        return;
    }
    let around = 24;
    let rows = 10;
    let start = g.positions.len() as u32;
    for j in 0..=rows {
        for i in 0..=around {
            let theta = i as f32 / around as f32 * TAU;
            let front = (-theta.cos()).max(0.0);
            let stop = if style == 2 {
                2.0 - front * 0.99
            } else {
                1.72 - front * 0.70
            };
            // A tiny nonzero polar ring avoids degenerate triangles and exposes no scalp.
            let phi = 0.015 + (stop - 0.015) * j as f32 / rows as f32;
            let curl = if style == 4 {
                1.0 + 0.06 * (theta * 9.0 + phi * 7.0).sin()
            } else {
                1.0
            };
            let radii = Vec3::new(0.090, 0.119, 0.101) * s;
            let dir = Vec3::new(phi.sin() * theta.sin(), phi.cos(), phi.sin() * theta.cos());
            let p = head + dir * radii * curl + Vec3::Y * 0.006 * s;
            let n = (dir / radii).normalize();
            g.positions.push(p.to_array());
            g.normals.push(n.to_array());
            g.uvs.push([theta * radii.x, phi * radii.y]);
        }
    }
    for j in 0..rows {
        for i in 0..around {
            let p = start + (j * (around + 1) + i) as u32;
            let q = p + (around + 1) as u32;
            g.indices.extend([p, q, p + 1, p + 1, q, q + 1]);
        }
    }
    // Crown and several directional locks prevent a perfectly featureless helmet.
    oval(
        g,
        Vec3::new(0.029, 0.012, 0.030) * s,
        head + Vec3::Y * 0.121 * s,
        Quat::IDENTITY,
    );
    if style == 3 {
        oval(
            g,
            Vec3::new(0.058, 0.061, 0.055) * s,
            head + Vec3::new(0.0, 0.045, 0.097) * s,
            Quat::IDENTITY,
        );
    }
    if style == 1 {
        oval(
            g,
            Vec3::new(0.064, 0.021, 0.044) * s,
            head + Vec3::new(-0.020, 0.10, -0.018) * s,
            Quat::from_rotation_z(-0.15),
        );
    }
    if style == 2 {
        for side in [-1.0, 1.0] {
            oval(
                g,
                Vec3::new(0.027, 0.087, 0.073) * s,
                head + Vec3::new(side * 0.075, -0.045, 0.022) * s,
                Quat::IDENTITY,
            );
        }
    }
}

pub fn build_human(h: &IndoorHuman) -> HumanAssembly {
    let mut a = HumanAssembly::default();
    let p = &h.joints;
    let s = h.stature / 1.75;
    let width = h.shoulder_width;
    let build = h.build;
    let pelvis = p[0];
    let waist = p[1];
    let chest = p[2];
    let neck = p[3];
    let head = p[4];
    // Trousers and the torso have anatomical waist/chest/shoulder silhouettes.
    loft(
        a.part(HumanSurface::Trousers),
        &[
            (pelvis - Vec3::Y * 0.075, 0.145 * build, 0.102 * build),
            (pelvis + Vec3::Y * 0.04, 0.16 * build, 0.105 * build),
            (waist - Vec3::Y * 0.025, 0.145 * build, 0.098 * build),
        ],
        20,
        Transform::IDENTITY,
        0.015,
    );
    loft(
        a.part(HumanSurface::Top),
        &[
            (waist - Vec3::Y * 0.02, width * 0.37, 0.107 * build),
            (waist + Vec3::Y * 0.055, width * 0.39, 0.112 * build),
            (chest - Vec3::Y * 0.09, width * 0.47, 0.133 * build),
            (chest + Vec3::Y * 0.012, width * 0.50, 0.124 * build),
            (chest + Vec3::Y * 0.047, width * 0.43, 0.107 * build),
            (chest + Vec3::Y * 0.070, width * 0.27, 0.083 * build),
        ],
        24,
        Transform::IDENTITY,
        0.013,
    );
    limb(
        a.part(HumanSurface::Skin),
        chest + Vec3::Y * 0.035,
        head - Vec3::Y * 0.060 * s,
        &[0.054 * s, 0.041 * s, 0.037 * s],
        0.95,
        0.0,
    );
    // Tailored collar/neck opening and front placket, with sleeve cuffs and buttons.
    let front = chest.z - 0.129 * build;
    if h.outfit == HumanOutfit::Blazer {
        a.part(HumanSurface::Shirt).cuboid(
            Vec3::new(0.112, 0.26 * s, 0.014),
            0.006,
            Transform::from_xyz(0.0, chest.y - 0.06, front - 0.006),
        );
        for side in [-1.0, 1.0] {
            a.part(HumanSurface::Top).cuboid(
                Vec3::new(0.060, 0.235 * s, 0.015),
                0.003,
                Transform::from_xyz(side * 0.069, chest.y - 0.043, front - 0.014)
                    .with_rotation(Quat::from_rotation_z(side * 0.24)),
            );
        }
        a.part(HumanSurface::Detail).cuboid(
            Vec3::new(0.034, 0.18 * s, 0.010),
            0.005,
            Transform::from_xyz(0.0, chest.y - 0.035, front - 0.023),
        );
    }
    if h.outfit != HumanOutfit::Knitwear {
        let collar = if h.outfit == HumanOutfit::Blazer {
            HumanSurface::Shirt
        } else {
            HumanSurface::Top
        };
        for side in [-1.0, 1.0] {
            a.part(collar).cuboid(
                Vec3::new(0.053, 0.065, 0.018) * s,
                0.007,
                Transform::from_translation(neck + Vec3::new(side * 0.043, -0.041, -0.042) * s)
                    .with_rotation(Quat::from_rotation_z(side * 0.30)),
            );
        }
        for row in 0..5 {
            oval(
                a.part(HumanSurface::Detail),
                Vec3::new(0.004, 0.004, 0.0025) * s,
                Vec3::new(0.0, waist.y + 0.025 + row as f32 * 0.055 * s, front - 0.006),
                Quat::IDENTITY,
            );
        }
    }
    for side in [-1.0, 1.0] {
        a.part(HumanSurface::Top).cuboid(
            Vec3::new(0.067, 0.010, 0.010) * s,
            0.002,
            Transform::from_xyz(side * width * 0.24, waist.y + 0.05, waist.z - 0.109 * build),
        );
    }
    for (base, side) in [(5, -1.0), (9, 1.0)] {
        let shoulder = p[base];
        let elbow = p[base + 1];
        let wrist = p[base + 2];
        // Sleeve heads overlap the sloping shoulder seam, avoiding separate
        // ball-joint silhouettes while retaining a shaped cloth sleeve.
        limb(
            a.part(HumanSurface::Top),
            shoulder + Vec3::new(-side * width * 0.10, 0.025 * s, 0.0),
            elbow,
            &[
                0.066 * build,
                0.079 * build,
                0.073 * build,
                0.059 * build,
                0.056 * build,
            ],
            1.04,
            0.030,
        );
        limb(
            a.part(HumanSurface::Top),
            elbow,
            wrist.lerp(elbow, 0.07),
            &[0.057 * build, 0.060 * build, 0.044 * build, 0.036 * build],
            0.94,
            0.027,
        );
        limb(
            a.part(HumanSurface::Skin),
            wrist.lerp(elbow, 0.09),
            p[base + 3],
            &[0.030 * build, 0.027 * build, 0.025 * build],
            0.80,
            0.0,
        );
        hand(&mut a, wrist, p[base + 3], side, s);
    }
    for base in [13, 17] {
        let hip = p[base];
        let knee = p[base + 1];
        let ankle = p[base + 2];
        limb(
            a.part(HumanSurface::Trousers),
            hip,
            knee,
            &[0.096 * build, 0.102 * build, 0.087 * build, 0.074 * build],
            1.0,
            0.024,
        );
        oval(
            a.part(HumanSurface::Trousers),
            Vec3::new(0.075 * build, 0.076 * build, 0.080 * build),
            knee,
            Quat::IDENTITY,
        );
        limb(
            a.part(HumanSurface::Trousers),
            knee,
            ankle,
            &[0.075 * build, 0.080 * build, 0.065 * build, 0.052 * build],
            0.93,
            0.028,
        );
        let centre = Vec3::new(ankle.x, 0.060, ankle.z - 0.045 * s);
        oval(
            a.part(HumanSurface::Shoes),
            Vec3::new(0.059 * build, 0.050, 0.130 * s),
            centre,
            Quat::IDENTITY,
        );
        a.part(HumanSurface::Detail).cuboid(
            Vec3::new(0.112 * build, 0.014, 0.242 * s),
            0.006,
            Transform::from_xyz(centre.x, 0.007, centre.z),
        );
        for row in 0..3 {
            a.part(HumanSurface::Detail).rod(
                centre + Vec3::new(-0.025, 0.044, -0.015 - row as f32 * 0.014),
                centre + Vec3::new(0.025, 0.044, -0.021 - row as f32 * 0.014),
                0.0018,
            );
        }
    }
    // A tapering jaw and cheek volumes, not a spherical head.
    loft(
        a.part(HumanSurface::Skin),
        &[
            (
                head + Vec3::new(0.0, -0.110, -0.006) * s,
                0.040 * s,
                0.053 * s,
            ),
            (
                head + Vec3::new(0.0, -0.079, -0.003) * s,
                0.064 * s,
                0.073 * s,
            ),
            (head + Vec3::new(0.0, -0.020, 0.0) * s, 0.082 * s, 0.088 * s),
            (
                head + Vec3::new(0.0, 0.040, 0.006) * s,
                0.082 * s,
                0.087 * s,
            ),
            (
                head + Vec3::new(0.0, 0.086, 0.011) * s,
                0.061 * s,
                0.065 * s,
            ),
            (
                head + Vec3::new(0.0, 0.110, 0.014) * s,
                0.025 * s,
                0.034 * s,
            ),
        ],
        24,
        Transform::IDENTITY,
        0.0,
    );
    oval(
        a.part(HumanSurface::Skin),
        Vec3::new(0.017, 0.030, 0.018) * s,
        head + Vec3::new(0.0, -0.003, -0.090) * s,
        Quat::from_rotation_x(-0.12),
    );
    oval(
        a.part(HumanSurface::Skin),
        Vec3::new(0.018, 0.010, 0.018) * s,
        head + Vec3::new(0.0, -0.019, -0.101) * s,
        Quat::IDENTITY,
    );
    oval(
        a.part(HumanSurface::Lip),
        Vec3::new(0.023, 0.0045, 0.005) * s,
        head + Vec3::new(0.0, -0.048, -0.080) * s,
        Quat::IDENTITY,
    );
    oval(
        a.part(HumanSurface::Detail),
        Vec3::new(0.018, 0.0016, 0.0055) * s,
        head + Vec3::new(0.0, -0.049, -0.081) * s,
        Quat::IDENTITY,
    );
    for side in [-1.0, 1.0] {
        oval(
            a.part(HumanSurface::Skin),
            Vec3::new(0.013, 0.026, 0.018) * s,
            head + Vec3::new(side * 0.081, -0.012, 0.004) * s,
            Quat::IDENTITY,
        );
        oval(
            a.part(HumanSurface::Lip),
            Vec3::new(0.006, 0.016, 0.011) * s,
            head + Vec3::new(side * 0.089, -0.012, -0.001) * s,
            Quat::IDENTITY,
        );
        let eye = head + Vec3::new(side * 0.032, 0.018, -0.084) * s;
        oval(
            a.part(HumanSurface::Eye),
            Vec3::new(0.014, 0.0055, 0.004) * s,
            eye,
            Quat::IDENTITY,
        );
        oval(
            a.part(HumanSurface::Hair),
            Vec3::new(0.0045, 0.0047, 0.003) * s,
            eye - Vec3::Z * 0.003 * s,
            Quat::IDENTITY,
        );
        oval(
            a.part(HumanSurface::Detail),
            Vec3::new(0.002, 0.003, 0.0015) * s,
            eye - Vec3::Z * 0.006 * s,
            Quat::IDENTITY,
        );
        a.part(HumanSurface::Hair).rod(
            eye + Vec3::new(-0.014, 0.012, 0.001) * s,
            eye + Vec3::new(0.014, 0.013, 0.001) * s,
            0.0025 * s,
        );
        if h.glasses {
            for y in [-0.012, 0.012] {
                a.part(HumanSurface::Detail).rod(
                    eye + Vec3::new(-0.020, y, -0.009) * s,
                    eye + Vec3::new(0.020, y, -0.009) * s,
                    0.0015 * s,
                );
            }
            for x in [-0.020, 0.020] {
                a.part(HumanSurface::Detail).rod(
                    eye + Vec3::new(x, -0.012, -0.009) * s,
                    eye + Vec3::new(x, 0.012, -0.009) * s,
                    0.0015 * s,
                );
            }
            a.part(HumanSurface::Detail).rod(
                eye + Vec3::new(side * 0.020, 0.004, -0.009) * s,
                head + Vec3::new(side * 0.085, 0.022, 0.027) * s,
                0.0015 * s,
            );
        }
    }
    if h.glasses {
        a.part(HumanSurface::Detail).rod(
            head + Vec3::new(-0.010, 0.021, -0.096) * s,
            head + Vec3::new(0.010, 0.021, -0.096) * s,
            0.0015 * s,
        );
    }
    hair_cap(a.part(HumanSurface::Hair), head, s, h.hairstyle);
    a
}
