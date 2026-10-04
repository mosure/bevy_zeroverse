//! AnnyBody adults with support-aware poses and procedural garment shells.
mod anatomy;
pub mod appearance;
pub(crate) mod body;
mod face;
pub mod footwear;
pub mod garments;
pub mod hair;
pub mod morphology;
mod population;
pub mod poses;
mod rig;
pub use population::populate;
use std::{
    collections::BTreeMap,
    f32::consts::{PI, TAU},
};

use bevy::{camera::primitives::Aabb, prelude::*};
use rand::Rng;
use serde::{Deserialize, Serialize};

use super::{
    geometry::Geometry,
    layout::{stream, IndoorManifest, ObjectKind, NEIGHBOR_DEPTH},
    materials::IndoorMaterials,
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
    SeatedTalking,
    StandingRelaxed,
    StandingPresenting,
    StandingConversation,
    StandingReading,
    StandingWalking,
}
impl HumanPoseKind {
    pub fn seated(self) -> bool {
        matches!(
            self,
            Self::SeatedWorking | Self::SeatedListening | Self::SeatedTalking
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanOutfit {
    Shirt,
    Knitwear,
    Blazer,
    Tee,
    Polo,
    Cardigan,
}
impl HumanOutfit {
    pub(super) fn open_front(self) -> bool {
        matches!(self, Self::Blazer | Self::Cardigan)
    }
    pub(super) fn collared(self) -> bool {
        matches!(self, Self::Shirt | Self::Blazer | Self::Polo)
    }
    pub(super) fn buttoned(self) -> bool {
        self.collared() || self == Self::Cardigan
    }
    pub(crate) fn knitted(self) -> bool {
        matches!(
            self,
            Self::Knitwear | Self::Cardigan | Self::Tee | Self::Polo
        )
    }
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
    #[serde(default)]
    pub head_yaw: f32,
    pub pose: HumanPoseKind,
    #[serde(default)]
    pub pose_program: Option<poses::PoseProgram>,
    #[serde(default)]
    pub appearance: Option<appearance::Appearance>,
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
        if let Some(color) = self.appearance.as_ref().and_then(|a| a.color(surface)) {
            return color;
        }
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
            HumanSurface::Hair | HumanSurface::Brow => [
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
            HumanSurface::Seam => [0.14, 0.14, 0.14],
            HumanSurface::Iris => [0.18, 0.12, 0.07],
            HumanSurface::Eyewear => [0.10, 0.075, 0.055],
            HumanSurface::Lens => [0.94, 0.98, 1.0],
            HumanSurface::FacialHair => [skin[0] * 0.65, skin[1] * 0.6, skin[2] * 0.6],
            HumanSurface::Sole | HumanSurface::ShoeDetail => [0.15, 0.14, 0.13],
        };
        Color::srgb(rgb[0], rgb[1], rgb[2])
    }
    fn material_key(&self, surface: HumanSurface) -> u64 {
        if self.appearance.is_some() {
            return self.seed;
        }
        let key = match surface {
            HumanSurface::Skin | HumanSurface::Lip => self.skin_tone,
            HumanSurface::Top => self.top_color,
            HumanSurface::Trousers => self.trouser_color,
            HumanSurface::Hair | HumanSurface::Brow => self.hair_color,
            HumanSurface::Shoes => self.shoe_color,
            _ => 0,
        };
        key as u64
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
    Seam,
    Iris,
    Brow,
    Eyewear,
    Lens,
    FacialHair,
    Sole,
    ShoeDetail,
}

#[derive(Default, Clone)]
pub struct HumanAssembly {
    pub parts: BTreeMap<HumanSurface, Geometry>,
    pub local_joints: Vec<Vec3>,
    pub body_measurements: morphology::BodyMeasurements,
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
/// reuse the same generated cloth maps and the shared AnnyBody reference.
pub(crate) fn cloth_finish(
    person: &IndoorHuman,
    surface: HumanSurface,
) -> (super::materials::Surface, usize) {
    let material = if surface == HumanSurface::Trousers {
        super::materials::Surface::FabricAlt
    } else {
        super::materials::Surface::Fabric
    };
    (
        material,
        super::materials::variants::slot(person.seed.wrapping_add(material as u64)),
    )
}
pub(crate) fn person_material(
    person: &IndoorHuman,
    surface: HumanSurface,
    indoor_materials: &IndoorMaterials,
    materials: &impl super::preparation::AssetStore<StandardMaterial>,
) -> StandardMaterial {
    let cloth = matches!(
        surface,
        HumanSurface::Top | HumanSurface::Trousers | HumanSurface::Shirt | HumanSurface::Seam
    );
    let mut material = if cloth {
        let mut mat = if person.outfit.knitted()
            && matches!(surface, HumanSurface::Top | HumanSurface::Seam)
        {
            materials.get(&indoor_materials.knit).unwrap().clone()
        } else if let Some(handle) = indoor_materials
            .variants
            .get(&cloth_finish(person, surface))
        {
            let mut mat = materials.get(handle).unwrap().clone();
            mat.uv_transform *= bevy::math::Affine2::from_scale(Vec2::splat(2.));
            mat
        } else {
            materials.get(&indoor_materials.cloth).unwrap().clone()
        };
        mat.perceptual_roughness = 1.;
        mat
    } else if matches!(surface, HumanSurface::Hair | HumanSurface::FacialHair) {
        materials.get(&indoor_materials.hair).unwrap().clone()
    } else if matches!(surface, HumanSurface::Skin | HumanSurface::Lip) {
        materials.get(&indoor_materials.skin).unwrap().clone()
    } else if surface == HumanSurface::Shoes {
        let leather = person
            .appearance
            .as_ref()
            .map_or(0.7, |a| a.footwear.leather);
        if leather >= 0.45 {
            materials
                .get(&indoor_materials.get(super::materials::Surface::Leather))
                .unwrap()
                .clone()
        } else {
            let mut m = materials.get(&indoor_materials.cloth).unwrap().clone();
            // Shoe UVs are metres, not Anny's normalized body atlas.
            m.uv_transform *= bevy::math::Affine2::from_scale(Vec2::splat(0.5));
            m
        }
    } else if matches!(surface, HumanSurface::Sole | HumanSurface::ShoeDetail) {
        materials
            .get(&indoor_materials.get(super::materials::Surface::Rubber))
            .unwrap()
            .clone()
    } else {
        StandardMaterial::default()
    };
    material.base_color = person.material_color(surface);
    if surface == HumanSurface::Lens {
        material.base_color = Color::srgba(0.94, 0.98, 1.0, 0.045);
        material.alpha_mode = AlphaMode::Blend;
        material.perceptual_roughness = 0.045;
        material.reflectance = 0.5;
        material.double_sided = true;
        material.cull_mode = None;
        return material;
    }
    if surface == HumanSurface::Brow {
        material.perceptual_roughness = 0.90;
        material.reflectance = 0.15;
        return material;
    }
    material.perceptual_roughness = if cloth {
        1.0
    } else {
        match surface {
            HumanSurface::Skin | HumanSurface::Lip => 0.58,
            HumanSurface::Eye => 0.20,
            HumanSurface::Iris => 0.23,
            HumanSurface::Hair => 0.60,
            HumanSurface::Shoes => 0.52,
            _ => 0.78,
        }
    };
    material.double_sided = false;
    material.cull_mode = Some(bevy::render::render_resource::Face::Back);
    if let Some(appearance) = &person.appearance {
        if cloth {
            // Preserve the spatial roughness map and only modulate it gently.
            material.perceptual_roughness = 0.72 + appearance.cloth_roughness * 0.25;
            material.uv_transform *= bevy::math::Affine2::from_scale_angle_translation(
                Vec2::splat(appearance.weave_scale),
                appearance.weave_rotation,
                Vec2::ZERO,
            );
            material.anisotropy_rotation -= appearance.weave_rotation;
        }
        if matches!(surface, HumanSurface::Skin | HumanSurface::Lip) {
            material.perceptual_roughness = appearance.skin_roughness;
            material.reflectance = 0.42;
        }
        if surface == HumanSurface::Hair {
            // Aggregate fibre reflection is directional in the groom's tangent
            // frame; curl broadens it continuously. This is still a surface PBR
            // approximation, not multiple scattering between individual fibres.
            material.perceptual_roughness =
                0.64 + 0.10 * (appearance.hair_curl / 0.06).clamp(0.0, 1.0);
        }
        if surface == HumanSurface::Eyewear {
            material.metallic = appearance.face.frame_metallic;
        }
        if surface == HumanSurface::Shoes {
            material.perceptual_roughness = 0.95 - appearance.footwear.leather * 0.15;
        }
    }
    if surface == HumanSurface::Hair {
        material.reflectance = 0.28;
        if person.hairstyle == hair::HairStyle::Afro as u8 {
            material.perceptual_roughness = 0.90;
            material.anisotropy_strength = 0.08;
        } else if person.hairstyle == hair::HairStyle::TightCurls as u8 {
            material.anisotropy_strength = 0.20;
        }
    }
    if surface == HumanSurface::Eyewear {
        material.perceptual_roughness = 0.27;
    }
    if surface == HumanSurface::FacialHair {
        material.perceptual_roughness = 0.88;
        material.reflectance = 0.22;
        material.anisotropy_strength = 0.0;
        material.uv_transform = bevy::math::Affine2::from_scale(Vec2::splat(24.0));
    }
    if matches!(
        surface,
        HumanSurface::Shoes | HumanSurface::Sole | HumanSurface::ShoeDetail
    ) {
        material.metallic = 0.;
        material.clearcoat = 0.;
        material.reflectance = 0.35;
    }
    material
}

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
                    local_joints: assembly.local_joints.clone(),
                },
                ObbTracked,
                ObbClass("person".into()),
                SemanticLabel::Person,
                Aabb::from_min_max(lo, hi),
            ))
            .id();
        for (surface, geometry) in assembly.parts {
            if geometry.indices.is_empty() {
                continue;
            }
            let handle = bank
                .entry((surface, person.material_key(surface)))
                .or_insert_with(|| {
                    let material = person_material(person, surface, indoor_materials, materials);
                    materials.add(material)
                })
                .clone();
            let mut entity = commands.spawn((
                Name::new(format!("person/{surface:?}")),
                Mesh3d(meshes.add(geometry.into_mesh())),
                MeshMaterial3d(handle),
                SemanticLabel::Person,
                IndoorHumanSurface(surface),
                OvoxelTracked,
                ChildOf(root),
            ));
            if surface == HumanSurface::Lens {
                entity.insert(bevy::light::NotShadowCaster);
            }
        }
    }
}

#[allow(clippy::type_complexity)]
pub fn update_human_poses(
    mut commands: Commands,
    people: Query<
        (Entity, &IndoorHumanInstance, &GlobalTransform),
        Or<(Changed<GlobalTransform>, Changed<IndoorHumanInstance>)>,
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
    let shoulder_width = morphology::shoulder_span(stature, build, rng.random_range(0.34..0.40));
    let program = poses::PoseProgram::sample(seed, pose);
    let joints = program.solve(stature, build, shoulder_width, pose.seated());
    let mut human = IndoorHuman {
        id,
        seed,
        position,
        yaw,
        stature,
        build,
        shoulder_width,
        head_yaw: rng.random_range(-0.40..0.40),
        pose,
        pose_program: Some(program),
        appearance: Some(appearance::Appearance::sample(seed)),
        chair,
        neighbor,
        outfit: [
            HumanOutfit::Shirt,
            HumanOutfit::Knitwear,
            HumanOutfit::Blazer,
            HumanOutfit::Tee,
            HumanOutfit::Polo,
            HumanOutfit::Cardigan,
        ][rng.random_range(0..6)],
        skin_tone: rng.random_range(0..8),
        top_color: rng.random_range(0..12),
        trouser_color: rng.random_range(0..8),
        hair_color: rng.random_range(0..6),
        hairstyle: rng.random_range(0..hair::HairStyle::ALL.len() as u8),
        shoe_color: rng.random_range(0..3),
        glasses: rng.random_bool(0.3),
        joints,
        bounds_min: Vec3::ZERO,
        bounds_max: Vec3::ZERO,
    };
    update_bounds(&mut human);
    human
}

/// Change support/pose before geometry and cameras are prepared, retaining the
/// same sampled phenotype, outfit and identity for an automatically moving actor.
pub(crate) fn standing_at(
    h: &IndoorHuman,
    position: Vec3,
    yaw: f32,
    neighbor: bool,
) -> IndoorHuman {
    let mut h = h.clone();
    h.pose = HumanPoseKind::StandingWalking;
    h.chair = None;
    h.neighbor = neighbor;
    h.position = position;
    h.yaw = yaw;
    let program = poses::PoseProgram::sample(h.seed, h.pose);
    h.joints = program.solve(h.stature, h.build, h.shoulder_width, false);
    h.pose_program = Some(program);
    update_bounds(&mut h);
    h
}

fn update_bounds(human: &mut IndoorHuman) {
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for (a, b, r) in collision_capsules(human) {
        lo = lo.min(a.min(b) - Vec3::splat(r));
        hi = hi.max(a.max(b) + Vec3::splat(r));
    }
    human.bounds_min = lo.with_y(0.0) - Vec3::new(0.055, 0.0, 0.055);
    human.bounds_max = hi + Vec3::splat(0.055);
}

fn collision_capsules(human: &IndoorHuman) -> Vec<(Vec3, Vec3, f32)> {
    // Anatomical arm roots shrink with stature; circulation and seated-person
    // reservations must still allow elbows/clothing and small pose changes.
    // Otherwise correcting short bodies unexpectedly packs occupied rooms more
    // densely and eliminates feasible multi-view camera paths.
    let planning_joints = human.pose_program.as_ref().map(|pose| {
        let span = human.shoulder_width.max(0.44 * human.build.sqrt());
        pose.solve(human.stature, human.build, span, human.pose.seated())
    });
    let p = planning_joints.as_deref().unwrap_or(&human.joints);
    let planning_shoulder = human.shoulder_width.max(0.44 * human.build.sqrt());
    let s = human.stature / 1.75;
    let leg_ease = human.appearance.as_ref().map_or(0., |a| {
        0.035 * a.garment.leg_straightness + a.garment.trouser_ease
    });
    let mut result = vec![
        (p[0], p[1], 0.15 * human.build),
        (p[1], p[2], planning_shoulder * 0.49),
        (p[3], p[4] + Vec3::Y * 0.03 * s, 0.155 * s),
    ];
    if hair::HairStyle::from_id(human.hairstyle).is_some_and(hair::HairStyle::falls) {
        let program = human
            .appearance
            .as_ref()
            .map_or_else(hair::HairProgram::default, |a| a.hair_program.clone());
        let torso_up = (p[2] - p[0]).normalize_or(Vec3::Y);
        let right = (p[9] - p[5]).normalize_or(Vec3::X);
        let back = right.cross(torso_up).normalize_or(Vec3::Z);
        let bob = matches!(
            hair::HairStyle::from_id(human.hairstyle),
            Some(hair::HairStyle::Bob | hair::HairStyle::AsymmetricBob)
        );
        let drop = if bob { 0.10 } else { program.drop_m + 0.13 };
        result.push((
            p[4] + back * 0.075 * s,
            p[3] - torso_up * drop * s + back * 0.11 * s,
            planning_shoulder * 0.58 + 0.04,
        ));
    }
    if human.hairstyle == hair::HairStyle::Afro as u8 {
        result.push((p[4], p[4] + Vec3::Y * 0.025 * s, 0.25 * s));
    }
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
            (p[base + 1], p[base + 2], 0.075 * human.build + leg_ease),
            (p[base + 2].with_y(0.05), p[base + 3], 0.083 * s),
        ]);
    }
    result
}

fn box_hit(a: Vec3, b: Vec3, r: f32, lo: Vec3, hi: Vec3) -> bool {
    super::layout::segment_hits_box(a, b, lo - Vec3::splat(r), hi + Vec3::splat(r))
}

pub fn placement_clear(scene: &IndoorManifest, person: &IndoorHuman) -> bool {
    let (lo, hi) = person.bounds();
    if !person.neighbor
        && scene.envelope.as_ref().is_some_and(|e| {
            !super::envelope::polygon::box_inside(&e.footprint, lo.xz(), hi.xz(), 0.30)
                || !e.support_clear(lo.with_y(person.position.y), hi)
                || [
                    lo.xz(),
                    hi.xz(),
                    Vec2::new(lo.x, hi.z),
                    Vec2::new(hi.x, lo.z),
                ]
                .into_iter()
                .any(|p| hi.y > scene.ceiling_height(p) - 0.15)
        })
    {
        return false;
    }
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
    if !person.neighbor
        && scene
            .program
            .as_ref()
            .is_some_and(|p| !p.portal_clear(lo, hi))
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
            || !human.head_yaw.is_finite()
            || !(0.82..=1.22).contains(&human.build)
            || !human.shoulder_width.is_finite()
            || !(0.26..=0.55).contains(&human.shoulder_width)
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
            || hair::HairStyle::from_id(human.hairstyle).is_none()
            || human.shoe_color >= 3
        {
            return Err("invalid human morphology or material palette".into());
        }
        if let Some(appearance) = &human.appearance {
            appearance.footwear.validate()?;
            appearance.hair_program.validate()?;
            if let Some(program) = &appearance.body_program {
                program.validate()?;
            }
            if appearance
                .body_gender
                .is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v))
            {
                return Err("invalid Anny gender anchor".into());
            }
            if !appearance.garment.leg_straightness.is_finite()
                || !(0.0..=1.).contains(&appearance.garment.leg_straightness)
                || !appearance.garment.hem_width.is_finite()
                || !(0.6..=1.7).contains(&appearance.garment.hem_width)
            {
                return Err("invalid trouser cut parameters".into());
            }
        }
    }
    Ok(())
}

// Geometry implementation follows below; each material retains its own mesh.

pub fn build_human(h: &IndoorHuman) -> HumanAssembly {
    body::build(h)
}
