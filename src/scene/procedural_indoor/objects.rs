mod catalog;
mod routing;
pub use catalog::{ObjectFamily, ObjectKind};
pub mod accessories;
pub mod audit;
pub mod bags;
pub mod chairs;
pub mod clocks;
pub mod computers;
pub mod containers;
pub mod desktop;
mod electrical;
pub mod equipment;
pub mod fans;
pub mod floor;
pub mod hospitality;
mod keyboard;
pub mod kitchen;
pub mod literature;
pub mod luminaires;
pub mod presentation;
pub mod public;
pub mod seating;
pub mod services;
pub mod standing_lamp;
pub mod storage;
pub mod tables;
pub mod utilities;
pub mod vessels;
pub mod wall;
use super::{
    geometry::{
        stock::{GrainAxis, TimberCuts},
        Geometry,
    },
    layout::{stream, IndoorObject},
    materials::{IndoorMaterials, Surface},
};
use crate::{
    annotation::obb::{ObbClass, ObbTracked},
    ovoxel::{OvoxelExcluded, OvoxelTracked},
    render::semantic::SemanticLabel,
};
use bevy::{camera::primitives::Aabb, light::NotShadowCaster, prelude::*};
use rand::Rng;
use std::{
    collections::BTreeMap,
    f32::consts::{FRAC_PI_2, PI, TAU},
};

/// Every assembly batches equal material/semantic parts while preserving object identity.
#[derive(Default)]
pub struct Assembly {
    pub(crate) motion_policy: Option<super::motion::IndoorMotionConfig>,
    pub(crate) joints: Vec<super::motion::Joint>,
    pub parts: BTreeMap<(Surface, String), Geometry>,
}

#[derive(Component, Debug)]
pub struct IndoorInstance {
    pub id: usize,
    pub kind: ObjectKind,
}

#[derive(Component, Debug)]
pub struct IndoorSurface(pub Surface);

/// Assembly keys can distinguish material variants while preserving one semantic class.
pub(super) fn part_label(label: &str) -> &str {
    label.split('#').next().unwrap_or(label)
}

impl Assembly {
    pub fn part(&mut self, surface: Surface, label: &str) -> &mut Geometry {
        self.parts.entry((surface, label.to_owned())).or_default()
    }
    pub fn box_part(&mut self, surface: Surface, label: &str, pos: Vec3, size: Vec3, bevel: f32) {
        self.part(surface, label)
            .cuboid(size, bevel, Transform::from_translation(pos));
    }
    fn board(
        &mut self,
        cuts: &mut TimberCuts,
        part: (Surface, &str),
        pos: Vec3,
        size: Vec3,
        bevel: f32,
        grain: GrainAxis,
    ) {
        cuts.cuboid(
            self.part(part.0, part.1),
            part.0,
            size,
            bevel,
            Transform::from_translation(pos),
            grain,
        );
    }
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for p in self.parts.values().flat_map(|g| &g.positions) {
            lo = lo.min(Vec3::from_array(*p));
            hi = hi.max(Vec3::from_array(*p));
        }
        (lo, hi)
    }
    pub fn spawn(
        self,
        parent: Entity,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &IndoorMaterials,
    ) {
        for ((surface, label), geometry) in self.parts {
            if geometry.indices.is_empty() {
                continue;
            }
            let semantic = SemanticLabel::from_label(part_label(&label))
                .expect("indoor semantic vocabulary is valid");
            let backlit =
                super::materials::window_textile::receives_transmitted_shadows(surface, &semantic);
            let sky = semantic == SemanticLabel::Sky;
            let mut entity = commands.spawn((
                Name::new(format!("{label}/{surface:?}")),
                Mesh3d(meshes.add(geometry.into_mesh())),
                MeshMaterial3d(materials.for_part(surface, &label)),
                semantic,
                IndoorSurface(surface),
                OvoxelTracked,
                ChildOf(parent),
            ));
            if backlit {
                entity.insert(bevy::light::TransmittedShadowReceiver);
            }
            if sky {
                entity.insert(super::architecture::background::SKY_TAG);
            }
            if label.split('#').any(|tag| tag == "exterior") {
                entity.insert(OvoxelExcluded);
            }
            if matches!(
                surface,
                Surface::Glass
                    | Surface::GlassInterior
                    | Surface::ContainerGlass
                    | Surface::Liquid
                    | Surface::Light
            ) {
                entity.insert(NotShadowCaster);
            }
        }
    }
}

pub fn spawn_object(
    object: &IndoorObject,
    parent: Entity,
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &IndoorMaterials,
) {
    let assembly = build_object(object);
    let (lo, hi) = assembly.bounds();
    let root = commands
        .spawn((
            Name::new(format!("{:?}_{}", object.kind, object.id)),
            crate::render::semantic::PreserveSemanticParts,
            object.transform(),
            Visibility::default(),
            ChildOf(parent),
            IndoorInstance {
                id: object.id,
                kind: object.kind,
            },
            ObbTracked,
            ObbClass(object.kind.class_name().to_owned()),
            SemanticLabel::from_label(object.kind.class_name()).expect("indoor object class"),
            Aabb::from_min_max(lo, hi),
        ))
        .id();
    assembly.spawn(root, commands, meshes, materials);
}

pub fn build_object(o: &IndoorObject) -> Assembly {
    build_object_with_motion(o, None)
}

pub(crate) fn build_object_with_motion(
    o: &IndoorObject,
    policy: Option<&super::motion::IndoorMotionConfig>,
) -> Assembly {
    build_object_at_version(o, policy, super::layout::GENERATOR_VERSION)
}

/// Rebuild an object with the archived motion policy and sheet registration.
pub fn build_manifest_object(scene: &super::layout::IndoorManifest, o: &IndoorObject) -> Assembly {
    build_object_at_version(
        o,
        scene.indoor_motion.as_ref().map(|m| &m.policy),
        scene.generator_version,
    )
}

pub(super) fn build_object_at_version(
    o: &IndoorObject,
    policy: Option<&super::motion::IndoorMotionConfig>,
    version: u32,
) -> Assembly {
    let mut a = Assembly {
        motion_policy: policy.cloned(),
        ..default()
    };
    routing::build(&mut a, o, version);
    if matches!(o.kind, ObjectKind::CoffeeMaker | ObjectKind::DrinkDispenser) {
        a.box_part(
            Surface::Rubber,
            o.kind.class_name(),
            Vec3::new(0., o.size.y * 0.10, -o.size.z * 0.5 + 0.004),
            Vec3::new(0.025, 0.020, 0.008),
            0.002,
        );
    }
    // Flat printed pages and phones use their horizontal face dimensions;
    // normalizing all six cuboid faces would crop portrait content horizontally.
    let registered_table =
        version >= tables::REGISTERED_SHEET_VERSION && o.kind.family() == ObjectFamily::Tables;
    for ((surface, label), geometry) in &mut a.parts {
        if registered_table
            && matches!(surface, Surface::Concrete | Surface::Plastic)
            && super::materials::finish_slot(*surface, label)
                .is_some_and(super::materials::worktop_finish::is_sheet_slot)
        {
            // These parts already have a cut in metres, including their edge and backing.
            continue;
        }
        if super::materials::rugs::key(label).is_some_and(|(_, binding)| !binding) {
            // Whole-rug coordinates are already normalized and outline-aligned.
            continue;
        } else if label.split('#').any(|tag| tag == "uv_local") {
            // Printed pages and manufactured keycaps carry their own normalized content
            // frame. Projecting the parent's X/Z coordinates would shear it.
            continue;
        } else if matches!(surface, Surface::PhoneScreen | Surface::PrintedPaper) {
            let mut lo = Vec2::splat(f32::INFINITY);
            let mut hi = Vec2::splat(f32::NEG_INFINITY);
            for p in &geometry.positions {
                let p = Vec2::new(p[0], p[2]);
                lo = lo.min(p);
                hi = hi.max(p);
            }
            let extent = (hi - lo).max(Vec2::splat(0.0001));
            for (uv, p) in geometry.uvs.iter_mut().zip(&geometry.positions) {
                *uv = ((Vec2::new(p[0], p[2]) - lo) / extent).to_array();
            }
        } else if matches!(
            surface,
            Surface::Screen | Surface::Whiteboard | Surface::Television
        ) {
            // Upright display faces receive the complete generated image.
            let mut lo = Vec2::splat(f32::INFINITY);
            let mut hi = Vec2::splat(f32::NEG_INFINITY);
            for uv in &geometry.uvs {
                let uv = Vec2::from_array(*uv);
                lo = lo.min(uv);
                hi = hi.max(uv);
            }
            let extent = (hi - lo).max(Vec2::splat(0.0001));
            for uv in &mut geometry.uvs {
                *uv = ((Vec2::from_array(*uv) - lo) / extent).to_array();
            }
        } else if matches!(
            surface,
            Surface::Wood
                | Surface::WoodEdge
                | Surface::Fabric
                | Surface::FabricAlt
                | Surface::Plastic
                | Surface::Metal
                | Surface::Concrete
        ) {
            let mut rng = stream(o.seed, 120 + *surface as u64);
            let phase = Vec2::new(rng.random_range(0.0..4.0), rng.random_range(0.0..4.0));
            for uv in &mut geometry.uvs {
                *uv = (Vec2::from_array(*uv) + phase).to_array();
            }
        }
    }
    // Object identities and pixels use the same taxonomy. Preserve explicit
    // contents (books/paper) on storage furniture, but replace builder-era
    // coarse furniture/prop classes before material batching.
    let class = o.kind.class_name();
    let mut labeled = BTreeMap::<(Surface, String), Geometry>::new();
    for ((surface, label), geometry) in a.parts {
        let old = part_label(&label);
        let label = if !o.kind.is_kitchen()
            && matches!(
                old,
                "other_prop" | "chair" | "sofa" | "table" | "cabinet" | "television"
            ) {
            label.replacen(old, class, 1)
        } else {
            label
        };
        // Several older part labels may converge on the same object class.
        match labeled.entry((surface, label)) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(geometry);
            }
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                entry.get_mut().append(geometry)
            }
        }
    }
    a.parts = labeled;
    // Keep a bounded finish palette, shared maps and separate semantic labels.
    // Independent object seeds avoid uniform mug/chassis/cover colors in a room.
    a.parts = a
        .parts
        .into_iter()
        .map(|((surface, label), geometry)| {
            let label = if super::materials::variants::supports(surface)
                && !label.contains("#finish")
                && !label.contains("#rug")
            {
                format!(
                    "{label}#finish{}",
                    super::materials::variants::object_slot(surface, o.seed, o.finish_seed)
                )
            } else {
                label
            };
            let label = if surface == Surface::Light && super::photometry::is_emitter(o.kind) {
                format!("{label}#source{}", o.id)
            } else {
                label
            };
            ((surface, label), geometry)
        })
        .fold(BTreeMap::new(), |mut parts, (key, geometry)| {
            // Explicit child finishes can coincide with the parent's assigned
            // finish. Preserve both meshes when their material/semantic keys
            // converge; collecting directly would silently discard one.
            match parts.entry(key) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(geometry);
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    entry.get_mut().append(geometry);
                }
            }
            parts
        });
    super::motion::construction::movable(&mut a, o);
    // Finish relabeling must keep articulation references aligned with meshes.
    for (id, joint) in a.joints.iter_mut().enumerate() {
        joint.parts = a
            .parts
            .keys()
            .filter(|(_, label)| super::motion::tag(label) == Some(id))
            .cloned()
            .collect();
    }
    a
}

fn wall_panel(a: &mut Assembly, o: &IndoorObject) {
    if o.kind == ObjectKind::Whiteboard {
        presentation::whiteboard(a, o);
    } else {
        presentation::artwork::build(a, o);
    }
}

#[cfg(test)]
mod catalog_tests;
#[cfg(test)]
mod coplanar_tests;
#[cfg(test)]
mod display_tests;
#[cfg(test)]
mod review_tests;

#[cfg(test)]
mod routing_tests;
