pub mod chairs;
pub mod computers;
pub mod tables;
mod utilities;
use super::{
    geometry::Geometry,
    layout::{stream, IndoorObject, ObjectKind},
    materials::{IndoorMaterials, Surface},
};
use crate::{
    annotation::obb::{ObbClass, ObbTracked},
    ovoxel::OvoxelTracked,
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
            let mut entity = commands.spawn((
                Name::new(format!("{label}/{surface:?}")),
                Mesh3d(meshes.add(geometry.into_mesh())),
                MeshMaterial3d(materials.for_part(surface, &label)),
                SemanticLabel::from_label(part_label(&label))
                    .expect("indoor semantic vocabulary is valid"),
                IndoorSurface(surface),
                OvoxelTracked,
                ChildOf(parent),
            ));
            if matches!(
                surface,
                Surface::Glass | Surface::GlassInterior | Surface::Light
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
            object.transform(),
            Visibility::default(),
            ChildOf(parent),
            IndoorInstance {
                id: object.id,
                kind: object.kind,
            },
            ObbTracked,
            ObbClass(object.kind.class_name().to_owned()),
            Aabb::from_min_max(lo, hi),
        ))
        .id();
    assembly.spawn(root, commands, meshes, materials);
}

pub fn build_object(o: &IndoorObject) -> Assembly {
    let mut a = Assembly::default();
    match o.kind {
        ObjectKind::Printer | ObjectKind::StorageBox | ObjectKind::CoatRack | ObjectKind::Bag => {
            utilities::build(&mut a, o)
        }
        ObjectKind::Table | ObjectKind::Desk | ObjectKind::CoffeeTable => tables::build(&mut a, o),
        ObjectKind::Chair => chairs::build(&mut a, o),
        ObjectKind::Sofa => sofa(&mut a, o),
        ObjectKind::Cabinet | ObjectKind::Bookcase => cabinet(&mut a, o),
        ObjectKind::Keyboard
        | ObjectKind::Mouse
        | ObjectKind::WaterBottle
        | ObjectKind::PenHolder => super::clutter::build(&mut a, o),
        ObjectKind::Plant => super::plants::build(&mut a, o),
        ObjectKind::TrashCan => bin(&mut a, o),
        ObjectKind::Display | ObjectKind::Monitor => display(&mut a, o),
        ObjectKind::Laptop => computers::laptop(&mut a, o),
        ObjectKind::Whiteboard | ObjectKind::WallArt => wall_panel(&mut a, o),
        ObjectKind::Mug => mug(&mut a, o),
        ObjectKind::Notebook | ObjectKind::Books => books(&mut a, o),
        ObjectKind::Clock => clock(&mut a, o),
        ObjectKind::FloorLamp => lamp(&mut a, o),
        ObjectKind::Rug => a.box_part(
            Surface::FabricAlt,
            "floormat",
            Vec3::Y * o.size.y * 0.5,
            o.size,
            0.002,
        ),
    }
    // Displays receive one complete generated dashboard per screen face,
    // independent of the panel's physical dimensions.
    for ((surface, _), geometry) in &mut a.parts {
        if *surface == Surface::Screen {
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
    a
}

fn sofa(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    for x in [-s.x * 0.5 + 0.14, s.x * 0.5 - 0.14] {
        for z in [-s.z * 0.5 + 0.14, s.z * 0.5 - 0.14] {
            a.part(Surface::Rubber, "sofa").cylinder(
                0.032,
                0.026,
                Transform::from_xyz(x, 0.013, z),
            );
            a.part(Surface::WoodEdge, "sofa").rod(
                Vec3::new(x, 0.025, z),
                Vec3::new(x, 0.18, z),
                0.032,
            );
        }
    }
    a.box_part(
        Surface::Fabric,
        "sofa",
        Vec3::new(0.0, 0.25, 0.0),
        Vec3::new(s.x, 0.22, s.z),
        0.06,
    );
    for x in [-s.x * 0.5 + 0.10, s.x * 0.5 - 0.10] {
        a.box_part(
            Surface::Fabric,
            "sofa",
            Vec3::new(x, 0.50, 0.0),
            Vec3::new(0.20, 0.44, s.z),
            0.06,
        );
    }
    let n = 3;
    let w = (s.x - 0.44) / n as f32;
    for i in 0..n {
        let x = (i as f32 - 1.0) * (w + 0.01);
        a.box_part(
            Surface::FabricAlt,
            "sofa",
            Vec3::new(x, 0.395, -0.055),
            Vec3::new(w, 0.12, s.z * 0.75),
            0.044,
        );
        a.box_part(
            Surface::Fabric,
            "sofa",
            Vec3::new(x, 0.63, s.z * 0.5 - 0.13),
            Vec3::new(w, 0.50, 0.22),
            0.065,
        );
    }
    for side in [-1.0, 1.0] {
        a.part(Surface::FabricAlt, "pillow").cuboid(
            Vec3::new(0.35, 0.34, 0.12),
            0.05,
            Transform::from_xyz(side * s.x * 0.29, 0.59, 0.12)
                .with_rotation(Quat::from_rotation_z(side * 0.20)),
        );
    }
    let scale_y = s.y / 0.88;
    for geometry in a.parts.values_mut() {
        for position in &mut geometry.positions {
            position[1] *= scale_y;
        }
        for normal in &mut geometry.normals {
            let n = Vec3::new(normal[0], normal[1] / scale_y, normal[2]).normalize();
            *normal = n.to_array();
        }
    }
}

fn cabinet(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    let label = o.kind.class_name();
    a.box_part(
        Surface::WoodEdge,
        label,
        Vec3::new(0.0, s.y * 0.5, -s.z * 0.5 + 0.012),
        Vec3::new(s.x, s.y, 0.024),
        0.003,
    );
    for x in [-s.x * 0.5 + 0.012, s.x * 0.5 - 0.012] {
        a.box_part(
            Surface::Wood,
            label,
            Vec3::new(x, s.y * 0.5, 0.0),
            Vec3::new(0.024, s.y, s.z),
            0.004,
        );
    }
    let shelves = if o.kind == ObjectKind::Bookcase { 5 } else { 2 };
    for i in 0..shelves {
        let y = 0.04 + i as f32 * (s.y - 0.052) / (shelves - 1) as f32;
        a.box_part(
            Surface::Wood,
            label,
            Vec3::new(0.0, y, 0.0),
            Vec3::new(s.x - 0.03, 0.024, s.z),
            0.004,
        );
        if o.kind == ObjectKind::Bookcase && i < shelves - 1 {
            let mut rng = stream(o.seed, i as u64);
            for b in 0..12 {
                if rng.random_bool(0.18) {
                    continue;
                }
                let h = rng.random_range(0.20..0.31);
                a.box_part(
                    [
                        Surface::FabricAlt,
                        Surface::Paper,
                        Surface::Accent,
                        Surface::WoodEdge,
                    ][b % 4],
                    "books",
                    Vec3::new(
                        -s.x * 0.43 + b as f32 * s.x * 0.073,
                        y + 0.012 + h * 0.5,
                        s.z * 0.06,
                    ),
                    Vec3::new(0.045, h, s.z * 0.76),
                    0.002,
                );
            }
        }
    }
    if o.kind == ObjectKind::Cabinet {
        let count = (s.x / 0.60).round().max(2.0) as usize;
        for i in 0..count {
            let w = (s.x - 0.04) / count as f32;
            let x = (i as f32 - (count - 1) as f32 * 0.5) * w;
            a.box_part(
                Surface::Wood,
                label,
                Vec3::new(x, s.y * 0.5 + 0.012, s.z * 0.5 - 0.013),
                Vec3::new(w - 0.004, s.y - 0.055, 0.026),
                0.003,
            );
            a.part(Surface::Metal, label).rod(
                Vec3::new(x + w * 0.30, s.y * 0.52, s.z * 0.5 + 0.018),
                Vec3::new(x + w * 0.30, s.y * 0.70, s.z * 0.5 + 0.018),
                0.006,
            );
        }
    }
}

fn bin(a: &mut Assembly, o: &IndoorObject) {
    let r = o.size.x * 0.5;
    let h = o.size.y;
    a.part(Surface::Metal, "other_prop").lathe(
        &[
            (0.0, 0.0),
            (r * 0.82, 0.0),
            (r, h * 0.90),
            (r * 0.94, h * 0.92),
            (r * 0.85, 0.03),
            (0.0, 0.03),
        ],
        36,
        Transform::IDENTITY,
    );
    a.part(Surface::Plastic, "other_prop").lathe(
        &[
            (r * 0.94, h * 0.88),
            (r, h * 0.93),
            (r * 0.98, h),
            (r * 0.58, h),
            (r * 0.55, h * 0.96),
        ],
        36,
        Transform::IDENTITY,
    );
    a.part(Surface::Rubber, "other_prop").cylinder(
        r * 0.8,
        0.005,
        Transform::from_xyz(0.0, h * 0.3, 0.0),
    );
}

fn display(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    if o.kind == ObjectKind::Display {
        a.box_part(
            Surface::Plastic,
            "television",
            Vec3::Y * s.y * 0.5,
            s,
            0.012,
        );
        a.box_part(
            Surface::Screen,
            "television",
            Vec3::new(0.0, s.y * 0.51, s.z * 0.5 + 0.001),
            Vec3::new(s.x - 0.045, s.y - 0.050, 0.002),
            0.0,
        );
        // Diagram bars and fine UI lines on the screen, made from geometry (no image asset).
        for i in 0..5 {
            let h = s.y * (0.12 + (i as f32 * 1.7).sin().abs() * 0.38);
            a.box_part(
                Surface::FabricAlt,
                "television",
                Vec3::new(
                    -s.x * 0.31 + i as f32 * s.x * 0.13,
                    s.y * 0.23 + h * 0.5,
                    s.z * 0.5 + 0.003,
                ),
                Vec3::new(s.x * 0.06, h, 0.001),
                0.0,
            );
        }
    } else {
        let label = o.kind.class_name();

        a.box_part(
            Surface::Metal,
            label,
            Vec3::new(0.0, 0.008, 0.01),
            Vec3::new(s.x, 0.016, s.z * 0.55),
            0.006,
        );
        let bottom = 0.065;
        {
            a.part(Surface::Metal, label).rod(
                Vec3::new(0.0, 0.02, 0.0),
                Vec3::new(0.0, 0.18, -0.07),
                0.012,
            );
        }
        let screen_h = s.y - bottom;
        a.box_part(
            Surface::Plastic,
            label,
            Vec3::new(0.0, bottom + screen_h * 0.5, -s.z * 0.36),
            Vec3::new(s.x, screen_h, 0.014),
            0.005,
        );
        a.box_part(
            Surface::Screen,
            label,
            Vec3::new(0.0, bottom + screen_h * 0.5, -s.z * 0.36 + 0.008),
            Vec3::new(s.x - 0.023, screen_h - 0.023, 0.001),
            0.0,
        );
    }
}

fn wall_panel(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    let label = o.kind.class_name();
    a.box_part(Surface::Chrome, label, Vec3::Y * s.y * 0.5, s, 0.006);
    a.box_part(
        if o.kind == ObjectKind::Whiteboard {
            Surface::Ceramic
        } else {
            Surface::Paper
        },
        label,
        Vec3::new(0.0, s.y * 0.5, s.z * 0.5 + 0.001),
        Vec3::new(s.x - 0.035, s.y - 0.035, 0.003),
        0.0,
    );
    let mut rng = stream(o.seed, 7);
    if o.kind == ObjectKind::Whiteboard {
        for row in 0..7 {
            let y = s.y * (0.78 - row as f32 * 0.09);
            for segment in 0..rng.random_range(2..6) {
                let x = s.x * (-0.36 + segment as f32 * 0.13);
                a.part(Surface::Ink, label).rod(
                    Vec3::new(x, y, s.z * 0.5 + 0.006),
                    Vec3::new(
                        x + s.x * rng.random_range(0.04..0.09),
                        y + rng.random_range(-0.012..0.012),
                        s.z * 0.5 + 0.006,
                    ),
                    0.002,
                );
            }
        }
        a.box_part(
            Surface::Chrome,
            label,
            Vec3::new(0.0, 0.013, 0.049),
            Vec3::new(s.x * 0.78, 0.025, 0.068),
            0.002,
        );
        for i in 0..3 {
            a.part(Surface::Ink, label).rod(
                Vec3::new(s.x * (-0.18 + i as f32 * 0.10), 0.031, 0.06),
                Vec3::new(s.x * (-0.12 + i as f32 * 0.10), 0.031, 0.06),
                0.006,
            );
        }
    } else {
        for i in 0..5 {
            let x = rng.random_range(-s.x * 0.27..s.x * 0.27);
            let y = rng.random_range(s.y * 0.2..s.y * 0.8);
            a.part(
                if i % 2 == 0 {
                    Surface::Art
                } else {
                    Surface::Wood
                },
                label,
            )
            .ellipsoid(
                Vec3::new(s.x * 0.16, s.y * 0.13, 0.002),
                Transform::from_xyz(x, y, s.z * 0.5 + 0.006 + i as f32 * 0.001),
            );
        }
    }
}

fn mug(a: &mut Assembly, o: &IndoorObject) {
    let h = o.size.y;
    let r = o.size.z * 0.44;
    a.part(Surface::Ceramic, "other_prop").lathe(
        &[
            (0.0, 0.0),
            (r * 0.83, 0.0),
            (r, h),
            (r * 0.87, h),
            (r * 0.72, 0.012),
            (0.0, 0.012),
        ],
        28,
        Transform::from_xyz(-0.012, 0.0, 0.0),
    );
    a.part(Surface::WoodEdge, "other_prop").cylinder(
        r * 0.86,
        0.002,
        Transform::from_xyz(-0.012, h * 0.76, 0.0),
    );
    let mut prev = Vec3::new(r - 0.014, h * 0.83, 0.0);
    for i in 1..=12 {
        let t = i as f32 / 12.0 * PI;
        let next = Vec3::new(
            r - 0.014 + 0.028 * t.sin(),
            h * 0.52 + h * 0.31 * t.cos(),
            0.0,
        );
        a.part(Surface::Ceramic, "other_prop")
            .rod(prev, next, 0.005);
        prev = next;
    }
}

fn books(a: &mut Assembly, o: &IndoorObject) {
    let n = if o.kind == ObjectKind::Books { 3 } else { 1 };
    for i in 0..n {
        let h = o.size.y / n as f32;
        let y = i as f32 * h;
        let cover = if i % 2 == 0 {
            Surface::FabricAlt
        } else {
            Surface::WoodEdge
        };
        let thickness = (h * 0.10).min(0.0025);
        // Separate covers and spine leave the page edges visible on three sides.
        // A solid cover-sized box would completely occlude the inner paper block.
        for cover_y in [y + thickness * 0.5, y + h - thickness * 0.5] {
            a.box_part(
                cover,
                o.kind.class_name(),
                Vec3::new(0.0, cover_y, 0.0),
                Vec3::new(o.size.x, thickness, o.size.z),
                thickness * 0.25,
            );
        }
        a.box_part(
            cover,
            o.kind.class_name(),
            Vec3::new(-o.size.x * 0.5 + thickness * 0.5, y + h * 0.5, 0.0),
            Vec3::new(thickness, h, o.size.z),
            thickness * 0.25,
        );
        a.box_part(
            Surface::Paper,
            o.kind.class_name(),
            Vec3::new(0.001, y + h * 0.5, 0.0),
            Vec3::new(o.size.x - 0.006, h - thickness * 2.0, o.size.z - 0.004),
            thickness * 0.20,
        );
    }
}

fn clock(a: &mut Assembly, o: &IndoorObject) {
    let r = o.size.x * 0.5;
    let tf = Transform::from_xyz(0.0, r, 0.0).with_rotation(Quat::from_rotation_x(FRAC_PI_2));
    a.part(Surface::Metal, "other_prop")
        .cylinder(r, o.size.z, tf);
    a.part(Surface::Paper, "other_prop").cylinder(
        r * 0.92,
        0.002,
        Transform::from_xyz(0.0, r, o.size.z * 0.5 + 0.001).with_rotation(tf.rotation),
    );
    for i in 0..12 {
        let angle = i as f32 * TAU / 12.0;
        let d = Vec3::new(angle.sin(), angle.cos(), 0.0);
        let c = Vec3::new(0.0, r, o.size.z * 0.5 + 0.004);
        a.part(Surface::Ink, "other_prop")
            .rod(c + d * r * 0.76, c + d * r * 0.84, 0.002);
    }
    let c = Vec3::new(0.0, r, o.size.z * 0.5 + 0.006);
    a.part(Surface::Ink, "other_prop")
        .rod(c, c + Vec3::new(-0.075, 0.04, 0.0), 0.004);
    a.part(Surface::Ink, "other_prop")
        .rod(c, c + Vec3::new(0.025, 0.11, 0.0), 0.003);
}

fn lamp(a: &mut Assembly, o: &IndoorObject) {
    let h = o.size.y;
    a.part(Surface::Metal, "lamp")
        .cylinder(0.17, 0.025, Transform::from_xyz(0.0, 0.0125, 0.0));
    a.part(Surface::Chrome, "lamp")
        .rod(Vec3::Y * 0.025, Vec3::Y * (h - 0.1), 0.012);
    a.part(Surface::FabricAlt, "lamp").lathe(
        &[(0.22, h - 0.30), (0.16, h), (0.15, h), (0.21, h - 0.30)],
        40,
        Transform::IDENTITY,
    );
    a.part(Surface::Light, "lamp")
        .ellipsoid(Vec3::splat(0.045), Transform::from_xyz(0.0, h - 0.22, 0.0));
}
