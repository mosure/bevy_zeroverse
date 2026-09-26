use super::super::{
    layout::{ArchitectureStyle, IndoorManifest},
    materials::Surface,
    objects::Assembly,
};
use bevy::prelude::*;

/// A real recessed bay, with a back, reveals and shelves; no intact wall behind
/// the opening. Its entire recess lies outside the camera/furniture envelope.
pub(super) fn rear_niche(a: &mut Assembly, scene: &IndoorManifest) {
    let Vec3 { x: w, y: h, z: d } = scene.room_size;
    let (left, right, bottom, top) = (-w * 0.5 + 0.6, -w * 0.5 + 2.0, 0.90, 2.32);
    let z = -d * 0.5;
    for (lo, hi) in [(-w * 0.5 - 0.24, left), (right, w * 0.5 + 0.24)] {
        a.box_part(
            Surface::Paint,
            "wall",
            Vec3::new((lo + hi) * 0.5, h * 0.5, z - 0.12),
            Vec3::new(hi - lo, h, 0.24),
            0.0,
        );
    }
    for (lo, hi) in [(0.0, bottom), (top, h)] {
        a.box_part(
            Surface::Paint,
            "wall",
            Vec3::new((left + right) * 0.5, (lo + hi) * 0.5, z - 0.12),
            Vec3::new(right - left, hi - lo, 0.24),
            0.0,
        );
    }
    a.box_part(
        Surface::Accent,
        "wall",
        Vec3::new((left + right) * 0.5, (bottom + top) * 0.5, z - 0.25),
        Vec3::new(right - left, top - bottom, 0.16),
        0.0,
    );
    for x in [left + 0.012, right - 0.012] {
        a.box_part(
            Surface::WoodEdge,
            "wall",
            Vec3::new(x, (bottom + top) * 0.5, z - 0.07),
            Vec3::new(0.024, top - bottom, 0.20),
            0.002,
        );
    }
    for y in [bottom, 1.42, 1.94, top] {
        a.box_part(
            Surface::Wood,
            "wall",
            Vec3::new((left + right) * 0.5, y, z - 0.07),
            Vec3::new(right - left, 0.035, 0.20),
            0.003,
        );
    }
}

pub(super) fn feature_wall(a: &mut Assembly, scene: &IndoorManifest) {
    let Vec3 { x: w, y: h, z: d } = scene.room_size;
    let back = -d * 0.5;
    match scene.architecture_style {
        ArchitectureStyle::Contemporary => {
            for i in 0..7 {
                a.box_part(
                    if i % 3 == 0 {
                        Surface::FabricAlt
                    } else {
                        Surface::Accent
                    },
                    "wall",
                    Vec3::new(-w * 0.38 + i as f32 * w * 0.11, h * 0.52, back + 0.035),
                    Vec3::new(w * 0.105, h * 0.78, 0.055),
                    0.008,
                );
            }
        }
        ArchitectureStyle::Timber => {
            a.box_part(
                Surface::Accent,
                "wall",
                Vec3::new(0.0, h * 0.5, back + 0.024),
                Vec3::new(w * 0.86, h - 0.3, 0.04),
                0.003,
            );
            let count = (w * 0.28 / 0.052) as usize;
            for i in 0..count {
                a.box_part(
                    Surface::Wood,
                    "wall",
                    Vec3::new(-w * 0.43 + i as f32 * 0.052, h * 0.5, back + 0.067),
                    Vec3::new(0.026, h - 0.35, 0.042),
                    0.002,
                );
            }
        }
        ArchitectureStyle::Industrial => {
            for i in 1..(w / 1.20) as usize {
                let x = -w * 0.5 + i as f32 * 1.20;
                a.box_part(
                    Surface::Metal,
                    "wall",
                    Vec3::new(x, h * 0.5, back + 0.003),
                    Vec3::new(0.007, h, 0.005),
                    0.0,
                );
            }
            for x in [-w * 0.35, w * 0.35] {
                a.part(Surface::Metal, "other_structure").rod(
                    Vec3::new(x, 0.15, back + 0.036),
                    Vec3::new(x, h - 0.15, back + 0.036),
                    0.013,
                );
            }
        }
        ArchitectureStyle::Classic => {
            // Shallow wainscot does not intrude into the 0.30 m placement margin.
            for z in [-d * 0.30, 0.0, d * 0.30] {
                a.box_part(
                    Surface::Wood,
                    "wall",
                    Vec3::new(w * 0.5 - 0.025, 0.48, z),
                    Vec3::new(0.042, 0.78, d * 0.27),
                    0.004,
                );
            }
            a.box_part(
                Surface::WoodEdge,
                "wall",
                Vec3::new(w * 0.5 - 0.040, 0.91, 0.0),
                Vec3::new(0.06, 0.055, d),
                0.005,
            );
        }
    }
    // Low wall plates ground the room visually without creating floor obstacles.
    for x in [-w * 0.25, w * 0.29] {
        a.box_part(
            Surface::Ceramic,
            "other_prop",
            Vec3::new(x, 0.28, back + 0.078),
            Vec3::new(0.15, 0.085, 0.015),
            0.004,
        );
        for offset in [-0.035, 0.035] {
            for dx in [-0.012, 0.012] {
                a.box_part(
                    Surface::Plastic,
                    "other_prop",
                    Vec3::new(x + offset + dx, 0.28, back + 0.087),
                    Vec3::new(0.006, 0.022, 0.003),
                    0.0,
                );
            }
        }
    }
}

pub(super) fn ceiling(a: &mut Assembly, scene: &IndoorManifest) {
    let Vec3 { x: w, y: h, z: d } = scene.room_size;
    if scene.ceiling_style == 3 {
        for i in 1..4 {
            let x = -w * 0.5 + w * i as f32 / 4.0;
            let z = -d * 0.5 + d * i as f32 / 4.0;
            a.box_part(
                Surface::Paint,
                "ceiling",
                Vec3::new(x, h - 0.065, 0.0),
                Vec3::new(0.18, 0.13, d),
                0.008,
            );
            a.box_part(
                Surface::Paint,
                "ceiling",
                Vec3::new(0.0, h - 0.065, z),
                Vec3::new(w, 0.13, 0.18),
                0.008,
            );
        }
    }
    if scene.architecture_style == ArchitectureStyle::Industrial {
        for x in [-w * 0.43, w * 0.43] {
            a.part(Surface::Metal, "other_structure").rod(
                Vec3::new(x, h - 0.225, -d * 0.45),
                Vec3::new(x, h - 0.225, d * 0.45),
                0.035,
            );
        }
        for z in [-d * 0.30, 0.0, d * 0.30] {
            a.box_part(
                Surface::Concrete,
                "ceiling",
                Vec3::new(0.0, h - 0.09, z),
                Vec3::new(w, 0.18, 0.19),
                0.005,
            );
        }
    }
}
