use super::super::{
    layout::{ArchitectureStyle, IndoorManifest},
    materials::Surface,
    objects::Assembly,
};
use bevy::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FinishParameters {
    pub ceiling_pitch: Vec2,
    pub panel_pitch: f32,
    pub panel_height: f32,
    pub reveal: f32,
}
impl FinishParameters {
    pub fn sample(seed: u64) -> Self {
        let mut rng = super::super::layout::stream(seed, 162);
        Self {
            ceiling_pitch: Vec2::new(rng.random_range(0.50..1.25), rng.random_range(0.50..1.25)),
            panel_pitch: rng.random_range(0.45..1.65),
            panel_height: rng.random_range(0.42..0.80),
            reveal: rng.random_range(0.015..0.045),
        }
    }
    pub fn for_scene(scene: &IndoorManifest) -> Self {
        scene
            .program
            .as_ref()
            .and_then(|p| p.finishes.clone())
            .unwrap_or_else(|| Self::sample(scene.seed))
    }
}

/// Dropped acoustic rafts articulate each functional zone. Fixtures hang below
/// their underside; the same drop bounds camera sampling and collision checks.
pub(super) fn zone_ceilings(a: &mut Assembly, scene: &IndoorManifest) {
    let Some(domain) = scene.domain() else {
        return;
    };
    if domain.ceiling_relief < 0.08 {
        return;
    }
    for zone in &scene.program.as_ref().unwrap().zones {
        let center = (zone.min + zone.max) * 0.5;
        let size = (zone.max - zone.min - Vec2::splat(0.65)) * domain.ceiling_coverage.sqrt();
        a.box_part(
            Surface::Ceiling,
            "ceiling",
            Vec3::new(
                center.x,
                scene.room_size.y - domain.ceiling_relief + 0.025,
                center.y,
            ),
            Vec3::new(size.x, 0.05, size.y),
            0.008,
        );
    }
}

/// Cross rails terminate at main rails. Their undersides never overlap; this is
/// the same butt joint used by suspended ceilings, rather than a depth bias.
pub(super) fn crossed_beams(
    a: &mut Assembly,
    scene: &IndoorManifest,
    pitch: Vec2,
    width: f32,
    height: f32,
    y: f32,
    surface: Surface,
) {
    let half = Vec2::new(scene.room_size.x, scene.room_size.z) * 0.5 - Vec2::splat(0.36);
    let nx = ((half.x * 2.0) / pitch.x).ceil() as usize;
    let nz = ((half.y * 2.0) / pitch.y).ceil() as usize;
    let dx = half.x * 2.0 / nx as f32;
    for i in 1..nx {
        a.box_part(
            surface,
            "ceiling",
            Vec3::new(-half.x + dx * i as f32, y, 0.0),
            Vec3::new(width, height, half.y * 2.0),
            0.0,
        );
    }
    for j in 1..nz {
        let z = -half.y + half.y * 2.0 * j as f32 / nz as f32;
        for i in 0..nx {
            let lo = -half.x + dx * i as f32 + if i == 0 { 0.0 } else { width * 0.5 };
            let hi = -half.x + dx * (i + 1) as f32 - if i + 1 == nx { 0.0 } else { width * 0.5 };
            a.box_part(
                surface,
                "ceiling",
                Vec3::new((lo + hi) * 0.5, y, z),
                Vec3::new(hi - lo, height, width),
                0.0,
            );
        }
    }
}

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
            Vec3::new(right - left - 0.048, 0.035, 0.20),
            0.003,
        );
    }
}

pub(super) fn feature_wall(a: &mut Assembly, scene: &IndoorManifest) {
    let Vec3 { x: w, y: h, z: d } = scene.room_size;
    let back = -d * 0.5;
    let finishes = FinishParameters::for_scene(scene);
    match scene.architecture_style {
        ArchitectureStyle::Contemporary => {
            let count = (w / finishes.panel_pitch).ceil() as usize;
            let pitch = w * 0.82 / count as f32;
            for i in 0..count {
                a.box_part(
                    if i % 3 == 0 {
                        Surface::FabricAlt
                    } else {
                        Surface::Accent
                    },
                    "wall",
                    Vec3::new(-w * 0.41 + (i as f32 + 0.5) * pitch, h * 0.52, back + 0.035),
                    Vec3::new(pitch - finishes.reveal, h * finishes.panel_height, 0.055),
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
        let f = FinishParameters::for_scene(scene);
        crossed_beams(
            a,
            scene,
            f.ceiling_pitch * 3.8,
            0.18,
            0.13,
            h - 0.065,
            Surface::Paint,
        );
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
