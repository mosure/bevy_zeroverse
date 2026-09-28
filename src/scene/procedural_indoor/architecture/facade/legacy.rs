//! Legacy single-wall manifests retain their original shell.
use super::super::details;
use super::*;
use crate::scene::procedural_indoor::layout::{ArchitectureStyle, NEIGHBOR_DEPTH};
pub(super) fn shell(a: &mut Assembly, scene: &IndoorManifest) {
    let Vec3 { x: w, y: h, z: d } = scene.room_size;
    let (hx, hz, t) = (w * 0.5, d * 0.5, 0.18);
    for z in [-hz - t * 0.5, hz + NEIGHBOR_DEPTH + t * 0.5] {
        if z < 0.0 && scene.architecture_style == ArchitectureStyle::Classic {
            details::rear_niche(a, scene);
        } else {
            a.box_part(
                if scene.architecture_style == ArchitectureStyle::Industrial {
                    Surface::Concrete
                } else {
                    Surface::Paint
                },
                "wall",
                Vec3::new(0.0, h * 0.5, z),
                Vec3::new(w + t * 2.0, h, t),
                0.0,
            );
        }
        a.box_part(
            Surface::WoodEdge,
            "wall",
            Vec3::new(0.0, 0.065, z - z.signum() * (t * 0.5 + 0.011)),
            Vec3::new(w, 0.13, 0.022),
            0.002,
        );
    }
    a.box_part(
        Surface::Paint,
        "wall",
        Vec3::new(hx + t * 0.5, h * 0.5, NEIGHBOR_DEPTH * 0.5),
        Vec3::new(t, h, d + NEIGHBOR_DEPTH),
        0.0,
    );
    a.box_part(
        Surface::WoodEdge,
        "wall",
        Vec3::new(hx - 0.01, 0.065, NEIGHBOR_DEPTH * 0.5),
        Vec3::new(0.024, 0.13, d + NEIGHBOR_DEPTH),
        0.002,
    );

    // Exterior wall is cut into piers, sill, lintel and inset panes. No coplanar wall behind glass.
    let sill = scene.window_sill;
    let top = scene.glazing_height;
    a.box_part(
        Surface::Paint,
        "wall",
        Vec3::new(-hx - t * 0.5, sill * 0.5, 0.0),
        Vec3::new(t, sill, d),
        0.0,
    );
    a.box_part(
        Surface::Paint,
        "wall",
        Vec3::new(-hx - t * 0.5, (top + h) * 0.5, 0.0),
        Vec3::new(t, h - top, d),
        0.0,
    );
    a.box_part(
        Surface::Paint,
        "wall",
        Vec3::new(-hx - t * 0.5, h * 0.5, hz + NEIGHBOR_DEPTH * 0.5),
        Vec3::new(t, h, NEIGHBOR_DEPTH),
        0.0,
    );
    a.box_part(
        Surface::WoodEdge,
        "wall",
        Vec3::new(-hx + 0.006, 0.065, 0.0),
        Vec3::new(0.025, 0.13, d),
        0.003,
    );
    let bay = d / scene.window_bays as f32;
    let pier = scene
        .domain()
        .map_or(0.15, |v| bay * v.facade_pier_fraction)
        .clamp(0.12, bay - 0.30);
    for i in 0..=scene.window_bays {
        let z = -hz + i as f32 * bay;
        a.box_part(
            Surface::Paint,
            "wall",
            Vec3::new(-hx - 0.03, (sill + top) * 0.5, z),
            Vec3::new(t + 0.10, top - sill, pier),
            0.003,
        );
    }
    for i in 0..scene.window_bays {
        let z = -hz + (i as f32 + 0.5) * bay;
        a.box_part(
            Surface::Glass,
            "window",
            Vec3::new(-hx - 0.055, (sill + top) * 0.5, z),
            Vec3::new(0.008, top - sill - 0.09, bay - pier - 0.04),
            0.0,
        );
        for dz in [-(bay - pier) * 0.5 + 0.025, 0.0, (bay - pier) * 0.5 - 0.025] {
            a.box_part(
                Surface::Metal,
                "window",
                Vec3::new(-hx - 0.052, (sill + top) * 0.5, z + dz),
                Vec3::new(0.070, top - sill - 0.08, 0.034),
                0.003,
            );
        }
        for y in [sill + 0.02, top - 0.02] {
            a.box_part(
                Surface::Metal,
                "window",
                Vec3::new(-hx - 0.052, y, z),
                Vec3::new(0.075, 0.04, bay - pier),
                0.003,
            );
        }
        a.box_part(
            Surface::Wood,
            "window",
            Vec3::new(-hx + 0.015, sill - 0.01, z),
            Vec3::new(0.32, 0.04, bay - pier + 0.03),
            0.007,
        );
        if scene.blinds {
            a.box_part(
                Surface::Metal,
                "blinds",
                Vec3::new(-hx + 0.12, top - 0.055, z),
                Vec3::new(0.06, 0.08, bay - pier - 0.03),
                0.004,
            );
            let coverage = scene.domain().map_or(0.65, |d| d.blind_coverage);
            let slats = (((top - sill - 0.18) * coverage / 0.065).floor() as usize).clamp(1, 64);
            for j in 0..slats {
                let y = top - 0.15 - j as f32 * 0.065;
                a.part(Surface::WoodEdge, "blinds").cuboid(
                    Vec3::new(0.072, 0.003, bay - pier - 0.05),
                    0.0,
                    Transform::from_xyz(-hx + 0.12, y, z).with_rotation(Quat::from_rotation_z(
                        scene.domain().map_or(-0.42, |d| d.blind_tilt),
                    )),
                );
            }
        }
    }
}
pub(super) fn backdrop(a: &mut Assembly, scene: &IndoorManifest) {
    let hx = scene.room_size.x * 0.5;
    // Exterior ground and compact generated neighboring facades: visible parallax, no backdrop image.
    a.box_part(
        Surface::Concrete,
        "floor",
        Vec3::new(-hx - 7.0, -0.17, 0.0),
        Vec3::new(13.0, 0.20, 32.0),
        0.0,
    );
    let mut rng = stream(scene.seed, 9);
    for i in 0..7 {
        let z = -15.0 + i as f32 * 5.0;
        let bh = rng.random_range(5.0..14.0);
        let x = -hx - rng.random_range(9.0..13.0);
        a.box_part(
            Surface::Concrete,
            "other_structure",
            Vec3::new(x, bh * 0.5, z),
            Vec3::new(3.0, bh, 3.8),
            0.015,
        );
        for floor in 0..(bh / 2.5) as usize {
            for col in [-1.0, 1.0] {
                a.box_part(
                    Surface::Glass,
                    "window",
                    Vec3::new(x + 1.51, 1.3 + floor as f32 * 2.5, z + col * 0.85),
                    Vec3::new(0.02, 1.15, 0.9),
                    0.0,
                );
            }
        }
    }
}
