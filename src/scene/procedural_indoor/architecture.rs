mod details;
use super::{
    layout::{stream, IndoorManifest, LightingMood, ObjectKind, NEIGHBOR_DEPTH},
    materials::{kelvin_rgb, Surface},
    objects::Assembly,
};
use bevy::{light::CascadeShadowConfigBuilder, prelude::*};
use rand::Rng;

pub fn architecture(scene: &IndoorManifest) -> Assembly {
    let mut a = Assembly::default();
    let w = scene.room_size.x;
    let h = scene.room_size.y;
    let d = scene.room_size.z;
    let hx = w * 0.5;
    let hz = d * 0.5;
    let t = 0.18;
    a.box_part(
        Surface::Floor,
        "floor",
        Vec3::new(0.0, -0.10, NEIGHBOR_DEPTH * 0.5),
        Vec3::new(w + t * 2.0, 0.20, d + NEIGHBOR_DEPTH + t),
        0.0,
    );
    a.box_part(
        Surface::Ceiling,
        "ceiling",
        Vec3::new(0.0, h + 0.07, NEIGHBOR_DEPTH * 0.5),
        Vec3::new(w + t * 2.0, 0.14, d + NEIGHBOR_DEPTH + t),
        0.0,
    );
    for z in [-hz - t * 0.5, hz + NEIGHBOR_DEPTH + t * 0.5] {
        if z < 0.0 && scene.architecture_style == super::layout::ArchitectureStyle::Classic {
            details::rear_niche(&mut a, scene);
        } else {
            a.box_part(
                if scene.architecture_style == super::layout::ArchitectureStyle::Industrial {
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
    for i in 0..=scene.window_bays {
        let z = -hz + i as f32 * bay;
        a.box_part(
            Surface::Paint,
            "wall",
            Vec3::new(-hx - 0.03, (sill + top) * 0.5, z),
            Vec3::new(t + 0.10, top - sill, 0.15),
            0.003,
        );
    }
    for i in 0..scene.window_bays {
        let z = -hz + (i as f32 + 0.5) * bay;
        a.box_part(
            Surface::Glass,
            "window",
            Vec3::new(-hx - 0.055, (sill + top) * 0.5, z),
            Vec3::new(0.008, top - sill - 0.09, bay - 0.19),
            0.0,
        );
        for dz in [-bay * 0.5 + 0.10, 0.0, bay * 0.5 - 0.10] {
            a.box_part(
                Surface::Metal,
                "window",
                Vec3::new(-hx - 0.052, (sill + top) * 0.5, z + dz),
                Vec3::new(0.070, top - sill, 0.034),
                0.003,
            );
        }
        for y in [sill + 0.02, top - 0.02] {
            a.box_part(
                Surface::Metal,
                "window",
                Vec3::new(-hx - 0.052, y, z),
                Vec3::new(0.075, 0.04, bay - 0.15),
                0.003,
            );
        }
        a.box_part(
            Surface::Wood,
            "window",
            Vec3::new(-hx + 0.015, sill - 0.01, z),
            Vec3::new(0.32, 0.04, bay - 0.12),
            0.007,
        );
        if scene.blinds {
            a.box_part(
                Surface::Metal,
                "blinds",
                Vec3::new(-hx + 0.12, top - 0.055, z),
                Vec3::new(0.06, 0.08, bay - 0.18),
                0.004,
            );
            for j in 0..10 {
                let y = top - 0.15 - j as f32 * 0.065;
                a.part(Surface::WoodEdge, "blinds").cuboid(
                    Vec3::new(0.072, 0.003, bay - 0.20),
                    0.0,
                    Transform::from_xyz(-hx + 0.12, y, z)
                        .with_rotation(Quat::from_rotation_z(-0.42)),
                );
            }
        }
    }

    // Shared partition to a second furnished room, with a 1.1 m open doorway.
    let door_left = scene.door_x - 0.55;
    let door_right = scene.door_x + 0.55;
    let partition_h = h - 0.22;
    a.box_part(
        Surface::Paint,
        "wall",
        Vec3::new(0.0, h - 0.11, hz),
        Vec3::new(w, 0.22, 0.16),
        0.0,
    );
    for (lo, hi) in [(-hx, door_left), (door_right, hx)] {
        let count = ((hi - lo) / 1.25).ceil() as u32;
        let segment = (hi - lo) / count as f32;
        for i in 0..count {
            let x = lo + (i as f32 + 0.5) * segment;
            a.box_part(
                Surface::Glass,
                "window",
                Vec3::new(x, partition_h * 0.5, hz),
                Vec3::new(segment - 0.045, partition_h - 0.06, 0.01),
                0.0,
            );
            // Fine safety manifestation band; glass itself remains transparent.
            a.box_part(
                Surface::Ceramic,
                "window",
                Vec3::new(x, 1.10, hz - 0.007),
                Vec3::new(segment - 0.07, 0.023, 0.002),
                0.0,
            );
        }
        for i in 0..=count {
            a.box_part(
                Surface::Metal,
                "window",
                Vec3::new(lo + i as f32 * segment, partition_h * 0.5, hz),
                Vec3::new(0.035, partition_h, 0.075),
                0.002,
            );
        }
        for y in [0.021, partition_h - 0.021] {
            a.box_part(
                Surface::Metal,
                "window",
                Vec3::new((lo + hi) * 0.5, y, hz),
                Vec3::new(hi - lo, 0.042, 0.075),
                0.003,
            );
        }
    }
    for x in [door_left, door_right] {
        a.box_part(
            Surface::WoodEdge,
            "door",
            Vec3::new(x, 1.16, hz),
            Vec3::new(0.075, 2.32, 0.17),
            0.004,
        );
    }
    a.box_part(
        Surface::WoodEdge,
        "door",
        Vec3::new(scene.door_x, 2.30, hz),
        Vec3::new(1.17, 0.09, 0.17),
        0.004,
    );
    a.box_part(
        Surface::Glass,
        "window",
        Vec3::new(scene.door_x, (partition_h + 2.35) * 0.5, hz),
        Vec3::new(1.05, partition_h - 2.35, 0.01),
        0.0,
    );
    // Open door leaf lies inside neighboring room, outside the clear approach.
    a.box_part(
        Surface::Wood,
        "door",
        Vec3::new(door_right - 0.025, 1.12, hz + 0.56),
        Vec3::new(0.045, 2.24, 1.07),
        0.007,
    );
    a.part(Surface::Chrome, "door").rod(
        Vec3::new(door_right - 0.06, 1.03, hz + 0.96),
        Vec3::new(door_right - 0.06, 1.03, hz + 0.83),
        0.011,
    );

    // Pilasters, ceiling perimeter reveals and a shallow acoustic feature wall.
    for x in [
        -hx + scene.column_width * 0.5,
        hx - scene.column_width * 0.5,
    ] {
        for z in [
            -hz + scene.column_width * 0.5,
            hz - scene.column_width * 0.5,
        ] {
            a.box_part(
                Surface::Concrete,
                "other_structure",
                Vec3::new(x, h * 0.5, z),
                Vec3::new(scene.column_width, h, scene.column_width),
                0.012,
            );
        }
    }
    for x in [-hx + 0.18, hx - 0.18] {
        a.box_part(
            Surface::Paint,
            "ceiling",
            Vec3::new(x, h - 0.07, 0.0),
            Vec3::new(0.36, 0.14, d),
            0.003,
        );
    }
    for z in [-hz + 0.18, hz - 0.18] {
        a.box_part(
            Surface::Paint,
            "ceiling",
            Vec3::new(0.0, h - 0.07, z),
            Vec3::new(w, 0.14, 0.36),
            0.003,
        );
    }
    details::feature_wall(&mut a, scene);
    if scene.ceiling_style == 0 {
        for i in 1..(w / 0.6) as usize {
            let x = -hx + i as f32 * 0.6;
            a.box_part(
                Surface::Metal,
                "ceiling",
                Vec3::new(x, h - 0.004, 0.0),
                Vec3::new(0.012, 0.008, d),
                0.0,
            );
        }
        for i in 1..(d / 0.6) as usize {
            let z = -hz + i as f32 * 0.6;
            a.box_part(
                Surface::Metal,
                "ceiling",
                Vec3::new(0.0, h - 0.004, z),
                Vec3::new(w, 0.008, 0.012),
                0.0,
            );
        }
    } else if scene.ceiling_style == 1 {
        for i in 0..16 {
            let x = -w * 0.34 + i as f32 * w * 0.68 / 15.0;
            a.box_part(
                Surface::Wood,
                "ceiling",
                Vec3::new(x, h - 0.10, 0.0),
                Vec3::new(0.035, 0.15, d * 0.74),
                0.003,
            );
        }
    }
    details::ceiling(&mut a, scene);
    // Supply grilles and recessed luminaire housings.
    for z in [-d * 0.28, d * 0.28] {
        a.box_part(
            Surface::Metal,
            "ceiling",
            Vec3::new(0.0, h - 0.012, z),
            Vec3::new(0.50, 0.02, 0.32),
            0.003,
        );
        for i in 0..9 {
            a.box_part(
                Surface::Paint,
                "ceiling",
                Vec3::new(0.0, h - 0.025, z - 0.13 + i as f32 * 0.032),
                Vec3::new(0.46, 0.012, 0.012),
                0.0,
            );
        }
    }
    for p in fixture_positions(scene) {
        a.box_part(
            Surface::Metal,
            "lamp",
            p,
            Vec3::new(1.05, 0.055, 0.27),
            0.006,
        );
        a.box_part(
            Surface::Light,
            "lamp",
            p - Vec3::Y * 0.030,
            Vec3::new(0.99, 0.008, 0.215),
            0.002,
        );
        for x in [-0.38, 0.38] {
            a.part(Surface::Chrome, "lamp").rod(
                p + Vec3::new(x, 0.028, 0.0),
                Vec3::new(p.x + x, h, p.z),
                0.002,
            );
        }
    }
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
    a
}

pub fn fixture_positions(scene: &IndoorManifest) -> Vec<Vec3> {
    let mut positions = Vec::new();
    let columns = (scene.room_size.x / 3.4).ceil().clamp(2.0, 4.0) as usize;
    let rows = (scene.room_size.z / 3.4).ceil().clamp(2.0, 4.0) as usize;
    for column in 0..columns {
        let x = ((column as f32 + 0.5) / columns as f32 - 0.5) * scene.room_size.x;
        for row in 0..rows {
            let z = ((row as f32 + 0.5) / rows as f32 - 0.5) * scene.room_size.z;
            positions.push(Vec3::new(x, scene.room_size.y - 0.25, z));
        }
    }
    positions.push(Vec3::new(
        0.0,
        scene.room_size.y - 0.18,
        scene.room_size.z * 0.5 + 1.5,
    ));
    positions
}

pub fn spawn_lights(
    scene: &IndoorManifest,
    quality: super::IndoorQuality,
    root: Entity,
    commands: &mut Commands,
) {
    let c = kelvin_rgb(scene.light_kelvin);
    for (i, p) in fixture_positions(scene).into_iter().enumerate() {
        commands.spawn((
            Name::new(format!("indoor_luminaire_{i}")),
            SpotLight {
                color: Color::srgb(c.x, c.y, c.z),
                // Bevy divides spot intensity by 4π before applying a squared
                // angular falloff. Normalize that cone so these are fixture lumens.
                intensity: spot_intensity_for_lumens(fixture_lumens(scene), 0.75, 1.35),
                range: 13.0,
                radius: 0.20,
                inner_angle: 0.75,
                outer_angle: 1.35,
                shadow_maps_enabled: quality.shadows() && i < 8,
                shadow_depth_bias: 0.015,
                // Bevy expresses normal bias in shadow texels, not metres.
                // Sub-texel offsets caused broad self-shadowing moire on walls.
                shadow_normal_bias: SpotLight::DEFAULT_SHADOW_NORMAL_BIAS,
                ..default()
            },
            Transform::from_translation(p - Vec3::Y * 0.06).looking_at(p - Vec3::Y, Vec3::Z),
            ChildOf(root),
        ));
    }
    let direction = sun_direction(scene);
    commands.spawn((
        Name::new("indoor_sun"),
        DirectionalLight {
            color: sun_color(scene),
            // An unshadowed sun would shine through the solid room shell.
            // Portable keeps the indoor fixtures and environment illumination.
            illuminance: if quality.shadows() {
                sun_illuminance(scene)
            } else {
                0.0
            },
            shadow_maps_enabled: quality.shadows(),
            shadow_depth_bias: 0.015,
            shadow_normal_bias: DirectionalLight::DEFAULT_SHADOW_NORMAL_BIAS,
            ..default()
        },
        CascadeShadowConfigBuilder {
            first_cascade_far_bound: 8.0,
            maximum_distance: 45.0,
            ..default()
        }
        .build(),
        Transform::from_translation(direction * 30.0).looking_at(Vec3::ZERO, Vec3::Y),
        ChildOf(root),
    ));
    for lamp in scene
        .objects
        .iter()
        .filter(|o| o.kind == ObjectKind::FloorLamp)
    {
        // The bulb sits within a real open fabric shade; its shadowed point light
        // illuminates the nearby seating and wall instead of merely glowing.
        commands.spawn((
            Name::new(format!("indoor_floor_lamp_{}", lamp.id)),
            PointLight {
                color: Color::srgb(1.0, 0.78, 0.57),
                intensity: 800.0,
                range: 5.0,
                radius: 0.035,
                shadow_maps_enabled: quality.shadows(),
                shadow_depth_bias: 0.01,
                shadow_normal_bias: PointLight::DEFAULT_SHADOW_NORMAL_BIAS,
                ..default()
            },
            Transform::from_translation(lamp.position + Vec3::Y * (lamp.size.y - 0.22)),
            ChildOf(root),
        ));
    }
}

pub(crate) fn spot_intensity_for_lumens(lumens: f32, inner: f32, outer: f32) -> f32 {
    let angular_integral = 1.0 - inner.cos() + (inner.cos() - outer.cos()) / 3.0;
    lumens * 2.0 / angular_integral
}

pub(crate) fn fixture_lumens(scene: &IndoorManifest) -> f32 {
    // Lumen-method design estimate (not a simulated lux measurement): E =
    // N*flux*utilization/area. More fixtures maintain plausible office lighting
    // as room area changes. Evening retains a slightly lower occupied level.
    let illuminance = if scene.lighting == LightingMood::Evening {
        300.0
    } else {
        450.0
    };
    let area = scene.room_size.x * scene.room_size.z;
    let count = (fixture_positions(scene).len() - 1) as f32;
    (illuminance * area / (count * 0.70)).clamp(2500.0, 6500.0)
}

pub(crate) fn sun_direction(scene: &IndoorManifest) -> Vec3 {
    Vec3::new(
        -scene.sun_elevation.cos() * scene.sun_azimuth.cos(),
        scene.sun_elevation.sin(),
        scene.sun_elevation.cos() * scene.sun_azimuth.sin(),
    )
    .normalize()
}

pub(crate) fn sun_color(scene: &IndoorManifest) -> Color {
    if scene.lighting == LightingMood::Evening {
        Color::srgb(1.0, 0.69, 0.43)
    } else {
        Color::srgb(1.0, 0.95, 0.87)
    }
}

pub(crate) fn sun_illuminance(scene: &IndoorManifest) -> f32 {
    match scene.lighting {
        LightingMood::Daylight => 18000.0,
        LightingMood::Overcast => 4200.0,
        LightingMood::Evening => 1100.0,
    }
}

#[cfg(test)]
mod lighting_tests {
    use super::*;

    #[test]
    fn spotlight_cone_integrates_to_declared_lumens() {
        let inner = 0.75_f32;
        let outer = 1.35_f32;
        let intensity = spot_intensity_for_lumens(4200.0, inner, outer);
        let intervals = 10000;
        let dx = 2.0 / intervals as f32;
        let integral = (0..intervals)
            .map(|i| {
                let cos_theta = -1.0 + (i as f32 + 0.5) * dx;
                ((cos_theta - outer.cos()) / (inner.cos() - outer.cos()))
                    .clamp(0.0, 1.0)
                    .powi(2)
                    * dx
            })
            .sum::<f32>()
            * std::f32::consts::TAU;
        let flux = intensity / (4.0 * std::f32::consts::PI) * integral;
        assert!((flux - 4200.0).abs() < 1.0);
    }
}
