pub mod details;
pub mod facade;
#[cfg(test)]
mod overlap_tests;
use super::{
    layout::{stream, IndoorManifest, LightingMood, ObjectKind, NEIGHBOR_DEPTH},
    materials::{kelvin_rgb, Surface},
    objects::Assembly,
};
use bevy::{light::CascadeShadowConfigBuilder, prelude::*};
use rand::Rng;

pub fn architecture(scene: &IndoorManifest) -> Assembly {
    if scene.envelope.is_some() {
        return super::envelope::construction::build(scene);
    }
    let mut a = Assembly::default();
    super::floorplan::build(scene, &mut a);
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
    facade::shell(&mut a, scene);

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
    for (lo, hi) in [(-hx + 0.006, door_left), (door_right, hx - 0.006)] {
        let count = ((hi - lo) / 1.25).ceil() as u32;
        let segment = (hi - lo) / count as f32;
        for i in 0..count {
            let x = lo + (i as f32 + 0.5) * segment;
            a.box_part(
                Surface::GlassInterior,
                "window",
                Vec3::new(x, partition_h * 0.5, hz),
                Vec3::new(segment - 0.045, partition_h - 0.06, 0.01),
                0.0,
            );
            // Applied safety bands end beside perpendicular partition jambs.
            // Their back faces otherwise coincide with the metal rail end caps.
            let mut spans = vec![(x - (segment - 0.07) * 0.5, x + (segment - 0.07) * 0.5)];
            if let Some(program) = &scene.program {
                for p in &program.partitions {
                    if p.axis != 0 || p.end < hz - 0.1 {
                        continue;
                    }
                    let cut = (
                        p.coordinate - p.thickness * 0.5 - 0.025,
                        p.coordinate + p.thickness * 0.5 + 0.025,
                    );
                    spans = spans
                        .into_iter()
                        .flat_map(|(lo, hi)| {
                            if hi <= cut.0 || lo >= cut.1 {
                                return vec![(lo, hi)];
                            }
                            let mut parts = Vec::new();
                            if lo < cut.0 {
                                parts.push((lo, cut.0));
                            }
                            if hi > cut.1 {
                                parts.push((cut.1, hi));
                            }
                            parts
                        })
                        .collect();
                }
            }
            for (lo, hi) in spans {
                if hi - lo > 0.01 {
                    a.box_part(
                        Surface::Ceramic,
                        "window",
                        Vec3::new((lo + hi) * 0.5, 1.10, hz - 0.007),
                        Vec3::new(hi - lo, 0.023, 0.002),
                        0.0,
                    );
                }
            }
        }
        for i in 0..=count {
            a.box_part(
                Surface::Metal,
                "window",
                Vec3::new(lo + i as f32 * segment, partition_h * 0.5, hz),
                Vec3::new(0.035, partition_h - 0.084, 0.075),
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
            Vec3::new(x, 2.255 * 0.5, hz),
            Vec3::new(0.075, 2.255, 0.17),
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
        Surface::GlassInterior,
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
            Vec3::new(w - 0.72, 0.14, 0.36),
            0.003,
        );
    }
    details::feature_wall(&mut a, scene);
    if scene.ceiling_style == 0 {
        let finishes = details::FinishParameters::for_scene(scene);
        details::crossed_beams(
            &mut a,
            scene,
            finishes.ceiling_pitch,
            0.012,
            0.008,
            h - 0.004,
            Surface::Metal,
        );
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
    details::zone_ceilings(&mut a, scene);
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
    fixture_geometry(&mut a, scene);
    facade::backdrop(&mut a, scene);
    a
}

pub(crate) fn fixture_geometry(a: &mut Assembly, scene: &IndoorManifest) {
    for (i, p) in fixture_positions(scene).into_iter().enumerate() {
        a.box_part(
            Surface::Metal,
            &format!("lamp#{i}"),
            p,
            fixture_size(scene),
            0.006,
        );
        a.box_part(
            Surface::Light,
            &format!("lamp#{i}"),
            p - Vec3::Y * 0.030,
            (fixture_size(scene) - Vec3::new(0.06, 0.0, 0.055)).with_y(0.008),
            0.002,
        );
        for x in [-fixture_size(scene).x * 0.36, fixture_size(scene).x * 0.36] {
            a.part(Surface::Chrome, &format!("lamp#{i}")).rod(
                p + Vec3::new(x, 0.028, 0.0),
                Vec3::new(
                    p.x + x,
                    scene
                        .envelope
                        .as_ref()
                        .and_then(|e| e.mezzanine.as_ref())
                        .filter(|m| m.deck.contains(p.xz()) && p.y < m.deck.height)
                        .map_or_else(
                            || scene.ceiling_height(Vec2::new(p.x + x, p.z)),
                            |m| m.deck.height - m.thickness,
                        ),
                    p.z,
                ),
                0.002,
            );
        }
    }
}

pub(crate) fn fixture_size(scene: &IndoorManifest) -> Vec3 {
    if let Some(program) = &scene.program {
        return Vec3::new(program.fixture_size.x, 0.055, program.fixture_size.y);
    }
    match scene.lighting_design % 3 {
        0 => Vec3::new(0.60, 0.055, 0.60),
        1 => Vec3::new(1.45, 0.055, 0.17),
        _ => Vec3::new(0.22, 0.055, 0.22),
    }
}

/// Close-mounted luminaires below the deck prevent the upper floor from
/// shadowing every light in the lower work area. Their anchors stop at the slab.
pub(crate) fn under_mezzanine_fixtures(scene: &IndoorManifest) -> Vec<Vec3> {
    let Some(m) = scene.envelope.as_ref().and_then(|e| e.mezzanine.as_ref()) else {
        return vec![];
    };
    let size = m.deck.max - m.deck.min;
    let count = (size / 2.5).ceil().max(Vec2::ONE).as_uvec2();
    let mut positions = Vec::new();
    for x in 0..count.x {
        for z in 0..count.y {
            let p = m.deck.min
                + size
                    * Vec2::new(
                        (x as f32 + 0.5) / count.x as f32,
                        (z as f32 + 0.5) / count.y as f32,
                    );
            positions.push(Vec3::new(p.x, m.deck.height - m.thickness - 0.10, p.y));
        }
    }
    positions
}

pub fn fixture_positions(scene: &IndoorManifest) -> Vec<Vec3> {
    if let Some(program) = &scene.program {
        let mut positions = Vec::new();
        for zone in &program.zones {
            let size = zone.max - zone.min;
            let counts = (size / program.light_spacing)
                .ceil()
                .max(Vec2::ONE)
                .as_uvec2();
            for x in 0..counts.x {
                for z in 0..counts.y {
                    let p = zone.min
                        + size * (Vec2::new(x as f32 + 0.5, z as f32 + 0.5) + program.light_phase)
                            / counts.as_vec2();
                    if scene.envelope.as_ref().is_none_or(|e| {
                        super::envelope::polygon::box_inside(
                            &e.footprint,
                            p - program.fixture_size * 0.5,
                            p + program.fixture_size * 0.5,
                            0.15,
                        )
                    }) {
                        positions.push(Vec3::new(
                            p.x,
                            scene.ceiling_height(p) - program.light_drop,
                            p.y,
                        ));
                    }
                }
            }
        }
        positions.extend(under_mezzanine_fixtures(scene));
        positions.push(Vec3::new(
            0.0,
            scene.ceiling_height(Vec2::new(0.0, scene.room_size.z * 0.5)) - 0.18,
            scene.room_size.z * 0.5 + 1.5,
        ));
        return positions;
    }
    let mut positions = Vec::new();
    let spacing = [3.4, 4.0, 2.8][scene.lighting_design as usize % 3];
    let columns = (scene.room_size.x / spacing).ceil().clamp(2.0, 4.0) as usize;
    let rows = (scene.room_size.z / spacing).ceil().clamp(2.0, 4.0) as usize;
    for column in 0..columns {
        let x = ((column as f32 + 0.5) / columns as f32 - 0.5) * scene.room_size.x;
        for row in 0..rows {
            let z = ((row as f32 + 0.5) / rows as f32 - 0.5) * scene.room_size.z;
            if scene.floor_plan == super::floorplan::FloorPlan::CornerCore
                && super::floorplan::obstacles(scene)
                    .iter()
                    .any(|(lo, hi)| x > lo.x && x < hi.x && z > lo.z && z < hi.z)
            {
                continue;
            }
            let drop = [0.10, 0.42, 0.06][scene.lighting_design as usize % 3];
            positions.push(Vec3::new(x, scene.room_size.y - drop, z));
        }
    }
    positions.push(Vec3::new(
        0.0,
        scene.ceiling_height(Vec2::new(0.0, scene.room_size.z * 0.5)) - 0.18,
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
    for (i, p) in fixture_positions(scene).into_iter().enumerate() {
        let (c, lumens) = fixture_photometry(scene, i);
        let (inner, outer) = fixture_angles(scene, i);
        commands.spawn((
            Name::new(format!("indoor_luminaire_{i}")),
            SpotLight {
                color: Color::srgb(c.x, c.y, c.z),
                // Bevy divides spot intensity by 4π before applying a squared
                // angular falloff. Normalize that cone so these are fixture lumens.
                intensity: spot_intensity_for_lumens(lumens, inner, outer),
                range: 13.0,
                radius: 0.20,
                inner_angle: inner,
                outer_angle: outer,
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
                intensity: floor_lamp_lumens(scene, lamp.seed),
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

/// Circuit-level dimming and colour temperature vary independently of placement.
pub(crate) fn fixture_angles(scene: &IndoorManifest, index: usize) -> (f32, f32) {
    let mut rng = stream(scene.seed, 192 + index as u64);
    (rng.random_range(0.48..0.90), rng.random_range(1.02..1.40))
}

pub(crate) fn fixture_photometry(scene: &IndoorManifest, index: usize) -> (Vec3, f32) {
    let mut rng = stream(scene.seed, 164 + index as u64);
    let mut kelvin = scene.light_kelvin + rng.random_range(-650.0..650.0);
    let mut flux = fixture_lumens(scene) * rng.random_range(0.72..1.18);
    if let Some(d) = scene.domain() {
        let position = fixture_positions(scene)[index];
        let uv = Vec2::new(
            position.x / scene.room_size.x,
            position.z / scene.room_size.z,
        ) * 2.0;
        flux *= (1.0 + d.photometry.fixture_gradient.dot(uv)).clamp(0.25, 1.75);
        kelvin += d.photometry.temperature_gradient * uv.x;
        // Independent circuits leave pools of light and unlit areas. Always keep
        // circuit zero energized; low-light scenes remain intentionally usable.
        if index > 0 && !rng.random_bool(d.photometry.active_fraction as f64) {
            flux = 0.0;
        } else {
            flux *= 1.0 - d.photometry.circuit_contrast * rng.random_range(0.0..1.0);
        }
    }
    (kelvin_rgb(kelvin.clamp(1800.0, 9000.0)), flux)
}

pub(crate) fn fixture_lumens(scene: &IndoorManifest) -> f32 {
    // Lumen-method design estimate (not a simulated lux measurement): E =
    // N*flux*utilization/area. More fixtures maintain plausible office lighting
    // as room area changes. Evening retains a slightly lower occupied level.
    let illuminance = scene.target_lux;
    let area = scene
        .envelope
        .as_ref()
        .map_or(scene.room_size.x * scene.room_size.z, |e| {
            super::envelope::polygon::area(&e.footprint)
                + e.mezzanine
                    .as_ref()
                    .map_or(0., |m| (m.deck.max - m.deck.min).element_product())
        });
    let count = (fixture_positions(scene).len() - 1) as f32;
    (illuminance * area / (count.max(1.0) * 0.70)).clamp(0.1, 16000.0)
}

pub(crate) fn floor_lamp_lumens(scene: &IndoorManifest, seed: u64) -> f32 {
    if scene.domain().is_none() {
        return 800.0;
    }
    scene.target_lux * stream(seed, 217).random_range(0.8..2.4)
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
    if let Some(d) = scene.domain() {
        let rgb = kelvin_rgb(d.photometry.sun_kelvin);
        return Color::srgb(rgb.x, rgb.y, rgb.z);
    }
    if scene.lighting == LightingMood::Evening {
        Color::srgb(1.0, 0.69, 0.43)
    } else {
        Color::srgb(1.0, 0.95, 0.87)
    }
}

pub(crate) fn sun_illuminance(scene: &IndoorManifest) -> f32 {
    if scene.domain().is_some() {
        return scene.daylight_lux;
    }
    match scene.lighting {
        LightingMood::Daylight => scene.daylight_lux,
        LightingMood::Overcast => scene.daylight_lux * 0.23,
        LightingMood::Evening => scene.daylight_lux * 0.06,
    }
}

#[cfg(test)]
mod lighting_tests {
    use super::*;

    #[test]
    fn spotlight_cone_integrates_to_declared_lumens() {
        for (inner, outer) in [(0.48_f32, 1.02_f32), (0.75, 1.35), (0.90, 1.40)] {
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
}
