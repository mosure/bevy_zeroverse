pub(crate) mod background;
pub mod ceiling;
pub mod details;
#[cfg(test)]
mod diversity_tests;
pub mod doors;
pub mod facade;
mod fixture_layout;
pub mod fixtures;
pub mod interior_glazing;
pub(crate) mod outdoors;
#[cfg(test)]
pub(crate) mod overlap_tests;
mod shared_openings;
pub mod trim;
use super::{
    layout::{stream, IndoorManifest, LightingMood, ObjectKind, NEIGHBOR_DEPTH},
    materials::Surface,
    objects::Assembly,
};
use crate::scene::procedural_indoor::photometry::source_color;
use bevy::{light::CascadeShadowConfigBuilder, prelude::*};
use rand::Rng;

pub fn architecture(scene: &IndoorManifest) -> Assembly {
    if scene.envelope.is_some() {
        return super::envelope::construction::build(scene);
    }
    let mut a = Assembly {
        motion_policy: scene.indoor_motion.as_ref().map(|m| m.policy.clone()),
        ..default()
    };
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
    let door_left = scene.door_x - scene.entry_width() * 0.5;
    let door_right = scene.door_x + scene.entry_width() * 0.5;
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
    if let Some(door) = &scene.entry_door {
        door.build(&mut a, true);
    } else {
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
    }

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
    let mut outside = Assembly::default();
    facade::backdrop(&mut outside, scene);
    for ((surface, label), g) in outside.parts {
        a.part(surface, &format!("{label}#exterior")).append(g);
    }
    background::append(&mut a, scene);
    a
}

pub(crate) fn fixture_geometry(a: &mut Assembly, scene: &IndoorManifest) {
    for (i, p) in fixture_positions(scene).into_iter().enumerate() {
        for (key, geometry) in fixtures::geometry(scene, i, p).parts {
            a.parts.entry(key).or_default().append(geometry);
        }
    }
}

pub(crate) fn fixture_size(scene: &IndoorManifest) -> Vec3 {
    if let Some(program) = &scene.program {
        let p = fixtures::parameters(scene);
        let mut r = stream(scene.seed, 0x4649585453495a45);
        let footprint = match p.shape {
            fixtures::FixtureShape::Ring => {
                // A ring is a luminaire in its own right, not a row of small
                // circles fitted into the narrow side of a fluorescent housing.
                let diameter = program.fixture_size.x * r.random_range(0.75..1.35);
                Vec2::splat(diameter.clamp(0.42, 1.90))
            }
            fixtures::FixtureShape::Drum => Vec2::splat(r.random_range(0.28..0.82)),
            fixtures::FixtureShape::Downlight => Vec2::splat(r.random_range(0.12..0.34)),
            _ => program.fixture_size,
        };
        return Vec3::new(footprint.x, p.depth_m, footprint.y);
    }
    match scene.lighting_design % 3 {
        0 => Vec3::new(0.60, 0.055, 0.60),
        1 => Vec3::new(1.45, 0.055, 0.17),
        _ => Vec3::new(0.22, 0.055, 0.22),
    }
}

/// Close-mounted luminaires below the deck prevent the upper floor from
/// shadowing every light in the lower work area. Their anchors stop at the slab.
pub(crate) fn fixture_half_extents(scene: &IndoorManifest, position: Vec3) -> Vec3 {
    ceiling::CeilingGrid::lamp_at(scene, position).map_or_else(
        || fixture_layout::half(scene, position),
        |grid| grid.lamp_size() * 0.5,
    )
}

pub(crate) fn under_mezzanine_fixtures(scene: &IndoorManifest) -> Vec<Vec3> {
    fixture_layout::under_mezzanine(scene)
}

pub fn fixture_positions(scene: &IndoorManifest) -> Vec<Vec3> {
    if scene.program.is_some() {
        let mut positions = fixture_layout::positions(scene);
        if let Some(grid) = ceiling::CeilingGrid::for_scene(scene) {
            grid.fit_positions(scene.room_size.y, &mut positions);
        }
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

/// Actual fixture bodies and their vertical suspensions, without mesh builds.
pub(crate) fn fixture_obstacles(scene: &IndoorManifest) -> impl Iterator<Item = (Vec3, Vec3)> + '_ {
    fixture_positions(scene).into_iter().map(|p| {
        let half = fixture_half_extents(scene, p) + Vec3::splat(0.015);
        let roof = scene
            .envelope
            .as_ref()
            .and_then(|e| e.mezzanine.as_ref())
            .filter(|m| m.deck.contains(p.xz()) && p.y < m.deck.height)
            .map_or_else(
                || {
                    [-1., 1.]
                        .into_iter()
                        .flat_map(|x| {
                            [-1., 1.].map(|z| {
                                scene.ceiling_height((p + half * Vec3::new(x, 0., z)).xz())
                            })
                        })
                        .fold(p.y + half.y, f32::max)
                },
                |m| m.deck.height - m.thickness,
            );
        (p - half, (p + half).with_y(roof + 0.02))
    })
}

pub fn spawn_lights(
    scene: &IndoorManifest,
    quality: super::IndoorQuality,
    root: Entity,
    commands: &mut Commands,
) {
    let positions = fixture_positions(scene);
    for (i, p, c, lumens) in powered_ceiling_fixtures(scene, &positions) {
        let (inner, outer) = fixtures::beam_angles(scene, i, p);
        commands.spawn((
            Name::new(format!("indoor_luminaire_{i}")),
            SpotLight {
                color: source_color(scene, c),
                // Bevy divides spot intensity by 4π before applying a squared
                // angular falloff. Normalize that cone so these are fixture lumens.
                intensity: spot_intensity_for_lumens(lumens, inner, outer),
                range: 13.0,
                // Match the finite highlight footprint to the emitting panel.
                // This remains Bevy's finite-radius spot approximation.
                radius: (fixtures::emitting_area_at(scene, p) / std::f32::consts::PI).sqrt() * 0.5,
                inner_angle: inner,
                outer_angle: outer,
                // Every emitting source must respect walls/decks. Truncating
                // this at eight leaked later fixtures through solid partitions.
                shadow_maps_enabled: quality.shadows(),
                shadow_depth_bias: 0.015,
                // Bevy expresses normal bias in shadow texels, not metres.
                // Sub-texel offsets caused broad self-shadowing moire on walls.
                shadow_normal_bias: SpotLight::DEFAULT_SHADOW_NORMAL_BIAS,
                ..default()
            },
            Transform::from_translation(fixtures::emitter(scene, p)).looking_to(
                fixtures::direction(scene, p),
                fixtures::rotation(scene, p) * Vec3::Z,
            ),
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
            // Deterministic contact-hardening shadow filtering. Cameras retain
            // Gaussian/non-temporal filtering so capture is independent of FPS
            // and history; this is a raster approximation, not a solar-disk tracer.
            #[cfg(not(target_arch = "wasm32"))]
            soft_shadow_size: quality.shadows().then_some(4.0),
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
        let power = super::photometry::lamp(scene, lamp);
        if power.lumens == 0. {
            continue;
        }
        let rest = Transform::from_translation(
            crate::scene::procedural_indoor::objects::standing_lamp::emitter(lamp),
        );
        let mut entity = commands.spawn((
            Name::new(format!("indoor_floor_lamp_{}", lamp.id)),
            PointLight {
                color: source_color(scene, power.color),
                intensity: power.lumens,
                range: 5.0,
                radius: 0.035,
                shadow_maps_enabled: quality.shadows(),
                shadow_depth_bias: 0.01,
                shadow_normal_bias: PointLight::DEFAULT_SHADOW_NORMAL_BIAS,
                ..default()
            },
            rest,
            ChildOf(root),
        ));
        if let Some(motion) = super::motion::AnimatedLight::for_object(scene, lamp.id, rest) {
            entity.insert(motion);
        }
    }
    for (lamp, range) in super::photometry::spot_fixtures(scene) {
        let (beam, power) = object_fixture_photometry(scene, lamp);
        if power.lumens == 0. {
            continue;
        }
        let rest = Transform::from_translation(beam.position).looking_to(beam.direction, Vec3::Z);
        let mut entity = commands.spawn((
            Name::new(if lamp.kind == ObjectKind::ExtractionHood {
                format!("indoor_extraction_hood_{}", lamp.id)
            } else {
                format!("indoor_desk_lamp_{}", lamp.id)
            }),
            SpotLight {
                color: source_color(scene, power.color),
                intensity: spot_intensity_for_lumens(power.lumens, beam.inner, beam.outer),
                range,
                radius: beam.radius,
                shadow_map_near_z: if matches!(
                    lamp.kind,
                    ObjectKind::DeskLamp | ObjectKind::ExtractionHood
                ) {
                    0.012
                } else {
                    SpotLight::DEFAULT_SHADOW_MAP_NEAR_Z
                },
                inner_angle: beam.inner,
                outer_angle: beam.outer,
                shadow_maps_enabled: quality.shadows(),
                shadow_depth_bias: 0.01,
                ..default()
            },
            rest,
            ChildOf(root),
        ));
        if let Some(motion) = super::motion::AnimatedLight::for_object(scene, lamp.id, rest) {
            entity.insert(motion);
        }
    }
    for (id, slot, light) in super::objects::equipment::signals::strings::practical_lights(scene) {
        let name = Name::new(format!("indoor_string_{id}_{slot}"));
        let intensity = light.bevy_intensity();
        // Shallow wall sockets can sit just centimetres from their backing.
        let shadow_map_near_z = (0.25 * light.radius).clamp(0.002, 0.012);
        let rest = Transform::from_translation(light.position);
        if let Some([inner, outer]) = light.cone {
            commands.spawn((
                name,
                SpotLight {
                    color: light.color,
                    intensity,
                    range: light.range,
                    radius: light.radius,
                    inner_angle: inner,
                    outer_angle: outer,
                    shadow_maps_enabled: quality.shadows(),
                    shadow_depth_bias: 0.008,
                    shadow_map_near_z,
                    ..default()
                },
                rest.looking_to(light.axis, Vec3::Z),
                ChildOf(root),
            ));
        } else {
            commands.spawn((
                name,
                PointLight {
                    color: light.color,
                    intensity,
                    range: light.range,
                    radius: light.radius,
                    shadow_maps_enabled: quality.shadows(),
                    shadow_depth_bias: 0.008,
                    shadow_map_near_z,
                    ..default()
                },
                rest,
                ChildOf(root),
            ));
        }
    }
}

pub(crate) fn spot_intensity_for_lumens(lumens: f32, inner: f32, outer: f32) -> f32 {
    let angular_integral = 1.0 - inner.cos() + (inner.cos() - outer.cos()) / 3.0;
    lumens * 2.0 / angular_integral
}

/// Native lighting and diffuse transport consume the same sampled emitter.
pub(crate) fn object_fixture_photometry(
    scene: &IndoorManifest,
    lamp: &super::layout::IndoorObject,
) -> (
    super::photometry::FixtureBeam,
    super::photometry::SourcePower,
) {
    if lamp.kind == ObjectKind::ExtractionHood {
        let deck = super::objects::public::service::hood::FilterDeck::installed(lamp);
        return (deck.beam(scene, lamp), deck.power(scene, lamp));
    }
    let beam = if lamp.kind == ObjectKind::DeskLamp {
        super::objects::accessories::lamps::beam(lamp)
    } else {
        super::objects::equipment::fixture_beam(lamp)
    };
    (beam, super::photometry::lamp(scene, lamp))
}

#[cfg(test)]
pub(crate) fn fixture_photometry(scene: &IndoorManifest, index: usize) -> (Vec3, f32) {
    fixture_photometry_at(scene, index, &fixture_positions(scene))
}

pub(crate) fn fixture_photometry_at(
    scene: &IndoorManifest,
    index: usize,
    positions: &[Vec3],
) -> (Vec3, f32) {
    let mut rng = stream(super::photometry::lighting_seed(scene), 164 + index as u64);
    let lamp_offset = rng.random_range(-1000.0..1000.0);
    let mut kelvin = scene.light_kelvin + lamp_offset;
    let primary = primary_fixture_count(scene, positions);
    let base = if index >= primary && scene.adjoining.is_some() && scene.envelope.is_some() {
        let area = scene.adjoining_space().area(scene.room_size.x);
        let count = (positions.len() - primary).max(1) as f32;
        // Circulation has a lower design illuminance than the occupied work area.
        (0.55 * scene.target_lux * area / (0.70 * count)).clamp(0.1, 12_000.)
    } else {
        fixture_lumens_for_count(scene, primary)
    };
    let mut flux = base * rng.random_range(0.50_f32..1.42).sqrt();
    if let Some(d) = scene.domain() {
        let position = positions[index];
        let uv = Vec2::new(
            position.x / scene.room_size.x,
            position.z / scene.room_size.z,
        ) * 2.0;
        flux *= (1.0 + d.photometry.fixture_gradient.dot(uv)).clamp(0.25, 1.75);
        if let Some(circuit) = d
            .photometry
            .ceiling_circuits
            .as_ref()
            .and_then(|c| c.for_mount(index))
        {
            // An installation shares a lamp batch; a small replacement-lamp tail
            // can be visibly warmer or cooler than the other heads on its switch.
            let spread = if rng.random_bool(0.075) { 0.92 } else { 0.12 };
            kelvin = scene.light_kelvin
                + circuit.batch_kelvin_offset
                + spread * lamp_offset
                + 0.45 * d.photometry.temperature_gradient * uv.x;
            let tint = circuit.batch_tint + rng.random_range(-0.008..0.008);
            return (
                d.photometry.illuminant_model.lamp_srgb(kelvin, tint),
                flux * circuit.dimmer,
            );
        }
        kelvin += d.photometry.temperature_gradient * uv.x;
        // Archived installations retain their per-mount switch and tint sequence.
        if index != energized_fixture(scene, primary)
            && !rng.random_bool(d.photometry.active_fraction as f64)
        {
            flux = 0.0;
        } else {
            flux *= 1.0 - d.photometry.circuit_contrast * rng.random_range(0.0..1.0);
        }
    }
    (
        scene
            .illuminant_model()
            .lamp_srgb(kelvin, rng.random_range(-0.045..0.045)),
        flux,
    )
}

/// The mount index also names its mesh, material and annotation. Dormant
/// fixtures keep those identities but contribute no analytic light or shadow.
pub(super) fn powered_ceiling_fixtures<'a>(
    scene: &'a IndoorManifest,
    positions: &'a [Vec3],
) -> impl Iterator<Item = (usize, Vec3, Vec3, f32)> + 'a {
    positions.iter().copied().enumerate().filter_map(|(i, p)| {
        let (color, lumens) = fixture_photometry_at(scene, i, positions);
        (lumens > 0.).then_some((i, p, color, lumens))
    })
}

pub(super) fn primary_fixture_count(scene: &IndoorManifest, positions: &[Vec3]) -> usize {
    positions.len()
        - positions
            .iter()
            .rev()
            .take_while(|p| p.z > scene.room_size.z * 0.5)
            .count()
}

pub(super) fn energized_fixture(scene: &IndoorManifest, count: usize) -> usize {
    if scene.domain().is_none_or(|d| d.photometry.meter.is_none()) {
        return 0;
    }
    stream(super::photometry::lighting_seed(scene), 0x454e455247495a45)
        .random_range(0..count.max(1))
}

pub(crate) fn fixture_lumens(scene: &IndoorManifest) -> f32 {
    fixture_lumens_for_count(
        scene,
        primary_fixture_count(scene, &fixture_positions(scene)),
    )
}
fn fixture_lumens_for_count(scene: &IndoorManifest, count: usize) -> f32 {
    // Lumen-method design estimate (not a simulated lux measurement): E =
    // N*flux*utilization/area. More fixtures maintain plausible office lighting
    // as room area changes. Evening retains a slightly lower occupied level.
    let illuminance = scene.target_lux;
    let area = fixture_floor_area(scene);
    let count = count as f32;
    (illuminance * area / (count.max(1.0) * 0.70)).clamp(0.1, 16000.0)
}

pub(super) fn fixture_floor_area(scene: &IndoorManifest) -> f32 {
    scene
        .envelope
        .as_ref()
        .map_or(scene.room_size.x * scene.room_size.z, |e| {
            super::envelope::polygon::area(&e.footprint)
                + e.mezzanine
                    .as_ref()
                    .map_or(0., |m| (m.deck.max - m.deck.min).element_product())
        })
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
        let rgb = d
            .photometry
            .illuminant_model
            .untinted_srgb(d.photometry.sun_kelvin);
        return source_color(scene, rgb);
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
    fn all_ceiling_emitters_keep_occlusion_beyond_eight_lights() {
        let scene = (0..64)
            .map(|seed| {
                IndoorManifest::generate_with_humans(
                    seed,
                    super::super::layout::IndoorLayout::Mixed,
                    0.65,
                    0,
                    0.,
                )
                .unwrap()
            })
            .find(|scene| {
                let mounts = fixture_positions(scene);
                let on = powered_ceiling_fixtures(scene, &mounts).count();
                on > 8 && on < mounts.len()
            })
            .unwrap();
        let mut world = World::new();
        let root = world.spawn_empty().id();
        let mut queue = bevy::ecs::world::CommandQueue::default();
        spawn_lights(
            &scene,
            super::super::IndoorQuality::Auto,
            root,
            &mut Commands::new(&mut queue, &world),
        );
        queue.apply(&mut world);
        let mut q = world.query::<(&Name, &SpotLight, &Transform)>();
        let ceiling: std::collections::BTreeMap<_, _> = q
            .iter(&world)
            .filter_map(|(name, light, tf)| {
                let index = name
                    .strip_prefix("indoor_luminaire_")?
                    .parse::<usize>()
                    .unwrap();
                Some((index, (light, tf)))
            })
            .collect();
        let mounts = fixture_positions(&scene);
        for (i, &p) in mounts.iter().enumerate() {
            let (_, lm) = fixture_photometry_at(&scene, i, &mounts);
            if lm == 0. {
                assert!(!ceiling.contains_key(&i));
                let geometry = fixtures::geometry(&scene, i, p);
                assert!(geometry.parts.keys().any(|(s, _)| *s == Surface::Light));
            } else {
                let (light, tf) = ceiling[&i];
                let (inner, outer) = fixtures::beam_angles(&scene, i, p);
                assert_eq!(light.intensity, spot_intensity_for_lumens(lm, inner, outer));
                assert_eq!(tf.translation, fixtures::emitter(&scene, p));
                assert!(light.shadow_maps_enabled);
            }
        }
        assert!(ceiling.len() > 8 && ceiling.len() < mounts.len());
    }

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

    #[test]
    fn desk_lamps_keep_the_swept_arm_valid_at_compact_workstations() {
        use super::super::objects::{audit, build_object};

        // Conference seed 57304 seats the rear pin of a tilted puck close to
        // the base. A longitudinal bow used to fold the tube inside itself.
        let mut lamp = audit::fixture(ObjectKind::DeskLamp, 11_291_326_550_121_891_613);
        lamp.size = Vec3::new(0.28978533, 0.31738603, 0.27421623);
        audit::check(&lamp, &build_object(&lamp)).unwrap();
        for seed in 0..128 {
            let mut r = stream(seed, 0x5441_534b_4e45_434b);
            lamp.seed = seed;
            lamp.size = Vec3::new(
                r.random_range(0.13..0.33),
                r.random_range(0.29..0.54),
                r.random_range(0.14..0.32),
            );
            audit::check(&lamp, &build_object(&lamp)).unwrap();
        }
    }

    #[test]
    fn desk_lamps_have_seeded_shadowed_downward_emitters_at_the_diffuser() {
        let scene = IndoorManifest::generate_with_humans(
            0,
            super::super::layout::IndoorLayout::Mixed,
            0.65,
            0,
            0.,
        )
        .unwrap();
        let mut lamp = scene.objects[0].clone();
        lamp.kind = ObjectKind::DeskLamp;
        lamp.size = Vec3::new(0.23, 0.41, 0.24);
        lamp.position = Vec3::new(1., 0.8, -1.);
        lamp.yaw = 0.7;
        let mut heads = std::collections::BTreeSet::new();
        let mut tilts = Vec::new();
        for seed in 0..96 {
            lamp.seed = seed;
            let (beam, power) = object_fixture_photometry(&scene, &lamp);
            assert_eq!((beam, power), object_fixture_photometry(&scene, &lamp));
            let (mut area, mut centroid) = (0., Vec3::ZERO);
            let world = lamp.transform();
            let mesh = super::super::objects::build_object(&lamp);
            for ((surface, _), g) in mesh.parts {
                if surface != Surface::Light {
                    continue;
                }
                for t in g.indices.as_chunks::<3>().0 {
                    let [a, b, c] =
                        t.map(|i| world.transform_point(Vec3::from_array(g.positions[i as usize])));
                    let cross = (b - a).cross(c - a);
                    assert!(cross.normalize().dot(beam.direction) > 0.99999);
                    let face = cross.length() * 0.5;
                    centroid += (a + b + c) * (face / 3.);
                    area += face;
                }
            }
            assert!((centroid / area + beam.direction * 0.003).distance(beam.position) < 1e-5);
            assert!(
                (area - super::super::objects::accessories::lamps::emitting_area(&lamp)).abs()
                    < 1e-6
            );
            let tilt = beam.tilt_from_down_degrees();
            assert!(tilt <= 47.01 && beam.inner < beam.outer && beam.radius > 0.);
            heads.insert(super::super::objects::accessories::lamp_parameters(&lamp).head);
            tilts.push(tilt);
        }
        assert_eq!(heads.len(), 4);
        assert!(tilts.iter().any(|&t| t < 0.01) && tilts.iter().any(|&t| t > 40.));
        let mut scene = scene;
        scene.objects = vec![lamp.clone()];
        let mut world = World::new();
        let root = world.spawn_empty().id();
        spawn_lights(
            &scene,
            super::super::IndoorQuality::Auto,
            root,
            &mut world.commands(),
        );
        world.flush();
        let sun = world.query::<&DirectionalLight>().single(&world).unwrap();
        assert!(sun.shadow_maps_enabled);
        #[cfg(not(target_arch = "wasm32"))]
        assert_eq!(sun.soft_shadow_size, Some(4.0));
        let (_, light, transform) = world
            .query::<(&Name, &SpotLight, &Transform)>()
            .iter(&world)
            .find(|(name, _, _)| name.as_str() == format!("indoor_desk_lamp_{}", lamp.id))
            .unwrap();
        assert!(light.shadow_maps_enabled);
        let (beam, power) = object_fixture_photometry(&scene, &lamp);
        assert!(Vec3::from(transform.forward()).distance(beam.direction) < 1e-6);
        assert_eq!(light.radius, beam.radius);
        assert!(transform.translation.distance(beam.position) < 1e-6);
        assert_eq!(
            light.intensity,
            spot_intensity_for_lumens(power.lumens, beam.inner, beam.outer)
        );
    }
}
