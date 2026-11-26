//! Bounded, stratified ceiling circuits. Unequal spacing and staggered centres
//! break repeated room-axis grids while retaining coverage and clear mounts.
use super::*;

pub(super) fn yaw(scene: &IndoorManifest, position: Vec3) -> f32 {
    if scene.program.is_none() {
        return 0.;
    }
    let zone = scene
        .program
        .as_ref()
        .unwrap()
        .zones
        .iter()
        .position(|z| position.xz().cmpge(z.min).all() && position.xz().cmple(z.max).all())
        .map_or(0, |index| index as u64 + 1);
    let mut circuit = stream(scene.seed, 7593 + zone);
    let heading = circuit.random_range(-std::f32::consts::PI..std::f32::consts::PI);
    let spread = circuit.random_range(0.01..0.32);
    let key = position.x.to_bits() as u64 | (position.z.to_bits() as u64) << 32;
    let mut item = stream(scene.seed ^ key, 7594);
    heading + item.random_range(-spread..spread)
}

pub(super) fn rotation(scene: &IndoorManifest, position: Vec3) -> Quat {
    rotation_for(scene, position, fixtures::parameters(scene).shape)
}

pub(super) fn rotation_for(
    scene: &IndoorManifest,
    position: Vec3,
    shape: fixtures::FixtureShape,
) -> Quat {
    let yaw = Quat::from_rotation_y(yaw(scene, position));
    if scene.program.is_none() {
        return yaw;
    }
    let key = position.x.to_bits() as u64 | (position.z.to_bits() as u64) << 32;
    let mut r = stream(scene.seed ^ key, 0x41494d4544464958);
    let directional = matches!(
        shape,
        fixtures::FixtureShape::Downlight | fixtures::FixtureShape::Tube
    );
    // Pendants may be subtly tilted; adjustable downlights include aimed spots.
    let angle = if directional && r.random_bool(0.70) {
        r.random_range(0.08..0.62)
    } else if r.random_bool(0.35) {
        r.random_range(0.025..0.15)
    } else {
        0.
    };
    yaw * Quat::from_axis_angle(
        Vec3::new(r.random::<f32>(), 0., r.random::<f32>()).normalize(),
        angle,
    )
}

pub(super) fn half(scene: &IndoorManifest, position: Vec3) -> Vec3 {
    let size = size(scene, position);
    Mat3::from_quat(rotation(scene, position)).abs() * (size * 0.5)
}

/// Realized dimensions are keyed only by x/z, so roof/drop calculation and
/// final geometry use the same sample. Legacy fixtures retain their sizes.
pub(super) fn size(scene: &IndoorManifest, position: Vec3) -> Vec3 {
    size_from(scene, position, super::fixture_size(scene))
}

fn size_from(scene: &IndoorManifest, position: Vec3, nominal: Vec3) -> Vec3 {
    if scene.program.is_none() {
        return nominal;
    }
    let key = position.x.to_bits() as u64 | (position.z.to_bits() as u64) << 32;
    let mut r = stream(scene.seed ^ key, 7596);
    let scale = r.random_range(0.58..1.42);
    let aspect = r.random_range(0.80..1.25);
    nominal * Vec3::new(scale * aspect, r.random_range(0.70..1.40), scale / aspect)
}

/// Distance between mounts depends on the throw to the occupied plane. Blend
/// that estimate with the circuit's sampled axial pitches so narrow-beam and
/// decorative fittings, tall halls and low offices retain different layouts.
fn roof_cells(extent: Vec2, requested: Vec2, throw: f32, shape: fixtures::FixtureShape) -> UVec2 {
    use fixtures::FixtureShape::*;
    let (spacing_to_height, max_pitch) = match shape {
        Downlight => (0.95, 3.4),
        Louver => (1.12, 4.6),
        Panel | Tube => (1.25, 4.8),
        Drum | Ring => (1.40, 5.5),
    };
    let coverage = (throw * spacing_to_height).clamp(1.8, 6.5);
    let nominal = requested.element_product().sqrt();
    // A compact downlight cannot cover an arbitrarily large task area merely
    // because the ceiling is high. Large rooms can still have many fixtures.
    let pitch = requested * ((0.5 * (nominal + coverage)).min(max_pitch) / nominal);
    // Choose the closest row count on each axis. Always rounding up added a
    // whole row and column to every partition just over a multiple of its pitch.
    (extent / pitch).round().max(Vec2::ONE).as_uvec2()
}

fn field(min: Vec2, max: Vec2, count: UVec2, phase: Vec2, seed: u64, domain: u64) -> Vec<Vec2> {
    let mut r = stream(seed, domain);
    let extent = max - min;
    let pitch = extent / count.as_vec2();
    let phase = phase + Vec2::new(r.random_range(-0.05..0.05), r.random_range(-0.05..0.05));
    let jitter = r.random_range(0.16..0.34);
    let stagger = r.random_range(-0.22..0.22);
    let mut points = Vec::with_capacity((count.x * count.y) as usize);
    for x in 0..count.x {
        for z in 0..count.y {
            let base = Vec2::new(x as f32 + 0.5, z as f32 + 0.5)
                + phase
                + Vec2::X * if z % 2 == 0 { stagger } else { -stagger };
            // Truncate the proposal interval, not the sampled position: edge
            // cells keep their coverage without snapped/clamped edge mass.
            let lo = (-base + Vec2::splat(0.001)).max(Vec2::splat(-jitter));
            let hi = (count.as_vec2() - base - Vec2::splat(0.001)).min(Vec2::splat(jitter));
            if !base.is_finite() || lo.cmpge(hi).any() {
                continue;
            }
            let offset = Vec2::new(r.random_range(lo.x..hi.x), r.random_range(lo.y..hi.y));
            points.push(min + (base + offset) * pitch);
        }
    }
    points
}

pub(super) fn under_mezzanine(scene: &IndoorManifest) -> Vec<Vec3> {
    let Some(m) = scene.envelope.as_ref().and_then(|e| e.mezzanine.as_ref()) else {
        return vec![];
    };
    let mut positions = Vec::new();
    for p in field(
        m.deck.min,
        m.deck.max,
        ((m.deck.max - m.deck.min) / 2.5)
            .ceil()
            .max(Vec2::ONE)
            .as_uvec2(),
        Vec2::ZERO,
        scene.seed,
        7595,
    ) {
        let mut position = Vec3::new(p.x, 0., p.y);
        let body = half(scene, position);
        position.y = m.underside_height() - (body.y + 0.035);
        let half_xz = body.xz();
        if (p - half_xz).cmpgt(m.deck.min + Vec2::splat(0.03)).all()
            && (p + half_xz).cmplt(m.deck.max - Vec2::splat(0.03)).all()
        {
            positions.push(position);
        }
    }
    positions
}

pub(super) fn positions(scene: &IndoorManifest) -> Vec<Vec3> {
    let program = scene.program.as_ref().expect("procedural ceiling program");
    let fixture = fixtures::parameters(scene);
    let nominal = super::fixture_size(scene);
    let mut positions: Vec<Vec3> = Vec::new();
    let mut occupied: Vec<(Vec3, Vec3)> = Vec::new();
    let mut obstacles = super::super::floorplan::obstacles(scene);
    if let Some(e) = &scene.envelope {
        obstacles.extend(e.structural_boxes(scene.room_size));
    }
    for (index, zone) in program.zones.iter().enumerate() {
        let extent = zone.max - zone.min;
        let legacy = scene.generator_version < 152;
        let center = (zone.min + zone.max) * 0.5;
        let count = if legacy {
            (extent / program.light_spacing)
                .ceil()
                .max(Vec2::ONE)
                .as_uvec2()
        } else {
            let floor = scene.envelope.as_ref().map_or(0., |e| {
                e.mezzanine
                    .as_ref()
                    .filter(|m| m.deck.contains(center))
                    .map_or_else(|| e.floor_height(center), |m| m.deck.height)
            });
            let drop = if fixture.shape == fixtures::FixtureShape::Downlight {
                nominal.y * 0.5
            } else {
                program.light_drop
            };
            roof_cells(
                extent,
                program.light_spacing,
                scene.ceiling_height(center) - floor - drop - 0.75,
                fixture.shape,
            )
        };
        let mut install = |p: Vec2| {
            let mut position = Vec3::new(p.x, 0., p.y);
            let body_half = Mat3::from_quat(rotation_for(scene, position, fixture.shape)).abs()
                * (size_from(scene, position, nominal) * 0.5);
            let half = body_half.xz();
            if scene.envelope.as_ref().is_some_and(|e| {
                e.courtyard
                    .is_some_and(|roof| roof.overlaps(p - half, p + half))
                    || !super::super::envelope::polygon::box_inside(
                        &e.footprint,
                        p - half,
                        p + half,
                        0.15,
                    )
            }) {
                return false;
            }
            let mut roof = [
                p - half,
                p + half,
                p + Vec2::new(half.x, -half.y),
                p + Vec2::new(-half.x, half.y),
            ]
            .map(|p| scene.ceiling_height(p))
            .into_iter()
            .fold(f32::INFINITY, f32::min);
            if let Some(e) = &scene.envelope {
                for soffit in &e.soffits {
                    if soffit.overlaps(p - half, p + half) {
                        roof = roof.min(soffit.height);
                    }
                }
            }
            let drop = if fixture.shape == fixtures::FixtureShape::Downlight {
                body_half.y + 0.008
            } else {
                let key = p.x.to_bits() as u64 | (p.y.to_bits() as u64) << 32;
                let mut suspension = stream(scene.seed ^ key, 0x44524f5056415259);
                (program.light_drop * suspension.random_range(0.65..1.45))
                    .max(body_half.y + 0.012)
                    .min(
                        (roof
                            - scene.envelope.as_ref().map_or(0., |e| e.floor_height(p))
                            - 2.10
                            - body_half.y)
                            .max(body_half.y + 0.012),
                    )
            };
            let clear = super::super::program::court::clear_height_at(scene, p - half, p + half);
            position.y = roof - clear.map_or(drop, |h| drop.min(roof - h - body_half.y - 0.008));
            if clear.is_some() && position.y + body_half.y > roof - 0.005 {
                return false;
            }
            if obstacles.iter().any(|(lo, hi)| {
                (position - body_half).cmplt(*hi + Vec3::splat(0.004)).all()
                    && (position + body_half).cmpgt(*lo - Vec3::splat(0.004)).all()
            }) {
                return false;
            }
            // Adjacent zone circuits may approach their shared edge. Reject
            // overlapping bodies before fixtures become ceiling reservations.
            if occupied.iter().any(|(center, half)| {
                (position - *center)
                    .abs()
                    .cmplt(body_half + *half + Vec3::splat(0.08))
                    .all()
            }) {
                return false;
            }
            occupied.push((position, body_half));
            positions.push(position);
            true
        };
        let mut fitted = false;
        for p in field(
            zone.min,
            zone.max,
            count,
            program.light_phase,
            scene.seed,
            7600 + index as u64,
        ) {
            fitted |= install(p);
        }
        // A small chamfered or obstructed room can lose its single sampled
        // cell. Prefer a central clear mount for coverage in that case.
        if !legacy && !fitted {
            let mut candidates = field(
                zone.min,
                zone.max,
                UVec2::splat(3),
                Vec2::ZERO,
                scene.seed,
                0x4649585245545259 ^ index as u64,
            );
            candidates.sort_unstable_by(|a, b| {
                a.distance_squared(center)
                    .total_cmp(&b.distance_squared(center))
            });
            for p in candidates {
                if install(p) {
                    break;
                }
            }
        }
    }
    positions.extend(under_mezzanine(scene));
    append_neighbor_fixtures(scene, &mut positions);
    positions
}

/// A larger circulation area can have a second luminaire. Append the adjoining
/// mounts after all primary mounts so their circuit budgets stay independent.
fn append_neighbor_fixtures(scene: &IndoorManifest, positions: &mut Vec<Vec3>) {
    let front = scene.room_size.z * 0.5;
    let legacy = Vec3::new(
        0.,
        scene.ceiling_height(Vec2::new(0., front)) - 0.18,
        front + 1.5,
    );
    if scene.generator_version < 122 {
        positions.push(legacy);
        return;
    }
    let bay = scene.adjoining_space();
    let hx = scene.room_size.x * 0.5;
    let start = positions.len();
    let modern = scene.adjoining.is_some() && scene.envelope.is_some();
    let mut circuit = stream(scene.seed, 0x4241_5943_4952_4355);
    let count =
        1 + usize::from(modern && bay.area(scene.room_size.x) > circuit.random_range(28.0..61.0));
    for i in 0..count {
        let mut r = stream(scene.seed, 0x4241594c49474854 + i as u64);
        if let Some(p) = find_bay_mount(scene, front, hx, bay, &positions[start..], &mut r) {
            positions.push(p);
        } else if !modern {
            positions.push(legacy);
        }
    }
}

fn find_bay_mount(
    scene: &IndoorManifest,
    front: f32,
    hx: f32,
    bay: super::super::layout::AdjoiningSpace,
    installed: &[Vec3],
    r: &mut impl Rng,
) -> Option<Vec3> {
    for _ in 0..8 {
        let x = if r.random_bool(0.45) {
            scene.door_x + r.random_range(-0.24..0.24) * scene.room_size.x
        } else {
            r.random_range(-0.78..0.78) * hx
        };
        let p = Vec3::new(
            x,
            0.,
            front + r.random_range(0.22..0.78) * bay.depth_over(x, x),
        );
        let h = half(scene, p);
        if !bay.contains_bounds(scene.room_size, (p - h).xz(), (p + h).xz(), 0.12) {
            continue;
        }
        if installed.iter().any(|q| {
            let separation = p.xz().distance(q.xz());
            separation < 1.65 || separation < h.xz().length() + half(scene, *q).xz().length() + 0.15
        }) {
            continue;
        }
        let roof = scene.minimum_ceiling_height(p.xz() - h.xz(), p.xz() + h.xz());
        let y = roof - h.y - r.random_range(0.012..0.18);
        if y - h.y >= 2.10
            && !scene
                .objects
                .iter()
                .filter(|o| o.neighbor && o.solid)
                .any(|o| {
                    let (lo, hi) = o.bounds();
                    let center = p.with_y(y);
                    (center - h).cmplt(hi + Vec3::splat(0.03)).all()
                        && (center + h).cmpgt(lo - Vec3::splat(0.03)).all()
                })
        {
            return Some(p.with_y(y));
        }
    }
    None
}

#[cfg(test)]
mod bay_tests {
    use super::*;

    #[test]
    fn circuit_pitch_tracks_mounting_height_optics_and_available_floor_area() {
        use fixtures::FixtureShape::{Downlight, Panel, Ring};
        let spacing = Vec2::splat(3.);
        assert_eq!(
            roof_cells(Vec2::splat(6.3), spacing, 2.4, Panel),
            UVec2::splat(2)
        );
        assert_eq!(roof_cells(Vec2::splat(1.), spacing, 4., Panel), UVec2::ONE);
        let hall = Vec2::new(24., 16.);
        let low = roof_cells(hall, spacing, 2., Ring);
        let high = roof_cells(hall, spacing, 5.5, Ring);
        let downlights = roof_cells(hall, spacing, 5.5, Downlight);
        assert!(high.cmplt(low).all() && high.cmplt(downlights).all());
        assert!(high.x * high.y > 12);
        let axial = roof_cells(Vec2::splat(20.), Vec2::new(2.1, 4.8), 3., Panel);
        assert!(axial.x > axial.y);
    }

    #[test]
    fn archived_dense_lecture_hall_replays_its_installed_mounts() {
        let mut hall = IndoorManifest::generate_with_humans(
            75155,
            super::super::super::layout::IndoorLayout::Mixed,
            0.65,
            0,
            0.25,
        )
        .unwrap();
        let current = positions(&hall);
        assert_eq!(current, positions(&hall));
        let current_primary = super::super::primary_fixture_count(&hall, &current);
        hall.generator_version = 151;
        let archive = positions(&hall);
        assert_eq!(super::super::primary_fixture_count(&hall, &archive), 50);
        assert!((9..40).contains(&current_primary));
    }

    #[test]
    fn bay_mounts_vary_within_the_slab_and_keep_furniture_clear() {
        let mut xs = [false; 3];
        let mut depths = Vec::new();
        for seed in 0..48 {
            let scene = IndoorManifest::generate_with_humans(
                seed,
                super::super::super::layout::IndoorLayout::Mixed,
                0.65,
                0,
                0.,
            )
            .unwrap();
            let mut mounts = Vec::new();
            append_neighbor_fixtures(&scene, &mut mounts);
            let mut replay = Vec::new();
            append_neighbor_fixtures(&scene, &mut replay);
            assert_eq!(mounts, replay);
            assert!(mounts.len() <= 2);
            for p in mounts {
                let h = half(&scene, p);
                let hx = scene.room_size.x * 0.5;
                let front = scene.room_size.z * 0.5;
                assert!(p.x.abs() + h.x <= hx && p.z - h.z > front);
                assert!(scene.adjoining_space().contains_bounds(
                    scene.room_size,
                    (p - h).xz(),
                    (p + h).xz(),
                    0.
                ));
                assert!(
                    p.y + h.y <= scene.minimum_ceiling_height((p - h).xz(), (p + h).xz()) + 0.001
                );
                for o in scene.objects.iter().filter(|o| o.neighbor && o.solid) {
                    let (lo, hi) = o.bounds();
                    assert!(
                        !(p - h).cmplt(hi).all() || !(p + h).cmpgt(lo).all(),
                        "bay light overlaps {:?} in {seed}",
                        o.kind
                    );
                }
                xs[if p.x < -0.2 * hx {
                    0
                } else if p.x > 0.2 * hx {
                    2
                } else {
                    1
                }] = true;
                depths.push(p.z - front);
            }
        }
        assert!(xs.into_iter().all(|x| x));
        depths.sort_by(f32::total_cmp);
        assert!(depths.last().unwrap() - depths[0] > 1.);
    }
}
