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
    let yaw = Quat::from_rotation_y(yaw(scene, position));
    if scene.program.is_none() {
        return yaw;
    }
    let key = position.x.to_bits() as u64 | (position.z.to_bits() as u64) << 32;
    let mut r = stream(scene.seed ^ key, 0x41494d4544464958);
    let directional = matches!(
        fixtures::parameters(scene).shape,
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
    let nominal = super::fixture_size(scene);
    if scene.program.is_none() {
        return nominal;
    }
    let key = position.x.to_bits() as u64 | (position.z.to_bits() as u64) << 32;
    let mut r = stream(scene.seed ^ key, 7596);
    let scale = r.random_range(0.58..1.42);
    let aspect = r.random_range(0.80..1.25);
    nominal * Vec3::new(scale * aspect, r.random_range(0.70..1.40), scale / aspect)
}

fn field(min: Vec2, max: Vec2, spacing: Vec2, phase: Vec2, seed: u64, domain: u64) -> Vec<Vec2> {
    let mut r = stream(seed, domain);
    let extent = max - min;
    let count = (extent / spacing).ceil().max(Vec2::ONE).as_uvec2();
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
        Vec2::splat(2.5),
        Vec2::ZERO,
        scene.seed,
        7595,
    ) {
        let mut position = Vec3::new(p.x, 0., p.y);
        position.y = m.underside_height() - (half(scene, position).y + 0.035);
        let half = half(scene, position).xz();
        if (p - half).cmpgt(m.deck.min + Vec2::splat(0.03)).all()
            && (p + half).cmplt(m.deck.max - Vec2::splat(0.03)).all()
        {
            positions.push(position);
        }
    }
    positions
}

pub(super) fn positions(scene: &IndoorManifest) -> Vec<Vec3> {
    let program = scene.program.as_ref().expect("procedural ceiling program");
    let fixture = fixtures::parameters(scene);
    let mut positions: Vec<Vec3> = Vec::new();
    let mut occupied: Vec<(Vec3, Vec3)> = Vec::new();
    let mut obstacles = super::super::floorplan::obstacles(scene);
    if let Some(e) = &scene.envelope {
        obstacles.extend(e.structural_boxes(scene.room_size));
    }
    for (index, zone) in program.zones.iter().enumerate() {
        for p in field(
            zone.min,
            zone.max,
            program.light_spacing,
            program.light_phase,
            scene.seed,
            7600 + index as u64,
        ) {
            let mut position = Vec3::new(p.x, 0., p.y);
            let body_half = half(scene, position);
            let half = body_half.xz();
            if scene.envelope.as_ref().is_some_and(|e| {
                !super::super::envelope::polygon::box_inside(&e.footprint, p - half, p + half, 0.15)
            }) {
                continue;
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
            position.y = roof - drop;
            if obstacles.iter().any(|(lo, hi)| {
                (position - body_half).cmplt(*hi + Vec3::splat(0.004)).all()
                    && (position + body_half).cmpgt(*lo - Vec3::splat(0.004)).all()
            }) {
                continue;
            }
            // Adjacent zone circuits may approach their shared edge. Reject
            // overlapping bodies before fixtures become ceiling reservations.
            if occupied.iter().any(|(center, half)| {
                (position - *center)
                    .abs()
                    .cmplt(body_half + *half + Vec3::splat(0.08))
                    .all()
            }) {
                continue;
            }
            occupied.push((position, body_half));
            positions.push(position);
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
