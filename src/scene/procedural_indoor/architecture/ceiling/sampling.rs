use super::*;
use rand::seq::SliceRandom;

/// Compute scene obstacles once, rather than allocating per proposal.
pub(super) struct Reservations {
    fixtures: Vec<(Vec3, Vec3)>,
    structure: Vec<(Vec3, Vec3)>,
    partitions: Vec<(Vec3, Vec3)>,
}
impl Reservations {
    pub(super) fn new(scene: &IndoorManifest) -> Self {
        Self {
            fixtures: fixture_bounds(scene),
            structure: scene
                .envelope
                .as_ref()
                .map_or_else(Vec::new, |e| e.structural_boxes(scene.room_size)),
            partitions: super::super::super::floorplan::obstacles(scene),
        }
    }
}

pub(super) fn fixture_bounds(scene: &IndoorManifest) -> Vec<(Vec3, Vec3)> {
    super::super::fixture_positions(scene)
        .into_iter()
        .map(|p| {
            let half = super::super::fixture_half_extents(scene, p) + Vec3::splat(0.04);
            (
                p - half,
                Vec3::new(
                    p.x + half.x,
                    scene.ceiling_height(p.xz()) + 0.02,
                    p.z + half.z,
                ),
            )
        })
        .collect()
}

pub(super) fn support_heights(scene: &IndoorManifest, center: Vec2, size: Vec2) -> [f32; 6] {
    let lo = center - size * 0.5;
    let hi = center + size * 0.5;
    let axis = if size.x > size.y { Vec2::X } else { Vec2::Y };
    let reach = axis * size.dot(axis) * 0.34;
    [
        lo,
        hi,
        Vec2::new(lo.x, hi.y),
        Vec2::new(hi.x, lo.y),
        center - reach,
        center + reach,
    ]
    .map(|p| scene.ceiling_height(p))
}

pub(crate) fn highest_floor(scene: &IndoorManifest, lo: Vec2, hi: Vec2) -> f32 {
    let corners = [lo, hi, Vec2::new(lo.x, hi.y), Vec2::new(hi.x, lo.y)];
    let h = corners
        .into_iter()
        .map(|p| scene.floor_height(p))
        .fold(f32::NEG_INFINITY, f32::max);
    // A rug can lie entirely between the query's corners. Its footprint is a
    // conservative reserve, consistent with fixture and furniture bounds.
    let h = scene
        .rug_supports
        .iter()
        .filter(|rug| lo.cmplt(rug.max).all() && hi.cmpgt(rug.min).all())
        .map(|rug| rug.top)
        .fold(h, f32::max);
    scene.envelope.as_ref().map_or(h, |e| {
        let h = e
            .floor_patches
            .iter()
            .filter(|p| p.overlaps(lo, hi))
            .map(|p| p.height)
            .fold(h, f32::max);
        // Mezzanine stairs are separate solids, not floor_patches. A narrow
        // flight can pass entirely inside this footprint without any corner
        // sampling its upper treads. Compute their maximum without allocating.
        e.mezzanine.as_ref().map_or(h, |m| {
            // floor_height intentionally queries the lower floor so objects
            // may occupy both storeys. Ceiling headroom needs the upper deck,
            // even before furniture or a rug is placed there (seed 65412811).
            let h = if m.deck.overlaps(lo, hi) {
                h.max(m.deck.height)
            } else {
                h
            };
            if lo.cmplt(m.stair_max).all() && hi.cmpgt(m.stair_min).all() {
                let axis = m.stair_axis;
                let tread = (m.stair_max[axis] - m.stair_min[axis]) / m.steps as f32;
                let steps = ((m.stair_max[axis] - lo[axis].max(m.stair_min[axis])) / tread)
                    .ceil()
                    .clamp(1., m.steps as f32);
                h.max(m.deck.height * steps / m.steps as f32)
            } else {
                h
            }
        })
    })
}

pub(super) fn admissible(
    scene: &IndoorManifest,
    m: &CeilingModule,
    reserved: &Reservations,
) -> bool {
    let (lo, hi) = m.body_bounds();
    let corners = [
        lo.xz(),
        hi.xz(),
        Vec2::new(lo.x, hi.z),
        Vec2::new(hi.x, lo.z),
    ];
    if lo.y < highest_floor(scene, lo.xz(), hi.xz()) + MIN_HEADROOM_M
        || corners
            .iter()
            .any(|&p| hi.y > scene.ceiling_height(p) - 0.002)
        || scene.envelope.as_ref().is_some_and(|e| {
            !super::super::super::envelope::polygon::box_inside(
                &e.footprint,
                lo.xz(),
                hi.xz(),
                // Window coverings occupy up to 20 cm inside the wall.
                if m.detail.is_some() { 0.24 } else { 0.16 },
            ) || e.soffits.iter().any(|p| {
                p.overlaps(lo.xz(), hi.xz())
                    && !(lo.xz().cmpge(p.min).all() && hi.xz().cmple(p.max).all())
            }) || e
                .mezzanine
                .as_ref()
                .is_some_and(|v| v.deck.overlaps(lo.xz(), hi.xz()))
        })
        || reserved
            .structure
            .iter()
            .any(|&b| overlap(m.bounds(), b, 0.))
        || reserved
            .partitions
            .iter()
            .any(|&b| overlap(m.bounds(), b, 0.025))
        || reserved
            .fixtures
            .iter()
            .any(|&b| overlap(m.bounds(), b, 0.035))
    {
        return false;
    }
    // One planar support field per module; splitting at soffit/roof breaks is
    // preferable to floating anchors or cutting the pipe into the roof.
    let roof = support_heights(scene, m.center.xz(), m.size.xz());
    let roof_max = roof.into_iter().fold(f32::NEG_INFINITY, f32::max);
    (m.support_top_m - (roof_max - 0.001)).abs() < 0.0001
        && roof_max - roof.into_iter().fold(f32::INFINITY, f32::min) < 0.32
}

pub(super) fn sample(scene: &IndoorManifest) -> CeilingInstallation {
    let mut rng = stream(scene.seed, 7220);
    let exposed = rng.random_bool(
        if scene.architecture_style == super::super::super::layout::ArchitectureStyle::Industrial {
            0.85
        } else {
            0.32
        },
    );
    let mut plan = CeilingInstallation {
        exposed_slab: exposed,
        modules: Vec::new(),
    };
    let reserved = Reservations::new(scene);
    let Some(program) = &scene.program else {
        return plan;
    };
    for zone in &program.zones {
        let lo = zone.min + Vec2::splat(0.28);
        let hi = zone.max - Vec2::splat(0.28);
        let extent = hi - lo;
        if extent.min_element() < 1.0 {
            continue;
        }
        let finish = rng.random_range(0..12);
        let decorative = if exposed {
            [
                CeilingSystem::Truss,
                CeilingSystem::Baffles,
                CeilingSystem::TimberSlats,
                CeilingSystem::OpenCell,
                CeilingSystem::Coffers,
            ][rng.random_range(0..5)]
        } else {
            [
                CeilingSystem::SuspendedTiles,
                CeilingSystem::SuspendedTiles,
                CeilingSystem::AcousticCloud,
                CeilingSystem::Baffles,
                CeilingSystem::TimberSlats,
                CeilingSystem::OpenCell,
                CeilingSystem::Coffers,
                CeilingSystem::Coffers,
            ][rng.random_range(0..8)]
        };
        // Shared rhythms and plenum height within a zone, with controlled changes
        // between zones. Short service runs may fit where a room-wide run cannot.
        let pitch = rng.random_range(1.7..3.4);
        let nx = (extent.x / pitch).ceil().clamp(1., 6.) as usize;
        let nz = (extent.y / pitch).ceil().clamp(1., 6.) as usize;
        let cell = extent / Vec2::new(nx as f32, nz as f32);
        let zone_gap = rng.random_range(0.06..0.42);
        let mut decoration = Vec::new();
        for index in 0..(nx * nz).min(24) {
            if !rng.random_bool(if exposed { 0.48 } else { 0.90 }) {
                continue;
            }
            let center =
                lo + cell * Vec2::new((index % nx) as f32 + 0.5, (index / nx) as f32 + 0.5);
            let shape = cell
                * if decorative == CeilingSystem::SuspendedTiles {
                    0.96
                } else {
                    rng.random_range(0.74..0.91)
                };
            let depth = match decorative {
                CeilingSystem::Truss => rng.random_range(0.25..0.64),
                CeilingSystem::Coffers => rng.random_range(0.18..0.48),
                CeilingSystem::AcousticCloud | CeilingSystem::SuspendedTiles => {
                    rng.random_range(0.10..0.18)
                }
                _ => rng.random_range(0.14..0.38),
            };
            decoration.push(module(
                scene, &mut rng, decorative, center, shape, depth, zone_gap, finish,
            ));
        }
        let axis = usize::from(rng.random_bool(0.5));
        let service_kinds: &[CeilingSystem] = if exposed {
            &[
                CeilingSystem::RoundDuct,
                CeilingSystem::RectangularDuct,
                CeilingSystem::SprinklerMain,
                CeilingSystem::Conduit,
                CeilingSystem::CableTray,
                CeilingSystem::Ventilation,
            ]
        } else {
            &[CeilingSystem::SprinklerMain, CeilingSystem::Ventilation]
        };
        let mut service_kinds = service_kinds.to_vec();
        service_kinds.shuffle(&mut rng);
        let lane_phase = rng.random_range(-0.18..0.18);
        for (lane, &system) in service_kinds.iter().enumerate() {
            if !rng.random_bool(if exposed { 0.85 } else { 0.65 }) {
                continue;
            }
            let mut shape = extent * rng.random_range(0.70..0.92);
            shape[1 - axis] = match system {
                CeilingSystem::RoundDuct | CeilingSystem::RectangularDuct => {
                    rng.random_range(0.48_f32..0.94)
                }
                CeilingSystem::SprinklerMain => rng.random_range(0.35..0.85),
                _ => rng.random_range(0.28..0.55),
            }
            .min(extent[1 - axis] * 0.28);
            if system == CeilingSystem::Ventilation {
                shape[axis] = rng.random_range(0.40_f32..1.20).min(extent[axis] * 0.40);
                shape[1 - axis] = rng
                    .random_range(0.32_f32..0.70)
                    .min(extent[1 - axis] * 0.40);
            }
            let mut center = (lo + hi) * 0.5;
            center[1 - axis] = lo[1 - axis]
                + extent[1 - axis] * (lane as f32 + 0.5 + lane_phase) / service_kinds.len() as f32;
            let depth = match system {
                CeilingSystem::RoundDuct | CeilingSystem::RectangularDuct => {
                    rng.random_range(0.28..0.62)
                }
                CeilingSystem::SprinklerMain => rng.random_range(0.14..0.36),
                _ => rng.random_range(0.12..0.25),
            };
            let gap = rng.random_range(0.035..0.32);
            // Concealed terminals mount against the ceiling. Only exposed
            // services suspend a plenum below it; carrying that long drop into
            // finished low rooms creates an implausibly hanging grille and
            // needlessly occupies usable camera headroom.
            let gap = if !exposed && system == CeilingSystem::Ventilation {
                0.006 + gap * 0.04
            } else {
                gap
            };
            // One deterministic retry budget; reuse the shape program while
            // shrinking a rejected run around columns/soffits/fixture supports.
            let mut candidate = module(scene, &mut rng, system, center, shape, depth, gap, finish);
            for _ in 0..3 {
                if accept(scene, &mut plan, candidate.clone(), &reserved) {
                    break;
                }
                shape[axis] *= 0.62;
                if shape[axis] < 0.65 {
                    break;
                }
                candidate = resize(scene, candidate, center, shape, depth, gap);
            }
        }
        for candidate in decoration {
            accept(scene, &mut plan, candidate, &reserved);
        }
    }
    plan
}

#[allow(clippy::too_many_arguments)]
fn module(
    scene: &IndoorManifest,
    rng: &mut impl Rng,
    system: CeilingSystem,
    center: Vec2,
    size: Vec2,
    depth: f32,
    gap: f32,
    finish: u8,
) -> CeilingModule {
    let pitch_m = match system {
        CeilingSystem::SuspendedTiles => rng.random_range(0.50..0.74),
        CeilingSystem::OpenCell => rng.random_range(0.18..0.42),
        CeilingSystem::Coffers => rng.random_range(0.65..1.5),
        _ => rng.random_range(0.16..0.34),
    };
    let m = CeilingModule {
        system,
        center: Vec3::ZERO,
        size: Vec3::ONE,
        support_top_m: 0.,
        pitch_m,
        thickness_m: rng.random_range(0.015..0.048),
        seed: rng.random(),
        detail: Some(CeilingDetail::sample(system, rng, finish)),
    };
    resize(scene, m, center, size, depth, gap)
}

fn resize(
    scene: &IndoorManifest,
    mut m: CeilingModule,
    center: Vec2,
    size: Vec2,
    depth: f32,
    gap: f32,
) -> CeilingModule {
    let roofs = support_heights(scene, center, size);
    let roof = roofs.into_iter().fold(f32::INFINITY, f32::min);
    // Keep real usable headroom instead of silently dropping all installations
    // in low rooms. Both body depth and plenum remain positive and recorded.
    let available =
        (roof - highest_floor(scene, center - size * 0.5, center + size * 0.5) - 2.36).max(0.10);
    let depth = depth.min((available - 0.025).max(0.10));
    let gap = gap.min((available - depth).max(0.008));
    m.center = Vec3::new(center.x, roof - gap - depth * 0.5, center.y);
    m.size = Vec3::new(size.x, depth, size.y);
    m.support_top_m = roofs.into_iter().fold(f32::NEG_INFINITY, f32::max) - 0.001;
    m
}

fn accept(
    scene: &IndoorManifest,
    plan: &mut CeilingInstallation,
    m: CeilingModule,
    reserved: &Reservations,
) -> bool {
    if plan.modules.len() >= 96
        || !admissible(scene, &m, reserved)
        || plan
            .modules
            .iter()
            .any(|other| overlap(other.bounds(), m.bounds(), 0.025))
    {
        return false;
    }
    plan.modules.push(m);
    true
}
