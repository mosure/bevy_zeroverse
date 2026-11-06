//! Required furnishings can turn into a side bay when a front-facing product
//! cannot fit beside the reserved aisle. Keep the sampled product dimensions
//! and run every alternative through the ordinary physical admission gate.
use super::*;
use rand::seq::SliceRandom;

#[derive(Clone, Copy)]
struct Bay {
    min: Vec2,
    max: Vec2,
}

/// Configuration-space proposals for narrow bays missed by uniform retries.
/// Rectangles describe possible object centers, not accepted placements. Exact
/// floor support, footprint, door access and ceiling checks remain in public_fit.
pub(super) fn anchor_points(
    scene: &IndoorManifest,
    index: usize,
    size: Vec3,
    yaw: f32,
    rng: &mut ChaCha8Rng,
) -> Vec<Vec2> {
    let p = &scene.assembly_diagnostics.programs[index];
    let (s, c) = yaw.sin_cos();
    let half = Vec2::new(
        size.x * c.abs() + size.z * s.abs(),
        size.x * s.abs() + size.z * c.abs(),
    ) * 0.5;
    let domain = Bay {
        min: p.min + half + Vec2::splat(0.02),
        max: p.max - half - Vec2::splat(0.02),
    };
    if domain.min.cmpge(domain.max).any() {
        return Vec::new();
    }
    let floor = scene.floor_height((p.min + p.max) * 0.5);
    let mut obstacles: Vec<_> = scene
        .columns()
        .into_iter()
        .chain(
            scene
                .objects
                .iter()
                .filter(|o| o.solid && !o.neighbor)
                .map(IndoorObject::bounds),
        )
        .filter(|(lo, hi)| lo.y < floor + size.y && hi.y > floor)
        .map(|(lo, hi)| Bay {
            min: lo.xz() - half - Vec2::splat(0.04),
            max: hi.xz() + half + Vec2::splat(0.04),
        })
        .collect();
    obstacles.extend(
        scene
            .functional_programs()
            .iter()
            .flat_map(|p| &p.reservations)
            .map(|r| Bay {
                min: r.min - half - Vec2::splat(0.005),
                max: r.max + half + Vec2::splat(0.005),
            }),
    );
    let mut bays = vec![domain];
    for blocked in obstacles {
        let mut next = Vec::new();
        for bay in bays {
            let lo = bay.min.max(blocked.min);
            let hi = bay.max.min(blocked.max);
            if lo.cmpge(hi).any() {
                next.push(bay);
                continue;
            }
            for (min, max) in [
                (bay.min, Vec2::new(lo.x, bay.max.y)),
                (Vec2::new(hi.x, bay.min.y), bay.max),
                (Vec2::new(lo.x, bay.min.y), Vec2::new(hi.x, lo.y)),
                (Vec2::new(lo.x, hi.y), Vec2::new(hi.x, bay.max.y)),
            ] {
                if (max - min).min_element() > 0.0001 {
                    next.push(Bay { min, max });
                }
            }
        }
        // Even adversarial manifests cannot turn recovery into an unbounded
        // polygon arrangement. Discarding proposals never waives admission.
        if next.len() > 256 {
            next.sort_by(|a, b| {
                let area = |b: &Bay| (b.max - b.min).element_product();
                area(b).total_cmp(&area(a))
            });
            next.truncate(256);
        }
        bays = next;
    }
    bays.shuffle(rng);
    bays.into_iter()
        .take(64)
        .map(|b| {
            b.min
                + (b.max - b.min)
                    * Vec2::new(rng.random_range(0.15..0.85), rng.random_range(0.15..0.85))
        })
        .collect()
}

impl IndoorManifest {
    pub(super) fn recover_display_anchor(
        &mut self,
        index: usize,
        size: Vec3,
        rng: &mut ChaCha8Rng,
    ) {
        let p = &self.assembly_diagnostics.programs[index];
        let (min, max) = (p.min, p.max);
        for yaw in [FRAC_PI_2, -FRAC_PI_2] {
            let extent = Vec2::new(size.z, size.x);
            let span = max - min - extent - Vec2::splat(0.04);
            if span.min_element() < 0. {
                continue;
            }
            for uv in variation::recovery_points(6, rng) {
                let q = min + extent * 0.5 + Vec2::splat(0.02) + span * uv;
                if self
                    .public_fit(
                        index,
                        ObjectKind::DisplayCase,
                        Vec3::new(q.x, 0., q.y),
                        size,
                        yaw,
                        rng,
                    )
                    .is_some()
                {
                    return;
                }
            }
        }
    }
}
