//! Physical presentation surfaces. Area and aspect are separate random draws;
//! installation clearance, rather than a catalogue of repeated widths, decides
//! whether the sampled panel fits its supporting wall.
use super::*;

pub(super) fn aspect(kind: ObjectKind, rng: &mut ChaCha8Rng) -> f32 {
    let (lo, hi): (f32, f32) = match kind {
        ObjectKind::Display => {
            // Most video panels retain plausible manufactured proportions;
            // portrait signage and ultrawide installations remain available.
            if rng.random_bool(0.23) {
                (0.50, 0.82)
            } else {
                (1.30, 2.65)
            }
        }
        ObjectKind::Whiteboard => (0.60, 3.8),
        _ => (0.30, 3.6),
    };
    rng.random_range(lo.ln()..hi.ln()).exp()
}

pub(crate) fn standalone_size(kind: ObjectKind, rng: &mut ChaCha8Rng) -> Vec3 {
    let a = aspect(kind, rng);
    let (low, high, max_height, depth): (f32, f32, f32, std::ops::Range<f32>) = match kind {
        ObjectKind::Display => (0.48, 1.65, 1.65, 0.035..0.10),
        ObjectKind::Whiteboard => (0.46, 1.85, 1.85, 0.07..0.16),
        _ => (0.22, 1.40, 1.95, 0.02..0.07),
    };
    let max_scale = high.min(max_height * a.sqrt());
    let scale = rng
        .random_range(low.min(max_scale * 0.8).ln()..max_scale.ln())
        .exp();
    Vec3::new(scale * a.sqrt(), scale / a.sqrt(), rng.random_range(depth))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::objects;
    #[test]
    fn physical_displays_span_sizes_aspects_and_keep_valid_geometry() {
        for kind in [
            ObjectKind::WallArt,
            ObjectKind::Whiteboard,
            ObjectKind::Display,
        ] {
            let mut min = Vec2::splat(f32::INFINITY);
            let mut max = Vec2::ZERO;
            for seed in 0..256 {
                let size = standalone_size(kind, &mut stream(seed, 811));
                assert_eq!(size, standalone_size(kind, &mut stream(seed, 811)));
                let stats = Vec2::new(size.x * size.y, size.x / size.y);
                min = min.min(stats);
                max = max.max(stats);
                let mut object = objects::audit::fixture(kind, seed);
                object.size = size;
                let assembly = objects::build_object(&object);
                objects::audit::check(&object, &assembly)
                    .unwrap_or_else(|e| panic!("{kind:?}/{seed}/{size:?}: {e}"));
            }
            assert!(
                max.x / min.x > 8.,
                "{kind:?}: insufficient area range {min:?}..{max:?}"
            );
            assert!(
                max.y / min.y > 4.,
                "{kind:?}: insufficient aspect range {min:?}..{max:?}"
            );
        }
    }
}
