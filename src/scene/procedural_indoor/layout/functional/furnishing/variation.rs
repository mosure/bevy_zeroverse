//! Shared manufacture with continuously sampled, smaller instance differences.
//! Sample before admission: never perturb accepted geometry through its neighbors.
use super::*;
use rand::seq::SliceRandom;

/// Cover the usable region without giving the lower-left corner or exact grid
/// coordinates priority. Bounded allocation, only used on recovery paths.
pub(super) fn recovery_points(n: usize, r: &mut ChaCha8Rng) -> Vec<Vec2> {
    let mut points = Vec::with_capacity(n * n);
    for row in 0..n {
        for col in 0..n {
            points.push(Vec2::new(
                (col as f32 + r.random_range(0.04..0.96)) / n as f32,
                (row as f32 + r.random_range(0.04..0.96)) / n as f32,
            ));
        }
    }
    points.shuffle(r);
    points
}

pub(super) fn local_alternatives(r: &mut ChaCha8Rng) -> [Vec2; 7] {
    let phase = r.random_range(-PI..PI);
    std::array::from_fn(|i| {
        if i == 0 {
            Vec2::ZERO
        } else {
            let angle =
                phase + (i - 1) as f32 * std::f32::consts::TAU / 6. + r.random_range(-0.24..0.24);
            Vec2::from_angle(angle) * r.random_range(0.28..0.92)
        }
    })
}

pub(super) fn anchor_yaw(kind: ObjectKind, r: &mut ChaCha8Rng) -> f32 {
    let range = match kind {
        ObjectKind::Lectern => 0.34,
        ObjectKind::TeachingBoard => 0.16,
        ObjectKind::InformationStand => 0.50,
        ObjectKind::ExhibitPlinth | ObjectKind::ExhibitionPanel => 0.60,
        // Service fronts and built-in cores keep their operational orientation.
        _ => return 0.,
    };
    r.random_range(-range..range)
}

pub(super) struct Collection {
    chair: Vec3,
    desk: Vec3,
    table_height: f32,
    table_depth: f32,
    variation: f32,
}

impl Collection {
    pub(super) fn sample(seed: u64, index: usize) -> Self {
        let mut r = stream(seed, 0x5055_424c_5349_5a45 + index as u64);
        Self {
            chair: Vec3::new(
                r.random_range(0.64..0.76),
                r.random_range(0.92..1.13),
                r.random_range(0.64..0.76),
            ),
            desk: Vec3::new(
                r.random_range(0.88..1.12),
                r.random_range(0.71..0.79),
                r.random_range(0.51..0.64),
            ),
            table_height: r.random_range(0.71..0.79),
            table_depth: r.random_range(0.68..0.94),
            // Retain coordinated sets, without imposing one catalogue size on
            // every room. These represent different products, not warped meshes.
            variation: r.random_range(0.0..0.065),
        }
    }

    fn instance(&self, size: Vec3, r: &mut ChaCha8Rng) -> Vec3 {
        size * Vec3::new(
            1. + r.random_range(-1.0..1.0) * self.variation,
            1. + r.random_range(-0.35..0.35) * self.variation,
            1. + r.random_range(-1.0..1.0) * self.variation,
        )
    }

    pub(super) fn chair(&self, r: &mut ChaCha8Rng) -> Vec3 {
        self.instance(self.chair, r)
    }

    pub(super) fn desk(&self, r: &mut ChaCha8Rng) -> Vec3 {
        self.instance(self.desk, r)
    }

    pub(super) fn table(&self, width: f32, r: &mut ChaCha8Rng) -> Vec3 {
        self.instance(Vec3::new(width, self.table_height, self.table_depth), r)
    }
}
