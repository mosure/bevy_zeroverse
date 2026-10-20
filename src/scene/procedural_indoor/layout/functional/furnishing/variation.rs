//! Shared manufacture with continuously sampled, smaller instance differences.
//! Sample before admission: never perturb accepted geometry through its neighbors.
use super::*;

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
