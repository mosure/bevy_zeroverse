use super::*;
use rand::seq::index;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Facing {
    Front,
    Beside,
    Behind,
}

pub(super) struct Placement {
    pub(super) row: usize,
    pub(super) slot: usize,
    pub(super) sku: usize,
    pub(super) facing: Facing,
    pub(super) base: Vec3,
    pub(super) yaw: f32,
}

impl Gondola {
    pub(super) fn place(&self, mut put: impl FnMut(Placement)) {
        let mut r = stream(self.seed, u64::from_le_bytes(*b"ShopFace"));
        let rows = self.decks.len() * 2;
        // Reserve one facing of each print lot somewhere on the entire
        // fixture. With distinct rows no two reservations can conflict.
        let mut anchors = vec![None; rows];
        for (sku, row) in index::sample(&mut r, rows, self.products.len())
            .iter()
            .enumerate()
        {
            anchors[row] = Some((r.random_range(0..self.facings), sku));
        }
        for (row, anchor) in anchors.into_iter().enumerate() {
            let deck = self.decks[row / 2];
            let sign = if row % 2 == 0 { -1. } else { 1. };
            let axis = if r.random_bool(0.5) { -1. } else { 1. };
            let row_fill = (self.stock + r.random_range(-0.13..0.12)).clamp(0.15, 0.99);
            let mut first = 0;
            while first < self.facings {
                let end = (first + r.random_range(1..=5)).min(self.facings);
                let run_anchor = anchor.filter(|(slot, _)| (first..end).contains(slot));
                let sku = run_anchor.map_or_else(|| r.random_range(0..4), |(_, sku)| sku);
                let size = self.products[sku].1;
                let fill = if r.random_bool(0.15) {
                    row_fill * r.random_range(0.18..0.68)
                } else {
                    row_fill
                };
                // These bound the rotated rectangle at every sampled yaw.
                // Pitch also covers the narrowest installed fixture.
                let swing = ((self.pitch - 0.003 - size.x) / size.z).clamp(0., 0.15);
                let width = size.x + size.z * swing;
                let depth = size.z + size.x * swing;
                // Small cans and bottles can present two facings in a wide bay.
                // Spend the same two-copy budget either across or behind it.
                let pair_spacing = width + r.random_range(0.006..0.018);
                let paired = width + pair_spacing + 0.003 <= self.pitch
                    && r.random_bool(f64::from(0.14 + 0.58 * row_fill));
                let spread = if paired { pair_spacing } else { 0. };
                let footprint = width + spread;
                let step = (footprint + r.random_range(0.008..0.045)).min(self.pitch);
                let count = end - first;
                let spare = self.pitch * count as f32 - step * (count - 1) as f32 - footprint;
                let start = -self.pitch * self.facings as f32 * 0.5
                    + self.pitch * first as f32
                    + footprint * 0.5
                    + spare.max(0.) * r.random_range(0.03..0.97);
                let near = self.spine * 0.5 + depth * 0.5 + 0.005;
                let far = self.spine * 0.5 + deck.projection - depth * 0.5 - 0.008;
                let setback = if r.random_bool(0.24) {
                    r.random_range(0.20..0.76)
                } else {
                    r.random_range(0.01..0.15)
                };
                let front = far - (far - near) * setback;
                let gap = r.random_range(0.005..0.022);
                for lane in first..end {
                    if run_anchor.is_none_or(|(slot, _)| lane != slot)
                        && !r.random_bool(fill.into())
                    {
                        continue;
                    }
                    let center = start + step * (lane - first) as f32;
                    let spread = if paired && r.random_bool(0.5) {
                        -spread
                    } else {
                        spread
                    };
                    let x = axis * (center - spread * 0.5);
                    let z = (front - r.random_range(0.0..0.021)).max(near);
                    let slot = if axis > 0. {
                        lane
                    } else {
                        self.facings - 1 - lane
                    };
                    let heading = if sign > 0. { 0. } else { PI };
                    let mut emit = |x, z, facing, r: &mut ChaCha8Rng| {
                        put(Placement {
                            row,
                            slot,
                            sku,
                            facing,
                            base: Vec3::new(x, deck.top, sign * z),
                            yaw: heading + swing * r.random_range(-1.0..1.0),
                        })
                    };
                    emit(x, z, Facing::Front, &mut r);
                    if r.random_bool(0.53) {
                        if paired {
                            emit(axis * (center + spread * 0.5), z, Facing::Beside, &mut r);
                        } else if z - depth - gap >= near {
                            emit(x, z - depth - gap, Facing::Behind, &mut r);
                        }
                    }
                }
                first = end;
            }
        }
    }
}
