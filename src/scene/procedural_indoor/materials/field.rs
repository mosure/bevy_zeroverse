//! Periodic substrate fields shared by mineral and coating programs.
use super::{hash, periodic_noise};
mod deposit;
mod periodic;
mod unit;
pub(super) use deposit::PreparedDeposit;
pub(super) use periodic::PreparedPeriodic;
pub(super) use unit::periodic_unit;

pub(super) fn smooth(t: f32) -> f32 {
    let t = t.clamp(0., 1.);
    t * t * (3. - 2. * t)
}
/// Integer, periodic feature counts derived from metres, not atlas resolution.
pub(super) fn frequency(period_m: f32, spacing_m: f32) -> u32 {
    (period_m / spacing_m).round().clamp(1., 192.) as u32
}

/// Approximate footprint integration for sparse round inclusions/air voids.
/// Features smaller than a texel preserve area instead of flashing at full contrast.
pub(super) fn disk(distance: f32, radius: f32, footprint: f32) -> f32 {
    let aa = footprint.max(0.002);
    let filtered = radius.max(aa);
    // The smooth radial edge integrates to pi * (filtered² + aa²/5).
    // Normalize it so even sub-texel inclusions keep their physical area.
    smooth((filtered + aa - distance) / (2. * aa)) * radius.powi(2)
        / (filtered.powi(2) + aa.powi(2) * 0.20)
}

pub(super) fn band(distance: f32, width: f32, footprint: f32) -> f32 {
    let aa = footprint.max(0.0001);
    let filtered = width.max(aa);
    smooth((filtered + aa - distance) / (2. * aa)) * width / filtered
}
pub(super) struct Cell {
    pub radius: f32,
    pub edge: f32,
    pub dye: f32,
    pub offset: [f32; 2],
    pub shape: f32,
}

/// Nearest feature only. Paint and mineral programs never consume the
/// distance to a second feature; keeping that field out prevents accidental
/// use of an omitted edge calculation.
pub(super) struct PrimaryCell {
    pub radius: f32,
    pub dye: f32,
    pub offset: [f32; 2],
    pub shape: f32,
}

impl From<Cell> for PrimaryCell {
    fn from(cell: Cell) -> Self {
        Self {
            radius: cell.radius,
            dye: cell.dye,
            offset: cell.offset,
            shape: cell.shape,
        }
    }
}
/// Jittered, wrapped cell centres. Colour and relief use the same cell identity.
pub(super) fn cellular([u, v]: [f32; 2], count: [u32; 2], seed: u64) -> Cell {
    let x = periodic_unit(u) * count[0] as f32;
    let y = periodic_unit(v) * count[1] as f32;
    let (ix, iy) = (x.floor() as i32, y.floor() as i32);
    let mut nearest = [f32::INFINITY; 2];
    let mut dye = 0.;
    let mut offset = [0.; 2];
    let mut shape = 0.;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let (cx, cy) = (ix + dx, iy + dy);
            let hx = cx.rem_euclid(count[0] as i32) as u32;
            let hy = cy.rem_euclid(count[1] as i32) as u32;
            let px = cx as f32 + 0.05 + 0.90 * hash(hx, hy, seed);
            let py = cy as f32 + 0.05 + 0.90 * hash(hx, hy, seed.wrapping_add(31));
            let d = (px - x).powi(2) + (py - y).powi(2);
            if d < nearest[0] {
                nearest = [d, nearest[0]];
                dye = hash(hx, hy, seed.wrapping_add(71));
                offset = [px - x, py - y];
                shape = hash(hx, hy, seed.wrapping_add(113));
            } else if d < nearest[1] {
                nearest[1] = d;
            }
        }
    }
    Cell {
        radius: nearest[0].sqrt(),
        edge: nearest[1].sqrt() - nearest[0].sqrt(),
        dye,
        offset,
        shape,
    }
}

/// Per-map absolute feature positions, with one neighbor halo. Each position
/// uses the original signed cell coordinate and original addition order;
/// translating a wrapped position would change f32 rounding at the seam.
pub(super) struct PreparedCellular {
    count: [u32; 2],
    seed: u64,
    attributes: Vec<[f32; 4]>,
}
impl PreparedCellular {
    pub(super) fn with_budget(count: [u32; 2], seed: u64, remaining: &mut usize) -> Option<Self> {
        if count.into_iter().any(|n| !(1..=192).contains(&n)) {
            return None;
        }
        let bytes = ((count[0] + 2) * (count[1] + 2)) as usize * std::mem::size_of::<[f32; 4]>();
        if bytes > *remaining {
            return None;
        }
        let prepared = Self::new(count, seed);
        // Account for Vec capacity, rather than assuming an allocator reports
        // precisely the requested capacity.
        let bytes = prepared.heap_bytes();
        if bytes > *remaining {
            return None;
        }
        *remaining -= bytes;
        Some(prepared)
    }

    pub(super) fn new(count: [u32; 2], seed: u64) -> Self {
        assert!(count.into_iter().all(|n| (1..=192).contains(&n)));
        let mut attributes = Vec::with_capacity(((count[0] + 2) * (count[1] + 2)) as usize);
        for cy in -1..=count[1] as i32 {
            for cx in -1..=count[0] as i32 {
                let x = cx.rem_euclid(count[0] as i32) as u32;
                let y = cy.rem_euclid(count[1] as i32) as u32;
                attributes.push([
                    cx as f32 + 0.05 + 0.90 * hash(x, y, seed),
                    cy as f32 + 0.05 + 0.90 * hash(x, y, seed.wrapping_add(31)),
                    hash(x, y, seed.wrapping_add(71)),
                    hash(x, y, seed.wrapping_add(113)),
                ]);
            }
        }
        Self {
            count,
            seed,
            attributes,
        }
    }
    fn heap_bytes(&self) -> usize {
        self.attributes.capacity() * std::mem::size_of::<[f32; 4]>()
    }
    #[cfg(test)]
    pub(super) fn bytes(&self) -> usize {
        self.heap_bytes()
    }
    pub(super) fn dyes(&self) -> impl ExactSizeIterator<Item = f32> + '_ {
        let [width, height] = self.count;
        (0..width * height).map(move |i| {
            self.attributes[((i / width + 1) * (width + 2) + i % width + 1) as usize][2]
        })
    }

    #[cfg(test)]
    pub(super) fn sample(&self, uv: [f32; 2]) -> Cell {
        self.select::<true, true>(uv).0
    }

    #[inline]
    pub(super) fn sample_edges(&self, uv: [f32; 2]) -> Cell {
        self.select::<true, false>(uv).0
    }

    #[inline]
    pub(super) fn sample_primary(&self, uv: [f32; 2]) -> PrimaryCell {
        self.select::<false, false>(uv).0.into()
    }

    #[inline]
    pub(super) fn sample_primary_indexed(&self, uv: [f32; 2]) -> (PrimaryCell, usize) {
        let (cell, index) = self.select::<false, true>(uv);
        (cell.into(), index)
    }

    #[inline]
    fn select<const EDGE: bool, const INDEX: bool>(&self, [u, v]: [f32; 2]) -> (Cell, usize) {
        let x = periodic_unit(u) * self.count[0] as f32;
        let y = periodic_unit(v) * self.count[1] as f32;
        let (ix, iy) = (x.floor() as i32, y.floor() as i32);
        // rem_euclid can round tiny negative inputs to exactly 1. Retain the
        // original evaluator if that puts the neighbor outside the halo.
        if !x.is_finite()
            || !y.is_finite()
            || ix < 0
            || iy < 0
            || ix >= self.count[0] as i32
            || iy >= self.count[1] as i32
        {
            return if INDEX {
                self.scalar_indexed([u, v])
            } else {
                (cellular([u, v], self.count, self.seed), 0)
            };
        }
        let mut nearest = [f32::INFINITY; 2];
        let mut dye = 0.;
        let mut offset = [0.; 2];
        let mut shape = 0.;
        let mut selected = 0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (cx, cy) = (ix + dx, iy + dy);
                let halo_index = ((cy + 1) as u32 * (self.count[0] + 2) + (cx + 1) as u32) as usize;
                let attributes = self.attributes[halo_index];
                let [px, py, _, _] = attributes;
                let d = (px - x).powi(2) + (py - y).powi(2);
                if d < nearest[0] {
                    nearest = [d, nearest[0]];
                    dye = attributes[2];
                    offset = [px - x, py - y];
                    shape = attributes[3];
                    if INDEX {
                        selected = halo_index;
                    }
                } else if EDGE && d < nearest[1] {
                    nearest[1] = d;
                }
            }
        }
        // Map the winning halo entry back to the original wrapped cell index
        // once, rather than wrapping all nine candidate cells per texel.
        let index = if INDEX {
            let pitch = self.count[0] as usize + 2;
            let hx = (selected % pitch) as i32 - 1;
            let hy = (selected / pitch) as i32 - 1;
            let wrap = |i: i32, n: u32| {
                if i < 0 {
                    n - 1
                } else if i == n as i32 {
                    0
                } else {
                    i as u32
                }
            };
            (wrap(hy, self.count[1]) * self.count[0] + wrap(hx, self.count[0])) as usize
        } else {
            0
        };
        (
            Cell {
                radius: nearest[0].sqrt(),
                edge: if EDGE {
                    nearest[1].sqrt() - nearest[0].sqrt()
                } else {
                    0.
                },
                dye,
                offset,
                shape,
            },
            index,
        )
    }

    fn scalar_indexed(&self, uv: [f32; 2]) -> (Cell, usize) {
        // The halo misses only the rare rounding case above. Compute its
        // original winner index independently so pigment lookup stays exact.
        let cell = cellular(uv, self.count, self.seed);
        let x = periodic_unit(uv[0]) * self.count[0] as f32;
        let y = periodic_unit(uv[1]) * self.count[1] as f32;
        let (ix, iy) = (x.floor() as i32, y.floor() as i32);
        let mut nearest = f32::INFINITY;
        let mut selected = 0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (cx, cy) = (ix + dx, iy + dy);
                let hx = cx.rem_euclid(self.count[0] as i32) as u32;
                let hy = cy.rem_euclid(self.count[1] as i32) as u32;
                let px = cx as f32 + 0.05 + 0.90 * hash(hx, hy, self.seed);
                let py = cy as f32 + 0.05 + 0.90 * hash(hx, hy, self.seed.wrapping_add(31));
                let d = (px - x).powi(2) + (py - y).powi(2);
                if d < nearest {
                    nearest = d;
                    selected = (hy * self.count[0] + hx) as usize;
                }
            }
        }
        (cell, selected)
    }
}
pub(super) fn deposit(uv: [f32; 2], count: [u32; 2], warp: f32, seed: u64) -> f32 {
    let [u, v] = uv;
    let x = u + (periodic_noise(u, v, 3, 5, seed) - 0.5) * warp;
    let y = v + (periodic_noise(u, v, 5, 3, seed.wrapping_add(17)) - 0.5) * warp;
    0.64 * periodic_noise(x, y, count[0], count[1], seed.wrapping_add(47))
        + 0.25 * periodic_noise(x, y, count[0] * 2, count[1] * 2, seed.wrapping_add(61))
        + 0.11 * periodic_noise(x, y, count[0] * 4, count[1] * 4, seed.wrapping_add(83))
}
pub(super) fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    // Pigmentation mixtures are interpolated in linear reflectance, not sRGB.
    [0, 1, 2].map(|i| {
        super::linear_to_srgb(
            super::srgb_to_linear(a[i]) * (1. - t) + super::srgb_to_linear(b[i]) * t,
        )
    })
}
pub(super) fn tint(c: [f32; 3], gain: f32) -> [f32; 3] {
    c.map(|c| super::linear_to_srgb((super::srgb_to_linear(c) * gain).clamp(0., 1.)))
}

#[cfg(test)]
mod tests {
    #[test]
    fn prepared_cell_attributes_preserve_positions_and_neighbor_selection() {
        for count in [[1, 1], [3, 5], [48, 48], [192, 192]] {
            for seed in [0, 43_084_584, u64::MAX] {
                let prepared = super::PreparedCellular::new(count, seed);
                let samples =
                    (0..32).flat_map(|y| (0..32).map(move |x| [x as f32 / 32., y as f32 / 32.]));
                for uv in samples.chain([
                    [-1. / 256., 1.],
                    [1., -1. / 256.],
                    [2.3, -0.4],
                    [-f32::EPSILON, f32::EPSILON],
                ]) {
                    let a = super::cellular(uv, count, seed);
                    let b = prepared.sample(uv);
                    assert_eq!(
                        [a.radius, a.edge, a.dye, a.offset[0], a.offset[1], a.shape]
                            .map(f32::to_bits),
                        [b.radius, b.edge, b.dye, b.offset[0], b.offset[1], b.shape]
                            .map(f32::to_bits),
                        "cell changed: count {count:?} seed {seed} uv {uv:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn filtered_inclusions_preserve_area_across_pixel_footprints() {
        let n = 256;
        let pitch = 2. / n as f32;
        for radius in [0.04_f32, 0.18, 0.32] {
            for footprint in [0.005, 0.05, 0.25] {
                let mut coverage = 0.;
                for y in 0..n {
                    for x in 0..n {
                        let u = -1. + (x as f32 + 0.5) * pitch;
                        let v = -1. + (y as f32 + 0.5) * pitch;
                        coverage += super::disk((u * u + v * v).sqrt(), radius, footprint);
                    }
                }
                let area = coverage * pitch * pitch;
                let expected = std::f32::consts::PI * radius * radius;
                assert!(
                    (area / expected - 1.).abs() < 0.03,
                    "inclusion coverage changed with filtering: r={radius} footprint={footprint} area={area}"
                );
            }
        }
    }
}

#[cfg(test)]
mod halo_replay;
