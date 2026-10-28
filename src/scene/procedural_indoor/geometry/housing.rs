//! Closed rounded-section lofts for manufactured shells and storage bodies.
use super::*;
const N: usize = 32;
impl Geometry {
    /// Each ring is (height, half-width/depth). Caps and side normals are separate.
    pub fn housing(&mut self, rings: &[(f32, Vec2)], exponent: f32, tf: Transform) {
        self.housing_sections(rings, exponent, tf, true);
    }

    /// Closed cross-section for a hollow rim. Repeat the first section last;
    /// unlike a solid housing, no disk caps fill or overlap its aperture.
    pub fn housing_loop(&mut self, rings: &[(f32, Vec2)], exponent: f32, tf: Transform) {
        debug_assert_eq!(rings.first(), rings.last());
        self.housing_sections(rings, exponent, tf, false);
    }

    fn housing_sections(
        &mut self,
        rings: &[(f32, Vec2)],
        exponent: f32,
        tf: Transform,
        caps: bool,
    ) {
        // Every ring shares its angular profile. Evaluate trig/powers once per
        // side, not again for each vertex, tangent and vertical normal sample.
        // Keep the same arithmetic order when applying metric ring dimensions.
        let outline: [Vec2; N] = std::array::from_fn(|i| {
            let t = TAU * i as f32 / N as f32;
            let f = |v: f32| {
                if v.abs() < 1e-6 {
                    0.
                } else {
                    v.signum() * v.abs().powf(2. / exponent)
                }
            };
            Vec2::new(f(t.cos()), f(t.sin()))
        });
        let point = |i: usize, ring: (f32, Vec2)| {
            let unit = outline[i % N];
            Vec3::new(ring.1.x * unit.x, ring.0, ring.1.y * unit.y)
        };
        for pair in rings.windows(2) {
            for i in 0..N {
                let p = point(i, pair[0]);
                let q = point(i, pair[1]);
                let r = point(i + 1, pair[1]);
                let s = point(i + 1, pair[0]);
                let start = self.positions.len() as u32;
                for (v, j, ring) in [
                    (p, i, pair[0]),
                    (q, i, pair[1]),
                    (r, i + 1, pair[1]),
                    (s, i + 1, pair[0]),
                ] {
                    let tangent = point(j + 1, ring) - point(j + N - 1, ring);
                    let vertical = point(j, pair[1]) - point(j, pair[0]);
                    self.vertex(
                        v,
                        vertical.cross(tangent).normalize(),
                        Vec2::new(j as f32 / N as f32 * TAU * pair[0].1.max_element(), v.y),
                        &tf,
                    );
                }
                self.indices
                    .extend([start, start + 1, start + 2, start, start + 2, start + 3]);
            }
        }
        if !caps {
            return;
        }
        for (ring, sign) in [(rings[0], -1.), (*rings.last().unwrap(), 1.)] {
            let start = self.positions.len() as u32;
            self.vertex(Vec3::Y * ring.0, Vec3::Y * sign, Vec2::ZERO, &tf);
            for i in 0..N {
                let p = point(i, ring);
                self.vertex(p, Vec3::Y * sign, p.xz(), &tf);
            }
            for i in 0..N as u32 {
                let a = start + 1 + i;
                let b = start + 1 + (i + 1) % N as u32;
                self.indices.extend(if sign > 0. {
                    [start, b, a]
                } else {
                    [start, a, b]
                });
            }
        }
    }
    /// Soft boxing with broad, flat contact caps and a smooth side/cap normal
    /// field. Keeps hard manufactured lofts in `housing` unchanged. Shared ring
    /// indexing avoids a welding map and extra allocations in the hot path.
    pub fn padded_housing(&mut self, rings: &[(f32, Vec2)], exponent: f32, tf: Transform) {
        let first = self.normals.len();
        self.housing(rings, exponent, tf);
        let caps = first + (rings.len() - 1) * N * 4;
        let bottom = self.normals[caps];
        let top = self.normals[caps + N + 1];
        for ring in 0..rings.len() {
            for col in 0..N {
                let prior = (col + N - 1) % N;
                let upper = first + ring * N * 4 + col * 4;
                let normal = if ring == 0 {
                    bottom
                } else if ring + 1 == rings.len() {
                    top
                } else {
                    (Vec3::from_array(self.normals[upper - N * 4 + 1])
                        + Vec3::from_array(self.normals[upper]))
                    .normalize()
                    .to_array()
                };
                if ring > 0 {
                    let band = first + (ring - 1) * N * 4;
                    self.normals[band + col * 4 + 1] = normal;
                    self.normals[band + prior * 4 + 2] = normal;
                }
                if ring + 1 < rings.len() {
                    let band = first + ring * N * 4;
                    self.normals[band + col * 4] = normal;
                    self.normals[band + prior * 4 + 3] = normal;
                }
            }
        }
    }
}
