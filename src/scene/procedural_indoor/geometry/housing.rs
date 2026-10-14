//! Closed rounded-section lofts for manufactured shells and storage bodies.
use super::*;
impl Geometry {
    /// Each ring is (height, half-width/depth). Caps and side normals are separate.
    pub fn housing(&mut self, rings: &[(f32, Vec2)], exponent: f32, tf: Transform) {
        const N: usize = 32;
        let point = |i: usize, ring: (f32, Vec2)| {
            let t = TAU * (i % N) as f32 / N as f32;
            let f = |v: f32| {
                if v.abs() < 1e-6 {
                    0.
                } else {
                    v.signum() * v.abs().powf(2. / exponent)
                }
            };
            Vec3::new(ring.1.x * f(t.cos()), ring.0, ring.1.y * f(t.sin()))
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
}
