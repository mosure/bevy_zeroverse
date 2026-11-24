//! Closed rounded-section lofts for manufactured shells and storage bodies.
use super::*;
const N: usize = 32;
/// Shared profile for inserts fitted to the actual polygonal housing.
pub(crate) fn housing_outline(exponent: f32) -> [Vec2; N] {
    let angles = primitives::tube_angles(N as u32);
    let power = 2. / exponent;
    std::array::from_fn(|i| {
        Vec2::new(
            profile_coordinate(angles[i].cos, power),
            profile_coordinate(angles[i].sin, power),
        )
    })
}

fn profile_coordinate(v: f32, power: f32) -> f32 {
    if v.abs() < 1e-6 {
        0.
    } else {
        v.signum() * v.abs().powf(power)
    }
}

/// Front edge at a horizontal station in the *tessellated* housing.
/// Used when fitting small features near the perimeter of a rounded product.
pub(crate) fn housing_front_z(half: Vec2, exponent: f32, x: f32) -> f32 {
    let x = x.abs();
    let angles = primitives::tube_angles(N as u32);
    let power = 2. / exponent;
    let mut prior = Vec2::new(0., -half.y);
    for a in &angles[N * 3 / 4 + 1..=N] {
        let next = half
            * Vec2::new(
                profile_coordinate(a.cos, power),
                profile_coordinate(a.sin, power),
            );
        if x <= next.x {
            return prior.y + (next.y - prior.y) * ((x - prior.x) / (next.x - prior.x));
        }
        prior = next;
    }
    0.
}
impl Geometry {
    /// Each ring is (height, half-width/depth). Caps and side normals are separate.
    pub fn housing(&mut self, rings: &[(f32, Vec2)], exponent: f32, tf: Transform) {
        self.housing_sections(rings, exponent, tf, true, None);
    }

    /// Closed loft with an independently finished top face, e.g. a handset display.
    pub(crate) fn housing_with_top(
        &mut self,
        top: &mut Geometry,
        rings: &[(f32, Vec2)],
        exponent: f32,
        tf: Transform,
    ) {
        self.housing_sections(rings, exponent, tf, true, Some(top));
    }

    /// Closed cross-section for a hollow rim. Repeat the first section last;
    /// unlike a solid housing, no disk caps fill or overlap its aperture.
    pub fn housing_loop(&mut self, rings: &[(f32, Vec2)], exponent: f32, tf: Transform) {
        debug_assert_eq!(rings.first(), rings.last());
        self.housing_sections(rings, exponent, tf, false, None);
    }

    fn housing_sections(
        &mut self,
        rings: &[(f32, Vec2)],
        exponent: f32,
        tf: Transform,
        caps: bool,
        mut separate_top: Option<&mut Geometry>,
    ) {
        if rings.len() < 2 {
            return;
        }
        // Every ring shares its angular profile. Evaluate trig/powers once per
        // side, not again for each vertex, tangent and vertical normal sample.
        // Keep the same arithmetic order when applying metric ring dimensions.
        let outline = housing_outline(exponent);
        let point = |i: usize, ring: (f32, Vec2)| {
            let unit = outline[i % N];
            Vec3::new(ring.1.x * unit.x, ring.0, ring.1.y * unit.y)
        };
        // Parameterize the patterned stock by metres along its widest section.
        // Equal polar angles are very unequal lengths across a rounded rectangle.
        let span = rings.iter().fold(Vec2::ZERO, |size, ring| size.max(ring.1));
        let mut along = [0.; N + 1];
        for i in 0..N {
            along[i + 1] = along[i] + ((outline[(i + 1) % N] - outline[i]) * span).length();
        }
        let mut lower: [Vec3; N] = std::array::from_fn(|i| point(i, rings[0]));
        let mut height = [0.; N];
        for pair in rings.windows(2) {
            let planar = pair[0].0 == pair[1].0;
            let upper: [Vec3; N] = std::array::from_fn(|i| point(i, pair[1]));
            let rise: [Vec3; N] = std::array::from_fn(|i| upper[i] - lower[i]);
            let next_height: [f32; N] = std::array::from_fn(|i| height[i] + rise[i].length());
            for i in 0..N {
                let start = self.positions.len() as u32;
                for (j, ring, distances) in [
                    (i, &lower, &height),
                    (i, &upper, &next_height),
                    (i + 1, &upper, &next_height),
                    (i + 1, &lower, &height),
                ] {
                    let k = j % N;
                    let p = ring[k];
                    let tangent = ring[(k + 1) % N] - ring[(k + N - 1) % N];
                    let uv = if planar {
                        p.xz()
                    } else {
                        Vec2::new(along[j], distances[k])
                    };
                    self.vertex(p, rise[k].cross(tangent).normalize(), uv, &tf);
                }
                self.indices
                    .extend([start, start + 1, start + 2, start, start + 2, start + 3]);
            }
            lower = upper;
            height = next_height;
        }
        if !caps {
            return;
        }
        for (ring, sign) in [(rings[0], -1.), (*rings.last().unwrap(), 1.)] {
            let face = if sign > 0. {
                separate_top.as_deref_mut().unwrap_or(self)
            } else {
                &mut *self
            };
            let start = face.positions.len() as u32;
            face.vertex(Vec3::Y * ring.0, Vec3::Y * sign, Vec2::ZERO, &tf);
            for i in 0..N {
                let p = point(i, ring);
                face.vertex(p, Vec3::Y * sign, p.xz(), &tf);
            }
            for i in 0..N as u32 {
                let a = start + 1 + i;
                let b = start + 1 + (i + 1) % N as u32;
                face.indices.extend(if sign > 0. {
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
        if rings.len() < 2 {
            return;
        }
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

#[cfg(test)]
mod profile_tests {
    use super::*;
    #[test]
    fn cached_housing_directions_replay_and_edge_queries_follow_actual_chords() {
        let half = Vec2::new(0.041, 0.076);
        for e in [2.0_f32, 2.7, 4., 6.41, 12., 17.73, 22.] {
            let outline = housing_outline(e);
            assert_eq!(outline, housing_outline(e));
            for (i, &actual) in outline.iter().enumerate() {
                let t = TAU * i as f32 / N as f32;
                let reference = |v: f32| {
                    if v.abs() < 1e-6 {
                        0.
                    } else {
                        v.signum() * v.abs().powf(2. / e)
                    }
                };
                // The scalar reference and the cached sin/cos pair can differ
                // by a few ULPs across platform math libraries.
                let expected = Vec2::new(reference(t.cos()), reference(t.sin()));
                assert!(actual.abs_diff_eq(expected, 4. * f32::EPSILON));
            }
            for i in 24..32 {
                let p = (outline[i] * 0.29 + outline[(i + 1) % 32] * 0.71) * half;
                assert!((housing_front_z(half, e, p.x) - p.y).abs() < 3e-6);
            }
        }
    }
}
