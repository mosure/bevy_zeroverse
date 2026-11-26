//! Plane cuts preserve winding and interpolate the existing normal and UV charts.
use super::*;

impl Geometry {
    /// Keep the nonnegative side. Clipping a triangle against one plane forms
    /// at most a quadrilateral, so this needs no per-triangle heap allocations.
    pub(crate) fn clip_half_space(&mut self, distance: impl Fn(Vec3) -> f64) {
        let source = std::mem::take(self);
        for tri in source.indices.as_chunks::<3>().0 {
            let input = tri.map(|i| {
                let i = i as usize;
                (
                    Vec3::from(source.positions[i]),
                    Vec3::from(source.normals[i]),
                    Vec2::from(source.uvs[i]),
                )
            });
            let distances = input.map(|v| distance(v.0));
            let mut output = [input[0]; 4];
            let mut count = 0;
            for i in 0..3 {
                let j = (i + 1) % 3;
                let (a, b) = (input[i], input[j]);
                let (da, db) = (distances[i], distances[j]);
                if da >= 0. {
                    output[count] = a;
                    count += 1;
                }
                if (da >= 0.) != (db >= 0.) {
                    // Start from the inside endpoint for identical intersections on
                    // shared edges. Round once to preserve slender bevels (seed 1013005).
                    let (a, b, da, db) = if da >= 0. {
                        (a, b, da, db)
                    } else {
                        (b, a, db, da)
                    };
                    let t = da / (da - db);
                    output[count] = (
                        (a.0.as_dvec3() + (b.0.as_dvec3() - a.0.as_dvec3()) * t).as_vec3(),
                        (a.1.as_dvec3() + (b.1.as_dvec3() - a.1.as_dvec3()) * t)
                            .as_vec3()
                            .normalize_or_zero(),
                        (a.2.as_dvec2() + (b.2.as_dvec2() - a.2.as_dvec2()) * t).as_vec2(),
                    );
                    count += 1;
                }
            }
            for i in 2..count {
                let v = [output[0], output[i - 1], output[i]];
                if (v[1].0 - v[0].0).cross(v[2].0 - v[0].0).length_squared() < 1e-14 {
                    continue;
                }
                let base = self.positions.len() as u32;
                for (p, n, uv) in v {
                    self.positions.push(p.to_array());
                    self.normals.push(n.to_array());
                    self.uvs.push(uv.to_array());
                }
                self.indices.extend([base, base + 1, base + 2]);
            }
        }
    }
}
