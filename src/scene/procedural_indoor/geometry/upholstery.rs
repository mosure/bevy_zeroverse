//! Closed, shaped upholstery with a shared smooth normal field and metric UVs.
use super::*;
use std::f32::consts::PI;

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct CushionProfile {
    /// Horizontal superellipse exponent: elliptical through rounded rectangular.
    pub roundness: f32,
    /// Vertical exponent; controls the crown independently of the plan shape.
    pub crown: f32,
    /// Shallow compression at the seating datum, in metres.
    pub dish: f32,
}

impl Geometry {
    /// A bounded superellipsoid, with a compressed upper surface. UV seams split
    /// the top/bottom cloth panels from the perimeter boxing, but share normals.
    pub fn cushion(&mut self, size: Vec3, profile: CushionProfile, tf: Transform) {
        const SIDES: usize = 32;
        const ROWS: usize = 12;
        let half = size * 0.5;
        let power = |x: f32, p: f32| {
            if x.abs() < 1e-6 {
                0.
            } else {
                x.signum() * x.abs().powf(p)
            }
        };
        let shape = profile.roundness.clamp(2., 10.);
        let crown = profile.crown.clamp(2., 6.);
        let dish = profile.dish.clamp(0., size.y * 0.20);
        let mut p = vec![Vec3::new(0., half.y - dish, 0.)];
        for row in 1..ROWS {
            let v = PI * row as f32 / ROWS as f32;
            let radial = v.sin().powf(2. / crown);
            let y = half.y * power(v.cos(), 2. / crown)
                - if row < ROWS / 2 {
                    dish * (1. - radial).powi(2)
                } else {
                    0.
                };
            for col in 0..=SIDES {
                // Exact shared seam positions are necessary for watertight export.
                let t = TAU * (col % SIDES) as f32 / SIDES as f32;
                p.push(Vec3::new(
                    half.x * radial * power(t.cos(), 2. / shape),
                    y,
                    half.z * radial * power(t.sin(), 2. / shape),
                ));
            }
        }
        p.push(Vec3::new(0., -half.y, 0.));
        let bottom = p.len() - 1;
        let mut triangles = Vec::with_capacity(2 * SIDES * (ROWS - 1));
        for col in 0..SIDES {
            triangles.push(([0, 1 + col + 1, 1 + col], 0));
        }
        for row in 0..ROWS - 2 {
            let band = if row < 3 {
                0
            } else if row >= 7 {
                2
            } else {
                1
            };
            for col in 0..SIDES {
                let a = 1 + row * (SIDES + 1) + col;
                let b = a + SIDES + 1;
                triangles.push(([a, a + 1, b + 1], band));
                triangles.push(([a, b + 1, b], band));
            }
        }
        for col in 0..SIDES {
            let a = bottom - (SIDES + 1) + col;
            triangles.push(([a, a + 1, bottom], 2));
        }
        let mut normals = vec![Vec3::ZERO; p.len()];
        for (tri, _) in &triangles {
            let n = (p[tri[1]] - p[tri[0]]).cross(p[tri[2]] - p[tri[0]]);
            for &i in tri {
                normals[i] += n;
            }
        }
        for row in 0..ROWS - 1 {
            let a = 1 + row * (SIDES + 1);
            let n = normals[a] + normals[a + SIDES];
            normals[a] = n;
            normals[a + SIDES] = n;
        }
        // Cloth boxing measures arclength around the actual plan profile.
        let equator = 1 + (ROWS / 2 - 1) * (SIDES + 1);
        let mut along = [0.; SIDES + 1];
        for i in 1..=SIDES {
            along[i] = along[i - 1] + p[equator + i].distance(p[equator + i - 1]);
        }
        let mut remap = vec![[None; 3]; p.len()];
        for (tri, band) in triangles {
            for i in tri {
                let index = if let Some(index) = remap[i][band] {
                    index
                } else {
                    let uv = if band == 1 {
                        Vec2::new(along[(i - 1) % (SIDES + 1)], p[i].y + half.y)
                    } else {
                        Vec2::new(p[i].x + half.x, p[i].z + half.z)
                    };
                    let index = self.positions.len() as u32;
                    self.vertex(p[i], normals[i].normalize(), uv, &tf);
                    remap[i][band] = Some(index);
                    index
                };
                self.indices.push(index);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cushions_are_closed_bounded_and_have_valid_cloth_frames() {
        for roundness in [2., 3.5, 8.] {
            for crown in [2., 3.6, 6.] {
                let mut g = Geometry::default();
                let size = Vec3::new(0.53, 0.11, 0.46);
                g.cushion(
                    size,
                    CushionProfile {
                        roundness,
                        crown,
                        dish: 0.013,
                    },
                    Transform::IDENTITY,
                );
                let mut edges = std::collections::BTreeMap::new();
                let key = |i: u32| g.positions[i as usize].map(|x| (x * 1e7).round() as i32);
                for t in g.indices.as_chunks::<3>().0 {
                    let p = t.map(|i| Vec3::from_array(g.positions[i as usize]));
                    let n = (p[1] - p[0]).cross(p[2] - p[0]);
                    assert!(n.length_squared() > 1e-15);
                    assert!(t
                        .iter()
                        .all(|i| n.dot(Vec3::from_array(g.normals[*i as usize])) > 0.));
                    for k in 0..3 {
                        let (a, b) = (key(t[k]), key(t[(k + 1) % 3]));
                        let (edge, direction) = if a < b { ((a, b), 1) } else { ((b, a), -1) };
                        let v = edges.entry(edge).or_insert((0, 0));
                        v.0 += 1;
                        v.1 += direction;
                    }
                }
                assert!(
                    edges.values().all(|v| *v == (2, 0)),
                    "closed oriented surface"
                );
                assert!(g.positions.iter().all(|p| Vec3::from_array(*p)
                    .abs()
                    .cmple(size * 0.5 + Vec3::splat(1e-6))
                    .all()));
                let mesh = g.into_mesh();
                let Some(VertexAttributeValues::Float32x4(t)) =
                    mesh.attribute(Mesh::ATTRIBUTE_TANGENT)
                else {
                    panic!("missing tangent frame")
                };
                assert!(t.iter().all(|t| Vec4::from_array(*t).is_finite()));
            }
        }
    }
}
