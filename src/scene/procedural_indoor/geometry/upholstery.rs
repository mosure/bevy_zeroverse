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
        self.cushion_panels::<33, 12>(size, profile, tf, None);
    }

    /// Replace the bottom panel with a distinct stock on the same closed mesh.
    pub(crate) fn cushion_panels<const COLUMNS: usize, const ROWS: usize>(
        &mut self,
        size: Vec3,
        profile: CushionProfile,
        tf: Transform,
        mut underside: Option<&mut Geometry>,
    ) {
        let sides = COLUMNS - 1;
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
        // Every latitude uses the same superellipse. Reuse its exact seam and
        // avoid evaluating two trigonometric powers per vertex of every chair.
        let outline: [Vec2; COLUMNS] = std::array::from_fn(|col| {
            let t = TAU * (col % sides) as f32 / sides as f32;
            Vec2::new(power(t.cos(), 2. / shape), power(t.sin(), 2. / shape))
        });
        let mut p = Vec::with_capacity(2 + (ROWS - 1) * COLUMNS);
        p.push(Vec3::new(0., half.y - dish, 0.));
        for row in 1..ROWS {
            let v = PI * row as f32 / ROWS as f32;
            let radial = v.sin().powf(2. / crown);
            let y = half.y * power(v.cos(), 2. / crown)
                - if row < ROWS / 2 {
                    dish * (1. - radial).powi(2)
                } else {
                    0.
                };
            for point in &outline {
                p.push(Vec3::new(
                    half.x * radial * point.x,
                    y,
                    half.z * radial * point.y,
                ));
            }
        }
        p.push(Vec3::new(0., -half.y, 0.));
        let bottom = p.len() - 1;
        let mut triangles = Vec::with_capacity(2 * sides * (ROWS - 1));
        for col in 0..sides {
            triangles.push(([0, 1 + col + 1, 1 + col], 0));
        }
        for row in 0..ROWS - 2 {
            let top_rows = (ROWS + 1) / 3 - 1;
            let band = if row < top_rows {
                0
            } else if row >= ROWS - 2 - top_rows {
                2
            } else {
                1
            };
            for col in 0..sides {
                let a = 1 + row * COLUMNS + col;
                let b = a + COLUMNS;
                triangles.push(([a, a + 1, b + 1], band));
                triangles.push(([a, b + 1, b], band));
            }
        }
        for col in 0..sides {
            let a = bottom - COLUMNS + col;
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
            let a = 1 + row * COLUMNS;
            let n = normals[a] + normals[a + sides];
            normals[a] = n;
            normals[a + sides] = n;
        }
        // Cloth boxing measures arclength around the actual plan profile.
        let equator = 1 + (ROWS / 2 - 1) * COLUMNS;
        let mut along = [0.; COLUMNS];
        for i in 1..=sides {
            along[i] = along[i - 1] + p[equator + i].distance(p[equator + i - 1]);
        }
        let mut remap = vec![[None; 3]; p.len()];
        for (tri, band) in triangles {
            let g: &mut Geometry = if band == 2 {
                underside.as_deref_mut().unwrap_or(self)
            } else {
                self
            };
            for i in tri {
                let index = if let Some(index) = remap[i][band] {
                    index
                } else {
                    let uv = if band == 1 {
                        Vec2::new(along[(i - 1) % COLUMNS], p[i].y + half.y)
                    } else {
                        Vec2::new(p[i].x + half.x, p[i].z + half.z)
                    };
                    let index = g.positions.len() as u32;
                    g.vertex(p[i], normals[i].normalize(), uv, &tf);
                    remap[i][band] = Some(index);
                    index
                };
                g.indices.push(index);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hard_underside_shares_the_exact_upholstered_seat_geometry() {
        let size = Vec3::new(0.51, 0.064, 0.46);
        let shape = CushionProfile {
            roundness: 4.2,
            crown: 3.6,
            dish: 0.006,
        };
        let pose = Transform::from_xyz(0., 0.65, 0.12).with_rotation(Quat::from_rotation_x(1.31));
        let mut one = Geometry::default();
        one.cushion_panels::<25, 8>(size, shape, pose, None);
        let mut face = Geometry::default();
        let mut shell = Geometry::default();
        face.cushion_panels::<25, 8>(size, shape, pose, Some(&mut shell));
        assert!(!shell.indices.is_empty() && !face.indices.is_empty());
        let triangles = |g: &Geometry| {
            g.indices
                .as_chunks::<3>()
                .0
                .iter()
                .map(|t| {
                    t.map(|i| {
                        let i = i as usize;
                        (
                            g.positions[i].map(f32::to_bits),
                            g.normals[i].map(f32::to_bits),
                            g.uvs[i].map(f32::to_bits),
                        )
                    })
                })
                .collect::<Vec<_>>()
        };
        let mut expected = triangles(&one);
        let mut split = triangles(&face);
        split.extend(triangles(&shell));
        expected.sort_unstable();
        split.sort_unstable();
        assert_eq!(expected, split);
    }

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
                assert!(g.positions.iter().all(|p| {
                    Vec3::from_array(*p)
                        .abs()
                        .cmple(size * 0.5 + Vec3::splat(1e-6))
                        .all()
                }));
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
