use super::*;

impl Geometry {
    /// Revolved profile with real inward flutes. Analytic derivatives keep the
    /// normal field consistent with the radial corrugation and wall taper.
    pub fn fluted_lathe(
        &mut self,
        profile: &[(f32, f32)],
        segments: u32,
        depth: f32,
        lobes: u32,
        tf: Transform,
    ) {
        for pair in profile.windows(2) {
            let delta = Vec2::new(pair[1].0 - pair[0].0, pair[1].1 - pair[0].1);
            if delta.length_squared() < 1e-12 {
                continue;
            }
            let first = self.positions.len() as u32;
            for &(r, y) in pair {
                for i in 0..=segments {
                    let angle = TAU * (i % segments) as f32 / segments as f32;
                    let phase = lobes as f32 * angle;
                    let f = 1. - depth * (0.5 + 0.5 * phase.cos());
                    let df = depth * 0.5 * lobes as f32 * phase.sin();
                    let (sin, cos) = angle.sin_cos();
                    let n = Vec3::new(
                        delta.y * (f * cos + df * sin),
                        -delta.x * f * f,
                        delta.y * (f * sin - df * cos),
                    )
                    .normalize();
                    self.vertex(
                        Vec3::new(r * f * cos, y, r * f * sin),
                        n,
                        Vec2::new(
                            TAU * i as f32 / segments as f32 * pair[0].0.max(pair[1].0),
                            y + r,
                        ),
                        &tf,
                    );
                }
            }
            for i in 0..segments {
                let a = first + i;
                let b = a + segments + 1;
                if pair[0].0 > 1e-5 {
                    self.indices.extend([a, b, a + 1]);
                }
                if pair[1].0 > 1e-5 {
                    self.indices.extend([a + 1, b, b + 1]);
                }
            }
        }
    }

    /// Continuous circular sweep for handles, goosenecks and binding wire.
    /// The caller provides distinct adjacent points; end caps are separate faces.
    pub fn tube(&mut self, points: &[Vec3], radius: f32, sides: u32) {
        if points.len() < 2 {
            return;
        }
        let start = self.positions.len() as u32;
        let closed = points[0].distance_squared(*points.last().unwrap()) < 1e-12;
        let mut distance = 0.;
        let mut frame = Vec3::ZERO;
        let mut frames = Vec::with_capacity(points.len());
        for (i, &p) in points.iter().enumerate() {
            let before = points[if closed && i == 0 {
                points.len() - 2
            } else {
                i.saturating_sub(1)
            }];
            let after = points[if closed && i == points.len() - 1 {
                1
            } else {
                (i + 1).min(points.len() - 1)
            }];
            let tangent = (after - before).normalize();
            frame = (frame - tangent * frame.dot(tangent))
                .try_normalize()
                .unwrap_or_else(|| tangent.any_orthonormal_vector());
            if i > 0 {
                distance += p.distance(points[i - 1]);
            }
            frames.push((tangent, frame, distance));
        }
        // A transported frame can return with a twist on a closed curve. Spread
        // its correction by arclength rather than leaving a crack at the seam.
        let twist = if closed {
            let (axis, last, _) = *frames.last().unwrap();
            axis.dot(last.cross(frames[0].1))
                .atan2(last.dot(frames[0].1))
        } else {
            0.
        };
        for (i, (&p, &(tangent, frame, along))) in points.iter().zip(&frames).enumerate() {
            let frame = Quat::from_axis_angle(tangent, twist * along / distance.max(1e-6)) * frame;
            let cross = tangent.cross(frame);
            for j in 0..=sides {
                let angle = TAU * j as f32 / sides as f32;
                let normal = frame * angle.cos() + cross * angle.sin();
                self.vertex(
                    p + radius * normal,
                    normal,
                    Vec2::new(angle * radius, along),
                    &Transform::IDENTITY,
                );
                // Position/normal welding is exact, while metric UVs retain the
                // intentional cloth/metal sweep seams at each wrap boundary.
                let index = self.positions.len() - 1;
                let source = if closed && i == points.len() - 1 {
                    Some(start as usize + j as usize)
                } else if j == sides {
                    Some(index - sides as usize)
                } else {
                    None
                };
                if let Some(source) = source {
                    self.positions[index] = self.positions[source];
                    self.normals[index] = self.normals[source];
                }
            }
        }
        for ring in 0..points.len() as u32 - 1 {
            for side in 0..sides {
                let i = start + ring * (sides + 1) + side;
                let j = i + sides + 1;
                self.indices.extend([i, i + 1, j + 1, i, j + 1, j]);
            }
        }
        // Closed rings have no coincident end-cap discs.
        if closed {
            return;
        }
        for (ring, sign) in [(0, -1.), (points.len() - 1, 1.)] {
            let neighbor = if ring == 0 { 1 } else { ring - 1 };
            let normal = (points[ring] - points[neighbor]).normalize();
            let center = self.positions.len() as u32;
            self.vertex(points[ring], normal, Vec2::ZERO, &Transform::IDENTITY);
            for side in 0..=sides {
                let p = Vec3::from_array(
                    self.positions[(start + ring as u32 * (sides + 1) + side) as usize],
                );
                self.vertex(p, normal, Vec2::ZERO, &Transform::IDENTITY);
            }
            for side in 0..sides {
                let a = center + 1 + side;
                self.indices.extend(if sign < 0. {
                    [center, a + 1, a]
                } else {
                    [center, a, a + 1]
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn closed_nonplanar_sweeps_weld_the_frame_and_uv_wraps() {
        let points: Vec<_> = (0..=64)
            .map(|i| {
                let t = TAU * (i % 64) as f32 / 64.;
                Vec3::new(
                    t.cos() * 0.35,
                    t.sin() * 0.46,
                    (t * 2.).sin() * 0.11 + (t * 3.).cos() * 0.04,
                )
            })
            .collect();
        let mut g = Geometry::default();
        g.tube(&points, 0.012, 8);
        for i in 0..=8 {
            assert_eq!(g.positions[i], g.positions[64 * 9 + i]);
            assert_eq!(g.normals[i], g.normals[64 * 9 + i]);
        }
        for ring in 0..=64 {
            assert_eq!(g.positions[ring * 9], g.positions[ring * 9 + 8]);
            assert_eq!(g.normals[ring * 9], g.normals[ring * 9 + 8]);
        }
        let key = |i: u32| g.positions[i as usize].map(f32::to_bits);
        let mut edges = std::collections::BTreeMap::new();
        for t in g.indices.as_chunks::<3>().0 {
            let p = t.map(|i| Vec3::from_array(g.positions[i as usize]));
            let n = (p[1] - p[0]).cross(p[2] - p[0]);
            assert!(n.length_squared() > 1e-15);
            assert!(t
                .iter()
                .all(|i| n.dot(Vec3::from_array(g.normals[*i as usize])) > 0.));
            for k in 0..3 {
                let (a, b) = (key(t[k]), key(t[(k + 1) % 3]));
                let (edge, sign) = if a < b { ((a, b), 1) } else { ((b, a), -1) };
                let v = edges.entry(edge).or_insert((0, 0));
                v.0 += 1;
                v.1 += sign;
            }
        }
        assert!(edges.values().all(|v| *v == (2, 0)));
    }
}
