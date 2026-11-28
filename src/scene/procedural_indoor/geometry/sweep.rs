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
        if segments >= 3 && profile.len() >= 2 {
            RevolvedOutline::round(segments, depth, lobes).sweep(
                self,
                profile,
                ProfileJoints::Creased,
                tf,
            );
        }
    }

    /// Continuous circular sweep for handles, goosenecks and binding wire.
    /// The caller provides distinct adjacent points; end caps are separate faces.
    pub fn tube(&mut self, points: &[Vec3], radius: f32, sides: u32) {
        self.tapered_tube(points, radius, radius, sides);
    }

    /// Sweep one connected skin along the path, with a linear taper in metres
    /// of arclength. Closed paths require matching radii at their welded seam.
    pub(crate) fn tapered_tube(
        &mut self,
        points: &[Vec3],
        root_radius: f32,
        tip_radius: f32,
        sides: u32,
    ) {
        self.sweep_tube::<false>(points, root_radius, tip_radius, sides, Vec3::ZERO);
    }

    /// Bent stock in a known plane. Its unit plane normal fixes the same
    /// angular seam on every ring, including across tight return bends.
    pub(crate) fn planar_tube(&mut self, points: &[Vec3], radius: f32, sides: u32, normal: Vec3) {
        self.sweep_tube::<true>(points, radius, radius, sides, normal);
    }

    fn sweep_tube<const PLANAR: bool>(
        &mut self,
        points: &[Vec3],
        root_radius: f32,
        tip_radius: f32,
        sides: u32,
        plane_normal: Vec3,
    ) {
        if points.len() < 2 || sides < 3 || root_radius <= 0. || tip_radius <= 0. {
            return;
        }
        let start = self.positions.len() as u32;
        let closed = points[0].distance_squared(*points.last().unwrap()) < 1e-12;
        if closed && (points.len() < 4 || root_radius != tip_radius) {
            return;
        }
        let mut distance = 0.;
        let mut frame = Vec3::ZERO;
        // Stems, petioles, spouts and cables commonly have just 2–8 nodes.
        let empty = (Vec3::ZERO, Vec3::ZERO, 0.);
        let mut local = [empty; 8];
        let mut long_path;
        let frames = if points.len() <= local.len() {
            &mut local[..points.len()]
        } else {
            long_path = vec![empty; points.len()];
            long_path.as_mut_slice()
        };
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
            // At a tapered branch elbow a long incoming segment must not tip
            // the cross-section backwards over its shorter outgoing segment.
            let direction =
                if (PLANAR || root_radius != tip_radius) && i > 0 && i + 1 < points.len() {
                    (p - before).normalize_or_zero() + (after - p).normalize_or_zero()
                } else {
                    after - before
                };
            let Some(tangent) = direction.try_normalize() else {
                return;
            };
            frame = if PLANAR {
                plane_normal
            } else if root_radius != tip_radius && i > 0 && points.len() > 2 {
                Quat::from_rotation_arc(frames[i - 1].0, tangent) * frame
            } else {
                (frame - tangent * frame.dot(tangent))
                    .try_normalize()
                    .unwrap_or_else(|| tangent.any_orthonormal_vector())
            };
            if i > 0 {
                distance += p.distance(points[i - 1]);
            }
            frames[i] = (tangent, frame, distance);
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
        let angles = primitives::tube_angles(sides);
        let slope = (tip_radius - root_radius) / distance;
        for (i, (&p, &(tangent, frame, along))) in points.iter().zip(frames.iter()).enumerate() {
            let frame = if closed {
                Quat::from_axis_angle(tangent, twist * along / distance) * frame
            } else {
                frame
            };
            let radius = root_radius + slope * along;
            let cross = tangent.cross(frame);
            for (j, angle) in angles.iter().enumerate() {
                let radial = frame * angle.cos + cross * angle.sin;
                self.vertex(
                    p + radius * radial,
                    radial - tangent * slope,
                    Vec2::new(angle.radians * radius, along),
                    &Transform::IDENTITY,
                );
                // Position/normal welding is exact, while metric UVs retain the
                // intentional cloth/metal sweep seams at each wrap boundary.
                let index = self.positions.len() - 1;
                let source = if closed && i == points.len() - 1 {
                    Some(start as usize + j)
                } else if j == sides as usize {
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
                let uv = if root_radius != tip_radius {
                    let (_, across, _) = frames[ring];
                    let offset = p - points[ring];
                    Vec2::new(offset.dot(across), offset.dot(normal.cross(across)))
                } else {
                    Vec2::ZERO
                };
                self.vertex(p, normal, uv, &Transform::IDENTITY);
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
    fn tube_rings_replay_the_angular_samples_at_cached_and_custom_resolutions() {
        for sides in [3, 6, 8, 10, 12, 20, 32, 47] {
            let ring = primitives::tube_angles(sides);
            let replay = primitives::tube_angles(sides);
            assert_eq!(ring.len(), sides as usize + 1);
            for (j, (sample, repeated)) in ring.iter().zip(replay.iter()).enumerate() {
                let angle = TAU * j as f32 / sides as f32;
                assert_eq!(sample.radians.to_bits(), angle.to_bits());
                // Scalar and paired libm calls can round differently on macOS.
                // Check accuracy against a wider reference and replay bit for bit.
                let (sin, cos) = f64::from(angle).sin_cos();
                for (value, second, expected) in [
                    (sample.cos, repeated.cos, cos as f32),
                    (sample.sin, repeated.sin, sin as f32),
                ] {
                    assert_eq!(value.to_bits(), second.to_bits());
                    assert!(
                        value.to_bits().abs_diff(expected.to_bits()) <= 1,
                        "ring {sides}, sample {j}: {value} versus {expected}"
                    );
                }
            }
        }
        let mut g = Geometry::default();
        g.tube(&[Vec3::ZERO, Vec3::Y], 0.013, 2);
        assert!(g.positions.is_empty());
    }

    #[test]
    fn bent_taper_has_one_ring_per_node_and_only_terminal_caps() {
        let points = [
            Vec3::ZERO,
            Vec3::Y * 0.21,
            Vec3::new(0.12, 0.42, 0.03),
            Vec3::new(0.22, 0.47, -0.01),
        ];
        let length: f32 = points.windows(2).map(|p| p[0].distance(p[1])).sum();
        let mut mesh = Geometry::default();
        mesh.tapered_tube(&points, 0.025, 0.009, 8);
        assert_eq!(mesh.indices.len() / 3, (points.len() - 1) * 16 + 16);
        assert_eq!(mesh.positions.len(), points.len() * 9 + 2 * 10);
        let mut along = 0.;
        for (i, &center) in points.iter().enumerate() {
            if i > 0 {
                along += center.distance(points[i - 1]);
            }
            let radius = 0.025 + (0.009 - 0.025) * along / length;
            let j = i * 9;
            assert_eq!(mesh.positions[j], mesh.positions[j + 8]);
            assert_eq!(mesh.normals[j], mesh.normals[j + 8]);
            for k in j..j + 9 {
                assert!(
                    (Vec3::from_array(mesh.positions[k]).distance(center) - radius).abs() < 1e-6
                );
                assert!((mesh.uvs[k][1] - along).abs() < 1e-6);
            }
        }
        for tri in mesh.indices.as_chunks::<3>().0 {
            let p = tri.map(|i| Vec3::from_array(mesh.positions[i as usize]));
            let normal = (p[1] - p[0]).cross(p[2] - p[0]);
            assert!(tri
                .iter()
                .all(|&i| normal.dot(Vec3::from_array(mesh.normals[i as usize])) > 0.));
        }
    }

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
