use super::*;

impl FanHead {
    pub(super) fn grille(&self, g: &mut Geometry, p: &FanProgram) {
        let rim = self.guard_radius;
        let lip = (p.wire * 1.6).min(rim * 0.035);
        let edge = self.rim_half_depth;
        let steps = p.spokes * 2;
        g.lathe(
            &[
                (rim - lip, -edge),
                (rim, -edge),
                (rim, edge),
                (rim - lip, edge),
                (rim - lip, -edge),
            ],
            steps as u32,
            Transform::from_rotation(Quat::from_rotation_x(FRAC_PI_2)),
        );
        for front in [true, false] {
            let (sign, apex, start, phase) = if front {
                (1., self.front_crown, self.badge_radius * 0.9, 0.)
            } else {
                (
                    -1.,
                    self.rear_crown,
                    self.motor_radius * 0.94,
                    self.rear_phase,
                )
            };
            // Seat the wire ends inside the actual faceted rolled rim, including
            // guards whose endpoints fall halfway between its vertices.
            let half_angle = PI / steps as f32;
            let offset =
                (phase + self.grille_sweep * sign).rem_euclid(2. * half_angle) - half_angle;
            let outer = (rim - lip * 0.55) * half_angle.cos() / offset.cos();
            let station = |u: f32, angle: f32| {
                let r = start.lerp(outer, u);
                let (sin, cos) = (angle + phase + self.grille_sweep * sign * u * u).sin_cos();
                Vec3::new(
                    cos * r,
                    sin * r,
                    sign * (edge - p.wire * 0.2).lerp(apex, 1. - u * u),
                )
            };
            // Every hoop has a station on every spoke, including swept guards.
            // Stamp one bent wire per face to avoid tessellating identical ribs.
            let segments = if p.rings == 1 { 4 } else { p.rings + 1 };
            let path: Vec<_> = (0..=segments)
                .map(|j| station(j as f32 / segments as f32, 0.))
                .collect();
            let mut spoke = Geometry::default();
            spoke.tube(&path, p.wire * 0.56, 5);
            g.positions.reserve(spoke.positions.len() * p.spokes);
            g.normals.reserve(spoke.normals.len() * p.spokes);
            g.uvs.reserve(spoke.uvs.len() * p.spokes);
            g.indices.reserve(spoke.indices.len() * p.spokes);
            for i in 0..p.spokes {
                let (s, c) = (TAU * i as f32 / p.spokes as f32).sin_cos();
                let turn = |&[x, y, z]: &[f32; 3]| [c * x - s * y, s * x + c * y, z];
                let offset = g.positions.len() as u32;
                g.positions.extend(spoke.positions.iter().map(turn));
                g.normals.extend(spoke.normals.iter().map(turn));
                g.uvs.extend_from_slice(&spoke.uvs);
                g.indices.extend(spoke.indices.iter().map(|i| i + offset));
            }
            for j in 1..=p.rings {
                let u = j as f32 / (p.rings + 1) as f32;
                let ring: Vec<_> = (0..=steps)
                    .map(|i| station(u, TAU * (i % steps) as f32 / steps as f32))
                    .collect();
                g.tube(&ring, p.wire * 0.71, 5);
            }
        }
    }
}
