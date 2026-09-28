use super::*;

impl Geometry {
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
            let cross = tangent.cross(frame);
            if i > 0 {
                distance += p.distance(points[i - 1]);
            }
            for j in 0..=sides {
                let angle = TAU * j as f32 / sides as f32;
                let normal = frame * angle.cos() + cross * angle.sin();
                self.vertex(
                    p + radius * normal,
                    normal,
                    Vec2::new(angle * radius, distance),
                    &Transform::IDENTITY,
                );
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
