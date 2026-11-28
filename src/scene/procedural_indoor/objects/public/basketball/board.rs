use super::*;

impl BasketballGoal {
    // The target, board and shoulders meet along shared edges on the same
    // front surface. There is no overlaid target decal to flicker at distance.
    pub(super) fn board_levels(self, size: Vec3) -> [(f32, f32); 8] {
        let rim = self.rim_height();
        let w = size.x * 0.5;
        let top = rim + 0.45;
        let shoulder = |factor| if self.arched { w * factor } else { w };
        [
            (rim - 0.15, shoulder(0.87)),
            (rim, w),
            (rim + 0.05, w),
            (top - 0.05, w),
            (top, w),
            (top + 0.30 * (size.y - top), shoulder(0.97)),
            (top + 0.74 * (size.y - top), shoulder(0.80)),
            (size.y, shoulder(self.shoulder)),
        ]
    }

    pub(super) fn build_board(self, a: &mut Assembly, size: Vec3) {
        let wood = (Surface::Wood, CLASS);
        let enamel = (Surface::Paint, sport::ENAMEL);
        let (base, ink) = if self.timber {
            (wood, enamel)
        } else {
            (enamel, (Surface::Ink, CLASS))
        };
        let front = self.face_z(size);
        let back = front - self.board_depth;
        let levels = self.board_levels(size);
        let corners = |w: f32| [-w, -0.295, -0.245, 0.245, 0.295, w];
        for (row, strip) in levels.windows(2).enumerate() {
            let [(low, wl), (high, wh)] = [strip[0], strip[1]];
            let (lower, upper) = (corners(wl), corners(wh));
            for col in 0..5 {
                let marking =
                    (1..=3).contains(&row) && (1..=3).contains(&col) && (row != 2 || col != 2);
                let (s, key) = if marking { ink } else { base };
                a.part(s, key).metric_quad(
                    [
                        Vec3::new(lower[col], low, front),
                        Vec3::new(lower[col + 1], low, front),
                        Vec3::new(upper[col + 1], high, front),
                        Vec3::new(upper[col], high, front),
                    ],
                    Vec3::Z,
                );
            }
            let g = a.part(base.0, base.1);
            g.metric_quad(
                [
                    Vec3::new(-wl, low, back),
                    Vec3::new(wl, low, back),
                    Vec3::new(wh, high, back),
                    Vec3::new(-wh, high, back),
                ],
                Vec3::NEG_Z,
            );
            for side in [-1., 1.] {
                g.metric_quad(
                    [
                        Vec3::new(side * wl, low, back),
                        Vec3::new(side * wl, low, front),
                        Vec3::new(side * wh, high, front),
                        Vec3::new(side * wh, high, back),
                    ],
                    Vec3::X * side,
                );
            }
        }
        let g = a.part(base.0, base.1);
        for ((y, w), out) in [
            (levels[0], Vec3::NEG_Y),
            (levels[levels.len() - 1], Vec3::Y),
        ] {
            g.metric_quad(
                [
                    Vec3::new(-w, y, front),
                    Vec3::new(w, y, front),
                    Vec3::new(w, y, back),
                    Vec3::new(-w, y, back),
                ],
                out,
            );
        }
    }
}
