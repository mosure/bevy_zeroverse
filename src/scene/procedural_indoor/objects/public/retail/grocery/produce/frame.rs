use super::*;

impl ProduceStand {
    pub(super) fn frame(
        &self,
        a: &mut Assembly,
        o: &IndoorObject,
        frame: Surface,
        cuts: &mut TimberCuts,
    ) {
        let s = o.size;
        let mut r = stream(o.seed, u64::from_le_bytes(*b"FruitLeg"));
        let thickness = r.random_range(0.024..0.038);
        let top = self.front_deck_m;
        let h = top - thickness;
        a.board(
            cuts,
            (frame, FIXTURE),
            Vec3::Y * (top - thickness * 0.5),
            Vec3::new(s.x, thickness, s.z),
            0.0015,
            GrainAxis::X,
        );
        if let Some(split) = self.row_split_m.filter(|_| self.rear_rise_m > 0.) {
            let depth = split + s.z * 0.5 - 0.009;
            a.board(
                cuts,
                (frame, FIXTURE),
                Vec3::new(0., top + self.rear_rise_m * 0.5, split - depth * 0.5),
                Vec3::new(s.x - 0.018, self.rear_rise_m, depth),
                0.002,
                GrainAxis::X,
            );
        }
        let x = s.x * r.random_range(0.34..0.43);
        let z = s.z * r.random_range(0.30..0.40);
        let t = r.random_range(0.034..0.055);
        match self.base {
            ProduceBase::OpenFrame => {
                for sign in [-1., 1.] {
                    for side in [-1., 1.] {
                        a.board(
                            cuts,
                            (frame, FIXTURE),
                            Vec3::new(sign * x, h * 0.5, side * z),
                            Vec3::new(t, h, t),
                            0.0015,
                            GrainAxis::Y,
                        );
                    }
                    a.board(
                        cuts,
                        (frame, FIXTURE),
                        Vec3::new(sign * x, h - 0.033, 0.),
                        Vec3::new(t * 0.77, 0.054, 2. * z - t),
                        0.001,
                        GrainAxis::Z,
                    );
                    a.board(
                        cuts,
                        (frame, FIXTURE),
                        Vec3::new(0., h - 0.040, sign * z),
                        Vec3::new(2. * x - t, 0.068, t * 0.77),
                        0.001,
                        GrainAxis::X,
                    );
                }
            }
            ProduceBase::EndPanels => {
                for sign in [-1., 1.] {
                    a.board(
                        cuts,
                        (frame, FIXTURE),
                        Vec3::new(sign * x, h * 0.5, 0.),
                        Vec3::new(t, h, 2. * z),
                        0.002,
                        GrainAxis::Y,
                    );
                }
                a.board(
                    cuts,
                    (frame, FIXTURE),
                    Vec3::new(0., h * r.random_range(0.24..0.43), 0.),
                    Vec3::new(2. * x - t, h * 0.24, 0.027),
                    0.0015,
                    GrainAxis::X,
                );
            }
            ProduceBase::Plinth => {
                let toe = r.random_range(0.058..0.10);
                let width = s.x - 0.075;
                let depth = s.z - 0.070;
                board(
                    a,
                    Surface::Rubber,
                    FIXTURE,
                    Vec3::Y * (toe * 0.5),
                    Vec3::new(width - 0.062, toe, depth - 0.058),
                );
                let y = (h + toe) * 0.5;
                a.board(
                    cuts,
                    (frame, FIXTURE),
                    Vec3::Y * y,
                    Vec3::new(width, h - toe, depth),
                    0.002,
                    GrainAxis::X,
                );
                // Door fronts stop shy of the carcass edges. Their real reveal exposes
                // the recessed face, with no coplanar decorative surface.
                let panels = (width / 0.67).round().clamp(2., 4.) as usize;
                let pitch = (width - 0.019) / panels as f32;
                for i in 0..panels {
                    a.board(
                        cuts,
                        (frame, FIXTURE),
                        Vec3::new(
                            (i as f32 + 0.5 - panels as f32 * 0.5) * pitch,
                            y,
                            depth * 0.5 + 0.005,
                        ),
                        Vec3::new(pitch - 0.006, h - toe - 0.024, 0.010),
                        0.001,
                        GrainAxis::Y,
                    );
                }
            }
        }
    }
}
