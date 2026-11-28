use super::*;

impl ProduceBin {
    pub(super) fn fit_to_bay(&mut self, r: &mut impl Rng) {
        let max_footprint = match self.style {
            CrateStyle::TimberSlats => Vec2::new(0.98, 0.85),
            CrateStyle::VentedTote => Vec2::new(0.83, 0.65),
            CrateStyle::FormedTray => Vec2::new(0.98, 0.83),
        };
        let bay = self.size_xz_m;
        self.size_xz_m = bay.min(max_footprint)
            * Vec2::new(r.random_range(0.95..1.0), r.random_range(0.90..1.0));
        let slack = bay - self.size_xz_m;
        self.base_m.x += slack.x * r.random_range(-0.48..0.48);
        // Staff usually front a returnable crate toward the customer, with space behind to lift it.
        self.base_m.z += slack.y * r.random_range(-0.20..0.48);
    }

    pub(super) fn crate_mesh(&self, a: &mut Assembly, cuts: &mut TimberCuts, resin: &str) {
        let half = self.size_xz_m * 0.5;
        let t = self.style.wall_m();
        let h = self.rim_height_m;
        if self.style == CrateStyle::FormedTray {
            // Pressed steel: outside bottom, flared lip, rolled return and the inside
            // floor form a single closed sheet. No cap spans the mouth.
            let foot = half * 0.94;
            a.part(Surface::Metal, FIXTURE).housing(
                &[
                    (0., foot),
                    (0.008, foot + Vec2::splat(0.002)),
                    (h, half),
                    (h, half - Vec2::splat(t)),
                    (t, foot - Vec2::splat(t)),
                ],
                8.,
                Transform::from_translation(self.base_m),
            );
            return;
        }
        let timber = self.style == CrateStyle::TimberSlats;
        let (surface, label) = if timber {
            (Surface::Wood, FIXTURE)
        } else {
            (Surface::Plastic, resin)
        };
        let mut part = |offset: Vec3, size: Vec3, grain| {
            a.board(
                cuts,
                (surface, label),
                self.base_m + offset,
                size,
                0.,
                grain,
            );
        };
        part(
            Vec3::Y * (t * 0.5),
            Vec3::new(half.x * 2., t, half.y * 2.),
            GrainAxis::X,
        );
        let low = h * 0.38;
        let high = h * 0.67;
        let span = h - high;
        for sign in [-1., 1.] {
            let x = sign * (half.x - t * 0.5);
            let z = sign * (half.y - t * 0.5);
            let end = 2. * (half.y - t);
            for (mid, height) in [((t + low) * 0.5, low - t), ((h + high) * 0.5, span)] {
                part(
                    Vec3::new(0., mid, z),
                    Vec3::new(2. * half.x, height, t),
                    GrainAxis::X,
                );
                part(
                    Vec3::new(x, mid, 0.),
                    Vec3::new(t, height, end),
                    GrainAxis::Z,
                );
            }
            let mid = (low + high) * 0.5;
            let band = high - low;
            if timber {
                // Inside cleats fasten the two slats, with the grain running vertically.
                for side in [-1., 1.] {
                    part(
                        Vec3::new(
                            x.signum() * (half.x - t - 0.007),
                            0.5 * (h + t),
                            side * (half.y - t - 0.007),
                        ),
                        Vec3::new(0.014, h - t, 0.014),
                        GrainAxis::Y,
                    );
                }
            } else {
                // Solid ends either side of the carrying slot and narrower webs
                // across the long sides actually interrupt the ventilation openings.
                let grip = (end * 0.40).min(0.13);
                let length = (end - grip) * 0.5;
                for side in [-1., 1.] {
                    part(
                        Vec3::new(x, mid, side * (grip + length) * 0.5),
                        Vec3::new(t, band, length),
                        GrainAxis::Z,
                    );
                }
                let n = (self.size_xz_m.x / 0.078).round().clamp(3., 12.) as usize;
                for i in 0..n {
                    let along = -half.x + (i as f32 + 0.5) * (2. * half.x / n as f32);
                    part(
                        Vec3::new(along, mid, z),
                        Vec3::new(0.010, band, t),
                        GrainAxis::Y,
                    );
                }
            }
        }
    }
}
