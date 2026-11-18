use super::*;

impl Builder<'_> {
    pub(super) fn sliding(&mut self, track: PanelTrack) {
        let (n, width, step) = track.panels(self.width() + 0.028);
        let top = self.hi.y - 0.055;
        let height = self.height();
        let right = self.p.stack == Stack::Right
            || self.p.stack == Stack::Pair && self.p.identity.is_multiple_of(2);
        let sign = if right { -1. } else { 1. };
        let parked = if right {
            self.hi.x - 0.016 - width * 0.5
        } else {
            self.lo.x + 0.016 + width * 0.5
        };
        for panel in 0..n {
            let x = parked + sign * panel as f32 * step * self.coverage;
            let z = 0.082 + 0.018 * panel as f32;
            let mut rng = stream(self.p.identity, 0x524545444d4154 + panel as u64);
            let pitch = track.reed_pitch_m * rng.random_range(0.90..1.10);
            let count = (height / pitch).round().clamp(3., 80.) as usize;
            let pitch = height / count as f32;
            let thickness = rng.random_range(0.0025..0.0045);
            let g = self.a.part(Surface::WoodEdge, &self.label);
            for row in 0..count {
                let y = top - (row as f32 + 0.5) * pitch;
                let fill = 1. - track.openness * rng.random_range(0.78..1.22);
                let uv_start = g.uvs.len();
                g.cuboid(
                    Vec3::new(width, pitch * fill, thickness),
                    0.,
                    Transform::from_translation(self.tf.transform_point(Vec3::new(x, y, z)))
                        .with_rotation(self.tf.rotation),
                );
                // Cut each splint from a different run of the timber grain.
                let offset = Vec2::new(rng.random_range(0.0..2.), rng.random_range(0.0..2.));
                for uv in &mut g.uvs[uv_start..] {
                    // Timber lengths follow V in the shared metric atlas.
                    *uv = [uv[1] + offset.x, uv[0] + offset.y];
                }
            }
            // Binding tapes touch the front of the splints. Separate depth
            // tracks leave clearance even with all panels parked together.
            for u in [-0.34, 0., 0.34] {
                self.a.part(self.p.fabric, &self.cloth_label).cuboid(
                    Vec3::new((width * 0.012).clamp(0.003, 0.009), height, 0.0016),
                    0.,
                    Transform::from_translation(self.tf.transform_point(Vec3::new(
                        x + u * width,
                        top - height * 0.5,
                        z + thickness * 0.5 + 0.0004,
                    )))
                    .with_rotation(self.tf.rotation),
                );
            }
            self.block(
                Surface::Metal,
                Vec3::new(x, top + 0.004, z),
                Vec3::new(width, 0.02, 0.011),
                0.001,
            );
            self.block(
                Surface::WoodEdge,
                Vec3::new(x, top - height, z),
                Vec3::new(width, 0.014, 0.009),
                0.,
            );
            if panel == n - 1 {
                let end = x + sign * (width * 0.5 - 0.016);
                self.tube(
                    Surface::Metal,
                    &[
                        Vec3::new(end, top, z + 0.006),
                        Vec3::new(end, top - height.min(0.88) * 0.86, z + 0.018),
                    ],
                    0.0028,
                );
            }
        }
    }
}
