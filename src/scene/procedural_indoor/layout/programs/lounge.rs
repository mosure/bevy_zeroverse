//! Conversation groups follow accepted furniture, not mirrored room coordinates.
use super::*;
use std::f32::consts::PI;

impl IndoorManifest {
    pub(super) fn furnish_lounge(&mut self, tf: Transform, extent: Vec2, zone: usize) {
        let mut r = stream(self.seed, 0x4c4f_554e_4745_0000 + zone as u64);
        let size = Vec3::new(
            (extent.x * r.random_range(0.53..0.83)).clamp(1.6, 3.4),
            r.random_range(0.78..1.08),
            if extent.min_element() > 3.7 && r.random_bool(0.45) {
                r.random_range(1.40..1.90)
            } else {
                r.random_range(0.80..1.08)
            },
        );
        let yaw = tf.rotation.to_euler(EulerRot::YXZ).0;
        let local = Vec3::new(
            r.random_range(-0.65..0.65) * ((extent.x - size.x) * 0.5 - 0.15).max(0.),
            0.,
            -extent.y * 0.5 + size.z * 0.5 + r.random_range(0.12..0.46),
        );
        let Some(sofa_id) = self.add(
            ObjectKind::Sofa,
            tf.transform_point(local),
            size,
            yaw + PI + r.random_range(-0.09..0.09),
            &mut r,
        ) else {
            return;
        };
        let sofa = self.objects[sofa_id].clone();
        let table_size = Vec3::new(
            (size.x * r.random_range(0.45..0.72)).clamp(0.72, 1.7),
            r.random_range(0.32..0.48),
            r.random_range(0.43..0.80),
        );
        let table_position = sofa.transform().transform_point(Vec3::new(
            r.random_range(-0.36..0.36) * (size.x - table_size.x),
            0.,
            -(size.z + table_size.z) * 0.5 - r.random_range(0.32..0.60),
        ));
        let table_yaw = sofa.yaw + PI + r.random_range(-0.23..0.23);
        let Some(table_id) = self.add(
            ObjectKind::CoffeeTable,
            table_position,
            table_size,
            table_yaw,
            &mut r,
        ) else {
            return;
        };

        // Some rooms really do contain matched, symmetrically arranged seats.
        // Otherwise vary occupancy, azimuth, setback and finish independently.
        let symmetric = r.random_bool(0.18);
        let seats = if symmetric {
            2
        } else if extent.x > 4.7 && r.random_bool(0.32) {
            3
        } else {
            r.random_range(1..=2)
        };
        let spread = r.random_range(0.62..1.45);
        let skew = r.random_range(-0.28..0.28);
        let shared_gap = r.random_range(0.52..0.78);
        let finish = if r.random_bool(0.58) {
            Some(sofa.seed)
        } else {
            None
        };
        for i in 0..seats {
            let t = if seats == 1 {
                r.random_range(-0.9..0.9)
            } else {
                i as f32 / (seats - 1) as f32 * 2. - 1.
            };
            let angle = t * spread
                + if symmetric {
                    0.
                } else {
                    skew + r.random_range(-0.22..0.22)
                };
            let gap = if symmetric {
                shared_gap
            } else {
                r.random_range(0.48..0.91)
            };
            let offset = Vec3::new(
                angle.sin() * (table_size.x * 0.5 + gap),
                0.,
                angle.cos() * (table_size.z * 0.5 + gap),
            );
            let position = table_position + Quat::from_rotation_y(table_yaw) * offset;
            let before = self.objects.len();
            self.chair_in_family_at_surface(
                position,
                table_yaw + angle,
                None,
                Some(table_id),
                &mut r,
            );
            if let Some(chair) = self.objects.get_mut(before) {
                chair.finish_seed = finish;
            }
        }
    }
}
