//! Heating installations are proposed after the main furnishings settle and
//! before people, motion and cameras. Accepted units are ordinary solid objects.
use super::*;

impl IndoorManifest {
    pub(crate) fn add_baseboard_heaters(&mut self) {
        let mut r = stream(self.seed, u64::from_le_bytes(*b"HeatPlac"));
        let wall_depth = self.envelope.as_ref().map_or(0.18, |e| e.wall_thickness());
        if !r.random_bool((0.12 + (wall_depth - 0.15) * 0.60).clamp(0.12, 0.46) as f64) {
            return;
        }
        let base = super::super::super::architecture::trim::TrimProgram::for_scene(self)
            .and_then(|t| t.base.as_ref())
            .map_or(0.028, |s| s.depth_m);
        for _ in 0..r.random_range(1..=4) {
            let size = super::super::super::objects::services::heater::dimensions(r.random());
            for _ in 0..8 {
                let (tf, span) = self.attachment_wall(&mut r);
                let reach = (span - size.x) * 0.5 - 0.16;
                if reach <= 0. {
                    continue;
                }
                let pos = tf.transform_point(Vec3::new(
                    r.random_range(-reach..reach),
                    0.,
                    base + 0.01 + size.z * 0.5,
                ));
                let yaw = tf.rotation.to_euler(EulerRot::YXZ).0;
                let o = self.floor_candidate(ObjectKind::BaseboardHeater, pos, size, yaw, &mut r);
                let (lo, hi) = o.bounds();
                if super::super::super::architecture::facade::overlaps_opening(self, lo, hi, 0.02)
                    || !self.placement_clear(&o, 0.06)
                {
                    continue;
                }
                self.objects.push(o);
                break;
            }
        }
    }
}
