//! Supplemental storage/greenery uses the real envelope, including oblique edges.
//! A fixed attempt budget bounds cost; ordinary placement rules remain authoritative.
use super::*;
use std::f32::consts::PI;

impl IndoorManifest {
    pub(super) fn furnish_perimeter_services(&mut self) {
        let mut r = stream(self.seed, 0x5045_5249_4d45_5445);
        let mut kinds = vec![ObjectKind::Cabinet, ObjectKind::Bookcase];
        for _ in 0..r.random_range(1..=4) {
            if r.random_bool((0.45 + self.density * 0.5) as f64) {
                kinds.push(ObjectKind::Plant);
            }
        }
        kinds.push(ObjectKind::TrashCan);
        if self.layout.prefers_soft_seating() || r.random_bool(0.45) {
            kinds.push(ObjectKind::FloorLamp);
        }
        use rand::seq::SliceRandom;
        kinds.shuffle(&mut r);
        for kind in kinds {
            let size = match kind {
                ObjectKind::Cabinet => Vec3::new(
                    r.random_range(0.65..1.85),
                    r.random_range(0.66..1.28),
                    r.random_range(0.33..0.59),
                ),
                ObjectKind::Bookcase => Vec3::new(
                    r.random_range(0.65..1.55),
                    r.random_range(1.05..2.08),
                    r.random_range(0.26..0.46),
                ),
                ObjectKind::Plant => {
                    let width = r.random_range(0.43..1.03);
                    Vec3::new(
                        width,
                        r.random_range(0.68..1.98),
                        width * r.random_range(0.78..1.13),
                    )
                }
                ObjectKind::FloorLamp => {
                    super::super::super::objects::standing_lamp::dimensions(r.random())
                }
                ObjectKind::TrashCan => {
                    super::super::super::objects::containers::bins::dimensions(&mut r)
                }
                _ => unreachable!(),
            };
            let storage = matches!(kind, ObjectKind::Cabinet | ObjectKind::Bookcase);
            for _ in 0..4 {
                let (wall, span) = self.attachment_wall(&mut r);
                let yaw = wall.rotation.to_euler(EulerRot::YXZ).0
                    + if storage {
                        r.random_range(-0.055..0.055)
                    } else {
                        r.random_range(-PI..PI)
                    };
                let relative = yaw - wall.rotation.to_euler(EulerRot::YXZ).0;
                let (sin, cos) = relative.sin_cos();
                let width = size.x * cos.abs() + size.z * sin.abs();
                let depth = size.z * cos.abs() + size.x * sin.abs();
                let available = (span - width) * 0.5 - 0.20;
                if available <= 0. {
                    continue;
                }
                let along = r.random_range(-1.0..1.0) * available;
                let gap = if storage {
                    r.random_range(0.10..0.27)
                } else {
                    r.random_range(0.15..0.54)
                };
                let position = wall.transform_point(Vec3::new(along, 0., depth * 0.5 + gap));
                if self.add(kind, position, size, yaw, &mut r).is_some() {
                    break;
                }
            }
        }
    }
}
