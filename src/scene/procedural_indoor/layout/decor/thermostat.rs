//! A room controller needs opaque backing, air circulation, and a clear place
//! to stand. Product sampling is independent of mounts and other wall objects.
use super::*;
use crate::scene::procedural_indoor::{
    envelope::polygon,
    objects::wall::{self, thermostat},
};

impl IndoorManifest {
    pub(in super::super) fn furnish_thermostat(&mut self) {
        let mut product = stream(self.seed, u64::from_le_bytes(*b"TstatLot"));
        let probability = match self.layout {
            IndoorLayout::ElevatorLanding => 0.23,
            IndoorLayout::Kitchen | IndoorLayout::Cafeteria => 0.39,
            _ => 0.65,
        };
        if !product.random_bool(probability) {
            return;
        }
        let mut o = self.candidate(
            ObjectKind::Thermostat,
            Vec3::ZERO,
            Vec3::ONE,
            0.,
            &mut product,
        );
        o.solid = false;
        thermostat::fit(&mut o);
        let mut location = stream(self.seed, u64::from_le_bytes(*b"TstatLoc"));
        let height = location.random_range(1.14..1.60);
        let offset = wall::mount_offset(&o).unwrap();
        let structure = self.camera_obstacles();
        for _ in 0..20 {
            let (tf, span) = match &self.entry_door {
                Some(d) if location.random_bool(0.32) => (
                    Transform::from_xyz(0., 0., d.center.z)
                        .with_rotation(Quat::from_rotation_y(std::f32::consts::PI)),
                    self.room_size.x,
                ),
                _ => self.attachment_wall(&mut location),
            };
            o.yaw = tf.rotation.to_euler(EulerRot::YXZ).0;
            o.position = tf.transform_point(Vec3::new(
                location.random_range(-0.47..0.47) * span,
                0.,
                offset,
            ));
            o.position.y =
                self.floor_height(thermostat::standing_point(&o).xz()) + height - o.size.y * 0.5;
            if self.thermostat_accessible(&o, &structure) {
                self.objects.push(o);
                return;
            }
        }
    }

    pub(crate) fn thermostat_accessible(
        &self,
        o: &IndoorObject,
        structure: &[(Vec3, Vec3)],
    ) -> bool {
        if o.solid || o.neighbor || o.support.is_some() || !self.wall_attachment_clear(o) {
            return false;
        }
        let stand = thermostat::standing_point(o);
        let floor = self.floor_height(stand.xz());
        let center = o.position + Vec3::Y * (o.size.y * 0.5);
        if !(1.10..=1.64).contains(&(center.y - floor)) || !self.in_primary_room(stand, 0.2) {
            return false;
        }
        let lo = Vec3::new(stand.x - 0.18, floor, stand.z - 0.18);
        let hi = Vec3::new(stand.x + 0.18, floor + 1.72, stand.z + 0.18);
        if self.envelope.as_ref().is_some_and(|e| {
            !e.support_clear(lo, hi)
                || !polygon::box_inside(&e.footprint, lo.xz(), hi.xz(), 0.004)
                || [
                    lo.xz(),
                    hi.xz(),
                    Vec2::new(lo.x, hi.z),
                    Vec2::new(hi.x, lo.z),
                ]
                .iter()
                .any(|&p| (e.floor_height(p) - floor).abs() > 0.045)
        }) {
            return false;
        }
        let hand = stand.with_y(center.y);
        let clearance = [
            o.bounds(),
            (lo + Vec3::Y * 0.085, hi),
            (
                center.min(hand) - Vec3::splat(0.047),
                center.max(hand) + Vec3::splat(0.047),
            ),
        ];
        let free = |bound| {
            clearance
                .iter()
                .all(|&part| electrical::installation_clear(part, std::slice::from_ref(&bound)))
        };
        if !structure.iter().copied().all(free) || !self.humans.iter().all(|h| free(h.bounds())) {
            return false;
        }
        self.objects
            .iter()
            .filter(|p| p.id != o.id && !p.neighbor)
            .all(|p| {
                let (a, b) = p.bounds();
                if matches!(
                    p.kind,
                    ObjectKind::BaseboardHeater
                        | ObjectKind::Range
                        | ObjectKind::Cooktop
                        | ObjectKind::Oven
                ) && center.xz().distance(center.xz().clamp(a.xz(), b.xz())) < 0.78
                {
                    return false;
                }
                free((a, b)) || p.obstacle_bounds().all(free)
            })
    }
}
