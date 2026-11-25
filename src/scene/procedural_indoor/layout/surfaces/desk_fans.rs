//! A personal fan may serve a seat at its workstation. Room circulation,
//! storage and parked fans retain freely sampled headings.
use super::*;
use crate::scene::procedural_indoor::objects::{accessories::fan, chairs};

impl IndoorManifest {
    pub(in crate::scene::procedural_indoor::layout) fn orient_desk_fans(&mut self) {
        for index in 0..self.objects.len() {
            let fan = &self.objects[index];
            if fan.kind != ObjectKind::DeskFan || fan.interaction_target.is_some() {
                continue;
            }
            if let Some((yaw, chair)) = self.seated_fan_heading(fan) {
                self.objects[index].yaw = yaw;
                self.objects[index].interaction_target = Some(chair);
            }
        }
    }

    fn seated_fan_heading(&self, object: &IndoorObject) -> Option<(f32, usize)> {
        let support = self.objects.get(object.support?)?;
        if !matches!(support.kind, ObjectKind::Table | ObjectKind::Desk) {
            return None;
        }
        let mut rng = stream(object.seed, u64::from_le_bytes(*b"AirToSit"));
        if !rng.random_bool(0.76) {
            return None;
        }
        let p = fan::parameters(object);
        let origin = object.transform().transform_point(p.center);
        let chair = self
            .objects
            .iter()
            .filter(|c| {
                c.kind == ObjectKind::Chair
                    && c.interaction_target == Some(support.id)
                    && c.neighbor == object.neighbor
                    && chairs::supports_upright_occupant(c)
            })
            .filter_map(|c| {
                let d = (c.position - origin).xz().length();
                let height = origin.y - c.position.y - d * p.pitch.tan();
                // Keep the sampled tilt. Only aim at a workstation when its ray
                // reaches the seated working volume above the actual cushion.
                let seat = chairs::seat_surface_height(c);
                ((0.32..=2.25).contains(&d) && (seat + 0.18..=seat + 0.94).contains(&height))
                    .then(|| (c, d * rng.random_range(0.78..1.22)))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))?
            .0;
        let offset = chair.transform().rotation
            * Vec3::new(
                chair.size.x * rng.random_range(-0.26..0.26),
                0.,
                chair.size.z * rng.random_range(-0.14..0.22),
            );
        let delta = chair.position + offset - origin;
        let yaw = delta.x.atan2(delta.z) - p.yaw;
        let direction = Quat::from_rotation_y(yaw) * p.head_transform().rotation * Vec3::Z;
        let end = origin + direction * (delta.xz().length() / direction.xz().length());
        let cushion_y = chair.position.y + chairs::seat_surface_height(chair);
        if !(cushion_y + 0.18..=cushion_y + 0.94).contains(&end.y) {
            return None;
        }
        let from = origin + direction * (p.depth * 0.5);
        if self
            .objects
            .iter()
            .filter(|other| {
                other.id != object.id
                    && other.id != chair.id
                    && other.id != support.id
                    && (other.solid || other.support == Some(support.id))
            })
            .any(|other| {
                let (lo, hi) = other.bounds();
                segment_hits_box(from, end, lo, hi)
            })
        {
            return None;
        }
        let mut proposal = object.clone();
        proposal.yaw = yaw;
        let (lo, hi) = proposal.bounds();
        if !self.prop_clear(&proposal, support, 0.012)
            || self.humans.iter().any(|human| {
                let (a, b) = human.bounds();
                lo.cmplt(b).all() && hi.cmpgt(a).all()
            })
        {
            return None;
        }
        Some((yaw, chair.id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::objects::audit::fixture;
    use std::f32::consts::PI;

    #[test]
    fn desk_fan_serves_a_clear_workstation_or_keeps_its_stashed_heading() {
        let mut room =
            IndoorManifest::generate_with_humans(42, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        room.objects.clear();
        let mut desk = fixture(ObjectKind::Desk, 0);
        desk.id = 0;
        desk.position = Vec3::ZERO;
        desk.size = Vec3::new(2., 0.76, 1.25);
        let mut seat = fixture(ObjectKind::Chair, 1);
        seat.id = 1;
        seat.interaction_target = Some(0);
        seat.variant = 0;
        seat.position = Vec3::new(0., 0., 1.16);
        let mut object = fixture(ObjectKind::DeskFan, 1);
        object.id = 2;
        object.support = Some(0);
        object.solid = false;
        object.position = Vec3::new(0.38, 0.76, 0.15);
        object.yaw = PI;
        room.objects.extend([desk, seat, object.clone()]);
        let mut aimed = 0;
        for seed in 0..96 {
            object.seed = seed;
            room.objects[2] = object.clone();
            room.orient_desk_fans();
            let fan = &room.objects[2];
            let p = fan::parameters(fan);
            if fan.interaction_target == Some(1) {
                aimed += 1;
                let axis = fan.transform().rotation * p.head_transform().rotation * Vec3::Z;
                let origin = fan.transform().transform_point(p.center);
                let crossing = origin + axis * ((1.16 - origin.z) / axis.z);
                assert!(
                    axis.z > 0. && crossing.x.abs() < 0.32 && (0.58..1.49).contains(&crossing.y)
                );
                assert!(room.prop_clear(fan, &room.objects[0], 0.012));
                let mut monitor = fixture(ObjectKind::Monitor, 2);
                monitor.id = 3;
                monitor.position = (origin + crossing) * 0.5 - Vec3::Y * 0.35;
                monitor.size = Vec3::new(0.85, 0.72, 0.045);
                monitor.support = Some(0);
                room.objects.push(monitor);
                room.objects[2] = object.clone();
                room.orient_desk_fans();
                assert_eq!(room.objects[2].interaction_target, None);
                assert_eq!(room.objects[2].yaw, PI);
                room.objects.pop();
            } else {
                assert_eq!(fan.yaw, PI);
            }
        }
        assert!((32..90).contains(&aimed), "{aimed}");
        room.objects[0].kind = ObjectKind::Cabinet;
        for seed in 0..32 {
            object.seed = seed;
            assert!(room.seated_fan_heading(&object).is_none());
        }
    }
}
