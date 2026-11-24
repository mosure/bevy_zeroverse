//! Optional visitor storage at the sides of an entrance. Sample the product
//! once and fit both its footprint and a free standing position beside it.
use super::*;
use crate::scene::procedural_indoor::{objects::utilities::racks, placement::rotated_half};

impl IndoorManifest {
    pub(in super::super) fn furnish_entry_coat_rack(&mut self) {
        let demand = match self.layout {
            IndoorLayout::Reception => 0.48,
            IndoorLayout::Conference | IndoorLayout::Lounge => 0.33,
            IndoorLayout::Coworking | IndoorLayout::Training | IndoorLayout::Studio => 0.22,
            IndoorLayout::Library | IndoorLayout::Gallery | IndoorLayout::OpenOffice => 0.12,
            _ => return,
        };
        let mut r = stream(self.seed, 0x454e_5452_5952_4143);
        if !r.random_bool(demand)
            || self
                .objects
                .iter()
                .any(|o| !o.neighbor && o.kind == ObjectKind::CoatRack)
        {
            return;
        }
        let Some(door) = &self.entry_door else { return };
        let wall = door.transform();
        let jamb = door.width * 0.5;
        let size = racks::dimensions(&mut r);
        let mut rack = self.floor_candidate(ObjectKind::CoatRack, Vec3::ZERO, size, 0., &mut r);
        rack.variant = racks::ENTRY_RACK;
        let first = if r.random_bool(0.5) { 1. } else { -1. };
        for i in 0..8 {
            let side = if i % 2 == 0 { first } else { -first };
            let relative_yaw = if size.x > size.z * 1.35 {
                r.random_range(-0.48..0.48)
            } else {
                r.random_range(-std::f32::consts::PI..std::f32::consts::PI)
            };
            rack.yaw = door.yaw + relative_yaw;
            let half = rotated_half(size.xz(), relative_yaw);
            let local = Vec3::new(
                side * (jamb + half.x + r.random_range(0.23..1.45)),
                0.,
                -half.y - r.random_range(0.37..1.10),
            );
            let center = wall.transform_point(local);
            rack.position = center.with_y(self.floor_height(center.xz()));
            if !self.placement_clear(&rack, 0.075) {
                continue;
            }
            // People reach the rail from the room, clear of the door swing.
            // Test at the same floor level, including recessed floors and stairs.
            let pad = wall.transform_point(local - Vec3::Z * (half.y + 0.31));
            let mut person_space = rack.clone();
            person_space.position = pad.with_y(rack.position.y);
            person_space.size = Vec3::new(0.48, 1.65, 0.48);
            person_space.yaw = door.yaw;
            if (self.floor_height(pad.xz()) - rack.position.y).abs() < 0.012
                && self.placement_clear(&person_space, 0.05)
            {
                self.objects.push(rack);
                return;
            }
        }
    }
}
