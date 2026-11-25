//! Controls on the room side of the entrance partition. Its sidelights are
//! constructed separately from exterior wall apertures.
use super::*;
use crate::scene::procedural_indoor::{
    architecture::{
        doors::{DoorInstallation, DoorMechanism},
        interior_glazing,
    },
    envelope::polygon,
    objects::wall,
};

pub(super) fn on_entry_face(door: &DoorInstallation, o: &IndoorObject) -> bool {
    wall::mount_offset(o).is_some_and(|offset| {
        (o.position.z - (door.center.z - offset)).abs() < 0.001
            && (o.transform().rotation * Vec3::Z).dot(Vec3::NEG_Z) > 0.99999
    })
}

#[cfg(test)]
mod tests;

fn solid_pier(
    scene: &IndoorManifest,
    door: &DoorInstallation,
    f: &facade::Facade,
    o: &IndoorObject,
) -> bool {
    let half_width = o.size.x * 0.5;
    // The shared wall's local +X points towards world -X. Include the full
    // plate and space for the jamb casing, aperture edge and fixings.
    let x = -o.position.x;
    x.abs() + half_width < scene.room_size.x * 0.5 - 0.08
        && (o.position.x - door.center.x).abs() - half_width > door.width * 0.5 + 0.09
        && !f.openings.iter().any(|w| {
            x + half_width > w.min.x - 0.065
                && x - half_width < w.max.x + 0.065
                && o.position.y + o.size.y > w.min.y - 0.065
                && o.position.y < w.max.y + 0.065
        })
}

impl IndoorManifest {
    pub(crate) fn entry_wall_attachment_clear(&self, o: &IndoorObject) -> bool {
        let Some(door) = self.entry_door.as_ref().filter(|d| on_entry_face(d, o)) else {
            return false;
        };
        let height = interior_glazing::partition_height(self);
        let f = interior_glazing::shared_facade(self, height);
        o.position.y >= 0. && o.position.y + o.size.y < height && solid_pier(self, door, &f, o)
    }

    pub(super) fn fit_entry_switch(
        &self,
        plate: &IndoorObject,
        height_above_floor: f32,
        occupied: &[(Vec3, Vec3)],
    ) -> Option<IndoorObject> {
        let door = self.entry_door.as_ref()?;
        let mut rng = stream(self.seed, 0x454e_5452_5953_5754);
        if !rng.random_bool(0.90) {
            return None;
        }
        let favored = match door.program.mechanism {
            // Both the hinged pivot and the parked sliding leaves occupy the
            // handed side. The opposite jamb is the natural first choice.
            DoorMechanism::Hinged | DoorMechanism::Sliding | DoorMechanism::Telescoping => {
                if rng.random_bool(0.92) {
                    -door.program.handedness
                } else {
                    door.program.handedness
                }
            }
            DoorMechanism::Paired | DoorMechanism::Revolving => {
                if rng.random_bool(0.5) {
                    -1.
                } else {
                    1.
                }
            }
        };
        let height = interior_glazing::partition_height(self);
        let glazing = interior_glazing::shared_facade(self, height);
        let mut portals = door.obstacles();
        if let Some(program) = &self.program {
            portals.extend(
                program
                    .partitions
                    .iter()
                    .flat_map(|p| p.portal_obstacles(self.room_size.y)),
            );
        }
        let mut o = plate.clone();
        o.yaw = std::f32::consts::PI;
        o.position.z = door.center.z - wall::mount_offset(&o)?;
        for side in [favored, -favored] {
            // First draw spans the near jamb; stratified alternatives find
            // narrow wall piers or a position beyond an obstruction.
            for (min, max) in [
                (0.12, 0.65),
                (0.095, 0.20),
                (0.20, 0.42),
                (0.42, 0.78),
                (0.78, 1.26),
            ] {
                o.position.x = door.center.x
                    + side * (door.width * 0.5 + o.size.x * 0.5 + rng.random_range(min..max));
                // Allow someone to stand at the control. Use the finished
                // structural floor under their feet, including steps.
                let standing = Vec2::new(o.position.x, door.center.z - 0.39);
                let floor = self
                    .envelope
                    .as_ref()
                    .map_or(0., |e| e.floor_height(standing));
                o.position.y = floor + height_above_floor;
                if o.position.y < 0.
                    || o.position.y + o.size.y >= height
                    || !solid_pier(self, door, &glazing, &o)
                    || !electrical::installation_clear(o.bounds(), occupied)
                {
                    continue;
                }
                let lo = Vec3::new(standing.x - 0.22, floor, standing.y - 0.25);
                let hi = Vec3::new(standing.x + 0.22, floor + 1.76, standing.y + 0.25);
                if self.envelope.as_ref().is_some_and(|e| {
                    !e.support_clear(lo, hi)
                        || !polygon::box_inside(&e.footprint, lo.xz(), hi.xz(), 0.005)
                }) {
                    continue;
                }
                // Ignore floor finishes below ankle level, but reject a
                // cabinet between the operator and the plate as well.
                let body = (lo + Vec3::Y * 0.08, hi);
                let hand_y = o.position.y + o.size.y * 0.5;
                let reach = (
                    Vec3::new(standing.x - 0.06, hand_y - 0.06, hi.z),
                    Vec3::new(standing.x + 0.06, hand_y + 0.06, o.position.z),
                );
                if [body, reach].into_iter().all(|bounds| {
                    electrical::installation_clear(bounds, occupied)
                        && electrical::installation_clear(bounds, &portals)
                }) && electrical::installation_clear(o.bounds(), &portals)
                {
                    return Some(o);
                }
            }
        }
        None
    }
}
