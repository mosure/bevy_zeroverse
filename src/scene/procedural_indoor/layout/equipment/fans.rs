use super::*;
use crate::scene::procedural_indoor::{architecture, objects::fans};

impl IndoorManifest {
    pub(in super::super::super) fn furnish_ceiling_fans(&mut self) {
        let mut r = stream(self.seed, 0x4346414e);
        if !r.random_bool(0.30 + 0.30 * self.density as f64) {
            return;
        }
        let wanted = if self.room_size.x * self.room_size.z > 55. && r.random_bool(0.45) {
            2
        } else {
            1
        };
        for _ in 0..wanted {
            for _ in 0..24 {
                let diameter = r.random_range(0.85..1.95);
                let height = r.random_range(0.28..0.85);
                let x = r.random_range(-0.38..0.38) * self.room_size.x;
                let z = r.random_range(-0.38..0.38) * self.room_size.z;
                let position = Vec3::new(x, 0., z);
                // A small flush canopy fits the lowest roof over its radius.
                // The rotor has its own, lower reserve on sloped ceilings.
                let roof = [-0.18, 0., 0.18]
                    .into_iter()
                    .flat_map(|dx| [-0.18, 0., 0.18].map(move |dz| Vec2::new(x + dx, z + dz)))
                    .map(|p| self.ceiling_height(p))
                    .fold(f32::INFINITY, f32::min);
                let mut o = self.candidate(
                    ObjectKind::CeilingFan,
                    position.with_y(roof - height - 0.004),
                    Vec3::new(diameter, height, diameter),
                    r.random_range(0.0..2. * PI),
                    &mut r,
                );
                o.solid = false;
                if !fan_clear(self, &o) {
                    continue;
                }
                self.objects.push(o);
                break;
            }
        }
    }
}

pub(crate) fn fan_clear(scene: &IndoorManifest, o: &IndoorObject) -> bool {
    if o.kind != ObjectKind::CeilingFan || o.solid || o.neighbor || o.support.is_some() {
        return false;
    }
    if !scene.in_primary_room(o.position, o.size.x.min(o.size.z) * 0.5 + 0.24) {
        return false;
    }
    let roof = [-0.18, 0., 0.18]
        .into_iter()
        .flat_map(|dx| [-0.18, 0., 0.18].map(move |dz| o.position.xz() + Vec2::new(dx, dz)))
        .map(|p| scene.ceiling_height(p))
        .fold(f32::INFINITY, f32::min);
    if (o.position.y + o.size.y + 0.004 - roof).abs() > 0.002 {
        return false;
    }
    let obstacles: Vec<_> = scene
        .camera_obstacles()
        .into_iter()
        .chain(architecture::fixture_positions(scene).into_iter().map(|p| {
            let half = architecture::fixture_half_extents(scene, p);
            (
                p - half - Vec3::Y * 0.01,
                Vec3::new(p.x + half.x, scene.ceiling_height(p.xz()), p.z + half.z),
            )
        }))
        .chain(
            scene
                .objects
                .iter()
                .filter(|other| other.id != o.id)
                // Use the same physical volumes in both directions: a tall
                // fan's empty space above its blades is not an obstacle to a
                // neighboring rotor. Cable paths likewise reserve segments.
                .flat_map(IndoorObject::obstacle_bounds),
        )
        .chain(scene.humans.iter().map(|h| h.bounds()))
        .collect();
    fans::local_reservations(o).into_iter().all(|(a, b)| {
        let (a, b) = (a + o.position, b + o.position);
        if a.y < architecture::ceiling::highest_floor(scene, a.xz(), b.xz()) + 2.1 {
            return false;
        }
        let corners = [a.xz(), b.xz(), Vec2::new(a.x, b.z), Vec2::new(b.x, a.z)];
        corners.into_iter().all(|p| {
            scene.in_primary_room(Vec3::new(p.x, a.y, p.y), 0.12)
                && a.y >= scene.floor_height(p) + 2.1
                && b.y < scene.ceiling_height(p) - 0.001
        }) && !obstacles.iter().any(|&(lo, hi)| {
            a.cmplt(hi + Vec3::splat(0.025)).all() && b.cmpgt(lo - Vec3::splat(0.025)).all()
        })
    })
}
