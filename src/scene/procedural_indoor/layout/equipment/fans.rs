use super::*;
use crate::scene::procedural_indoor::{
    architecture::{self, ceiling::CeilingGrid},
    objects::fans,
};

#[derive(Clone, Copy)]
struct FanStock {
    variant: u32,
    finish: u64,
    diameter: f32,
    drop: f32,
}

impl FanStock {
    fn sample(seed: u64, ordinal: u64) -> Self {
        let mut r = stream(seed, u64::from_le_bytes(*b"FanModel") ^ ordinal);
        let diameter = r.random_range(0.85..1.95);
        Self {
            variant: fans::installed_variant(r.random(), diameter),
            finish: r.random(),
            diameter,
            drop: r.random_range(0.28..0.85),
        }
    }
}

impl IndoorManifest {
    pub(in super::super::super) fn furnish_ceiling_fans(&mut self) {
        let mut r = stream(self.seed, 0x4346414e);
        if !r.random_bool(0.30 + 0.30 * self.density as f64) {
            return;
        }
        let pair = self.room_size.x * self.room_size.z > 55. && r.random_bool(0.45);
        let matching = pair && r.random_bool(0.73);
        let stock = FanStock::sample(self.seed, 0);
        let mut obstacles = fan_obstacles(self, None);
        let mut first_anchor: Option<Vec2> = None;
        for ordinal in 0..=u64::from(pair) {
            let model = if ordinal == 0 || matching {
                stock
            } else {
                FanStock::sample(self.seed, ordinal)
            };
            let mut fan = self.candidate(
                ObjectKind::CeilingFan,
                Vec3::ZERO,
                Vec3::new(model.diameter, model.drop, model.diameter),
                r.random_range(-PI..PI),
                &mut r,
            );
            fan.variant = model.variant;
            fan.finish_seed = Some(model.finish);
            fan.solid = false;
            let floor_rise = super::super::floor::area::MAX_THICKNESS_M;
            let radius = Vec2::splat(model.diameter * 0.5);
            for attempt in 0..24 {
                let sample = Vec2::new(r.random_range(-0.38..0.38), r.random_range(-0.38..0.38))
                    * self.room_size.xz();
                let center = if let Some(first) = first_anchor.filter(|_| matching && attempt < 4) {
                    -first + sample * 0.22
                } else {
                    sample
                };
                if !self.in_primary_room(Vec3::new(center.x, 0., center.y), radius.x + 0.24) {
                    continue;
                }
                let Some(roof) = super::canopy_support_height(self, center, fans::canopy(&fan))
                else {
                    continue;
                };
                // A downrod can be shortened at installation. Reserve future rug
                // thickness across the entire disk, including raised landings.
                let floor =
                    architecture::ceiling::highest_floor(self, center - radius, center + radius);
                let drop = model.drop.min(roof - floor - floor_rise - 2.1 - 1e-5);
                if drop < 0.28 {
                    continue;
                }
                fan.size.y = drop;
                fan.position = Vec3::new(center.x, roof - drop, center.y);
                if !volumes_clear(self, &fan, floor_rise, &obstacles) {
                    continue;
                }
                obstacles.extend(fan.obstacle_bounds());
                first_anchor = Some(center);
                self.objects.push(fan);
                break;
            }
        }
    }
}

fn fan_obstacles(scene: &IndoorManifest, exclude: Option<usize>) -> Vec<(Vec3, Vec3)> {
    scene
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
                .filter(|o| Some(o.id) != exclude)
                .flat_map(IndoorObject::obstacle_bounds),
        )
        .chain(scene.humans.iter().map(|h| h.bounds()))
        .collect()
}

pub(crate) fn fan_clear(scene: &IndoorManifest, o: &IndoorObject) -> bool {
    if o.kind != ObjectKind::CeilingFan
        || o.solid
        || o.neighbor
        || o.support.is_some()
        || !scene.in_primary_room(o.position, o.size.x.min(o.size.z) * 0.5 + 0.24)
    {
        return false;
    }
    let center = o.position.xz();
    let top = o.position.y + o.size.y;
    let contact = if fans::installed(o) {
        super::canopy_support_height(scene, center, fans::canopy(o))
    } else {
        // Archived mounting datum.
        Some(
            [-0.18, 0., 0.18]
                .into_iter()
                .flat_map(|dx| [-0.18, 0., 0.18].map(move |dz| center + Vec2::new(dx, dz)))
                .map(|p| scene.ceiling_height(p))
                .fold(f32::INFINITY, f32::min)
                - 0.004,
        )
    };
    contact.is_some_and(|y| (top - y).abs() < 0.002)
        && volumes_clear(scene, o, 0., &fan_obstacles(scene, Some(o.id)))
}

fn volumes_clear(
    scene: &IndoorManifest,
    o: &IndoorObject,
    floor_rise: f32,
    obstacles: &[(Vec3, Vec3)],
) -> bool {
    let center = o.position.xz();
    let top = o.position.y + o.size.y;
    let grid = CeilingGrid::for_scene(scene);
    for (index, (a, b)) in fans::local_reservations(o).into_iter().enumerate() {
        let (a, b) = (a + o.position, b + o.position);
        if a.y < architecture::ceiling::highest_floor(scene, a.xz(), b.xz()) + floor_rise + 2.1 {
            return false;
        }
        if fans::installed(o) {
            let ceiling = scene.minimum_ceiling_height(a.xz(), b.xz());
            if index == 0 {
                if b.y >= ceiling - grid.map_or(0.001, |g| g.tile.drop_m + 0.001) {
                    return false;
                }
            } else if grid.is_none() {
                let c = fans::canopy(o);
                if top - c.pad_height_m
                    > scene.minimum_ceiling_height(center - c.radius_m, center + c.radius_m)
                    || top - c.height_m > ceiling
                {
                    return false;
                }
            }
        }
        let corners = [a.xz(), b.xz(), Vec2::new(a.x, b.z), Vec2::new(b.x, a.z)];
        if corners.into_iter().any(|p| {
            !scene.in_primary_room(Vec3::new(p.x, a.y, p.y), 0.12)
                || a.y < scene.floor_height(p) + floor_rise + 2.1
                || (!fans::installed(o) && b.y >= scene.ceiling_height(p) - 0.001)
        }) || obstacles
            .iter()
            .any(|&(lo, hi)| a.cmplt(hi + 0.025).all() && b.cmpgt(lo - 0.025).all())
        {
            return false;
        }
    }
    true
}
