//! Near-window hardscape. Separate pavers/kerbs give metric perspective cues;
//! all strips remain outside the primary room and neighboring room footprint.
use super::*;
mod access;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
pub(super) struct Entrance {
    pub threshold: Vec2,
    pub outward: Vec2,
    pub width: f32,
}

struct Strip {
    min: Vec2,
    max: Vec2,
    outward: Vec2,
}
impl Strip {
    fn axis(&self) -> Vec2 {
        self.outward.abs().yx()
    }
    fn length(&self) -> f32 {
        (self.max - self.min).dot(self.axis())
    }
    fn lip(&self) -> Vec2 {
        (self.min + self.max) * 0.5
            + self.outward * ((self.max - self.min).dot(self.outward.abs()) * 0.5)
    }
}

fn strips(scene: &IndoorManifest, width: f32) -> [Strip; 4] {
    let half = scene.room_size.xz() * 0.5;
    let lo = Vec2::new(-half.x - 0.55, -half.y - 0.55);
    let hi = Vec2::new(
        half.x + 0.55,
        half.y + scene.adjoining_space().max_depth() + 0.55,
    );
    // Four disjoint perimeter strips with mitre-free corner allocation.
    [
        (
            Vec2::new(lo.x - width, lo.y - width),
            Vec2::new(lo.x, hi.y + width),
            Vec2::NEG_X,
        ),
        (
            Vec2::new(hi.x, lo.y - width),
            Vec2::new(hi.x + width, hi.y + width),
            Vec2::X,
        ),
        (
            Vec2::new(lo.x, lo.y - width),
            Vec2::new(hi.x, lo.y),
            Vec2::NEG_Y,
        ),
        (
            Vec2::new(lo.x, hi.y),
            Vec2::new(hi.x, hi.y + width),
            Vec2::Y,
        ),
    ]
    .map(|(min, max, outward)| Strip { min, max, outward })
}

pub(super) fn measure(
    scene: &IndoorManifest,
    sites: &[Parcel],
    rows: &[midground::Terrace],
    entrances: &[(usize, Entrance)],
    out: &mut Vec<(&'static str, f32)>,
) {
    let sides = strips(scene, District::sample(scene.seed).paving_width);
    let walks = access::plan(&sides, sites, rows, entrances);
    out.push(("exterior_door_access_walks", walks.len() as f32));
    for walk in walks {
        out.push(("exterior_access_walk_length_m", walk.length()));
        out.push((
            "exterior_access_walk_landing_width_m",
            walk.end[0].distance(walk.end[1]),
        ));
    }
}

pub(super) fn build(
    a: &mut Assembly,
    scene: &IndoorManifest,
    sites: &[Parcel],
    terraces: &[midground::Terrace],
    entrances: &[(usize, Entrance)],
) {
    let mut rng = stream(scene.seed, 6370);
    let district = District::sample(scene.seed);
    let width = district.paving_width;
    let finish = super::super::super::materials::exterior::label(
        Surface::Concrete,
        "floor",
        scene.seed ^ 6370,
    );
    let strips = strips(scene, width);
    let paths = access::plan(&strips, sites, terraces, entrances);
    if !paths.is_empty() {
        access::pave(a, &paths, &finish);
    }
    let pitch = district.paving_pitch;
    for (index, strip) in strips.into_iter().enumerate() {
        let Strip { min, max, outward } = strip;
        let extent = max - min;
        let axis = strip.axis();
        let cross = Vec2::new(axis.y, axis.x);
        let count = (extent.dot(axis) / pitch).ceil() as usize;
        let cell = extent.dot(axis) / count as f32;
        for i in 0..count {
            let center =
                (min + max) * 0.5 + axis * ((i as f32 + 0.5) * cell - extent.dot(axis) * 0.5);
            let size = axis * (cell - 0.009) + cross * (width - 0.008);
            a.box_part(
                Surface::Concrete,
                &finish,
                Vec3::new(center.x, -0.035, center.y),
                Vec3::new(size.x, 0.07, size.y),
                0.003,
            );
        }
        // A continuous curb borders the outer edge; expansion gaps separate blocks.
        let mut gaps: Vec<_> = paths
            .iter()
            .filter(|p| p.side == index)
            .map(|p| p.opening)
            .collect();
        gaps.sort_by(|a, b| a[0].total_cmp(&b[0]));
        for i in 0..count {
            let start = i as f32 * cell - extent.dot(axis) * 0.5;
            for (lo, hi) in access::intervals(start + 0.006, start + cell - 0.006, &gaps) {
                let center =
                    (min + max) * 0.5 + axis * ((lo + hi) * 0.5) + outward * (width * 0.5 - 0.035);
                let size = axis * (hi - lo) + cross * 0.055;
                a.box_part(
                    Surface::Concrete,
                    "other_structure",
                    Vec3::new(center.x, 0.0325, center.y),
                    Vec3::new(size.x, 0.065, size.y),
                    0.006,
                );
            }
        }
        // Occasional short metal railing on the remote half of a walkway.
        // Railings are capped, with posts anchored into the paving below them.
        if !rng.random_bool(district.railing_probability as f64) {
            continue;
        }
        let base = (min + max) * 0.5 + outward * (width * 0.5 - 0.08);
        let half = extent.dot(axis) * 0.26;
        let height = rng.random_range(0.75..1.15);
        let point = |p: Vec2, y| Vec3::new(p.x, y, p.y);
        for (lo, hi) in access::intervals(-half, half, &gaps) {
            if hi - lo < 0.40 {
                continue;
            }
            let (start, end) = (base + axis * lo, base + axis * hi);
            for y in [height * 0.48, height] {
                a.part(Surface::Metal, "other_structure").rod(
                    point(start, y),
                    point(end, y),
                    0.016,
                );
            }
            let posts = start.distance(end).ceil() as usize;
            for i in 0..=posts {
                let p = start.lerp(end, i as f32 / posts as f32);
                a.part(Surface::Metal, "other_structure").rod(
                    point(p, 0.),
                    point(p, height),
                    0.018,
                );
            }
        }
    }
}
