//! Streets beyond the individual near-window parcels. Reserve actual footprint
//! and roof clearance inside the finite sky; open campus clearings can remain.
use super::*;
use crate::scene::procedural_indoor::footprint::Footprint;

#[derive(Debug, PartialEq)]
pub(super) struct Terrace {
    pub center: Vec2,
    pub size: Vec2,
    pub front: Vec2,
    pub apron: f32,
    pub eaves_limit: f32,
    pub seed: u64,
    pub finish_seed: u64,
}

impl Terrace {
    pub fn yaw(&self) -> f32 {
        self.front.x.atan2(self.front.y)
    }
    pub fn axis(&self) -> Vec2 {
        Vec2::new(self.front.y, -self.front.x)
    }
    pub fn footprint(&self) -> Footprint {
        Footprint::oriented(
            self.center + self.front * (self.apron * 0.5),
            self.size + Vec2::new(0., self.apron),
            self.yaw(),
        )
    }
}

pub(super) fn plan(
    scene: &IndoorManifest,
    windows: &[(Vec2, Vec2)],
    sites: &[Parcel],
) -> Vec<Terrace> {
    let mut rows: Vec<Terrace> = Vec::new();
    if windows.is_empty() {
        return rows;
    }
    let mut rng = stream(scene.seed, 0x5354_5245_4554_524f);
    let district = District::sample(scene.seed);
    let target = ((district.building_fraction * 3.5 + rng.random_range(-0.9..1.0)) as usize).min(3);
    if target == 0 {
        return rows;
    }
    let half = scene.room_size.xz() * 0.5;
    let neighboring = scene.adjoining_space().max_depth();
    let primary = Footprint::oriented(
        Vec2::Y * neighboring * 0.5,
        half * 2. + Vec2::new(0., neighboring),
        0.,
    );
    let occupied: Vec<_> = sites
        .iter()
        .map(|p| Footprint::oriented(p.center, p.size, p.yaw))
        .collect();
    let occluders: Vec<_> = sites
        .iter()
        .map(|p| {
            let (s, c) = p.yaw.sin_cos();
            (
                p.center,
                Vec2::new(c, -s) * (p.size.x * 0.5),
                Vec2::new(s, c) * (p.size.y * 0.5),
                if p.kind == 0 { 1. } else { 0.4 },
            )
        })
        .collect();
    let mut candidates = Vec::new();
    // The analytic sky sphere is tessellated; keep even the roof corners
    // inside its inscribed surface. The building and sidewalk are on grade.
    let sky = super::super::background::radius(scene) * 0.965;
    for i in 0..64 {
        let (eye, facing) = windows[rng.random_range(0..windows.len())];
        // Seek the two flanks of a window corridor as well as its centre. A
        // near building often already intercepts the central ray.
        let bearing = (match i % 3 {
            0 => -0.55,
            1 => 0.55,
            _ => 0.,
        }) + rng.random_range(-0.32..0.32);
        let ray = Vec2::from_angle(bearing).rotate(facing);
        let front = -Vec2::from_angle(rng.random_range(-0.18..0.18)).rotate(facing);
        let size = Vec2::new(rng.random_range(5.0..9.6), rng.random_range(2.9..5.1));
        let radius = sky - rng.random_range(2.9..5.0);
        let along = eye.dot(ray);
        let distance = (along * along + radius * radius - eye.length_squared()).sqrt() - along;
        if !distance.is_finite() || distance < 11.0 {
            continue;
        }
        let center = eye + ray * distance;
        let axis = Vec2::new(front.y, -front.x);
        let max_radius = [-1., 1.]
            .into_iter()
            .flat_map(|u| {
                [-1., 1.]
                    .map(|v| (center + (axis * (u * size.x) + front * (v * size.y)) * 0.5).length())
            })
            .fold(0., f32::max);
        let eaves_limit = (sky * sky - max_radius * max_radius).max(0.).sqrt() - 0.30;
        if eaves_limit < 3.05 {
            continue;
        }
        let apron = rng.random_range(0.82..1.55);
        let finish_seed = sites
            .iter()
            .min_by(|p, q| {
                p.center
                    .distance_squared(center)
                    .total_cmp(&q.center.distance_squared(center))
            })
            .map_or(scene.seed, |p| p.seed);
        let row = Terrace {
            center,
            size,
            front,
            apron,
            eaves_limit,
            seed: rng.random(),
            finish_seed,
        };
        let plot = row.footprint();
        if !plot.overlaps(primary, 1. + district.paving_width)
            && occupied.iter().all(|p| !plot.overlaps(*p, 0.55))
        {
            // Score the actual width visible past near lots. A centre alone
            // can clear a parked garden yet leave the entire terrace obscured.
            let front_center = center + row.front * size.y * 0.5;
            let mut exposure = 0.;
            for u in [-0.40, 0., 0.40] {
                let sight = front_center + axis * (u * size.x) - eye;
                let length = sight.length();
                let dir = sight / length;
                let cross = Vec2::new(dir.y, -dir.x);
                let transmission = occluders
                    .iter()
                    .fold(1_f32, |value, (at, bx, bz, opacity)| {
                        let d = *at - eye;
                        let t = d.dot(dir);
                        let radius = bx.dot(cross).abs() + bz.dot(cross).abs();
                        if t > 0. && t < length && d.dot(cross).abs() < radius {
                            value * (1. - opacity)
                        } else {
                            value
                        }
                    });
                exposure += transmission * dir.dot(facing).max(0.);
            }
            candidates.push((exposure * size.x / distance, row));
        }
    }
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (_, row) in candidates {
        if rows
            .iter()
            .all(|p| !row.footprint().overlaps(p.footprint(), 0.65))
        {
            rows.push(row);
            if rows.len() == target {
                break;
            }
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::layout::IndoorLayout;
    use std::collections::BTreeSet;

    #[test]
    fn window_streets_fit_the_real_sky_and_keep_near_plots_clear() {
        let mut counts = BTreeSet::new();
        let mut built = 0;
        for seed in (57000..57096).chain([44214, 57496]) {
            let s = IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.)
                .unwrap();
            let windows = window_rays(&s);
            let sites = parcels_at(&s, &windows);
            let rows = plan(&s, &windows, &sites);
            assert_eq!(rows, plan(&s, &windows, &sites));
            assert!(plan(&s, &[], &sites).is_empty());
            assert!(rows.len() <= 3);
            counts.insert(rows.len());
            let bg = super::super::super::background::radius(&s);
            for (i, row) in rows.iter().enumerate() {
                let plot = row.footprint();
                assert!(sites
                    .iter()
                    .all(|p| !plot.overlaps(Footprint::oriented(p.center, p.size, p.yaw), 0.54)));
                assert!(rows[..i]
                    .iter()
                    .all(|p| !plot.overlaps(p.footprint(), 0.64)));
                let mut a = Assembly::default();
                buildings::terrace::build(&mut a, row);
                super::super::qualification::check(&a);
                let all = a.parts.values();
                let tris: usize = all.clone().map(|g| g.indices.len() / 3).sum();
                assert!((60..=2200).contains(&tris), "seed {seed}, {tris} triangles");
                assert!(all
                    .flat_map(|g| &g.positions)
                    .all(|v| Vec3::from_array(*v).length() < bg * 0.966));
                assert!(a
                    .parts
                    .keys()
                    .any(|(s, l)| *s == Surface::Glass && l.starts_with("window#finish")));
                assert!(a
                    .parts
                    .keys()
                    .any(|(s, l)| *s == Surface::Paint && l.starts_with("door#finish")));
                assert!(a
                    .parts
                    .keys()
                    .any(|(s, l)| *s == Surface::Concrete && l.starts_with("floor#finish")));
                built += 1;
            }
        }
        assert!(
            built > 75 && counts.len() >= 3,
            "built {built} rows, counts {counts:?}"
        );
    }
}
