//! A building across a street occupies a range of directions through a window.
//! Allocate some existing buildings along underrepresented ground-storey
//! frontages, retaining the district inventory and naturally open courtyards.
use super::*;

const BEARINGS: [f32; 7] = [-1.02, -0.68, -0.34, 0., 0.34, 0.68, 1.02];
const SAMPLES: usize = BEARINGS.len() * 3;

struct Plot {
    center: Vec2,
    half: Vec2,
    axes: [Vec2; 2],
}

impl Plot {
    fn new(p: &Parcel) -> Self {
        let (s, c) = p.yaw.sin_cos();
        Self {
            center: p.center,
            half: p.size * 0.5,
            axes: [Vec2::new(c, -s), Vec2::new(s, c)],
        }
    }

    fn intersects(&self, origin: Vec2, ray: Vec2) -> bool {
        let delta = self.center - origin;
        let mut near = 0_f32;
        let mut far = f32::INFINITY;
        for (axis, half) in self.axes.iter().zip(self.half.to_array()) {
            let distance = delta.dot(*axis);
            let direction = ray.dot(*axis);
            if direction.abs() < 1e-6 {
                if distance.abs() > half {
                    return false;
                }
            } else {
                let a = (distance - half) / direction;
                let b = (distance + half) / direction;
                near = near.max(a.min(b));
                far = far.min(a.max(b));
                if near >= far {
                    return false;
                }
            }
        }
        true
    }
}

struct Aperture {
    window: WindowView,
    weight: f32,
    plots: [u32; SAMPLES],
}

pub(super) struct Horizon {
    apertures: Vec<Aperture>,
    total_weight: f32,
}

impl Horizon {
    pub(super) fn new(sites: &[Parcel], windows: &[WindowView]) -> Self {
        let plots: Vec<_> = sites.iter().map(Plot::new).collect();
        let directions = BEARINGS.map(Vec2::from_angle);
        let apertures: Vec<_> = windows
            .iter()
            .filter_map(|w| {
                // Eye-level apertures carry the neighborhood view. High clerestories
                // can legitimately open entirely to the sky.
                let height = w.head.min(2.4) - w.sill.max(0.85);
                if height <= 0. {
                    return None;
                }
                let tangent = Vec2::new(-w.normal.y, w.normal.x);
                let directions = directions.map(|d| d.rotate(w.normal));
                let plots = std::array::from_fn(|i| {
                    let lateral = [-0.34, 0., 0.34][i / BEARINGS.len()] * w.width;
                    let eye = w.center + tangent * lateral;
                    let ray = directions[i % BEARINGS.len()];
                    plots.iter().enumerate().fold(0, |mask, (j, plot)| {
                        mask | (u32::from(plot.intersects(eye, ray)) << j)
                    })
                });
                Some(Aperture {
                    window: *w,
                    weight: height * w.width,
                    plots,
                })
            })
            .collect();
        Self {
            total_weight: apertures.iter().map(|a| a.weight).sum(),
            apertures,
        }
    }

    /// Fraction of sampled directions with building footprints, and a placement
    /// score with diminishing returns for filling additional rays of one opening.
    fn score(&self, buildings: u32) -> (f32, f32) {
        let mut covered = 0.;
        let mut utility = 0.;
        for a in &self.apertures {
            let f = a
                .plots
                .iter()
                .filter(|&&mask| mask & buildings != 0)
                .count() as f32
                / SAMPLES as f32;
            covered += a.weight * f;
            utility += a.weight * f * (1. - 0.45 * f);
        }
        (covered / self.total_weight, utility / self.total_weight)
    }
    pub(super) fn coverage(&self, sites: &[Parcel]) -> Option<f32> {
        (self.total_weight > 0.).then(|| self.score(building_mask(sites)).0)
    }

    /// Angular sampling distribution for the further street. An open strip of
    /// a picture window sees more of the neighborhood than a punched window.
    pub(super) fn street_bearings(&self, sites: &[Parcel]) -> Vec<StreetBearing> {
        let buildings = building_mask(sites);
        let mut mass = 0.;
        let mut bearings = Vec::with_capacity(self.apertures.len() * BEARINGS.len());
        for (opening, a) in self.apertures.iter().enumerate() {
            for (j, angle) in BEARINGS.into_iter().enumerate() {
                let open = (0..3)
                    .filter(|&k| a.plots[k * BEARINGS.len() + j] & buildings == 0)
                    .count() as f32
                    / 3.;
                if open > 0. {
                    mass += a.weight * open * angle.cos();
                    bearings.push(StreetBearing {
                        window: a.window,
                        opening,
                        angle,
                        cumulative_weight: mass,
                    });
                }
            }
        }
        bearings
    }

    pub(super) fn assign(&self, sites: &mut [Parcel], seed: u64, building_fraction: f32) {
        let mut buildings = building_mask(sites);
        let count = buildings.count_ones() as usize;
        if count == 0 || count == sites.len() || self.total_weight <= 0. {
            return;
        }
        let mut rng = stream(seed, u64::from_le_bytes(*b"StFronts"));
        // An open green or campus is an intentional part of the mixture. The
        // other districts have a continuous frontage target, never a per-view quota.
        let target = if rng.random_bool(0.13) {
            0.
        } else {
            0.23 + 0.29 * building_fraction + rng.random_range(0.0..0.30)
        };
        if target == 0. {
            return;
        }
        for _ in 0..2 {
            let (coverage, utility) = self.score(buildings);
            if coverage >= target {
                break;
            }
            let mut best = None;
            let mut best_utility = utility + 0.012;
            for (i, _) in sites.iter().enumerate().filter(|(_, p)| p.kind == 0) {
                for (j, _) in sites.iter().enumerate().filter(|(_, p)| p.kind != 0) {
                    let trial = buildings ^ (1 << i) ^ (1 << j);
                    let (next, score) = self.score(trial);
                    if next > coverage && score > best_utility {
                        best = Some((i, j, trial));
                        best_utility = score;
                    }
                }
            }
            let Some((i, j, trial)) = best else {
                break;
            };
            sites[i].kind = sites[j].kind;
            sites[j].kind = 0;
            buildings = trial;
        }
    }
}

pub(super) struct StreetBearing {
    pub window: WindowView,
    pub opening: usize,
    pub angle: f32,
    pub cumulative_weight: f32,
}

fn building_mask(sites: &[Parcel]) -> u32 {
    sites
        .iter()
        .enumerate()
        .fold(0, |mask, (i, p)| mask | (u32::from(p.kind == 0) << i))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coverage(sites: &[Parcel], windows: &[WindowView]) -> Option<f32> {
        Horizon::new(sites, windows).coverage(sites)
    }

    fn window(width: f32) -> WindowView {
        WindowView {
            center: Vec2::ZERO,
            normal: Vec2::Y,
            width,
            sill: 0.64,
            head: 2.55,
        }
    }
    fn plot(x: f32, z: f32, kind: u8) -> Parcel {
        Parcel {
            center: Vec2::new(x, z),
            size: Vec2::new(4.6, 3.2),
            yaw: 0.23,
            kind,
            seed: z.to_bits() as u64,
        }
    }

    #[test]
    fn measured_frontage_counts_ground_storeys_across_the_actual_aperture() {
        let sites = [plot(0., 7., 0), plot(0., -5., 1)];
        let view = window(2.7);
        let fraction = coverage(&sites, &[view]).unwrap();
        assert!(fraction > 0.10 && fraction < 0.8);
        assert_eq!(
            coverage(
                &sites,
                &[WindowView {
                    normal: -view.normal,
                    ..view
                }]
            ),
            Some(0.)
        );
        assert_eq!(
            coverage(
                &sites,
                &[WindowView {
                    sill: 2.7,
                    head: 3.5,
                    ..view
                }]
            ),
            None
        );
        assert_eq!(coverage(&sites, &[]), None);
        let axis = Vec2::from_angle(0.54);
        let sites = sites.map(|mut p| {
            p.center = axis.rotate(p.center);
            p.yaw -= 0.54;
            p
        });
        assert_eq!(
            coverage(
                &sites,
                &[WindowView {
                    normal: axis.rotate(view.normal),
                    ..view
                }]
            ),
            Some(fraction)
        );
    }

    #[test]
    fn distant_streets_sample_visible_angular_width_and_skip_covered_strips() {
        let windows = [
            window(4.),
            WindowView {
                normal: Vec2::NEG_Y,
                ..window(1.)
            },
            WindowView {
                sill: 2.7,
                head: 3.2,
                ..window(8.)
            },
        ];
        let mut sites = [plot(0., 7., 2)];
        let h = Horizon::new(&sites, &windows);
        let b = h.street_bearings(&sites);
        let wide_mass = b
            .iter()
            .rfind(|s| s.opening == 0)
            .unwrap()
            .cumulative_weight;
        assert!((wide_mass / b.last().unwrap().cumulative_weight - 0.8).abs() < 1e-5);
        assert_eq!(b.len(), 2 * BEARINGS.len());
        sites[0].kind = 0;
        let b = h.street_bearings(&sites);
        assert!(!b.iter().any(|s| s.opening == 0 && s.angle == 0.));
        assert!(b.iter().any(|s| s.opening == 1 && s.angle == 0.));
        assert!(b
            .windows(2)
            .all(|s| s[0].cumulative_weight < s[1].cumulative_weight));
    }

    #[test]
    fn allocation_replays_and_retains_inventory_plots_open_districts_and_clearances() {
        let source = [
            plot(0., -8., 0),
            plot(-1., 7., 3),
            plot(7., 6., 2),
            plot(-7., 9., 1),
        ];
        let windows = [window(5.6)];
        let mut changed = 0;
        for seed in 0..64 {
            let mut sites = source.clone();
            let horizon = Horizon::new(&source, &windows);
            horizon.assign(&mut sites, seed, 0.56);
            let mut replay = source.clone();
            horizon.assign(&mut replay, seed, 0.56);
            let kinds = |p: &[Parcel]| p.iter().map(|p| p.kind).collect::<Vec<_>>();
            assert_eq!(kinds(&sites), kinds(&replay));
            let mut inventory = kinds(&sites);
            inventory.sort_unstable();
            assert_eq!(inventory, [0, 1, 2, 3]);
            for (a, b) in source.iter().zip(&sites) {
                assert_eq!(
                    (a.center, a.size, a.yaw, a.seed),
                    (b.center, b.size, b.yaw, b.seed)
                );
            }
            if kinds(&sites) != kinds(&source) {
                changed += 1;
                assert!(coverage(&sites, &windows).unwrap() > coverage(&source, &windows).unwrap());
            }
        }
        assert!((35..64).contains(&changed));
    }
}
