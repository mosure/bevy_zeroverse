//! Maker rooms share the Workshop profile with office-based seminars.
use super::*;
use crate::scene::procedural_indoor::{
    objects::public::{self, workbench::Bench},
    program::Zone,
};

impl IndoorManifest {
    /// Return the occupied zone. Other rooms in a mixed-purpose suite still receive their
    /// own activity. A rejected maker layout falls through to the ordinary workshop program.
    pub(super) fn furnish_workshop(&mut self, zones: &[Zone]) -> Option<usize> {
        let mut r = stream(self.seed, u64::from_le_bytes(*b"ShopSite"));
        if r.random_bool(0.22) {
            return None;
        }
        let (lo, hi) = self.primary_room_bounds();
        let index = zones.iter().position(|z| z.min == lo && z.max == hi)?;
        let extent = hi - lo;
        let area = extent.element_product();
        let capacity = (area * (0.65 + self.density * 0.5) / r.random_range(6.5..10.))
            .floor()
            .clamp(1., 4.) as usize;
        let count = r.random_range(1..=capacity);
        let kit: u64 = r.random();
        let share = r.random_bool(0.55);
        let preferred = r.random_range(0..4);
        let structure = PlacementStructure::new();
        let mut admitted = false;
        for _ in 0..count {
            let mut bench = self.floor_candidate(
                ObjectKind::Workbench,
                Vec3::ZERO,
                public::dimensions(ObjectKind::Workbench, r.random()),
                0.,
                &mut r,
            );
            if share && r.random_bool(0.8) {
                bench.finish_seed = Some(kit);
            }
            let wall_run = r.random_bool(0.61);
            for attempt in 0..16 {
                let (site, yaw) = if (wall_run && attempt < 9) || r.random_bool(0.34) {
                    let side = if attempt < 5 && share {
                        preferred
                    } else {
                        r.random_range(0..4)
                    };
                    let (at, inward) = super::zone_edge(
                        lo,
                        hi,
                        side,
                        r.random_range(0.10..0.90),
                        bench.size.z * 0.5 + r.random_range(0.13..0.45),
                    );
                    (at, inward + r.random_range(-0.16..0.16))
                } else {
                    (
                        lo + extent
                            * Vec2::new(r.random_range(0.10..0.90), r.random_range(0.10..0.90)),
                        zones[index].orientation
                            + r.random_range(-std::f32::consts::PI..std::f32::consts::PI),
                    )
                };
                bench.position = Vec3::new(site.x, self.floor_height(site), site.y);
                bench.yaw = yaw;
                let (min, max) = super::access::envelope(&bench).unwrap().bounds();
                if min.xz().cmplt(lo + 0.035).any()
                    || max.xz().cmpgt(hi - 0.035).any()
                    || !self.placement_clear_with_structure(&bench, 0.045, false, &structure)
                {
                    continue;
                }
                self.objects.push(bench);
                admitted = true;
                break;
            }
        }
        if admitted {
            Some(index)
        } else {
            None
        }
    }

    pub(super) fn shop_storage(&mut self, zone: &Zone) {
        let mut r = stream(self.seed, u64::from_le_bytes(*b"ShopBins"));
        for _ in 0..r.random_range(1..=3) {
            let kind = if r.random_bool(0.52) {
                ObjectKind::Locker
            } else {
                ObjectKind::StorageBox
            };
            let size = if kind == ObjectKind::Locker {
                public::dimensions(kind, r.random())
            } else {
                Vec3::new(
                    r.random_range(0.38..0.71),
                    r.random_range(0.23..0.44),
                    r.random_range(0.29..0.51),
                )
            };
            for _ in 0..6 {
                let (site, inward) = super::zone_edge(
                    zone.min,
                    zone.max,
                    r.random_range(0..4),
                    r.random_range(0.08..0.92),
                    size.z * 0.5 + r.random_range(0.14..0.36),
                );
                if self
                    .add(
                        kind,
                        Vec3::new(site.x, 0., site.y),
                        size,
                        inward + r.random_range(-0.19..0.19),
                        &mut r,
                    )
                    .is_some()
                {
                    break;
                }
            }
        }
    }

    pub(in crate::scene::procedural_indoor::layout) fn dress_workbench(
        &mut self,
        bench: &IndoorObject,
    ) {
        let mut r = stream(bench.seed, u64::from_le_bytes(*b"ShopTask"));
        let activity = self.clutter.sample(bench.seed);
        if activity <= 0. {
            return;
        }
        let p = Bench::sample(bench);
        if let Some(v) = p
            .vise
            .filter(|_| r.random_bool(f64::from(0.37 + 0.59 * activity)))
        {
            let size = Vec3::new(
                r.random_range(0.31..0.68),
                r.random_range(0.093..0.225),
                v.opening_m,
            );
            let local = Vec3::new(
                v.center_x_m + r.random_range(-0.033..0.033),
                Bench::clamped_bottom(bench),
                p.front_z_m + v.opening_m * 0.5,
            );
            let turn = if r.random_bool(0.5) {
                std::f32::consts::PI
            } else {
                0.
            };
            let stock = self.candidate(
                ObjectKind::TimberOffcut,
                bench.transform().transform_point(local),
                size,
                bench.yaw + turn,
                &mut r,
            );
            self.admit_supported_prop(bench, stock);
        }
        let d = p.deck(bench);
        let samples = (3.8 * activity + r.random::<f32>()).floor() as usize;
        for _ in 0..samples {
            let kind = match r.random_range(0..10) {
                0 => ObjectKind::Notepad,
                1 | 2 => ObjectKind::Pencil,
                _ => ObjectKind::TimberOffcut,
            };
            let size = match kind {
                ObjectKind::Pencil => Vec3::new(0.012, 0.012, r.random_range(0.132..0.184)),
                ObjectKind::Notepad => Vec3::new(
                    r.random_range(0.11..0.20),
                    r.random_range(0.006..0.018),
                    r.random_range(0.16..0.26),
                ),
                _ => public::dimensions(kind, r.random()),
            };
            let at = d.min
                + d.size() * Vec2::new(r.random_range(0.12..0.88), r.random_range(0.16..0.84));
            self.supported_prop(
                bench,
                kind,
                Vec3::new(at.x, 0., at.y),
                size,
                r.random_range(-2.75..2.75),
                &mut r,
            );
        }
    }
}
