//! Cafeteria return stations fit removable ware to shelves or paired runners.
//! All heights are measured from the floor. Local +Z is the loading side.
use super::*;
use crate::scene::procedural_indoor::placement::SurfacePattern;
mod frame;
mod ware;
pub use ware::{ReturnWare, WareKind};
#[cfg(test)]
mod tests;

pub const INTRODUCED: u32 = 250;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Bearing {
    Shelf,
    RimGuides,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Load {
    pub bay: usize,
    pub level: usize,
    /// Bottom of the first tray's feet, in station coordinates.
    pub origin: Vec3,
    pub yaw: f32,
    pub count: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TrayStation {
    pub bearing: Bearing,
    pub enclosed: bool,
    pub ware: ReturnWare,
    pub uprights_m: f32,
    pub deck_m: f32,
    pub toe_height_m: f32,
    pub bay_centers_m: Vec<f32>,
    pub guide_clearance_m: f32,
    /// Upper contact surface of each shelf or guide flange.
    pub bearings_m: Vec<f32>,
    pub loads: Vec<Load>,
}

impl TrayStation {
    pub(super) fn fit(o: &IndoorObject, case: &CaseProgram) -> Self {
        let mut design = stream(o.seed, u64::from_le_bytes(*b"TrayRack"));
        let bearing = if case.structure.is_multiple_of(2) {
            Bearing::Shelf
        } else {
            Bearing::RimGuides
        };
        let uprights_m = design
            .random_range(0.018_f32..0.033)
            .min(o.size.x.min(o.size.z) * 0.055);
        let deck_m = design.random_range(0.009_f32..0.021).min(o.size.y * 0.035);
        let toe_height_m = o.size.y * case.base_fraction;
        let guide_clearance_m = design.random_range(0.0035..0.007);
        let interior = o.size.xz() - Vec2::splat(2. * uprights_m);
        let ware = ReturnWare::fit(
            o,
            interior - Vec2::new(2. * (uprights_m + guide_clearance_m + 0.008), 0.025),
        );
        // Leave room for two complete channels and their webs even in a double bay.
        let pitch = ware.size_m.x + 2. * (guide_clearance_m + uprights_m + frame::WEB) + 0.012;
        let bays = if 2. * pitch <= interior.x && design.random_bool(0.88) {
            2
        } else {
            1
        };
        let (spacing, shift) = match bearing {
            Bearing::Shelf => ((interior.x - 0.006) / bays as f32, 0.),
            Bearing::RimGuides => (
                pitch,
                design.random_range(-0.45..0.45) * (interior.x - pitch * bays as f32).max(0.),
            ),
        };
        let bay_centers_m = (0..bays)
            .map(|i| shift + (i as f32 - (bays - 1) as f32 * 0.5) * spacing)
            .collect();
        let first = match bearing {
            Bearing::Shelf => toe_height_m + deck_m,
            Bearing::RimGuides => toe_height_m + deck_m + ware.under_lip_m() + 0.014,
        };
        let ceiling = o.size.y - deck_m;
        let free = ceiling - first;
        let desired_pitch = match bearing {
            Bearing::Shelf => ware.height_m + design.random_range(0.16..0.32),
            Bearing::RimGuides => ware.height_m + design.random_range(0.058..0.145),
        };
        let max_levels = if bearing == Bearing::Shelf { 5 } else { 9 };
        let levels = (free / desired_pitch).floor().clamp(1., max_levels as f32) as usize;
        let bearings_m = (0..levels)
            .map(|i| first + i as f32 * free / levels as f32)
            .collect();
        let mut station = Self {
            bearing,
            enclosed: case.structure < 2,
            ware,
            uprights_m,
            deck_m,
            toe_height_m,
            bay_centers_m,
            guide_clearance_m,
            bearings_m,
            loads: Vec::new(),
        };
        if case.contents {
            station.stock(o);
        }
        station
    }

    fn stock(&mut self, o: &IndoorObject) {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"TrayLoad"));
        let fill = r.random_range(0.22..0.96);
        let pattern = SurfacePattern::sample(&mut r);
        let half = o.size.xz() * 0.5 - Vec2::splat(self.uprights_m + 0.003);
        let bays = self.bay_centers_m.len();
        for (level, &y) in self.bearings_m.iter().enumerate() {
            let above = self.bearings_m.get(level + 1).copied().unwrap_or(o.size.y) - self.deck_m;
            for bay in 0..bays {
                if !r.random_bool(fill) {
                    continue;
                }
                let (center, bottom, yaw, count) = match self.bearing {
                    Bearing::RimGuides => {
                        let play = self.guide_clearance_m.min(self.ware.lip_m * 0.28);
                        let yaw = r.random_range(-0.55..0.55) * play / (0.5 * self.ware.size_m.y);
                        let free_z = (half.y
                            - crate::scene::procedural_indoor::placement::rotated_half(
                                self.ware.size_m,
                                yaw,
                            )
                            .y
                            - 0.005)
                            .max(0.);
                        let center = Vec2::new(
                            self.bay_centers_m[bay] + r.random_range(-0.35..0.35) * play,
                            r.random_range(-0.94..0.94) * free_z,
                        );
                        (center, y - self.ware.under_lip_m(), yaw, 1)
                    }
                    Bearing::Shelf => {
                        let cell = Vec2::new(self.bay_centers_m[bay], 0.);
                        let extent = Vec2::new(half.x / bays as f32, half.y);
                        let bounds = (cell - extent, cell + extent);
                        // The final straight insertion is available on a tight shelf.
                        let Some(p) = (0..5).find_map(|i| {
                            let yaw = if i == 4 {
                                0.
                            } else {
                                r.random_range(-0.55..0.55)
                            };
                            pattern.propose(bounds, self.ware.size_m, yaw, 0.005, &mut r)
                        }) else {
                            continue;
                        };
                        let spare = above - y - self.ware.height_m - 0.012;
                        if spare < 0. {
                            continue;
                        }
                        let max =
                            (1 + (spare / self.ware.nesting_pitch_m()).floor() as usize).min(5);
                        let count =
                            if self.ware.kind == WareKind::ServingTray && r.random_bool(0.48) {
                                r.random_range(1..=max)
                            } else {
                                1
                            };
                        (p.center, y, p.yaw, count)
                    }
                };
                self.loads.push(Load {
                    bay,
                    level,
                    origin: Vec3::new(center.x, bottom, center.y),
                    yaw,
                    count,
                });
            }
        }
    }

    pub(super) fn build(&self, a: &mut Assembly, o: &IndoorObject, case: &CaseProgram) {
        self.frame(a, o, case);
        if !self.loads.is_empty() {
            self.ware.install(a, &self.loads);
        }
    }
}
