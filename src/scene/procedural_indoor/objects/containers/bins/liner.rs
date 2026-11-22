use super::*;
use crate::scene::procedural_indoor::geometry::housing_outline;
pub use crate::scene::procedural_indoor::materials::bin_film::FilmFinish;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum LinerMount {
    OverRim,
    Tucked,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct Liner {
    pub mount: LinerMount,
    pub finish: FilmFinish,
    pub folds: u32,
    pub gather_m: f32,
    pub roll_m: f32,
    pub hem_drop_m: f32,
    pub phase: f32,
    pub twist: f32,
}

impl Liner {
    pub(super) fn sample(seed: u64, p: &BinProgram) -> Option<Self> {
        // A closed lid hides the film. A divided basket cannot take one bag
        // across the divider. Bare open and recycling bins also remain common.
        let frequency = match p.lid {
            0 | 4 => 0.72,
            5 if p.waste_stream == WasteStream::General => 0.84,
            5 => 0.38,
            _ => return None,
        };
        let mut r = stream(
            seed,
            u64::from_le_bytes(*b"BinLiner") ^ p.waste_stream as u64,
        );
        if !r.random_bool(frequency) {
            return None;
        }
        let half = p.body.xz().min_element() * 0.5;
        Some(Self {
            mount: if p.lid == 0 && r.random_bool(0.74) {
                LinerMount::OverRim
            } else {
                LinerMount::Tucked
            },
            finish: if r.random_bool(0.68) {
                FilmFinish::Charcoal
            } else {
                FilmFinish::Pearl
            },
            folds: r.random_range(4..=8),
            gather_m: half * r.random_range(0.045..0.16),
            roll_m: (half * r.random_range(0.014..0.032)).min(0.0045),
            hem_drop_m: p.body.y * r.random_range(0.055..0.15),
            phase: r.random_range(0.0..TAU),
            twist: r.random_range(-0.68..0.68),
        })
    }

    pub(super) fn build(self, g: &mut Geometry, p: &BinProgram) {
        let outline = housing_outline(p.squareness);
        let top = p.body.y;
        let half = p.body.xz() * 0.5;
        let rim = p.rim_inner();
        // Each crease runs down the same sheet and turns slightly with depth.
        // Compute angular harmonics once per vertex for the whole profile.
        let mut hem = [Vec3::ZERO; 32];
        let mut outside = hem;
        let mut crest = hem;
        let mut mouth = hem;
        let mut neck = hem;
        let mut middle = hem;
        let mut foot = hem;
        let neck_y = top - p.rim_height;
        let mid_y = p.floor + (neck_y - p.floor) * 0.53;
        let foot_y = p.floor + self.roll_m;
        let mid_half = p.cavity_half(mid_y);
        let foot_half = p.cavity_half(foot_y);
        for (i, u) in outline.into_iter().enumerate() {
            let angle = TAU * i as f32 / 32.;
            let gather =
                angle * self.folds as f32 + self.phase + 0.57 * (2. * angle - self.phase).sin();
            let crease = |level: f32| 0.5 + 0.5 * (gather + self.twist * level).cos();
            let point = |radii: Vec2, y: f32| Vec3::new(u.x * radii.x, y, u.y * radii.y);
            if self.mount == LinerMount::OverRim {
                let fullness = self.roll_m * (0.42 + 0.74 * crease(0.));
                hem[i] = point(
                    half * 1.015 + fullness,
                    top - self.hem_drop_m * (0.56 + 0.44 * crease(0.)),
                );
                outside[i] = point(half * 1.015 + self.roll_m * 0.45, top - self.roll_m * 0.35);
                crest[i] = point(half, top + self.roll_m);
                mouth[i] = point(rim - self.roll_m * 0.30, top + self.roll_m * 0.52);
            } else {
                // The upper edge is captured behind the inside return of the rigid rim.
                mouth[i] = point(rim + p.wall * 0.25, top - p.rim_height * 0.38);
            }
            neck[i] = point(
                rim - self.roll_m - self.gather_m * (0.15 + 0.47 * crease(0.)),
                neck_y,
            );
            middle[i] = point(
                mid_half - self.roll_m - self.gather_m * (0.25 + 0.95 * crease(0.6)),
                mid_y,
            );
            foot[i] = point(
                foot_half - self.roll_m - self.gather_m * (0.20 + 0.72 * crease(1.)),
                foot_y,
            );
        }
        if self.mount == LinerMount::OverRim {
            shell::band(g, &hem, &outside, false);
            shell::band(g, &outside, &crest, false);
            shell::band(g, &crest, &mouth, true);
        }
        shell::band(g, &mouth, &neck, false);
        shell::band(g, &neck, &middle, false);
        shell::band(g, &middle, &foot, false);
        shell::bag_bottom(g, &foot, p.floor + self.roll_m * 0.36);
    }
}
