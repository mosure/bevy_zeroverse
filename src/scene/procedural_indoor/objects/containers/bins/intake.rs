use super::*;
use crate::scene::procedural_indoor::geometry::housing_outline;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum InletKind {
    GeneralWaste,
    Paper,
    BottlesAndCans,
    MixedRecycling,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct StationInlet {
    pub kind: InletKind,
    /// Clear throat, excluding the molded bevel, measured in local X and Z.
    pub width_depth_m: Vec2,
    pub offset_m: Vec2,
    pub squareness: f32,
    pub bevel_m: f32,
}

impl StationInlet {
    pub(super) fn sample(seed: u64, p: &BinProgram) -> Option<Self> {
        if p.lid != 5 {
            return None;
        }
        let mut r = stream(
            seed,
            u64::from_le_bytes(*b"BinInlet") ^ p.waste_stream as u64,
        );
        let lid = p.body.xz() * 0.96;
        let kind = if p.waste_stream == WasteStream::General {
            InletKind::GeneralWaste
        } else {
            match r.random_range(0..10) {
                0..=3 => InletKind::Paper,
                4..=6 => InletKind::BottlesAndCans,
                _ => InletKind::MixedRecycling,
            }
        };
        let (mut throat, squareness) = match kind {
            InletKind::BottlesAndCans => {
                let diameter = r
                    .random_range(0.095_f32..0.17)
                    .min(lid.min_element() * 0.70);
                (Vec2::splat(diameter), 2.)
            }
            InletKind::Paper => {
                let long = lid.max_element() * r.random_range(0.60..0.82);
                let short = r
                    .random_range(0.032_f32..0.075)
                    .min(lid.min_element() * 0.38)
                    .min(long * 0.37);
                (
                    if lid.x >= lid.y {
                        Vec2::new(long, short)
                    } else {
                        Vec2::new(short, long)
                    },
                    r.random_range(4.5..9.),
                )
            }
            InletKind::GeneralWaste => (
                lid * Vec2::new(r.random_range(0.49..0.74), r.random_range(0.44..0.71)),
                r.random_range(2.2..6.0),
            ),
            InletKind::MixedRecycling => (
                lid * Vec2::new(r.random_range(0.50..0.73), r.random_range(0.35..0.58)),
                r.random_range(2.0..4.6),
            ),
        };
        // Inscribe the throat and its lip even in the roundest body; leave a
        // continuous load-bearing land at the short ends of a paper slot.
        let relative = (throat / lid).length();
        throat *= (0.89 / relative).min(1.);
        let bevel = (p.lid_height * r.random_range(0.10..0.25))
            .min(throat.min_element() * 0.075)
            .min(lid.min_element() * 0.018);
        let offset = lid * Vec2::new(r.random_range(-0.032..0.032), r.random_range(-0.038..0.038));
        Some(Self {
            kind,
            width_depth_m: throat,
            offset_m: offset,
            squareness,
            bevel_m: bevel,
        })
    }

    pub(super) fn build(self, g: &mut Geometry, p: &BinProgram, underside: f32) {
        let outer = housing_outline(p.squareness);
        let cut = housing_outline(self.squareness);
        let half = p.body.xz() * 0.5;
        let throat = self.width_depth_m * 0.5;
        let top = underside + p.lid_height;
        let base = shell::ring(&outer, half * 1.015, Vec2::ZERO, underside);
        let edge = shell::ring(&outer, half * 0.96, Vec2::ZERO, top);
        let mouth = shell::ring(&cut, throat + self.bevel_m, self.offset_m, top);
        let lip = shell::ring(&cut, throat, self.offset_m, top - self.bevel_m);
        let exit = shell::ring(&cut, throat, self.offset_m, underside);
        shell::band(g, &base, &edge, false);
        shell::band(g, &edge, &mouth, true);
        shell::band(g, &mouth, &lip, false);
        shell::band(g, &lip, &exit, false);
        shell::band(g, &exit, &base, true);
    }
}
