//! Table linen is cut from venue stock. Folding and setting it down leave
//! variations in the crease, free edges and grain of each individual piece.
use super::*;
use crate::scene::procedural_indoor::objects::tables::LinenWeave;

mod cloth;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Fold {
    Book,
    Triangle,
    Tent,
}

impl Fold {
    const ALL: [Self; 3] = [Self::Book, Self::Triangle, Self::Tent];

    pub(crate) fn sample(r: &mut impl Rng, upright: f64) -> Self {
        if r.random_bool(upright) {
            Self::Tent
        } else if r.random_bool(0.32) {
            Self::Triangle
        } else {
            Self::Book
        }
    }

    pub(crate) fn folded_size(self, cut_m: f32, r: &mut impl Rng) -> Vec3 {
        let (span, rise, length) = match self {
            Self::Book => (0.22..0.32, 0.0040..0.010, 0.43..0.55),
            Self::Triangle => (0.31..0.47, 0.0035..0.009, 0.49..0.69),
            Self::Tent => (0.25..0.41, 0.062..0.125, 0.33..0.50),
        };
        Vec3::new(
            cut_m * r.random_range(span),
            r.random_range(rise),
            cut_m * r.random_range(length),
        )
    }

    pub(crate) fn variant(self, weave: LinenWeave) -> u32 {
        self as u32 | (weave as u32) << 2
    }
}

pub(crate) fn weave(o: &IndoorObject) -> LinenWeave {
    LinenWeave::ALL[((o.variant >> 2) & 3) as usize]
}

pub(crate) fn fixture(seed: u64) -> (u32, Vec3) {
    let mut r = stream(seed, u64::from_le_bytes(*b"linenCut"));
    let fold = Fold::ALL[(seed % 3) as usize];
    let weave = LinenWeave::ALL[(seed / 3 % 4) as usize];
    let cut = r.random_range(0.32..0.48);
    (fold.variant(weave), fold.folded_size(cut, &mut r))
}

#[derive(Debug, serde::Serialize)]
pub(crate) struct Napkin {
    pub(crate) fold: Fold,
    pub(crate) weave: LinenWeave,
    ply_m: f32,
    crease_offset: f32,
    free_edge: f32,
    corner_lift: f32,
    grain_angle_rad: f32,
    hem_m: f32,
}

impl Napkin {
    pub(crate) fn for_object(o: &IndoorObject) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"linenLay"));
        Self {
            fold: Fold::ALL[(o.variant & 3) as usize % Fold::ALL.len()],
            weave: weave(o),
            ply_m: r
                .random_range(0.00048_f32..0.00094)
                .min(o.size.y * 0.13)
                .min(o.size.x * 0.04),
            crease_offset: r.random_range(-0.075..0.075),
            free_edge: r.random_range(0.72..0.95),
            corner_lift: r.random_range(-0.24..0.24),
            grain_angle_rad: r.random_range(-0.15..0.15),
            hem_m: r.random_range(0.0045..0.010),
        }
    }

    pub(crate) fn build(&self, a: &mut Assembly, size: Vec3) {
        let label = match self.weave {
            LinenWeave::Plain => "towel#finish880#uv_local",
            LinenWeave::Herringbone => "towel#finish881#uv_local",
            LinenWeave::AwningStripe => "towel#finish882#uv_local",
            LinenWeave::BistroCheck => "towel#finish883#uv_local",
        };
        self.cloth(a.part(Surface::Fabric, label), size);
    }
}
