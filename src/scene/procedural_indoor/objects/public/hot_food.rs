//! Countertop bain-maries. +Z faces the queue and the controls face the server.
//! One full-size Gastronorm opening accepts one full, two half, three third,
//! or one half and two quarter pans. A second well can carry another division.
use super::*;
use crate::scene::procedural_indoor::materials::broth;
mod gratin;
mod mesh;
mod produce;
#[cfg(test)]
mod tests;
pub(super) use mesh::build;

pub(crate) const VERSION: u32 = 293;
/// Center of the rear cable gland, at its outside face in object space.
pub(crate) fn power_port(o: &IndoorObject) -> Vec3 {
    Vec3::new(o.size.x * 0.31, 0.069, -o.size.z * 0.5 + 0.006)
}

const GN: Vec2 = Vec2::new(0.530, 0.325);
pub(super) fn dimensions(r: &mut impl Rng) -> Vec3 {
    let wells = if r.random_bool(0.42) { 2. } else { 1. };
    Vec3::new(
        wells * (GN.x + r.random_range(0.036..0.078)),
        if r.random_bool(0.62) {
            r.random_range(0.57..0.79)
        } else {
            r.random_range(0.238..0.306)
        },
        r.random_range(0.397..0.487),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Contents {
    Empty,
    Covered,
    TomatoSoup,
    PeaSoup,
    SquashSoup,
    RoastPotatoes,
    GlazedCarrots,
    PotatoGratin,
}
impl Contents {
    fn sample(r: &mut impl Rng) -> Self {
        match r.random_range(0..100) {
            0..12 => Self::Empty,
            12..32 => Self::Covered,
            32..45 => Self::TomatoSoup,
            45..54 => Self::PeaSoup,
            54..62 => Self::SquashSoup,
            62..78 => Self::RoastPotatoes,
            78..90 => Self::GlazedCarrots,
            _ => Self::PotatoGratin,
        }
    }
    fn soup(self) -> Option<broth::Recipe> {
        Some(match self {
            Self::TomatoSoup => broth::Recipe::Tomato,
            Self::PeaSoup => broth::Recipe::Pea,
            Self::SquashSoup => broth::Recipe::Squash,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Pan {
    pub center: Vec2,
    /// Allocated GN cell including the seam around its rolled lip.
    pub size: Vec2,
    pub depth: f32,
    pub contents: Contents,
    /// Liquid or casserole depth, or stocking density for pieces. Empty and covered pans report zero.
    pub fill_fraction: f32,
    seed: u64,
}
impl Pan {
    fn outer(&self) -> Vec2 {
        self.size * 0.5 - Vec2::splat(0.002)
    }
    fn inner(&self) -> Vec2 {
        self.outer() - Vec2::splat(0.009)
    }
    fn bottom(&self) -> Vec2 {
        self.inner() - Vec2::splat(0.014)
    }
    fn floor(&self, deck: f32) -> f32 {
        deck - self.depth
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Guard {
    pub thickness: f32,
    pub clearance: f32,
    pub setback: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct HotFoodWell {
    pub wells: usize,
    pub deck_height: f32,
    pub guard: Option<Guard>,
    pub pans: Vec<Pan>,
}
impl HotFoodWell {
    pub fn for_object(o: &IndoorObject) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"WellTool"));
        let wells = if o.size.x > 0.92 { 2 } else { 1 };
        let guard = (o.size.y > 0.45).then(|| Guard {
            thickness: if r.random_bool(0.52) { 0.005 } else { 0.006 },
            clearance: r.random_range(0.135..0.206),
            setback: r.random_range(0.100..0.185),
        });
        let deck_height = if guard.is_some() {
            r.random_range(0.192..0.249)
        } else {
            o.size.y - 0.043
        };
        let mut pans = Vec::with_capacity(3 * wells);
        let pitch = o.size.x / wells as f32;
        let gn = GN.min(Vec2::new(pitch - 0.018, o.size.z - 0.060));
        for n in 0..wells {
            let center = Vec2::new(pitch * (n as f32 + 0.5) - o.size.x * 0.5, 0.);
            let split = r.random_range(0..5);
            let columns = [1, 2, 2, 3, 2][split];
            let quarter = r.random_range(0..2);
            for i in 0..columns {
                let rows = if split == 4 && i == quarter { 2 } else { 1 };
                for j in 0..rows {
                    let size = gn / Vec2::new(columns as f32, rows as f32);
                    let at = center - gn * 0.5 + size * Vec2::new(i as f32 + 0.5, j as f32 + 0.5);
                    let draw = r.random_range(0..3);
                    let contents = Contents::sample(&mut r);
                    let depth =
                        if matches!(contents, Contents::GlazedCarrots | Contents::RoastPotatoes) {
                            [0.065, 0.065, 0.100][draw]
                        } else {
                            [0.065, 0.100, 0.150][draw]
                        };
                    let stock = r.random_range(0.27..0.87);
                    pans.push(Pan {
                        center: at,
                        size,
                        depth,
                        contents,
                        fill_fraction: if matches!(contents, Contents::Empty | Contents::Covered) {
                            0.
                        } else {
                            stock
                        },
                        seed: r.random(),
                    });
                }
            }
        }
        Self {
            wells,
            deck_height,
            guard,
            pans,
        }
    }
    pub fn has_soup(&self) -> bool {
        self.pans.iter().any(|p| p.contents.soup().is_some())
    }
}
