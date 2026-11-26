//! Acoustic inserts fitted to the free metal between the keys, hinge and deck edge.
use super::{palmrest::DeckSkin, *};

mod mesh;
#[cfg(test)]
mod tests;
pub(crate) const INTRODUCED: u32 = 253;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GrilleLayout {
    KeyboardFlanks,
    RearSoundbar,
    OffsetRear,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Opening {
    /// X and Z measured from the center of the laptop base, excluding the lid sweep.
    pub deck_center_xz_m: Vec2,
    pub size_xz_m: Vec2,
    #[serde(skip)]
    panel: usize,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct SpeakerBank {
    pub layout: GrilleLayout,
    pub facing: Surface,
    pub light_finish: bool,
    pub recess_m: f32,
    pub edge_radius_fraction: f32,
    pub texture_scale: f32,
    pub first: Opening,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub second: Option<Opening>,
}

impl LaptopProgram {
    /// Visible, top-facing acoustic inserts for this manifest version.
    pub fn speaker_bank(&self, object: &IndoorObject, version: u32) -> Option<SpeakerBank> {
        if !self.speaker_grilles || version < INTRODUCED || object.kind != ObjectKind::Laptop {
            return None;
        }
        let deck = Deck::fit(object, self);
        SpeakerBank::fit(
            object,
            self,
            deck,
            &DeckSkin::fit(object, self, deck.depth),
            &Housing::for_object(object),
            version,
        )
    }
}

// This collar is part of the deck triangulation, outside the visible cut.
const COLLAR: f32 = 0.00065;

impl SpeakerBank {
    pub(super) fn fit(
        o: &IndoorObject,
        p: &LaptopProgram,
        deck: Deck,
        skin: &DeckSkin,
        housing: &Housing,
        version: u32,
    ) -> Option<Self> {
        if !p.speaker_grilles || version < INTRODUCED {
            return None;
        }
        let mut r = stream(o.seed, u64::from_le_bytes(*b"DeckVent"));
        let mut layout = match r.random_range(0..10) {
            0..=4 => GrilleLayout::KeyboardFlanks,
            5..=7 => GrilleLayout::RearSoundbar,
            _ => GrilleLayout::OffsetRear,
        };
        let d = deck.depth;
        // The hinge barrels have a 6 mm radius. Key geometry starts at or
        // beyond 6.5% of the base depth, even on a six-row keyboard.
        let rear_limit = d * 0.065 - 0.0014;
        let rear_start = 0.0076_f32.max(p.lid_m * 0.5 + 0.001);
        if rear_limit - rear_start < 0.003 {
            layout = GrilleLayout::KeyboardFlanks;
        }
        let (first, second) = if layout == GrilleLayout::KeyboardFlanks {
            let edge = skin.panel(11)[0].x;
            let free = edge - o.size.x * p.keyboard_fraction * 0.5 - 0.0015;
            let offset = free * r.random_range(0.14..0.36);
            let width = (free - offset) * r.random_range(0.44..0.89);
            let length = d * r.random_range(0.27..0.69);
            let z = d * r.random_range(-0.14..0.09);
            let right = Opening::fit(
                skin,
                11,
                Vec2::new(edge - offset - width * 0.5, z),
                Vec2::new(width, length),
            )?;
            let left = Opening::fit(
                skin,
                5,
                Vec2::new(-right.deck_center_xz_m.x, z),
                Vec2::new(width, length),
            )?;
            // Inset touchpads can narrow one rail. Fit a matched stereo pair
            // to the length available on both sides of this particular deck.
            let z_min = (right.deck_center_xz_m.y - right.size_xz_m.y * 0.5)
                .max(left.deck_center_xz_m.y - left.size_xz_m.y * 0.5);
            let z_max = (right.deck_center_xz_m.y + right.size_xz_m.y * 0.5)
                .min(left.deck_center_xz_m.y + left.size_xz_m.y * 0.5);
            if z_max <= z_min {
                return None;
            }
            let matched = |mut slot: Opening| {
                slot.deck_center_xz_m.y = (z_min + z_max) * 0.5;
                slot.size_xz_m.y = z_max - z_min;
                slot
            };
            (matched(left), Some(matched(right)))
        } else {
            let width = (rear_limit - rear_start) * r.random_range(0.45..0.90);
            let z = -d * 0.5 + r.random_range(rear_start..rear_limit - width) + width * 0.5;
            let (x, length) = match layout {
                GrilleLayout::RearSoundbar => (
                    o.size.x * r.random_range(-0.055..0.055),
                    o.size.x * r.random_range(0.36..0.78),
                ),
                _ => {
                    let side = if r.random_bool(0.5) { 1. } else { -1. };
                    (
                        side * o.size.x * r.random_range(0.13..0.31),
                        o.size.x * r.random_range(0.13..0.34),
                    )
                }
            };
            (
                Opening::fit(skin, 8, Vec2::new(x, z), Vec2::new(length, width))?,
                None,
            )
        };
        let mut stock = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"GrilCoat"),
        );
        let facing = if stock.random_bool(0.78) {
            Surface::Metal
        } else {
            Surface::Fabric
        };
        let light_finish = stock.random_bool(if housing.tone() == "pearl" {
            0.67
        } else {
            0.23
        });
        Some(Self {
            layout,
            first,
            second,
            facing,
            light_finish,
            recess_m: r.random_range(0.00038..0.00115),
            edge_radius_fraction: r.random_range(0.10..0.44),
            texture_scale: stock.random_range(1.05..2.25),
        })
    }

    pub fn openings(&self) -> impl Iterator<Item = &Opening> {
        std::iter::once(&self.first).chain(self.second.as_ref())
    }

    pub(super) fn opening_at(&self, panel: usize) -> Option<&Opening> {
        self.openings().find(|opening| opening.panel == panel)
    }

    pub(crate) fn label(&self) -> &'static str {
        if self.light_finish {
            "laptop#finish481#uv_local"
        } else {
            "laptop#finish480#uv_local"
        }
    }
}

impl Opening {
    /// Clip the length against both sloping sides of the actual palmrest
    /// trapezoid at both edges of the opening. No triangulation search.
    fn fit(skin: &DeckSkin, panel: usize, center: Vec2, size: Vec2) -> Option<Self> {
        let q = skin.panel(panel);
        let (u, v) = if panel == 8 { (1, 0) } else { (0, 1) };
        let mut lo = f32::NEG_INFINITY;
        let mut hi = f32::INFINITY;
        for side in [-1., 1.] {
            let x = center[u] + side * (size[u] * 0.5 + COLLAR);
            let t = (x - q[0][u]) / (q[3][u] - q[0][u]);
            if !(t > 0. && t < 1.) {
                return None;
            }
            let a = q[0].lerp(q[3], t)[v];
            let b = q[1].lerp(q[2], t)[v];
            lo = lo.max(a.min(b) + 2. * COLLAR);
            hi = hi.min(a.max(b) - 2. * COLLAR);
        }
        let low = (center[v] - size[v] * 0.5).max(lo);
        let high = (center[v] + size[v] * 0.5).min(hi);
        if high - low < 0.004 || size[u] < 0.001 {
            return None;
        }
        let mut cut = Self {
            panel,
            deck_center_xz_m: center,
            size_xz_m: size,
        };
        cut.deck_center_xz_m[v] = (high + low) * 0.5;
        cut.size_xz_m[v] = high - low;
        Some(cut)
    }
}
