//! Pinboard construction in metric depth. Contents and validation use the same
//! exposed panel face; the reserved box is not an invariant solid silhouette.
use super::*;

pub(crate) const TACKBOARD_VERSION: u32 = 212;

#[derive(Debug, Clone, Copy, serde::Serialize, PartialEq, Eq)]
pub enum Facing {
    Textile,
    NaturalCork,
    ToastedCork,
}

impl Facing {
    pub fn for_board(o: &IndoorObject, version: u32) -> Self {
        if version < TACKBOARD_VERSION {
            return Self::Textile;
        }
        let mut r = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"PinStock"),
        );
        match r.random_range(0..100) {
            0..=38 => Self::Textile,
            39..=84 => Self::NaturalCork,
            _ => Self::ToastedCork,
        }
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, PartialEq)]
pub struct BoardSection {
    pub backing_m: f32,
    pub depth_m: f32,
    pub face_z_m: f32,
    pub bevel_fraction: f32,
    pub frame_surface: Surface,
}

pub fn parameters(o: &IndoorObject) -> BoardSection {
    let mut r = stream(o.seed, 0x50494e424f415244);
    let depth = o.size.z * r.random_range(0.42..0.98);
    BoardSection {
        backing_m: depth * r.random_range(0.12..0.25),
        depth_m: depth,
        face_z_m: -o.size.z * 0.5 + depth * r.random_range(0.48..0.82),
        bevel_fraction: r.random_range(0.04..0.38),
        frame_surface: [Surface::WoodEdge, Surface::Metal, Surface::Plastic][r.random_range(0..3)],
    }
}
