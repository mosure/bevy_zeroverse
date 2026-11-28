//! Movable ensemble furniture. The reader or pianist sits on the local +Z side.
use super::*;
use crate::scene::procedural_indoor::materials::paper::score;

mod keyboard;
mod stand;
pub(crate) mod timpani;
pub(super) use keyboard::build as piano;
pub use keyboard::Keyboard;
pub(super) use stand::build as stand;
pub use stand::Stand;
pub use timpani::{BowlStock, Timpani};

pub(super) fn dimensions(kind: ObjectKind, r: &mut impl Rng) -> Vec3 {
    match kind {
        ObjectKind::MusicStand => Vec3::new(
            r.random_range(0.47..0.60),
            r.random_range(1.02..1.40),
            r.random_range(0.40..0.58),
        ),
        ObjectKind::StagePiano => {
            let white_keys = match r.random_range(0..10) {
                0..=2 => 36,
                3..=4 => 43,
                _ => 52,
            };
            Vec3::new(
                white_keys as f32 * r.random_range(0.0226..0.0237) + r.random_range(0.055..0.108),
                r.random_range(0.75..0.85),
                r.random_range(0.38..0.54),
            )
        }
        _ => unreachable!("ensemble equipment"),
    }
}

/// A reading desk/piano presents its +Z face to a chair looking along local -Z.
/// Check the actual fitted transforms; JSON replays and independent validators
/// do not have to reconstruct the original samples.
pub(crate) fn faces_player(fixture: &IndoorObject, player: &IndoorObject) -> bool {
    let to_player = player.position.xz() - fixture.position.xz();
    let distance = to_player.length();
    let forward = |o: &IndoorObject| Vec2::new(o.yaw.sin(), o.yaw.cos());
    player.kind == ObjectKind::Chair
        && (0.59..1.02).contains(&distance)
        && forward(fixture).dot(to_player) > 0.94 * distance
        && forward(player).dot(to_player) > 0.94 * distance
}

#[cfg(test)]
mod tests;
