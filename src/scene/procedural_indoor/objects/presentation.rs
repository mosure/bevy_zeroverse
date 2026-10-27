//! Mounted presentation surfaces share physical construction and optical targets.
use super::*;
mod board;
mod contents;
pub mod projection;
pub use board::{parameters, BoardProgram};
pub use contents::{Magnet, TrayItem};

pub(super) fn face(a: &mut Assembly, surface: Surface, label: &str, lo: Vec3, hi: Vec3) {
    let g = a.part(surface, label);
    let i = g.positions.len() as u32;
    g.positions.extend([
        [lo.x, lo.y, lo.z],
        [hi.x, lo.y, lo.z],
        [hi.x, hi.y, lo.z],
        [lo.x, hi.y, lo.z],
    ]);
    g.normals.extend([[0., 0., 1.]; 4]);
    g.uvs.extend([[0., 1.], [1., 1.], [1., 0.], [0., 0.]]);
    g.indices.extend([i, i + 1, i + 2, i, i + 2, i + 3]);
}
pub(super) fn whiteboard(a: &mut Assembly, o: &IndoorObject) {
    board::build(a, o);
}
pub(super) fn screen(a: &mut Assembly, o: &IndoorObject) {
    projection::build(a, o);
}

#[cfg(test)]
mod tests;
