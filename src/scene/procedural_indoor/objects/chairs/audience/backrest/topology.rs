//! Reusable connectivity for solid and pierced shells. On a pierced shell,
//! entire grid lines follow the four sides of the grip, keeping even a high,
//! off-centre aperture inside its own local patch.
use super::*;
use std::sync::LazyLock;

const COLS: usize = 10;
const SOLID_ROWS: usize = 6;
const OPEN_ROWS: usize = 10;
const LEFT: usize = 3;
const RIGHT: usize = 7;
const BOTTOM: usize = 4;
const TOP: usize = 8;
const RING: usize = 2 * (RIGHT - LEFT + TOP - BOTTOM);
pub(super) const MAX_VERTICES: usize = (COLS + 1) * (OPEN_ROWS + 1) + RING;

pub(super) struct Topology {
    pub(super) nodes: Vec<Vec2>,
    pub(super) slot_start: usize,
    pub(super) triangles: Vec<[u32; 3]>,
    pub(super) perimeter: Vec<usize>,
    pub(super) hole: Vec<usize>,
}

pub(super) fn get(open: bool) -> &'static Topology {
    static SOLID: LazyLock<Topology> = LazyLock::new(|| Topology::new(false));
    static OPEN: LazyLock<Topology> = LazyLock::new(|| Topology::new(true));
    if open {
        &OPEN
    } else {
        &SOLID
    }
}

fn perimeter(
    left: usize,
    right: usize,
    bottom: usize,
    top: usize,
) -> impl Iterator<Item = (usize, usize)> {
    (left..right)
        .map(move |i| (i, bottom))
        .chain((bottom..top).map(move |j| (right, j)))
        .chain((left + 1..=right).rev().map(move |i| (i, top)))
        .chain((bottom + 1..=top).rev().map(move |j| (left, j)))
}

/// -2, -1, 1 and 2 correspond to the outer edge, near and far sides of the
/// rectangular patch, and the opposite outer edge, respectively.
fn patch_axis(i: usize, left: usize, right: usize, end: usize) -> f32 {
    if i < left {
        -2. + i as f32 / left as f32
    } else if i > right {
        1. + (i - right) as f32 / (end - right) as f32
    } else {
        -1. + 2. * (i - left) as f32 / (right - left) as f32
    }
}

impl Topology {
    fn new(open: bool) -> Self {
        let rows = if open { OPEN_ROWS } else { SOLID_ROWS };
        let mut nodes = Vec::with_capacity(MAX_VERTICES);
        let mut indices = [[0; COLS + 1]; OPEN_ROWS + 1];
        for (row, line) in indices.iter_mut().take(rows + 1).enumerate() {
            for (col, id) in line.iter_mut().enumerate() {
                if open && (LEFT + 1..RIGHT).contains(&col) && (BOTTOM + 1..TOP).contains(&row) {
                    continue;
                }
                *id = nodes.len();
                nodes.push(if open {
                    Vec2::new(
                        patch_axis(col, LEFT, RIGHT, COLS),
                        patch_axis(row, BOTTOM, TOP, rows),
                    )
                } else {
                    Vec2::new(2. * col as f32 / COLS as f32 - 1., row as f32 / rows as f32)
                });
            }
        }
        let mut triangles = Vec::with_capacity(2 * COLS * rows);
        let mut quad = |a: usize, b: usize, c: usize, d: usize| {
            triangles.extend([
                [a as u32, b as u32, c as u32],
                [a as u32, c as u32, d as u32],
            ]);
        };
        for row in 0..rows {
            for col in 0..COLS {
                if open && (BOTTOM..TOP).contains(&row) && (LEFT..RIGHT).contains(&col) {
                    continue;
                }
                quad(
                    indices[row][col],
                    indices[row][col + 1],
                    indices[row + 1][col + 1],
                    indices[row + 1][col],
                );
            }
        }
        let slot_start = nodes.len();
        let mut hole = Vec::new();
        if open {
            let border: Vec<_> = perimeter(LEFT, RIGHT, BOTTOM, TOP)
                .map(|(i, j)| indices[j][i])
                .collect();
            for (i, &outer) in border.iter().enumerate() {
                nodes.push(nodes[outer]);
                let j = (i + 1) % border.len();
                quad(outer, border[j], slot_start + j, slot_start + i);
            }
            // An inner rim faces the opening, opposite the outer edge.
            hole.extend((slot_start..nodes.len()).rev());
        }
        Self {
            nodes,
            slot_start,
            triangles,
            perimeter: perimeter(0, COLS, 0, rows)
                .map(|(i, j)| indices[j][i])
                .collect(),
            hole,
        }
    }
}
