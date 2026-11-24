//! The two supported upholstery grids have fixed connectivity and chart seams.
//! Store their index buffers once; physical arclength and shading stay per mesh.
use super::*;

pub(super) struct ChartVertex {
    pub source: usize,
    pub boxing: bool,
}

pub(super) struct Grid<const C: usize, const R: usize> {
    pub angles: [Vec2; C],
    pub latitudes: [Vec2; R],
    pub faces: Vec<[usize; 3]>,
    pub vertices: Vec<ChartVertex>,
    pub indices: Vec<u32>,
    pub underside_vertex: usize,
    pub underside_index: usize,
}

impl<const C: usize, const R: usize> Grid<C, R> {
    pub fn new() -> Self {
        let sides = C - 1;
        let angles = std::array::from_fn(|i| {
            let t = TAU * (i % sides) as f32 / sides as f32;
            Vec2::new(t.cos(), t.sin())
        });
        let latitudes = std::array::from_fn(|row| {
            let v = PI * row as f32 / R as f32;
            Vec2::new(v.cos(), v.sin())
        });
        let bottom = 1 + (R - 1) * C;
        let n = 2 * sides * (R - 1);
        let mut grid = Self {
            angles,
            latitudes,
            faces: Vec::with_capacity(n),
            vertices: Vec::with_capacity(bottom + 2 * C + 1),
            indices: Vec::with_capacity(n * 3),
            underside_vertex: 0,
            underside_index: 0,
        };
        let mut charts = vec![[None; 3]; bottom + 1];
        let mut face = |g: &mut Self, tri: [usize; 3], band: usize| {
            g.faces.push(tri);
            for source in tri {
                let index = *charts[source][band].get_or_insert_with(|| {
                    let index = g.vertices.len() as u32;
                    g.vertices.push(ChartVertex {
                        source,
                        boxing: band == 1,
                    });
                    index
                });
                g.indices.push(index);
            }
        };
        for col in 0..sides {
            face(&mut grid, [0, 2 + col, 1 + col], 0);
        }
        let top_rows = (R + 1) / 3 - 1;
        for row in 0..R - 2 {
            let first_bottom_row = R - 2 - top_rows;
            let band = if row < top_rows {
                0
            } else if row < first_bottom_row {
                1
            } else {
                2
            };
            if row == first_bottom_row {
                grid.underside_vertex = grid.vertices.len();
                grid.underside_index = grid.indices.len();
            }
            for col in 0..sides {
                let a = 1 + row * C + col;
                let b = a + C;
                face(&mut grid, [a, a + 1, b + 1], band);
                face(&mut grid, [a, b + 1, b], band);
            }
        }
        for col in 0..sides {
            let a = bottom - C + col;
            face(&mut grid, [a, a + 1, bottom], 2);
        }
        grid
    }
}
