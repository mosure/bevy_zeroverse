//! A slip pocket sewn to the inside rear panel along its sides and bottom.
//! The backed cloth has an open lip; only the stitched edges touch the lining.
use super::*;

pub(super) fn build(g: &mut Geometry, shell: &Shell, size: Vec3, p: &BagProgram) {
    const COLS: usize = 8;
    const ROWS: usize = 3;
    let width = size.x * p.pocket_width * 0.9;
    let height = size.y * p.body_height * p.pocket_height;
    let gap = size.z * p.pocket_depth * 0.72;
    let gauge = 0.0010;
    let point = |i: usize, j: usize| {
        let x = i as f32 / COLS as f32;
        let y = j as f32 / ROWS as f32;
        let crown = 4. * x * (1. - x);
        let u = 0.055 + (2. * x - 1.) * p.pocket_width;
        let t = 0.78 - p.pocket_height * (1. - y) - 0.035 * crown * y;
        Shell::panel(&shell.inside, 1, u, t) + Vec3::Z * (-0.00045 + gap * crown * y)
    };
    let mut points = [[Vec3::ZERO; COLS + 1]; ROWS + 1];
    for (j, row) in points.iter_mut().enumerate() {
        for (i, v) in row.iter_mut().enumerate() {
            *v = point(i, j);
        }
    }
    // Share the sampled cut between both faces and the four edge bindings.
    for outer in [false, true] {
        let start = g.positions.len() as u32;
        for (j, row) in points.iter().enumerate() {
            for (i, &v) in row.iter().enumerate() {
                let dx = row[(i + 1).min(COLS)] - row[i.saturating_sub(1)];
                let dy = points[(j + 1).min(ROWS)][i] - points[j.saturating_sub(1)][i];
                let n = dx.cross(dy).normalize();
                g.positions
                    .push((if outer { v + Vec3::Z * gauge } else { v }).to_array());
                g.normals.push((if outer { n } else { -n }).to_array());
                g.uvs.push([
                    width * i as f32 / COLS as f32,
                    height * j as f32 / ROWS as f32,
                ]);
                if i < COLS && j < ROWS {
                    let a = start + (j * (COLS + 1) + i) as u32;
                    let d = a + COLS as u32 + 1;
                    g.indices.extend(if outer {
                        [a, a + 1, d + 1, a, d + 1, d]
                    } else {
                        [a, d + 1, a + 1, a, d, d + 1]
                    });
                }
            }
        }
    }
    let mut hem =
        |a, b, normal| g.metric_quad([a, b, b + Vec3::Z * gauge, a + Vec3::Z * gauge], normal);
    for i in 0..COLS {
        hem(points[0][i], points[0][i + 1], Vec3::NEG_Y);
        hem(points[ROWS][i], points[ROWS][i + 1], Vec3::Y);
    }
    for j in 0..ROWS {
        hem(points[j][0], points[j + 1][0], Vec3::NEG_X);
        hem(points[j][COLS], points[j + 1][COLS], Vec3::X);
    }
}
