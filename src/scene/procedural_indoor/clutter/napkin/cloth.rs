use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;

pub(super) const ROWS: usize = 11;
pub(super) const COLS: usize = 13;
const CREASE: usize = 6;
type Grid = [[Vec3; COLS]; ROWS];

impl Napkin {
    /// One section along the crease, following the cut from one free hem to
    /// the other. The triangle tapers along the fold. The tent has two feet.
    fn section(&self, size: Vec3, t: f32) -> [Vec3; COLS] {
        let bow = 1. - t * t;
        let curl = self.corner_lift * t;
        let points = if self.fold == Fold::Tent {
            let ridge = self.crease_offset * (0.55 + 0.45 * t);
            [
                (-0.47 + bow * 0.026, 0.024 * bow),
                (-0.33, 0.22 + curl * 0.20),
                (ridge - 0.082, 0.85 - bow * 0.07),
                (ridge, 1. - bow * 0.12),
                (ridge + 0.089, 0.86 - bow * 0.10),
                (0.32 + curl * 0.14, 0.25 + curl * 0.30),
                (0.47 - bow * 0.02, 0.035 * (1. + t)),
            ]
        } else {
            let tip = self.crease_offset * 2.;
            let span = if self.fold == Fold::Triangle {
                let side = if t < tip { 1. + tip } else { 1. - tip };
                0.045 + 0.89 * (1. - (t - tip).abs() / side)
            } else {
                0.93 - 0.037 * bow
            };
            let left = -0.47 + self.crease_offset * 0.20 * t;
            let x = |u: f32| left + u * span;
            [
                (x(1.), 0.03 + bow * 0.07),
                (x(0.40), 0.025 * (1. + t)),
                (x(0.042), 0.13 + bow * 0.09),
                (x(0.), 0.49 + bow * 0.07),
                (x(0.039), 0.84 + bow * 0.07),
                (x(0.39), 0.57 + bow * 0.18),
                (x(self.free_edge), 0.61 + curl),
            ]
        };
        // The envelope is the finished width, including both turned edges.
        let cut_width = size.x - 4. * self.ply_m;
        let [a, b, c, d, e, f, g] =
            points.map(|(x, y)| Vec3::new(x * cut_width, y * size.y, t * size.z * 0.5));
        // Keep the free hems a physical width, including where the ends of the
        // triangular fold get too narrow for the full turn-under.
        let edge = |hem: Vec3, next: Vec3| {
            let step = next - hem;
            let turn = step * (self.hem_m / step.length()).min(0.55);
            [0.68, 1., 1.22].map(|s| hem + s * turn)
        };
        let [a1, a2, a3] = edge(a, b);
        let [g1, g2, g3] = edge(g, f);
        [a, a1, a2, a3, b, c, d, e, f, g3, g2, g1, g]
    }

    pub(super) fn cloth(&self, mesh: &mut Geometry, size: Vec3) {
        let turn = (2. * self.hem_m / size.z).min(0.23);
        let end = [0.68, 1., 1.22].map(|s| 1. - s * turn);
        let mid: Grid = [
            -1., -end[0], -end[1], -end[2], -0.48, 0., 0.48, end[2], end[1], end[0], 1.,
        ]
        .map(|t| self.section(size, t));
        let mut layers = [mid; 2];
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for i in 0..ROWS {
            for j in 0..COLS {
                let normal = surface_normal(&mid, i, j);
                // A turned edge has extra plies and a pressed stitch line.
                // Thicken the outer face, leaving the space between the leaves
                // open. Both the hem and its crease use the original sheet.
                let border = i.min(ROWS - 1 - i).min(j.min(COLS - 1 - j));
                let relief = match border {
                    0 => 0.72,
                    1 => 1.48,
                    2 => -0.12,
                    _ => 0.,
                };
                for (layer, offset) in layers.iter_mut().zip([0.5 + relief, -0.5]) {
                    let p = mid[i][j] + normal * (offset * self.ply_m);
                    // Keep each cut end in its cross-section. Its hem stays
                    // inside the admitted length even when the cloth twists.
                    layer[i][j] = Vec3::new(p.x, p.y, mid[i][j].z);
                    lo = lo.min(p.y);
                    hi = hi.max(p.y);
                }
            }
        }
        let cut = Mat2::from_angle(self.grain_angle_rad);
        let base = mesh.positions.len() as u32;
        mesh.positions.reserve(2 * ROWS * COLS);
        for (side, layer) in layers.iter_mut().enumerate() {
            for p in layer.iter_mut().flatten() {
                p.y = (p.y - lo) * (size.y / (hi - lo));
            }
            for (i, row) in mid.iter().enumerate() {
                // Use unfolded distance to keep the weave continuous over the
                // ridge, with the same warp direction on the facing leaves.
                let mut distance = [0.; COLS];
                for j in 1..COLS {
                    distance[j] = distance[j - 1] + row[j - 1].distance(row[j]);
                }
                for j in 0..COLS {
                    let p = layer[i][j];
                    let normal = surface_normal(layer, i, j);
                    mesh.positions.push(p.to_array());
                    mesh.normals
                        .push((if side == 0 { normal } else { -normal }).to_array());
                    mesh.uvs
                        .push((cut * Vec2::new(distance[j] - distance[CREASE], p.z)).to_array());
                    if i + 1 < ROWS && j + 1 < COLS {
                        let a = base + ((side * ROWS + i) * COLS + j) as u32;
                        let b = a + COLS as u32;
                        let indices = if side == 0 {
                            [a, b, b + 1, a, b + 1, a + 1]
                        } else {
                            [a, b + 1, b, a, a + 1, b + 1]
                        };
                        mesh.indices.extend(indices);
                    }
                }
            }
        }
        let [top, bottom] = &layers;
        let mut hem = |(i, j): (usize, usize), (k, l): (usize, usize)| {
            let p = [top[i][j], bottom[i][j], bottom[k][l], top[k][l]];
            let normal = (p[1] - p[0]).cross(p[3] - p[0]).normalize();
            mesh.metric_quad(p, normal);
        };
        for j in 1..COLS {
            hem((0, j), (0, j - 1));
            hem((ROWS - 1, j - 1), (ROWS - 1, j));
        }
        for i in 1..ROWS {
            hem((i - 1, 0), (i, 0));
            hem((i, COLS - 1), (i - 1, COLS - 1));
        }
    }
}

fn surface_normal(grid: &Grid, i: usize, j: usize) -> Vec3 {
    let at = grid[i][j];
    // Equal-weight tangents preserve the rounded fold between two long leaves.
    let before = (at - grid[i][j.saturating_sub(1)]).normalize_or_zero();
    let after = (grid[i][(j + 1).min(COLS - 1)] - at).normalize_or_zero();
    let along = grid[(i + 1).min(ROWS - 1)][j] - grid[i.saturating_sub(1)][j];
    along.cross(before + after).normalize()
}
