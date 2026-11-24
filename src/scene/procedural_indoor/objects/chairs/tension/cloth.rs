//! Closed, two-sided textile stock. The weave uses the regular metric PBR
//! material. Only silhouette, broad bends and clear apertures need triangles.
use super::*;

/// A full width cut or a horizontal strap follows the same lumbar and shoulder curve.
fn horizontal_row<const N: usize>(f: &Frame, v: f32, ease: f32) -> [Vec3; N] {
    let half_chord = f.chord(v * 2. - 1.);
    std::array::from_fn(|i| {
        let along = 2. * i as f32 / (N - 1) as f32 - 1.;
        f.local(along * half_chord, v) + Vec3::Z * (ease * (1. - along * along))
    })
}

pub(super) fn sling(g: &mut Geometry, f: &Frame, gauge_m: f32, ease_m: f32) {
    // Both cut ends tuck into the rim. A short chord in place of a collapsed
    // pole keeps the triangles and the UV chart regular.
    let hem = (0.0035 / f.height).min(0.04);
    let centers: [[_; 9]; 13] = std::array::from_fn(|row| {
        let v = hem + (1. - 2. * hem) * 0.5 * (1. - (PI * row as f32 / 12.).cos());
        horizontal_row(f, v, ease_m * (PI * v).sin())
    });
    patch(g, f, gauge_m, &centers);
}

pub(super) fn webbing(
    g: &mut Geometry,
    f: &Frame,
    direction: WebbingDirection,
    nominal_pitch_m: f32,
    fill_fraction: f32,
    gauge_m: f32,
    ease_m: f32,
) {
    let span = match direction {
        WebbingDirection::Horizontal => f.height,
        WebbingDirection::Vertical => f.width,
    };
    let bands = (span / nominal_pitch_m).round().clamp(2., 18.) as usize;
    for band in 0..bands {
        let ordinate = |t| (band as f32 + 0.5 + (t - 0.5) * fill_fraction) / bands as f32;
        match direction {
            WebbingDirection::Horizontal => {
                let centers: [[_; 11]; 3] = std::array::from_fn(|row| {
                    horizontal_row(f, ordinate(row as f32 * 0.5), ease_m)
                });
                patch(g, f, gauge_m, &centers);
            }
            WebbingDirection::Vertical => {
                let columns: [_; 3] = std::array::from_fn(|col| {
                    let x = 2. * ordinate(col as f32 * 0.5) - 1.;
                    (x, f.chord(x))
                });
                let centers: [[_; 3]; 11] = std::array::from_fn(|row| {
                    let along = row as f32 / 5. - 1.;
                    columns.map(|(x, chord)| {
                        f.local(x, 0.5 + 0.5 * along * chord)
                            + Vec3::Z * (ease_m * (1. - along * along))
                    })
                });
                patch(g, f, gauge_m, &centers);
            }
        }
    }
}

/// Append both skins of one cut piece and close its edges. Adjacent cells share
/// vertices and shading normals. UV coordinates stay in metres across the bend.
fn patch<const COLS: usize, const ROWS: usize>(
    g: &mut Geometry,
    f: &Frame,
    gauge: f32,
    centers: &[[Vec3; COLS]; ROWS],
) {
    let (nx, ny) = (COLS - 1, ROWS - 1);
    let count = ROWS * COLS;
    let base = g.positions.len() as u32;
    let vertex = |x: usize, y: usize| base + 2 * (y * COLS + x) as u32;
    let vertex_count = 2 * count + 8 * (nx + ny);
    g.positions.reserve(vertex_count);
    g.normals.reserve(vertex_count);
    g.uvs.reserve(vertex_count);
    g.indices.reserve(12 * (nx * ny + nx + ny));
    for y in 0..=ny {
        for x in 0..=nx {
            let u = centers[y][(x + 1).min(nx)] - centers[y][x.saturating_sub(1)];
            let v = centers[(y + 1).min(ny)][x] - centers[y.saturating_sub(1)][x];
            let normal = u.cross(v).normalize();
            let center = centers[y][x];
            for sign in [-1., 1.] {
                g.positions.push(
                    f.transform
                        .transform_point(center + normal * (sign * gauge * 0.5))
                        .to_array(),
                );
                g.normals
                    .push((f.transform.rotation * (normal * sign)).to_array());
                g.uvs.push([center.x, center.y]);
            }
            if x < nx && y < ny {
                let a = vertex(x, y);
                let b = vertex(x + 1, y);
                let c = vertex(x + 1, y + 1);
                let d = vertex(x, y + 1);
                g.indices
                    .extend([a, d, b, b, d, c, a + 1, b + 1, d + 1, b + 1, c + 1, d + 1]);
            }
        }
    }
    let mut edge = |a: u32, b: u32| {
        let [lo, next, hi_next, hi] =
            [a, b, b + 1, a + 1].map(|i| Vec3::from_array(g.positions[i as usize]));
        let outside = (next - lo).cross(hi - lo);
        g.metric_quad([lo, next, hi_next, hi], outside);
    };
    for x in 0..nx {
        edge(vertex(x, 0), vertex(x + 1, 0));
        edge(vertex(x + 1, ny), vertex(x, ny));
    }
    for y in 0..ny {
        edge(vertex(nx, y), vertex(nx, y + 1));
        edge(vertex(0, y + 1), vertex(0, y));
    }
}
