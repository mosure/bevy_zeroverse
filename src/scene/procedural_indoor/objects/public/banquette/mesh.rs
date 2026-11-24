use super::*;
use cover::{SewnBack, Stitch};

/// Sampling follows the stitches in physical space. A cropped repeat near an
/// end never creates a needle-thin triangle beside a rolled-edge sample.
fn knots(length: f32, repeat: Option<(f32, f32)>, edge: f32, corner: f32) -> Vec<f32> {
    let half = length * 0.5;
    let mut axis = vec![-half, half];
    for inset in [0.34 * corner, corner, edge * 0.5, edge] {
        axis.extend([-half + inset, half - inset]);
    }
    if let Some((step, origin)) = repeat {
        let first = ((-half - origin) / step).ceil() as i32;
        let last = ((half - origin) / step).floor() as i32;
        axis.extend((first..=last).map(|i| origin + i as f32 * step));
    } else {
        axis.extend([-length * 0.25, 0., length * 0.25]);
    }
    axis.sort_unstable_by(f32::total_cmp);
    axis.retain(|&x| x >= -half && x <= half);
    // Reserve the precise outer limits even if a stitch falls next to a cut.
    let mut used = 1;
    for i in 1..axis.len() {
        if axis[i] - axis[used - 1] >= 0.0008 && half - axis[i] >= 0.0008 {
            axis[used] = axis[i];
            used += 1;
        }
    }
    axis.truncate(used);
    axis.push(half);
    axis
}

// An elliptic grid maps just the square corner patches onto quarter circles.
// Straight edges and the internal stitch grid retain their physical positions.
fn rounded_point(p: Vec2, half: Vec2, radius: f32) -> Vec2 {
    let offset = (p.abs() - (half - radius)).max(Vec2::ZERO);
    if offset.min_element() <= 0. {
        return p;
    }
    let uv = offset / radius;
    let scale = Vec2::new(
        (1. - 0.5 * uv.y * uv.y).sqrt(),
        (1. - 0.5 * uv.x * uv.x).sqrt(),
    );
    p - p.signum() * offset * (Vec2::ONE - scale)
}

impl SewnBack {
    pub(super) fn build(&self, g: &mut Geometry, tf: Transform) {
        let (rx, ry) = match self.stitch {
            Stitch::Channels {
                vertical,
                pitch_m,
                offset_m,
            } => {
                let period = Some((pitch_m / 8., offset_m));
                if vertical {
                    (period, None)
                } else {
                    (None, period)
                }
            }
            Stitch::Diamonds {
                pitch_m, offset_m, ..
            } => (
                Some((pitch_m.x / 8., offset_m.x)),
                Some((pitch_m.y / 4., offset_m.y)),
            ),
        };
        let xs = knots(self.size_m.x, rx, self.edge_roll_m, self.corner_radius_m);
        let ys = knots(self.size_m.y, ry, self.edge_roll_m, self.corner_radius_m);
        let nx = xs.len();
        let ny = ys.len();
        let start = g.positions.len();
        let n = nx * ny;
        g.positions.reserve(n);
        g.normals.reserve(n);
        g.uvs.reserve(n);
        g.indices.reserve(6 * (nx - 1) * (ny - 1));
        let half = self.size_m * 0.5;
        for (j, &y) in ys.iter().enumerate() {
            for (i, &x) in xs.iter().enumerate() {
                let chart = Vec2::new(x, y);
                let coord = rounded_point(chart, half, self.corner_radius_m);
                let z = self.depth(coord);
                let point = tf.transform_point(coord.extend(z));
                let index = start + j * nx + i;
                let arc = |at: usize| point.distance(Vec3::from(g.positions[at]));
                let u = if i == 0 {
                    0.
                } else {
                    g.uvs[index - 1][0] + arc(index - 1)
                };
                let v = if j == 0 {
                    0.
                } else {
                    g.uvs[index - nx][1] + arc(index - nx)
                };
                g.positions.push(point.to_array());
                g.uvs.push([u, v]);
            }
        }
        let axis = tf.rotation * Vec3::Z;
        let first_face = g.indices.len();
        for j in 0..ny - 1 {
            for i in 0..nx - 1 {
                let a = (start + j * nx + i) as u32;
                let b = a + 1;
                let c = a + nx as u32;
                let d = c + 1;
                // Follow the deeper pair through the cell. Both families of
                // diagonal valleys get an edge, including on cropped diamonds.
                let pos = |k: u32| Vec3::from(g.positions[k as usize]);
                if ((pos(a) - pos(b)) + (pos(d) - pos(c))).dot(axis) < 0. {
                    g.indices.extend([a, b, d, a, d, c]);
                } else {
                    g.indices.extend([a, b, c, b, d, c]);
                }
            }
        }
        // Average incident face normals with equal weight. Closely cropped
        // stitches still contribute to the shading at their shared vertices.
        g.normals.resize(start + n, [0.; 3]);
        for tri in g.indices[first_face..].as_chunks::<3>().0 {
            let pos = |k| Vec3::from(g.positions[k as usize]);
            let face = (pos(tri[1]) - pos(tri[0]))
                .cross(pos(tri[2]) - pos(tri[0]))
                .normalize();
            for &vertex in tri {
                let normal = &mut g.normals[vertex as usize];
                *normal = (Vec3::from(*normal) + face).to_array();
            }
        }
        for normal in &mut g.normals[start..] {
            *normal = Vec3::from(*normal).normalize().to_array();
        }
        // Upholstery wraps behind the exposed front to a tack line inside the
        // structural board. Only the visible faces are meshed.
        let mut along = 0.;
        let mut side = |a: usize, b: usize| {
            let p = Vec3::from(g.positions[start + a]);
            let q = Vec3::from(g.positions[start + b]);
            let dp = (p - tf.translation).dot(axis) + 0.001;
            let dq = (q - tf.translation).dot(axis) + 0.001;
            let rear = p - axis * dp;
            let end = q - axis * dq;
            let tangent = end - rear;
            let n = tangent.cross(axis).normalize();
            let next = along + tangent.length();
            g.vertex_quad(
                [rear, end, q, p],
                [n; 4],
                [[along, 0.], [next, 0.], [next, dq], [along, dp]],
            );
            along = next;
        };
        for i in 0..nx - 1 {
            side(i, i + 1);
        }
        for j in 0..ny - 1 {
            side(j * nx + nx - 1, (j + 1) * nx + nx - 1);
        }
        for i in (1..nx).rev() {
            side((ny - 1) * nx + i, (ny - 1) * nx + i - 1);
        }
        for j in (1..ny).rev() {
            side(j * nx, (j - 1) * nx);
        }
        self.buttons(|xy, r| {
            let valley = self.depth(xy);
            // Seat the shank in the tuck and expose the cap above the cloth
            // at its rim. A cap placed at the mathematical valley disappears
            // into the four rising folds on deeply tufted stock.
            let rim = (0..8)
                .map(|i| {
                    self.depth(xy + r * Vec2::from_angle(i as f32 * std::f32::consts::FRAC_PI_4))
                })
                .fold(valley, f32::max)
                - valley;
            let at = tf.mul_transform(
                Transform::from_translation(xy.extend(valley))
                    .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
            );
            g.lathe(
                &[
                    (r * 0.80, -r * 0.20),
                    (r, rim),
                    (r * 0.86, rim + r * 0.24),
                    (0., rim + r * 0.40),
                ],
                12,
                at,
            );
        });
    }
}
