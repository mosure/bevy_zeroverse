//! A waste fitting joins both faces of the basin sheet. Slots expose the recessed
//! throat to RGB and geometry cameras; the underside encloses the fitting.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Strainer {
    Basket,
    BarGrate,
    Open,
}

#[derive(Clone, Debug, Serialize)]
pub struct WasteFitting {
    pub strainer: Strainer,
    pub seat_radius_m: f32,
    pub depth_ratio: f32,
    pub webs: u32,
    pub azimuth: f32,
    pub web_fraction: f32,
}

impl WasteFitting {
    pub(super) fn sample(seed: u64, index: u32) -> Self {
        let mut r = stream(seed, 0x5741_5354_4500 + u64::from(index));
        let strainer = match r.random_range(0..10) {
            0 => Strainer::Open,
            1..=3 => Strainer::BarGrate,
            _ => Strainer::Basket,
        };
        Self {
            strainer,
            seat_radius_m: if strainer == Strainer::Basket {
                r.random_range(0.032..0.052)
            } else {
                r.random_range(0.018..0.040)
            },
            depth_ratio: r.random_range(0.68..1.04),
            webs: match strainer {
                Strainer::Basket => r.random_range(6..=10),
                Strainer::BarGrate => r.random_range(2..=4),
                Strainer::Open => 0,
            },
            azimuth: r.random_range(0.0..TAU),
            web_fraction: r.random_range(0.11..0.19),
        }
    }

    pub(super) fn build<const N: usize>(
        &self,
        a: &mut Assembly,
        p: &program::Basin,
        bowl: Surface,
        floor: &[Vec3; N],
        bottom: &[Vec3; N],
    ) {
        let r = p.drain_radius;
        let center = Vec3::new(p.drain.x, p.floor_m, p.drain.y);
        let radial: [Vec3; N] = std::array::from_fn(|i| {
            let (sin, cos) = (TAU * i as f32 / N as f32).sin_cos();
            Vec3::new(cos, 0., sin)
        });
        let ring = |radius: f32, y: f32| radial.map(|v| center + v * radius + Vec3::Y * y);
        let shell = a.part(
            bowl,
            crate::scene::procedural_indoor::materials::worktop_finish::BASIN_LABEL,
        );
        pierced_floor(shell, floor, &ring(r, 0.), &radial, false);
        pierced_floor(shell, bottom, &ring(r, -p.wall_m), &radial, true);

        let depth = self.depth_ratio * r;
        let metal = a.part(Surface::Chrome, "sink");
        for (sign, profile) in [
            (
                1.,
                [
                    (r, 0.),
                    (r * 0.82, -r * 0.06),
                    (r * 0.68, -r * 0.24),
                    (r * 0.60, -depth),
                ],
            ),
            (
                -1.,
                [
                    (r, -p.wall_m),
                    (r * 0.86, -p.wall_m - r * 0.06),
                    (r * 0.75, -p.wall_m - r * 0.24),
                    (r * 0.67, -depth - r * 0.09),
                ],
            ),
        ] {
            for pair in profile.windows(2) {
                let [(r0, y0), (r1, y1)] = [pair[0], pair[1]];
                let a = ring(r0, y0);
                let b = ring(r1, y1);
                let slope = Vec2::new(y1 - y0, r0 - r1).normalize() * sign;
                let normals = radial.map(|v| v * slope.x + Vec3::Y * slope.y);
                for i in 0..N {
                    let j = (i + 1) % N;
                    let mut q = [a[i], b[i], b[j], a[j]];
                    let mut ns = [normals[i], normals[i], normals[j], normals[j]];
                    if sign < 0. {
                        q.reverse();
                        ns.reverse();
                    }
                    metal.vertex_quad(q, ns, q.map(|v| v.xz().to_array()));
                }
            }
            let (radius, y) = profile[3];
            let edge = ring(radius, y);
            let idx = metal.positions.len() as u32;
            for v in std::iter::once(center + Vec3::Y * y).chain(edge) {
                metal.positions.push(v.to_array());
                metal.normals.push([0., sign, 0.]);
                metal.uvs.push(v.xz().to_array());
            }
            for i in 0..N as u32 {
                let b = idx + 1 + i;
                let c = idx + 1 + (i + 1) % N as u32;
                metal
                    .indices
                    .extend(if sign > 0. { [idx, c, b] } else { [idx, b, c] });
            }
        }
        self.insert(metal, center, r);
    }

    fn insert(&self, g: &mut Geometry, center: Vec3, r: f32) {
        let at = center - Vec3::Y * (r * 0.22);
        let width = self.web_fraction * r;
        match self.strainer {
            Strainer::Open => {}
            Strainer::Basket => {
                // Each pressed web terminates inside the seat and the central lift button.
                for i in 0..self.webs {
                    let yaw = self.azimuth + TAU * i as f32 / self.webs as f32;
                    let rot = Quat::from_rotation_y(yaw);
                    g.cuboid(
                        Vec3::new(width, r * 0.074, r * 0.59),
                        0.,
                        Transform::from_translation(at + rot * Vec3::Z * (r * 0.46))
                            .with_rotation(rot),
                    );
                }
                g.lathe(
                    &[
                        (0., -r * 0.08),
                        (r * 0.245, -r * 0.08),
                        (r * 0.245, r * 0.065),
                        (r * 0.15, r * 0.14),
                        (0., r * 0.14),
                    ],
                    16,
                    Transform::from_translation(at),
                );
            }
            Strainer::BarGrate => {
                let rot = Quat::from_rotation_y(self.azimuth);
                for i in 0..self.webs {
                    let x = r * 0.76 * ((i as f32 + 0.5) / self.webs as f32 - 0.5);
                    let half = ((r * 0.76).powi(2) - x * x).sqrt();
                    g.cuboid(
                        Vec3::new(width, r * 0.08, 2. * half),
                        0.,
                        Transform::from_translation(at + rot * Vec3::X * x).with_rotation(rot),
                    );
                }
            }
        }
    }
}

/// Stitch nested convex outlines in order of edge normal. Each inner edge fans
/// to the supporting outer vertex, so off-centre drains in wide bowls retain a
/// circular mouth without crossing triangles or subdividing the wall boundary.
/// Both outlines begin at their +X support and run counterclockwise in XZ.
fn pierced_floor<const N: usize>(
    g: &mut Geometry,
    outer: &[Vec3; N],
    hole: &[Vec3; N],
    radial: &[Vec3; N],
    underside: bool,
) {
    let base = g.positions.len() as u32;
    let normal = if underside { Vec3::NEG_Y } else { Vec3::Y };
    for v in outer.iter().chain(hole) {
        g.positions.push(v.to_array());
        g.normals.push(normal.to_array());
        g.uvs.push(v.xz().to_array());
    }
    let (mut i, mut j) = (0, 0);
    for end in [N / 2, N] {
        while i < end || j < end {
            let a = (i % N) as u32;
            let b = (N + j % N) as u32;
            let before = j == end
                || (i < end
                    && (outer[(i + 1) % N] - outer[i]).dot(radial[j] + radial[(j + 1) % N]) >= 0.);
            let c = if before {
                i += 1;
                (i % N) as u32
            } else {
                j += 1;
                (N + j % N) as u32
            };
            let face = if underside { [a, c, b] } else { [a, b, c] };
            g.indices.extend(face.map(|i| base + i));
        }
    }
}
