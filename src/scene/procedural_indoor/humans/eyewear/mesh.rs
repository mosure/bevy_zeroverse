use super::{
    fit::{temple_path, Fit, Lens, SAMPLES},
    Rim,
};
use crate::scene::procedural_indoor::{
    geometry::Geometry,
    humans::{face::projection::Patch, HumanAssembly, HumanSurface},
};
use bevy::prelude::*;

impl Fit<'_> {
    pub(in super::super) fn append(
        &self,
        posed: &[Vec3],
        origin: Vec3,
        rotation: Quat,
        skin: &Patch,
        temples: &[Patch; 2],
        mesh: &mut HumanAssembly,
    ) {
        let p = self.program;
        let stock = p.rim_radius_m * self.scale;
        let start =
            [HumanSurface::Eyewear, HumanSurface::Lens].map(|s| mesh.part(s).positions.len());
        for lens in &self.lenses {
            optical_face(lens, mesh.part(HumanSurface::Lens));
            let g = mesh.part(HumanSurface::Eyewear);
            if p.rim != Rim::DrillMounted {
                let full = p.rim == Rim::Full;
                let n = if full { SAMPLES } else { SAMPLES / 2 + 1 };
                let rim: Vec<_> = lens.offsets[..n]
                    .iter()
                    .map(|&offset| {
                        let top = (offset.y / (0.5 * p.lens_height_m * self.scale)).clamp(-1., 1.);
                        Knot {
                            point: lens.point(offset),
                            axis: lens.normal(offset),
                            radii: Vec2::new(stock * 0.82, stock * (1. + p.brow_weight * top)),
                        }
                    })
                    .collect();
                sweep(g, &rim, full);
            } else {
                for index in [2, 14] {
                    g.small_ellipsoid(
                        Vec3::splat(0.0015 * self.scale),
                        Transform::from_translation(lens.mount(index, p.rim)),
                    );
                }
            }
            let path = temple_path(
                lens,
                p,
                self.scale,
                posed,
                origin,
                rotation,
                &temples[usize::from(lens.side > 0.)],
            );
            // A flat temple widens at the hinge, narrows at the ear and turns
            // down behind it. Both the bridge and arm terminate on fitted mounts.
            let arm: Vec<_> = path
                .iter()
                .zip([0.75, 1.48, 1.42, 1.33, 1.24, 0.96, 0.87, 0.78, 0.90, 0.73])
                .map(|(&point, w)| Knot {
                    point,
                    axis: Vec3::Y,
                    radii: Vec2::new(stock * w * 1.48, stock * 0.75),
                })
                .collect();
            sweep(g, &arm, false);
            let xy = Vec2::new(lens.side * p.bridge_width_m * 0.43, -0.004) * self.scale;
            if let Some((z, n)) = skin.front(xy) {
                let centre = xy.extend(z) + n * (0.0010 * self.scale);
                let at = lens.mount(14, p.rim);
                g.tapered_tube(
                    &[
                        at,
                        at.lerp(centre, 0.52) - Vec3::Z * (0.0018 * self.scale),
                        centre,
                    ],
                    stock * 0.74,
                    stock * 0.57,
                    6,
                );
                let up = (Vec3::Y - Vec3::Y.dot(n) * n).normalize_or(Vec3::X);
                let orientation = Quat::from_mat3(&Mat3::from_cols(up.cross(n), up, n));
                // Clear nose pads on thinner wire frames, moulded saddles on heavier stock.
                let pad = if p.rim_radius_m < 0.00135 {
                    HumanSurface::Lens
                } else {
                    HumanSurface::Eyewear
                };
                mesh.part(pad).small_ellipsoid(
                    Vec3::new(0.0025, 0.0047, 0.0012) * self.scale,
                    Transform::from_translation(centre).with_rotation(orientation),
                );
            }
        }
        mesh.part(HumanSurface::Eyewear)
            .tube(&self.bridge(skin), stock * 0.86, 8);
        for (s, first) in [HumanSurface::Eyewear, HumanSurface::Lens]
            .into_iter()
            .zip(start)
        {
            let g = mesh.part(s);
            for p in &mut g.positions[first..] {
                *p = (origin + rotation * Vec3::from_array(*p)).to_array();
            }
            for n in &mut g.normals[first..] {
                *n = (rotation * Vec3::from_array(*n)).to_array();
            }
        }
    }
}

// One curved optical surface. Extra transparent backfaces would blend the
// tint twice. The physical curvature supplies a moving, coherent highlight.
fn optical_face(l: &Lens, g: &mut Geometry) {
    let base = g.positions.len() as u32;
    let mut vertex = |xy: Vec2| {
        g.positions.push(l.point(xy).to_array());
        g.normals.push(l.normal(xy).to_array());
        g.uvs.push(xy.to_array());
    };
    vertex(Vec2::ZERO);
    for radius in [0.5, 1.] {
        for offset in l.offsets {
            vertex(offset * radius);
        }
    }
    for i in 0..SAMPLES as u32 {
        let next = (i + 1) % SAMPLES as u32;
        let a = base + 1 + i;
        let b = base + 1 + next;
        let c = a + SAMPLES as u32;
        let d = b + SAMPLES as u32;
        for mut tri in [[base, b, a], [a, b, d], [a, d, c]] {
            if l.side < 0. {
                tri.swap(1, 2);
            }
            g.indices.extend(tri);
        }
    }
}

struct Knot {
    point: Vec3,
    axis: Vec3,
    radii: Vec2,
}

// A shared ring at each joint eliminates the coincident discs and hard facets
// of adjacent capped cylinders. The rim closes with position-identical vertices.
fn sweep(g: &mut Geometry, path: &[Knot], closed: bool) {
    const SIDES: usize = 8;
    let base = g.positions.len() as u32;
    let rings = path.len() + usize::from(closed);
    let mut arc = 0.;
    let mut axes = [Vec3::ZERO; 2];
    for row in 0..rings {
        let i = row % path.len();
        let k = &path[i];
        let prev = if i == 0 && closed {
            path.len() - 1
        } else {
            i.saturating_sub(1)
        };
        let next = if closed {
            (i + 1) % path.len()
        } else {
            (i + 1).min(path.len() - 1)
        };
        let tangent = (path[next].point - path[prev].point).normalize_or(Vec3::Z);
        let u = (k.axis - tangent * tangent.dot(k.axis))
            .try_normalize()
            .unwrap_or_else(|| tangent.any_orthonormal_vector());
        let v = tangent.cross(u);
        if row > 0 {
            arc += k.point.distance(path[prev].point);
        }
        for j in 0..=SIDES {
            let angle = std::f32::consts::TAU * (j % SIDES) as f32 / SIDES as f32;
            let (s, c) = angle.sin_cos();
            g.positions
                .push((k.point + u * (c * k.radii.x) + v * (s * k.radii.y)).to_array());
            g.normals.push(
                (u * (c / k.radii.x) + v * (s / k.radii.y))
                    .normalize()
                    .to_array(),
            );
            g.uvs.push([
                std::f32::consts::TAU * j as f32 / SIDES as f32 * k.radii.max_element(),
                arc,
            ]);
        }
        if row == 0 {
            axes[0] = -tangent;
        }
        if row + 1 == rings {
            axes[1] = tangent;
        }
    }
    for row in 0..rings - 1 {
        for j in 0..SIDES {
            let a = base + (row * (SIDES + 1) + j) as u32;
            let b = a + (SIDES + 1) as u32;
            g.indices.extend([a, a + 1, b, b, a + 1, b + 1]);
        }
    }
    if !closed {
        for (end, normal) in axes.into_iter().enumerate() {
            let row = end * (rings - 1);
            let k = &path[row];
            let tangent = normal.any_orthonormal_vector();
            let bitangent = normal.cross(tangent);
            let centre = g.positions.len() as u32;
            g.positions.push(k.point.to_array());
            g.normals.push(normal.to_array());
            g.uvs.push([0., 0.]);
            for j in 0..SIDES {
                let p = g.positions[base as usize + row * (SIDES + 1) + j];
                let d = Vec3::from_array(p) - k.point;
                g.positions.push(p);
                g.normals.push(normal.to_array());
                g.uvs.push([d.dot(tangent), d.dot(bitangent)]);
            }
            for j in 0..SIDES as u32 {
                let a = centre + 1 + j;
                let b = centre + 1 + (j + 1) % SIDES as u32;
                g.indices.extend(if end == 0 {
                    [centre, b, a]
                } else {
                    [centre, a, b]
                });
            }
        }
    }
}
