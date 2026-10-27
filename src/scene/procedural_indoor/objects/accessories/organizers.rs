//! Unequal desktop compartments and document trays with supported, fitted contents.
use super::*;
use crate::scene::procedural_indoor::{materials::variants, placement::rotated_half};

#[derive(Debug, Clone, serde::Serialize)]
pub struct OrganizerProgram {
    pub weights: Vec<f32>,
    pub heights: Vec<f32>,
    pub filled: Vec<bool>,
    pub front_fraction: f32,
    pub wall_fraction: f32,
    pub finish: Surface,
}

pub fn parameters(o: &IndoorObject) -> OrganizerProgram {
    let mut r = stream(o.seed, 0x4f5247414e495a45);
    let count = if o.kind == ObjectKind::DocumentTray {
        r.random_range(1..=4)
    } else {
        r.random_range(1..=5)
    };
    OrganizerProgram {
        weights: (0..count).map(|_| r.random_range(0.55..1.60)).collect(),
        heights: (0..count).map(|_| r.random_range(0.36..0.96)).collect(),
        filled: (0..count).map(|_| r.random_bool(0.66)).collect(),
        front_fraction: r.random_range(0.12..0.78),
        wall_fraction: r.random_range(0.014..0.032),
        finish: [Surface::Metal, Surface::Plastic, Surface::Wood][r.random_range(0..3)],
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let label = o.kind.class_name();
    let sum = p.weights.iter().sum::<f32>();
    let trays = o.kind == ObjectKind::DocumentTray;
    let narrowest = p.weights.iter().copied().fold(f32::INFINITY, f32::min) / sum;
    let t =
        (s.min_element() * p.wall_fraction).min(if trays { s.y } else { s.x } * narrowest * 0.12);
    if trays && p.weights.len() > 1 {
        let last = s.y * p.weights.last().unwrap() / sum;
        let frame_height = s.y - last + t + (last - t) * p.front_fraction;
        // A single connected frame supports every tier; unequal spacing models
        // mixed letter/document storage without arbitrary floating shelves.
        for x in [-1., 1.] {
            for z in [-1., 1.] {
                a.box_part(
                    p.finish,
                    label,
                    Vec3::new(
                        x * (s.x * 0.5 - t * 1.6),
                        frame_height * 0.5,
                        z * (s.z * 0.5 - t * 1.6),
                    ),
                    Vec3::new(t, frame_height, t),
                    t * 0.15,
                );
            }
        }
    } else if !trays {
        a.box_part(
            p.finish,
            label,
            Vec3::Y * t * 0.5,
            Vec3::new(s.x, t, s.z),
            t * 0.2,
        );
    }
    let mut cursor = if trays { 0. } else { -s.x * 0.5 };
    let mut r = stream(o.seed, 0x4f5247434f4e5445);
    for (i, &weight) in p.weights.iter().enumerate() {
        let width = if trays { s.x } else { s.x * weight / sum };
        let height = if trays {
            s.y * weight / sum
        } else {
            s.y * p.heights[i]
        };
        let base = if trays { cursor } else { 0. };
        let x = if trays { 0. } else { cursor + width * 0.5 };
        let wall_h = if trays {
            (height - t) * p.front_fraction
        } else {
            height - t
        };
        if trays {
            a.box_part(
                p.finish,
                label,
                Vec3::new(x, base + t * 0.5, 0.),
                Vec3::new(width, t, s.z),
                t * 0.2,
            );
        }
        // Each bay owns its side walls; a physical seam keeps adjacent unequal
        // heights from producing overlapping coplanar divider faces.
        for side in [-1., 1.] {
            a.box_part(
                p.finish,
                label,
                Vec3::new(
                    x + side * (width * 0.5 - t * 0.6),
                    base + t + wall_h * 0.5,
                    0.,
                ),
                Vec3::new(t, wall_h, s.z),
                t * 0.12,
            );
        }
        a.box_part(
            p.finish,
            label,
            Vec3::new(x, base + t + wall_h * 0.5, -s.z * 0.5 + t * 0.5),
            Vec3::new(width - t * 2.2, wall_h, t),
            t * 0.12,
        );
        if !trays {
            a.box_part(
                p.finish,
                label,
                Vec3::new(x, t + wall_h * p.front_fraction * 0.5, s.z * 0.5 - t * 0.5),
                Vec3::new(width - t * 2.2, wall_h * p.front_fraction, t),
                t * 0.12,
            );
        }
        if p.filled[i] {
            let inner = Vec2::new(width - t * 4.4, s.z - t * 4.4);
            if trays || r.random_bool(0.42) {
                let yaw = r.random_range(-0.18_f32..0.18);
                let size =
                    inner * Vec2::new(r.random_range(0.56..0.88), r.random_range(0.60..0.90));
                let half = rotated_half(size, yaw);
                let fit = (inner * 0.5 / half).min_element().min(1.);
                let size = size * fit * 0.97;
                let half = rotated_half(size, yaw);
                let freedom = inner * 0.5 - half;
                let pos = Vec3::new(
                    x + freedom.x * r.random_range(-1.0..1.0),
                    base + t,
                    freedom.y * r.random_range(-1.0..1.0),
                );
                let h = (height - t * 2.) * r.random_range(0.08..0.36);
                let finish = variants::slot(o.seed ^ i as u64);
                let mut page = Geometry::default();
                page.cuboid(Vec3::new(size.x, h, size.y), t * 0.1, Transform::IDENTITY);
                let tf = Transform::from_translation(pos + Vec3::Y * h * 0.5)
                    .with_rotation(Quat::from_rotation_y(yaw));
                for ((point, normal), uv) in page
                    .positions
                    .iter_mut()
                    .zip(&mut page.normals)
                    .zip(&mut page.uvs)
                {
                    let local = Vec3::from_array(*point);
                    *uv = (local.xz() / size + Vec2::splat(0.5)).to_array();
                    *point = tf.transform_point(local).to_array();
                    *normal = (tf.rotation * Vec3::from_array(*normal)).to_array();
                }
                a.part(
                    Surface::PrintedPaper,
                    &format!("paper#finish{finish}#uv_local"),
                )
                .append(page);
            } else {
                let count = r.random_range(1..=(inner.x / 0.010).floor().clamp(1., 3.) as usize);
                for j in 0..count {
                    let xx = x + inner.x * ((j as f32 + 0.5) / count as f32 - 0.5) * 0.66;
                    let z = inner.y * r.random_range(-0.25..0.22);
                    let h = s.y * r.random_range(0.84..0.97);
                    let radius = (inner.x / count as f32 * 0.12).min(0.0035);
                    a.part(
                        if j % 2 == 0 {
                            Surface::Wood
                        } else {
                            Surface::Ink
                        },
                        label,
                    )
                    .rod(Vec3::new(xx, t, z), Vec3::new(xx, h, z), radius);
                }
            }
        }
        cursor += if trays { height } else { width };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn printed_content_rotates_with_its_supported_page() {
        let mut rotated = 0;
        for seed in 0..32 {
            let o = crate::scene::procedural_indoor::objects::audit::fixture(
                ObjectKind::DocumentTray,
                seed,
            );
            let a = crate::scene::procedural_indoor::objects::build_object(&o);
            for ((surface, _), g) in &a.parts {
                if *surface != Surface::PrintedPaper {
                    continue;
                }
                assert!(g
                    .uvs
                    .iter()
                    .flatten()
                    .all(|&u| (-0.00001..=1.00001).contains(&u)));
                for tri in g.indices.as_chunks::<3>().0 {
                    if tri.iter().any(|&i| g.normals[i as usize][1] < 0.99) {
                        continue;
                    }
                    let p = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
                    let uv = tri.map(|i| Vec2::from_array(g.uvs[i as usize]));
                    let du = uv[1] - uv[0];
                    let dv = uv[2] - uv[0];
                    let determinant = du.perp_dot(dv);
                    if determinant.abs() < 1e-8 {
                        continue;
                    }
                    let tangent =
                        (((p[1] - p[0]) * dv.y - (p[2] - p[0]) * du.y) / determinant).normalize();
                    rotated += usize::from(tangent.z.abs() > 0.01);
                    break;
                }
            }
        }
        assert!(
            rotated > 20,
            "page content snapped back to room axes: {rotated}"
        );
    }
}
