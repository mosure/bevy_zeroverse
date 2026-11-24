//! Generator 227 and earlier wrapped one printed cuboid around each paper pile.
use super::*;
use crate::scene::procedural_indoor::{materials::variants, placement::rotated_half};

pub(super) fn paper(a: &mut Assembly, o: &IndoorObject, b: &Bay, r: &mut impl Rng) {
    let Bay {
        index: i,
        x,
        base,
        inner,
        height,
        t,
        ..
    } = *b;
    let yaw = r.random_range(-0.18_f32..0.18);
    let size = inner * Vec2::new(r.random_range(0.56..0.88), r.random_range(0.60..0.90));
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
}

pub(super) fn tools(a: &mut Assembly, o: &IndoorObject, b: &Bay, r: &mut impl Rng) {
    let Bay { inner, x, t, .. } = *b;
    let s = o.size;
    let label = o.kind.class_name();
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

pub(super) fn tray_frame(a: &mut Assembly, o: &IndoorObject, p: &OrganizerProgram, t: f32) {
    let label = o.kind.class_name();
    let s = o.size;
    let sum = p.weights.iter().sum::<f32>();
    let last = s.y * p.weights.last().unwrap() / sum;
    let frame_height = s.y - last + t + (last - t) * p.front_fraction;
    for xx in [-1., 1.] {
        for z in [-1., 1.] {
            a.box_part(
                p.finish,
                label,
                Vec3::new(
                    xx * (s.x * 0.5 - t * 1.6),
                    frame_height * 0.5,
                    z * (s.z * 0.5 - t * 1.6),
                ),
                Vec3::new(t, frame_height, t),
                t * 0.15,
            );
        }
    }
}
