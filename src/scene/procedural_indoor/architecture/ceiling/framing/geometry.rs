use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;

fn face(g: &mut Geometry, points: [Vec3; 4], outside: Vec3) {
    let mut p = points;
    if (p[1] - p[0]).cross(p[2] - p[0]).dot(outside) < 0. {
        p.reverse();
    }
    let normal = (p[1] - p[0]).cross(p[2] - p[0]).normalize();
    let u = (p[1] - p[0]).normalize();
    let v = normal.cross(u);
    let base = g.positions.len() as u32;
    for point in p {
        g.positions.push(point.to_array());
        g.normals.push(normal.to_array());
        g.uvs.push([point.dot(u), point.dot(v)]);
    }
    g.indices
        .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn cell(
    g: &mut Geometry,
    m: &FrameMember,
    upper: [f32; 2],
    lower: [f32; 2],
    top: f32,
    bottom: f32,
) {
    let a = [
        m.point(0., upper[0], top),
        m.point(1., upper[0], top),
        m.point(1., upper[1], top),
        m.point(0., upper[1], top),
    ];
    let b = [
        m.point(0., lower[0], bottom),
        m.point(1., lower[0], bottom),
        m.point(1., lower[1], bottom),
        m.point(0., lower[1], bottom),
    ];
    face(g, a, Vec3::Y);
    face(g, b, -Vec3::Y);
    let center = (m.top[0] + m.top[2]) * 0.5;
    for i in 0..4 {
        let j = (i + 1) % 4;
        face(
            g,
            [a[i], a[j], b[j], b[i]],
            ((a[i] + a[j]) * 0.5 - center).with_y(0.),
        );
    }
}

pub(super) fn build(plan: &CeilingFraming) -> Assembly {
    let mut a = Assembly::default();
    for m in &plan.members {
        let label = format!("ceiling#finish{}", m.finish);
        let g = a.part(
            m.material,
            if super::super::super::super::materials::variants::supports(m.material) {
                &label
            } else {
                "ceiling"
            },
        );
        match m.section {
            FrameSection::Solid => cell(g, m, [0., 1.], [0., 1.], 0., m.depth),
            FrameSection::Fluted => cell(g, m, [0., 1.], [0.22, 0.78], 0., m.depth),
            FrameSection::ISection => {
                let flange = (m.depth * 0.09).clamp(0.008, 0.025).min(m.depth * 0.24);
                cell(g, m, [0., 1.], [0., 1.], 0., flange);
                cell(g, m, [0.43, 0.57], [0.43, 0.57], flange, m.depth - flange);
                cell(g, m, [0., 1.], [0., 1.], m.depth - flange, m.depth);
            }
            FrameSection::OpenWeb => {
                let width = m.top[0].distance(m.top[3]);
                let radius = (width * 0.12).min(m.depth * 0.075);
                let count = (m.top[0].distance(m.top[1]) / m.web_pitch)
                    .ceil()
                    .clamp(2., 24.) as usize;
                for side in [0.22, 0.78] {
                    for height in [radius, m.depth - radius] {
                        g.tube(
                            &[m.point(0., side, height), m.point(1., side, height)],
                            radius,
                            8,
                        );
                    }
                    for i in 0..count {
                        let lo = i as f32 / count as f32;
                        let hi = (i + 1) as f32 / count as f32;
                        let (d0, d1) = if i % 2 == 0 {
                            (radius, m.depth - radius)
                        } else {
                            (m.depth - radius, radius)
                        };
                        g.tube(
                            &[m.point(lo, side, d0), m.point(hi, side, d1)],
                            radius * 0.6,
                            6,
                        );
                    }
                    for t in [0., 1.] {
                        g.tube(
                            &[m.point(t, side, radius), m.point(t, side, m.depth - radius)],
                            radius * 0.65,
                            6,
                        );
                    }
                }
                for i in 0..=count {
                    let t = i as f32 / count as f32;
                    g.tube(
                        &[m.point(t, 0.22, radius), m.point(t, 0.78, radius)],
                        radius * 0.55,
                        6,
                    );
                }
            }
        }
    }
    a
}

// Only open-web hits need triangles: a solid box would anchor hangers in air
// between chords. This bounded query is paid at attachment construction, not
// per render frame, and only after its narrow member footprint matched.
pub(super) fn attachment(member: &FrameMember, p: Vec3) -> Option<f32> {
    let a = build(&CeilingFraming {
        roof_material: member.material,
        members: vec![member.clone()],
    });
    a.parts
        .values()
        .flat_map(|g| {
            g.indices.as_chunks::<3>().0.iter().filter_map(|indices| {
                let [a, b, c] = indices.map(|i| Vec3::from(g.positions[i as usize]).as_dvec3());
                let origin = p.as_dvec3();
                let direction = bevy::math::DVec3::Y;
                let ab = b - a;
                let ac = c - a;
                let h = direction.cross(ac);
                let det = ab.dot(h);
                if det.abs() < 1e-12 {
                    return None;
                }
                let s = origin - a;
                let u = s.dot(h) / det;
                let q = s.cross(ab);
                let v = direction.dot(q) / det;
                let t = ac.dot(q) / det;
                (u >= -1e-8 && v >= -1e-8 && u + v <= 1.00000001 && t > 0.003)
                    .then_some(p.y + t as f32)
            })
        })
        .min_by(f32::total_cmp)
}
