//! A pinned sheet has one printed face. Ink, image and bare paper partition
//! it, with the same physical curl, including at oblique viewing angles.
use super::*;

struct Mark {
    bounds: Rect,
    surface: Surface,
}

pub(in super::super) fn build(
    a: &mut Assembly,
    o: &IndoorObject,
    note: &Pinup,
    face: f32,
    version: u32,
) {
    let mut r = stream(note.seed, 5731);
    let size = note.size;
    let has_image = r.random_bool(0.40);
    let left = r.random_range(-0.44..-0.24);
    let right = r.random_range(0.26..0.44);
    let top = r.random_range(0.22..0.40);
    let lines = r.random_range(2..=9);
    let pitch = r.random_range(0.38..0.65) / lines as f32;
    let mut stock = stream(note.seed, u64::from_le_bytes(*b"NoteForm"));
    let thick = stock
        .random_range(0.00015_f32..0.00042)
        .min(o.size.z * 0.025);
    let looseness = stock.random::<f32>();
    let lift = (size.y * 0.032 * looseness * looseness)
        .min(0.004)
        .min(o.size.z * 0.08);
    let bend = size.y * stock.random_range(-0.36..-0.13);
    let ink_height = (size.y * stock.random_range(0.004..0.014)).min(pitch * size.y * 0.35);
    let mut marks = Vec::with_capacity(lines + 1);
    if has_image {
        marks.push(Mark {
            bounds: Rect::from_corners(
                Vec2::new(left, -0.35) * size,
                Vec2::new(right, 0.12) * size,
            ),
            surface: Surface::Art,
        });
    }
    for line in 0..if has_image { 2 } else { lines } {
        let length = size.x * r.random_range((right - left) * 0.45..right - left);
        let y = size.y * (top - line as f32 * pitch);
        if !has_image || y - ink_height * 0.5 > 0.12 * size.y {
            marks.push(Mark {
                bounds: Rect::from_corners(
                    Vec2::new(left * size.x, y - ink_height * 0.5),
                    Vec2::new(left * size.x + length, y + ink_height * 0.5),
                ),
                surface: Surface::Ink,
            });
        }
    }

    let label = o.kind.class_name();
    let art = format!(
        "{label}#finish{}",
        super::super::super::super::materials::variants::object_slot(Surface::Art, note.seed, None)
    );
    let tf = Transform::from_xyz(note.center.x, note.center.y, face)
        .with_rotation(Quat::from_rotation_z(note.yaw));
    let half = size * 0.5;
    let height = |y: f32| lift * ((bend - y) / (bend + half.y)).clamp(0., 1.);
    let point = |p: Vec2, front: bool| {
        tf.transform_point(p.extend(height(p.y) + if front { thick } else { 0. }))
    };
    let patch = |a: &mut Assembly, x0, x1, y0, y1, mark: Option<&Mark>| {
        if x0 >= x1 {
            return;
        }
        let corners = [
            Vec2::new(x0, y0),
            Vec2::new(x1, y0),
            Vec2::new(x1, y1),
            Vec2::new(x0, y1),
        ];
        let world = corners.map(|v| point(v, true));
        let surface = mark.map_or(Surface::Paper, |m| m.surface);
        let g = a.part(surface, if surface == Surface::Art { &art } else { label });
        let normal =
            tf.rotation * Vec3::new(0., (height(y0) - height(y1)) / (y1 - y0), 1.).normalize();
        let uvs = corners.map(|p| {
            if let Some(m) = mark.filter(|m| m.surface == Surface::Art) {
                let uv = (p - m.bounds.min) / m.bounds.size();
                [uv.x, 1. - uv.y]
            } else {
                p.to_array()
            }
        });
        g.vertex_quad(world, [normal; 4], uvs);
    };
    let mut rows = vec![-half.y, bend, half.y];
    for m in &marks {
        rows.extend([m.bounds.min.y, m.bounds.max.y]);
    }
    rows.sort_unstable_by(f32::total_cmp);
    rows.dedup();
    for row in rows.windows(2) {
        let (y0, y1) = (row[0], row[1]);
        let mid = (y0 + y1) * 0.5;
        if let Some(m) = marks
            .iter()
            .find(|m| mid >= m.bounds.min.y && mid < m.bounds.max.y)
        {
            patch(a, -half.x, m.bounds.min.x, y0, y1, None);
            patch(a, m.bounds.min.x, m.bounds.max.x, y0, y1, Some(m));
            patch(a, m.bounds.max.x, half.x, y0, y1, None);
        } else {
            patch(a, -half.x, half.x, y0, y1, None);
        }
    }
    // Only the lifted part has an exposed reverse side. Cut the lateral edges
    // at the bend so the edge, face and reverse meet on the same fold.
    let g = a.part(Surface::Paper, label);
    let rear = [
        Vec2::new(-half.x, -half.y),
        Vec2::new(-half.x, bend),
        Vec2::new(half.x, bend),
        Vec2::new(half.x, -half.y),
    ]
    .map(|p| point(p, false));
    g.metric_quad(rear, (rear[1] - rear[0]).cross(rear[2] - rear[0]));
    let rim = [
        -half,
        Vec2::new(half.x, -half.y),
        Vec2::new(half.x, bend),
        half,
        Vec2::new(-half.x, half.y),
        Vec2::new(-half.x, bend),
    ];
    for (i, &u) in rim.iter().enumerate() {
        let v = rim[(i + 1) % rim.len()];
        g.metric_quad(
            [
                point(u, true),
                point(u, false),
                point(v, false),
                point(v, true),
            ],
            tf.rotation * (-(v - u).perp()).extend(0.),
        );
    }
    if version >= super::super::board_section::TACKBOARD_VERSION {
        for pin in anchors(note) {
            pushpin(a, o, note.seed, point(pin, true));
        }
        return;
    }
    let pin = Vec2::new(size.x * r.random_range(-0.22..0.22), size.y * 0.41);
    let pin_depth = o.size.z * 0.012;
    a.part(Surface::Chrome, label).ellipsoid(
        Vec3::new(0.002, 0.002, pin_depth),
        Transform::from_translation(point(pin, true) + Vec3::Z * pin_depth * 0.55),
    );
}

fn anchors(note: &Pinup) -> impl Iterator<Item = Vec2> {
    let mut r = stream(note.seed, u64::from_le_bytes(*b"PinHoles"));
    let s = note.size;
    let paired = s.x >= 0.14
        && r.random_bool(f64::from(
            (0.08 + 1.85 * (s.x - 0.14) + 0.7 * (s.y - 0.16).max(0.)).clamp(0., 0.86),
        ));
    let first_x = if paired {
        -s.x * 0.5 + r.random_range(0.009..0.026)
    } else {
        s.x * r.random_range(-0.20..0.20)
    };
    let top = |inset: f32| s.y * 0.5 - inset.min(s.y * 0.24);
    let first = Vec2::new(first_x, top(r.random_range(0.007..0.023)));
    let second = paired.then(|| {
        Vec2::new(
            s.x * 0.5 - r.random_range(0.009..0.026),
            top(r.random_range(0.007..0.023)),
        )
    });
    std::iter::once(first).chain(second)
}

/// The pin enters the flat, attached portion of the sheet. Only the exposed
/// head is meshed. Very shallow installations constrain its height.
fn pushpin(a: &mut Assembly, o: &IndoorObject, seed: u64, contact: Vec3) {
    let mut r = stream(seed, u64::from_le_bytes(*b"PinHeads"));
    let domed = r.random_bool(0.46);
    let radius = r.random_range(0.0031..0.0057);
    let room = o.size.z * 0.5 - contact.z;
    let height = (if domed {
        r.random_range(0.0013_f32..0.0029)
    } else {
        r.random_range(0.005..0.011)
    })
    .min(room * 0.96);
    let (surface, profile) = if domed {
        (
            Surface::Chrome,
            [
                (radius, -0.00010),
                (radius, height * 0.21),
                (radius * 0.81, height * 0.64),
                (radius * 0.45, height * 0.93),
                (0., height),
            ],
        )
    } else {
        (
            Surface::Plastic,
            [
                (radius, -0.00010),
                (radius * 0.96, height * 0.15),
                (radius * 0.51, height * 0.66),
                (radius * 0.80, height * 0.93),
                (0., height),
            ],
        )
    };
    a.part(surface, o.kind.class_name()).lathe(
        &profile,
        10,
        Transform::from_translation(contact)
            .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{
        architecture::overlap_tests::intersection_area, objects::audit::fixture,
    };

    #[test]
    fn both_pushpin_heads_fit_inside_the_sheet_and_touch_its_flat_upper_face() {
        let mut paired = 0;
        for seed in 0..96 {
            for size in [Vec2::new(0.07, 0.038), Vec2::new(0.297, 0.42)] {
                let note = Pinup {
                    center: Vec2::ZERO,
                    size,
                    yaw: 0.,
                    seed,
                };
                let points: Vec<_> = anchors(&note).collect();
                if points.len() == 2 {
                    paired += 1;
                    assert!(size.x > 0.14 && points[1].x - points[0].x > size.x * 0.5);
                }
                for p in points {
                    assert!((size * 0.5 - p.abs()).min_element() > 0.0057 && p.y > 0.);
                }
            }
        }
        assert!((16..85).contains(&paired));
    }

    #[test]
    fn ink_and_photos_share_one_continuous_piece_of_paper() {
        let o = fixture(ObjectKind::NoticeBoard, 0);
        let mut photos = 0;
        let mut plain = 0;
        for seed in 0..32 {
            let note = Pinup {
                center: Vec2::new(0.35, 0.42),
                size: Vec2::new(0.18, 0.14),
                yaw: 0.47,
                seed,
            };
            let mut a = Assembly::default();
            build(
                &mut a,
                &o,
                &note,
                0.012,
                bevy_zeroverse_capture::GENERATOR_VERSION,
            );
            let mut faces = Vec::new();
            let mut area = 0.;
            let mut photo = false;
            let mut front = 0f32;
            let mut back = f32::MAX;
            for ((s, _), g) in &a.parts {
                if !matches!(s, Surface::Paper | Surface::Art | Surface::Ink) {
                    continue;
                }
                for t in g.indices.as_chunks::<3>().0 {
                    let p = t.map(|i| Vec3::from(g.positions[i as usize]));
                    let cross = (p[1] - p[0]).cross(p[2] - p[0]);
                    if cross.z <= 1e-10 {
                        continue;
                    }
                    let q = p.map(|p| p.xy());
                    for &prev in &faces {
                        assert!(intersection_area(q, prev) < 2e-8, "overprint seed {seed}");
                    }
                    area += cross.z * 0.5;
                    for v in p {
                        front = front.max(v.z);
                        back = back.min(v.z);
                    }
                    if *s == Surface::Art {
                        photo = true;
                        for i in t {
                            assert!(g.uvs[*i as usize]
                                .iter()
                                .all(|v| (-1e-5..=1.00001).contains(v)));
                        }
                    }
                    assert!(cross.normalize().dot(Vec3::from(g.normals[t[0] as usize])) > 0.999);
                    faces.push(q);
                }
            }
            assert!((area - note.size.element_product()).abs() < 2e-7);
            assert!(back > 0.012 && front <= 0.012 + 0.0045);
            if photo {
                photos += 1;
            } else {
                plain += 1;
            }
        }
        assert!(photos >= 4 && plain >= 4);
    }
}
