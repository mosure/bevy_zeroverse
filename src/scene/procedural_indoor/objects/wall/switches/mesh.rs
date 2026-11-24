//! The opening, its pocket and its moving face occupy distinct surfaces.
//! All dimensions and switch poses are local to the installed plate.
use super::*;
use crate::scene::procedural_indoor::objects::electrical::{edge, pierced, Rect};

pub(super) fn build(b: &SwitchBank, a: &mut Assembly, class: &str) {
    let plate = b.label(b.finish, class);
    let cap = b.label(Surface::Plastic, class);
    let openings: [Rect; 3] = std::array::from_fn(|i| Rect::new(b.center(i), b.module_size));
    let openings = &openings[..b.modules.len()];
    cover(a.part(b.finish, &plate), b, openings);
    for (i, (&control, &cut)) in b.modules.iter().zip(openings).enumerate() {
        let c = b.center(i);
        let s = b.module_size - 2. * b.gap_m;
        let base = b.face_z_m - b.gap_m * 0.42;
        let insert = Rect::new(c, s);
        let dark = a.part(Surface::Ink, class);
        edge(dark, cut, b.face_z_m, cut, b.pocket_z_m, true);
        cut.face(dark, b.pocket_z_m);
        match control {
            Actuator::Paddle { angle_rad } => {
                rocker(a.part(Surface::Plastic, &cap), b, c, s, angle_rad);
            }
            Actuator::Split { angles_rad } => {
                let width = (s.x - b.gap_m) * 0.5;
                for (j, angle) in angles_rad.into_iter().enumerate() {
                    let x = (j as f32 - 0.5) * (width + b.gap_m);
                    rocker(
                        a.part(Surface::Plastic, &cap),
                        b,
                        c + Vec2::X * x,
                        Vec2::new(width, s.y),
                        angle,
                    );
                }
            }
            Actuator::Toggle { angle_rad } => {
                let hole = Rect::new(c, s * Vec2::new(0.32, 0.36));
                carrier(a, &cap, class, insert, &[hole], base, b.pocket_z_m);
                let radius = (s.x * 0.093).min(0.0048);
                // The tilted basal ring must fit above the wall contact even on a
                // thin faceplate. The length below is measured from this pivot.
                let root = (base - (base - b.pocket_z_m) * 0.64)
                    .max(b.rear_z_m + radius * 0.76 * angle_rad.sin().abs() + 0.00004);
                // The pivot sits in the well, including when viewed under a thrown lever.
                a.part(Surface::Plastic, &cap).lathe(
                    &[
                        (radius * 0.91, b.pocket_z_m - 0.00012 - root),
                        (radius * 0.95, 0.),
                        (radius * 0.70, radius * 0.40),
                        (0., radius * 0.63),
                    ],
                    12,
                    Transform::from_translation(c.extend(root))
                        .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                );
                let length = ((b.front_limit() - root - radius * angle_rad.sin().abs())
                    / angle_rad.cos())
                .min(0.019);
                let axis = Quat::from_rotation_x(FRAC_PI_2 - angle_rad);
                a.part(Surface::Plastic, &cap).lathe(
                    &[
                        (radius * 0.76, 0.),
                        (radius * 0.83, length * 0.31),
                        (radius, length * 0.80),
                        (radius * 0.86, length * 0.95),
                        (0., length),
                    ],
                    12,
                    Transform::from_translation(c.extend(root)).with_rotation(axis),
                );
            }
            Actuator::Rotary { angle_rad } => {
                carrier(a, &cap, class, insert, &[], base, b.pocket_z_m);
                let radius = s.min_element() * 0.417;
                let height = (b.front_limit() - base - 0.0007).min(radius * 0.86);
                let axis = Transform::from_translation(c.extend(base))
                    .with_rotation(Quat::from_rotation_x(FRAC_PI_2));
                a.part(Surface::Plastic, &cap).lathe(
                    &[
                        (radius * 0.69, -0.00020),
                        (radius * 0.73, height * 0.14),
                        (radius, height * 0.22),
                        (radius * 0.96, height * 0.87),
                        (radius * 0.84, height),
                        (0., height),
                    ],
                    24,
                    axis,
                );
                // A raised index is seated in the cap. Its azimuth is the actual dimmer setting.
                let rot = Quat::from_rotation_z(angle_rad);
                a.part(Surface::Ink, class).cuboid(
                    Vec3::new((radius * 0.10).min(0.0016), radius * 0.41, 0.00048),
                    0.00012,
                    Transform::from_translation(
                        c.extend(base + height + 0.00010) + rot * Vec3::Y * radius * 0.57,
                    )
                    .with_rotation(rot),
                );
            }
            Actuator::Slider { level } => {
                let track = s * Vec2::new(0.16, 0.79);
                carrier(
                    a,
                    &cap,
                    class,
                    insert,
                    &[Rect::new(c, track)],
                    base,
                    b.pocket_z_m,
                );
                let thumb = s * Vec2::new(0.62, 0.159);
                let y = c.y + (level - 0.5) * (track.y - thumb.y);
                let xy = Vec2::new(c.x, y);
                let top = base + (b.front_limit() - base).min(0.0058);
                let underside = base + (top - base) * 0.19;
                let depth = underside - b.pocket_z_m;
                a.box_part(
                    Surface::Plastic,
                    &cap,
                    xy.extend(b.pocket_z_m + depth * 0.5),
                    Vec3::new(track.x * 0.74, thumb.y * 0.64, depth + 0.00018),
                    0.00025,
                );
                a.box_part(
                    Surface::Plastic,
                    &cap,
                    xy.extend((underside + top) * 0.5),
                    thumb.extend(top - underside),
                    0.0008,
                );
            }
            Actuator::Buttons { latched } => {
                let size = s * Vec2::new(0.79, 0.355);
                let holes = [-0.227, 0.227].map(|v| Rect::new(c + Vec2::Y * s.y * v, size));
                carrier(a, &cap, class, insert, &holes, base, b.pocket_z_m);
                for (j, hole) in holes.into_iter().enumerate() {
                    let center = (hole.lo + hole.hi) * 0.5;
                    let top = if latched == Some(j) {
                        base - b.gap_m * 0.61
                    } else {
                        base + (b.front_limit() - base).min(0.0028)
                    };
                    paddle_mesh(
                        a.part(Surface::Plastic, &cap),
                        Rect::new(center, size - b.gap_m * 1.5),
                        b.pocket_z_m,
                        top,
                        0.,
                        b.gap_m * 0.52,
                    );
                }
            }
        }
    }
    if b.screws {
        for side in [-1., 1.] {
            let margin = (b.case_size.y - b.module_size.y) * 0.5;
            let c = Vec2::new(
                0.,
                b.case_size.y * 0.5 + side * (b.module_size.y + margin) * 0.5,
            );
            let radius = (margin * 0.20).min(0.00255);
            a.part(Surface::Metal, class).lathe(
                &[
                    (radius, -0.00012),
                    (radius, 0.00023),
                    (radius * 0.65, 0.00054),
                    (0., 0.00058),
                ],
                12,
                Transform::from_translation(c.extend(b.face_z_m))
                    .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
            );
        }
    }
}

fn carrier(
    a: &mut Assembly,
    cap: &str,
    class: &str,
    insert: Rect,
    holes: &[Rect],
    front: f32,
    floor: f32,
) {
    let g = a.part(Surface::Plastic, cap);
    edge(g, insert, floor, insert, front, false);
    pierced(g, insert, front, holes);
    for &hole in holes {
        edge(a.part(Surface::Ink, class), hole, front, hole, floor, true);
    }
}

fn rocker(g: &mut Geometry, b: &SwitchBank, c: Vec2, size: Vec2, angle: f32) {
    let slope = angle.tan();
    let middle = b.rocker_low() + slope.abs() * size.y * 0.5;
    paddle_mesh(
        g,
        Rect::new(c, size),
        b.pocket_z_m,
        middle,
        slope,
        b.gap_m * 0.71,
    );
}

/// Pivoted paddle, with a beveled rim. The skirt enters the well on both
/// ends; the inclined face cannot occlude the plate outside its opening.
fn paddle_mesh(g: &mut Geometry, body: Rect, floor: f32, mid: f32, slope: f32, bevel: f32) {
    let cy = (body.lo.y + body.hi.y) * 0.5;
    let inner = Rect {
        lo: body.lo + bevel,
        hi: body.hi - bevel,
    };
    let heights = |v: Vec2| mid + (v.y - cy) * slope;
    let ring = body.corners();
    let top = inner.corners();
    for i in 0..4 {
        let j = (i + 1) % 4;
        let p = ring[i];
        let q = ring[j];
        let side = Vec3::new(q.y - p.y, p.x - q.x, 0.);
        let a = p.extend(heights(p) - bevel);
        let b = q.extend(heights(q) - bevel);
        g.metric_quad([p.extend(floor), q.extend(floor), b, a], side);
        g.metric_quad(
            [
                a,
                b,
                top[j].extend(heights(top[j])),
                top[i].extend(heights(top[i])),
            ],
            Vec3::Z,
        );
    }
    g.metric_quad(top.map(|p| p.extend(heights(p))), Vec3::Z);
}

/// Three chord segments per corner. The central band contains every opening;
/// cap bands meet it exactly, including when the faceplate is seen edge-on.
fn cover(g: &mut Geometry, b: &SwitchBank, holes: &[Rect]) {
    let center = Vec2::Y * (b.case_size.y * 0.5);
    let rear = b.rear_z_m;
    let face = b.face_z_m;
    let bevel = b.edge_m.min(b.corner_m * 0.62);
    let rows = |inset: f32| {
        let h = b.case_size.xy() * 0.5 - inset;
        let r = b.corner_m - inset;
        let offsets = [(1., 0.), (0.5, 0.133_974_6), (0.133_974_6, 0.5), (0., 1.)];
        std::array::from_fn::<_, 8, _>(|i| {
            let (dx, dy) = offsets[if i < 4 { i } else { 7 - i }];
            let sign = if i < 4 { -1. } else { 1. };
            (h.x - r * dx, center.y + sign * (h.y - r * dy))
        })
    };
    let outer = rows(0.);
    let inner = rows(bevel);
    let strip = |(w0, y0): (f32, f32), (w1, y1): (f32, f32)| {
        [
            Vec2::new(-w0, y0),
            Vec2::new(w0, y0),
            Vec2::new(w1, y1),
            Vec2::new(-w1, y1),
        ]
    };
    for i in 0..7 {
        let back = strip(outer[i], outer[i + 1]);
        g.metric_quad(back.map(|p| p.extend(rear)), Vec3::NEG_Z);
        let front = strip(inner[i], inner[i + 1]);
        if i == 3 {
            pierced(
                g,
                Rect {
                    lo: front[0],
                    hi: front[2],
                },
                face,
                holes,
            );
        } else {
            g.metric_quad(front.map(|p| p.extend(face)), Vec3::Z);
        }
        for j in 0..4 {
            if (j == 0 && i != 0) || (j == 2 && i != 6) {
                continue;
            }
            let k = (j + 1) % 4;
            let v = back[k] - back[j];
            let n = Vec3::new(v.y, -v.x, 0.);
            let shoulder = face - bevel;
            g.metric_quad(
                [
                    back[j].extend(rear),
                    back[k].extend(rear),
                    back[k].extend(shoulder),
                    back[j].extend(shoulder),
                ],
                n,
            );
            g.metric_quad(
                [
                    back[j].extend(shoulder),
                    back[k].extend(shoulder),
                    front[k].extend(face),
                    front[j].extend(face),
                ],
                n,
            );
        }
    }
}
