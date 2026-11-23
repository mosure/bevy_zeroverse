//! A machined blank with cut-through lights and blind finger pockets.
//! All surfaces and texture coordinates are constructed in the moving leaf frame.
use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;

fn corners(r: Rect, z: f32) -> [Vec3; 4] {
    [
        r.min,
        Vec2::new(r.max.x, r.min.y),
        r.max,
        Vec2::new(r.min.x, r.max.y),
    ]
    .map(|v| v.extend(z))
}

fn inset(r: Rect, width: f32) -> Rect {
    let offset = Vec2::splat(width.min(r.size().min_element() * 0.22));
    Rect::from_corners(r.min + offset, r.max - offset)
}

fn quad(g: &mut Geometry, tf: Transform, local: [Vec3; 4], outward: Vec3) {
    let base = g.positions.len();
    g.metric_quad(local, outward);
    for i in base..g.positions.len() {
        let p = Vec3::from_array(g.positions[i]);
        let n = Vec3::from_array(g.normals[i]);
        // Keep a single veneer/paint chart across the flat pieces of the blank.
        if n.z.abs() > 0.9999 {
            g.uvs[i] = [p.x * n.z.signum(), p.y];
        }
        g.positions[i] = tf.transform_point(p).to_array();
        g.normals[i] = (tf.rotation * n).to_array();
    }
}

fn face(g: &mut Geometry, tf: Transform, r: Rect, z: f32, side: f32) {
    quad(g, tf, corners(r, z), side * Vec3::Z);
}

/// Two rectangular sections can form a machined rebate, a glazing stop, or a
/// reveal. 'inside' selects the visible cavity on a vertical span.
fn ring(
    g: &mut Geometry,
    tf: Transform,
    from: (Rect, f32),
    to: (Rect, f32),
    side: f32,
    inside: bool,
) {
    let a = corners(from.0, from.1);
    let b = corners(to.0, to.1);
    for (i, edge) in [Vec3::NEG_Y, Vec3::X, Vec3::Y, Vec3::NEG_X]
        .into_iter()
        .enumerate()
    {
        let next = (i + 1) % 4;
        let n = if from.0 == to.0 {
            if inside {
                -edge
            } else {
                edge
            }
        } else {
            side * Vec3::Z
        };
        quad(g, tf, [a[i], a[next], b[next], b[i]], n);
    }
}

fn blank_face(
    g: &mut Geometry,
    tf: Transform,
    outer: Rect,
    mut holes: [Option<Rect>; 2],
    z: f32,
    side: f32,
) {
    // The lock pocket lies within the latch stile, beside every panel opening.
    holes.sort_by(|a, b| {
        a.map_or(f32::INFINITY, |r| r.min.x)
            .total_cmp(&b.map_or(f32::INFINITY, |r| r.min.x))
    });
    let mut strip = |x0, x1, y0, y1| {
        if x1 > x0 && y1 > y0 {
            face(
                g,
                tf,
                Rect::from_corners(Vec2::new(x0, y0), Vec2::new(x1, y1)),
                z,
                side,
            );
        }
    };
    let mut x = outer.min.x;
    for hole in holes.into_iter().flatten() {
        strip(x, hole.min.x, outer.min.y, outer.max.y);
        strip(hole.min.x, hole.max.x, outer.min.y, hole.min.y);
        strip(hole.min.x, hole.max.x, hole.max.y, outer.max.y);
        x = hole.max.x;
    }
    strip(x, outer.max.x, outer.min.y, outer.max.y);
}

pub(super) fn build(
    a: &mut Assembly,
    p: &DoorProgram,
    leaf: Leaf,
    fit: &Fit,
    j: &Joinery,
    index: usize,
    labels: &Labels,
) {
    let tf = leaf.transform;
    let half = leaf.size.z * 0.5;
    let outer = Rect::from_center_size(Vec2::ZERO, leaf.size.truncate());
    let bevel = 0.0014;
    let face_rect = inset(outer, bevel);
    let g = a.part(p.finish, &labels.leaf);
    ring(
        g,
        tf,
        (outer, -half + bevel),
        (outer, half - bevel),
        1.,
        false,
    );
    for side in [-1., 1.] {
        let z = side * half;
        ring(
            g,
            tf,
            (outer, z - side * bevel),
            (face_rect, z),
            side,
            false,
        );
        let cup = (Fit::handle(p, index, side) == DoorHandle::Recessed).then_some(fit.pocket);
        blank_face(g, tf, face_rect, [fit.opening, cup], z, side);
    }
    let Some(hole) = fit.opening else { return };
    match p.panel {
        DoorPanel::Glazed | DoorPanel::VisionLite => {
            let gauge = j.glass.metres();
            let seal = inset(hole, j.bead);
            let clear = inset(seal, j.seal);
            for side in [-1., 1.] {
                let glass_z = j.inset + side * gauge * 0.5;
                let bead_z = glass_z + side * 0.0025;
                ring(
                    a.part(p.finish, &labels.leaf),
                    tf,
                    (hole, side * half),
                    (seal, bead_z),
                    side,
                    true,
                );
                ring(
                    a.part(Surface::Rubber, &labels.seal),
                    tf,
                    (seal, bead_z),
                    (clear, glass_z),
                    side,
                    true,
                );
            }
            ring(
                a.part(Surface::Rubber, &labels.seal),
                tf,
                (clear, j.inset - gauge * 0.5),
                (clear, j.inset + gauge * 0.5),
                1.,
                true,
            );
            // Slip the pane edge into the seal, as on a dry-glazed leaf. Through-
            // glass annotation still sees the opaque reveal from oblique angles.
            let glass_size = clear.size() + Vec2::splat(0.002);
            a.part(Surface::GlassInterior, &labels.glass).cuboid(
                glass_size.extend(gauge),
                0.,
                tf.mul_transform(Transform::from_translation(clear.center().extend(j.inset))),
            );
        }
        DoorPanel::Recessed => {
            // Cross rails and the differently sized floating fields meet along a
            // shared boundary. A closed solid door cannot leak sky between bays.
            for side in [-1., 1.] {
                let mut bottom = hole.min.y;
                for i in 0..j.bays {
                    let t = (i + 1) as f32 / j.bays as f32;
                    let joint = hole.min.y + hole.height() * (t + j.balance * t * (1. - t));
                    let top = if i + 1 == j.bays {
                        hole.max.y
                    } else {
                        joint - fit.stile * 0.5
                    };
                    let field = Rect::from_corners(
                        Vec2::new(hole.min.x, bottom),
                        Vec2::new(hole.max.x, top),
                    );
                    let inner = inset(field, j.shoulder);
                    let z = side * (half - j.recess);
                    ring(g, tf, (field, side * half), (inner, z), side, true);
                    face(g, tf, inner, z, side);
                    if i + 1 < j.bays {
                        bottom = joint + fit.stile * 0.5;
                        face(
                            g,
                            tf,
                            Rect::from_corners(
                                Vec2::new(hole.min.x, top),
                                Vec2::new(hole.max.x, bottom),
                            ),
                            side * half,
                            side,
                        );
                    }
                }
            }
        }
        DoorPanel::Louvered => {
            ring(g, tf, (hole, -half), (hole, half), 1., true);
            let count = (hole.height() / j.pitch).round().max(1.) as usize;
            let pitch = hole.height() / count as f32;
            let (sin, cos) = j.slope.sin_cos();
            let height = (pitch * 0.83).min((leaf.size.z * 0.94 - j.gauge * cos) / sin);
            let rotation = Quat::from_rotation_x(j.slope);
            let mut cuts = p
                .finish
                .is_cut_timber()
                .then(|| TimberCuts::new(Joinery::stock_seed(p), 1 + index as u64));
            for i in 0..count {
                let center = Vec3::new(hole.center().x, hole.min.y + (i as f32 + 0.5) * pitch, 0.);
                blade(
                    g,
                    tf.mul_transform(Transform::from_translation(center).with_rotation(rotation)),
                    hole.width(),
                    height,
                    j.gauge,
                    cuts.as_mut().map_or(Vec2::ZERO, TimberCuts::origin),
                );
            }
        }
        DoorPanel::Flush => unreachable!(),
    }
}

/// Exposed, bevelled leading and trailing edges. Both ends meet the stiles,
/// which close them; internal blade end caps would never be visible.
fn blade(g: &mut Geometry, tf: Transform, width: f32, height: f32, gauge: f32, origin: Vec2) {
    let y = height * 0.5;
    let z = gauge * 0.5;
    let b = z * 0.57;
    let profile = [
        Vec2::new(-y + b, -z),
        Vec2::new(y - b, -z),
        Vec2::new(y, -z + b),
        Vec2::new(y, z - b),
        Vec2::new(y - b, z),
        Vec2::new(-y + b, z),
        Vec2::new(-y, z - b),
        Vec2::new(-y, -z + b),
    ];
    // The grain follows the length of the timber blade. Unroll the section
    // across the grain so the bevels continue into both broad faces.
    let mut distance = 0.;
    for (i, &start) in profile.iter().enumerate() {
        let end = profile[(i + 1) % profile.len()];
        let delta = end - start;
        let next = distance + delta.length();
        let n = (tf.rotation * Vec3::new(0., delta.y, -delta.x).normalize()).to_array();
        let half = width * 0.5;
        let base = g.positions.len() as u32;
        for (x, yz, u) in [
            (-half, end, next),
            (half, end, next),
            (half, start, distance),
            (-half, start, distance),
        ] {
            g.positions
                .push(tf.transform_point(Vec3::new(x, yz.x, yz.y)).to_array());
            g.normals.push(n);
            g.uvs.push((origin + Vec2::new(u, -x)).to_array());
        }
        g.indices
            .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        distance = next;
    }
}

pub(super) fn finger_cup(g: &mut Geometry, leaf: Leaf, fit: &Fit, side: f32) {
    let tf = leaf.transform;
    let outer = fit.pocket;
    let mouth = inset(outer, 0.0032);
    let bottom = inset(mouth, 0.0028);
    let face_z = side * leaf.size.z * 0.5;
    let lip_z = face_z + side * 0.0022;
    let bottom_z = face_z - side * (leaf.size.z * 0.32).min(0.014);
    ring(g, tf, (outer, face_z), (mouth, lip_z), side, true);
    ring(g, tf, (mouth, lip_z), (bottom, bottom_z), side, true);
    face(g, tf, bottom, bottom_z, side);
}
