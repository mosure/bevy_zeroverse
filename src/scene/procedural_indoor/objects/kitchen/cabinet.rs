//! Panel construction with visible interiors; glazed leaves contain real apertures.
use super::*;

pub(super) fn shell(
    a: &mut Assembly,
    size: Vec3,
    bottom: f32,
    z: f32,
    finish: Surface,
    label: &str,
) {
    open_shell(a, size, bottom, z, finish, label);
    let t = 0.018_f32.min(size.min_element() * 0.08);
    a.box_part(
        finish,
        label,
        Vec3::new(0., bottom + size.y - t * 0.5, z),
        Vec3::new(size.x - 2. * t, t, size.z),
        0.002,
    );
}

/// Sink bases must leave the basin aperture open all the way into the cabinet.
pub(super) fn open_shell(
    a: &mut Assembly,
    size: Vec3,
    bottom: f32,
    z: f32,
    finish: Surface,
    label: &str,
) {
    let t = 0.018_f32.min(size.min_element() * 0.08);
    let h = size.y;
    for x in [-1., 1.] {
        a.box_part(
            finish,
            label,
            Vec3::new(x * (size.x - t) * 0.5, bottom + h * 0.5, z),
            Vec3::new(t, h, size.z),
            0.003,
        );
    }
    a.box_part(
        finish,
        label,
        Vec3::new(0., bottom + t * 0.5, z),
        Vec3::new(size.x - 2. * t, t, size.z),
        0.002,
    );
    a.box_part(
        finish,
        label,
        Vec3::new(0., bottom + h * 0.5, z - size.z * 0.5 + t * 0.5),
        Vec3::new(size.x - 2. * t, h - 2. * t, t),
        0.001,
    );
}

pub(super) fn panel(
    a: &mut Assembly,
    size: Vec2,
    tf: Transform,
    finish: Surface,
    label: &str,
    glass: bool,
) {
    let t = 0.022;
    if glass {
        let edge = (size.min_element() * 0.12).clamp(0.018, 0.045);
        for x in [-1., 1.] {
            box_at(
                a,
                finish,
                label,
                Vec3::new(x * (size.x - edge) * 0.5, 0., 0.),
                Vec3::new(edge, size.y, t),
                tf,
            );
        }
        for y in [-1., 1.] {
            box_at(
                a,
                finish,
                label,
                Vec3::new(0., y * (size.y - edge) * 0.5, 0.),
                Vec3::new(size.x - 2. * edge, edge, t),
                tf,
            );
        }
        box_at(
            a,
            Surface::GlassInterior,
            label,
            Vec3::ZERO,
            Vec3::new(
                size.x - 2. * edge - 0.002,
                size.y - 2. * edge - 0.002,
                0.004,
            ),
            tf,
        );
    } else {
        box_at(
            a,
            finish,
            label,
            Vec3::ZERO,
            Vec3::new(size.x, size.y, t),
            tf,
        );
        // A raised centre field leaves a real reveal, without coincident faces.
        box_at(
            a,
            finish,
            label,
            Vec3::new(0., 0., 0.013),
            Vec3::new(size.x * 0.79, size.y * 0.83, 0.004),
            tf,
        );
    }
}

/// Fill a built-in appliance surround around door apertures. Disjoint cells
/// avoid the hidden solid facade and coincident overlaid panels of a box model.
pub(super) fn aperture_front(
    a: &mut Assembly,
    surface: Surface,
    label: &str,
    bounds: [Vec2; 2],
    apertures: &[[Vec2; 2]],
    z: f32,
) {
    let [lo, hi] = bounds;
    let mut xs = vec![lo.x, hi.x];
    let mut ys = vec![lo.y, hi.y];
    for [a, b] in apertures {
        xs.extend([a.x.clamp(lo.x, hi.x), b.x.clamp(lo.x, hi.x)]);
        ys.extend([a.y.clamp(lo.y, hi.y), b.y.clamp(lo.y, hi.y)]);
    }
    xs.sort_by(f32::total_cmp);
    xs.dedup();
    ys.sort_by(f32::total_cmp);
    ys.dedup();
    for x in xs.windows(2) {
        for y in ys.windows(2) {
            let c = Vec2::new((x[0] + x[1]) * 0.5, (y[0] + y[1]) * 0.5);
            if apertures
                .iter()
                .any(|[a, b]| c.cmpge(*a).all() && c.cmple(*b).all())
            {
                continue;
            }
            if x[1] - x[0] < 0.001 || y[1] - y[0] < 0.001 {
                continue;
            }
            a.box_part(
                surface,
                label,
                Vec3::new(c.x, c.y, z),
                Vec3::new(x[1] - x[0], y[1] - y[0], 0.022),
                0.0005,
            );
        }
    }
}
pub(super) fn handle(
    a: &mut Assembly,
    tf: Transform,
    x: f32,
    y: f32,
    p: &KitchenProgram,
    horizontal: bool,
    label: &str,
) {
    let style = p.handles;
    let label = if label == "cabinet" {
        "cabinet_hardware"
    } else {
        label
    };
    let s = p.hardware;
    if style == 4 {
        return;
    }
    if style == 3 {
        box_at(
            a,
            Surface::Metal,
            label,
            Vec3::new(x, y, 0.019),
            Vec3::new(
                if horizontal { 0.11 } else { 0.024 },
                if horizontal { 0.024 } else { 0.11 },
                0.012,
            ),
            tf,
        );
    } else if style == 1 {
        a.part(s, label).ellipsoid(
            Vec3::new(0.012, 0.012, 0.015),
            Transform::from_translation(tf.transform_point(Vec3::new(x, y, 0.033)))
                .with_rotation(tf.rotation),
        );
    } else {
        let delta =
            if horizontal { Vec3::X } else { Vec3::Y } * if style == 2 { 0.033 } else { 0.055 };
        let c = Vec3::new(x, y, 0.036);
        pipe(
            a,
            s,
            label,
            &[
                tf.transform_point(c - delta - Vec3::Z * 0.018),
                tf.transform_point(c - delta),
                tf.transform_point(c + delta),
                tf.transform_point(c + delta - Vec3::Z * 0.018),
            ],
            0.005,
        );
    }
}
pub(super) fn doors(
    a: &mut Assembly,
    width: f32,
    bottom: f32,
    height: f32,
    front: f32,
    p: &KitchenProgram,
    label: &str,
) {
    if p.door_style == 2 {
        return;
    }
    let sliding = p.cabinet_motion == 1;
    let lift = p.cabinet_motion == 2 && bottom > 1.0;
    let count = if sliding {
        2
    } else if lift {
        1
    } else if p.paired {
        2
    } else {
        1
    };
    let leaf = (width - 0.045) / count as f32;
    for i in 0..count {
        let side = if i == 0 { -1. } else { 1. };
        let tf = if sliding {
            let x = (i as f32 - 0.5) * leaf
                + if i == 0 {
                    leaf * p.opening_radians / 1.35
                } else {
                    0.
                };
            Transform::from_xyz(x, bottom + height * 0.5, front + 0.014 + i as f32 * 0.035)
        } else if lift {
            let q = Quat::from_rotation_x(-p.opening_radians);
            Transform::from_translation(
                Vec3::new(0., bottom + height - 0.022, front + 0.014)
                    + q * Vec3::NEG_Y * (height - 0.045) * 0.5,
            )
            .with_rotation(q)
        } else {
            let hinge = Vec3::new(
                side * (width * 0.5 - 0.022),
                bottom + height * 0.5,
                front + 0.014,
            );
            let q = Quat::from_rotation_y(side * p.opening_radians);
            Transform::from_translation(hinge + q * Vec3::new(-side * leaf * 0.5, 0., 0.))
                .with_rotation(q)
        };
        use super::super::super::motion::{construction as motion, MotionKind};
        let seed = (width.to_bits() as u64) ^ ((bottom.to_bits() as u64) << 32) ^ i as u64;
        let moving = a.motion_selected(MotionKind::Cabinet, seed);
        let mut part = Assembly::default();
        let dest = if moving { &mut part } else { &mut *a };
        front_panel(
            dest,
            Vec2::new(leaf - 0.005, height - if lift { 0.045 } else { 0.006 }),
            tf,
            p,
            label,
        );
        let mut hardware = Assembly::default();
        handle(
            &mut hardware,
            tf,
            if lift {
                0.
            } else {
                -side * (leaf * 0.5 - 0.055)
            },
            if lift { -height * 0.5 + 0.075 } else { 0. },
            p,
            lift,
            label,
        );
        for ((_, label), g) in hardware.parts {
            dest.part(p.hardware, &label).append(g);
        }
        // Visible hinge barrels only on hinged fronts; short cylinders have
        // finite thickness and stay attached when the leaf is open.
        if !sliding && !lift {
            for y in [-0.32, 0.32] {
                dest.part(p.hardware, "cabinet_hardware").cylinder(
                    0.005,
                    0.035,
                    Transform::from_translation(tf.transform_point(Vec3::new(
                        side * leaf * 0.5,
                        y * height,
                        -0.008,
                    )))
                    .with_rotation(tf.rotation),
                );
            }
        }
        if moving {
            let end = motion::endpoint(seed, p.opening_radians, 1.35);
            let delta = end - p.opening_radians;
            let (pivot, axis, angle, translation) = if sliding {
                (
                    Vec3::ZERO,
                    Vec3::Y,
                    0.,
                    Vec3::X * if i == 0 { leaf * delta / 1.35 } else { 0. },
                )
            } else if lift {
                (
                    Vec3::new(0., bottom + height - 0.022, front + 0.014),
                    Vec3::X,
                    -delta,
                    Vec3::ZERO,
                )
            } else {
                (
                    Vec3::new(
                        side * (width * 0.5 - 0.022),
                        bottom + height * 0.5,
                        front + 0.014,
                    ),
                    Vec3::Y,
                    side * delta,
                    Vec3::ZERO,
                )
            };
            if angle.abs() + translation.length() > 1e-5 {
                let mut joint =
                    motion::rigid(MotionKind::Cabinet, seed, pivot, axis, angle, translation);
                joint.frame = tf;
                a.articulated(part, joint);
            } else {
                a.append_assembly(part);
            }
        }
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &KitchenProgram) {
    let (d, z) = body(o, p);
    let w = o.size.x - 0.012;
    let h = o.size.y;
    let toe = (h * 0.08).min(0.11);
    a.box_part(
        Surface::Rubber,
        "cabinet",
        Vec3::new(0., toe * 0.5, z - 0.025),
        Vec3::new(w - 0.07, toe, d - 0.06),
        0.004,
    );
    if o.kind == ObjectKind::KitchenCounter {
        shell(
            a,
            Vec3::new(w, h - toe - 0.035, d),
            toe,
            z,
            p.finish,
            "cabinet",
        );
        a.box_part(
            p.top,
            "counter",
            Vec3::new(0., h - 0.017, z),
            Vec3::new(o.size.x, 0.034, d + 0.012),
            0.004,
        );
        let front = z + d * 0.5;
        if p.counter_doors {
            shelves(a, o, p, Vec3::new(w, h - toe - 0.04, d), toe, z);
            doors(a, w, toe, h - toe - 0.045, front, p, "cabinet");
            return;
        }
        let rows = p.shelves;
        for row in 0..rows {
            let rh = (h - toe - 0.06) / rows as f32;
            let tf = Transform::from_xyz(0., toe + rh * (row as f32 + 0.5), front + 0.014);
            front_panel(a, Vec2::new(w - 0.014, rh - 0.006), tf, p, "cabinet");
            handle(a, tf, 0., 0., p, true, "cabinet");
            if row + 1 == rows && matches!(p.handles, 0 | 2) && o.seed.is_multiple_of(3) {
                // A drying towel drapes over the upper pull, not a knob or a
                // toe-level drawer. Keep both fabric sides outside the front.
                let width = if p.handles == 2 { 0.058 } else { 0.095 };
                props::hanging_towel(a, tf.translation + Vec3::Z * 0.036, width, rh * 0.80);
            }
        }
        return;
    }
    if o.kind == ObjectKind::KitchenCupboard {
        // A fitted dresser: supported base run, backsplash, and mounted upper cabinet.
        let top = h * 0.425;
        shell(a, Vec3::new(w, top - toe, d), toe, z, p.finish, "cabinet");
        a.box_part(
            p.top,
            "counter",
            Vec3::new(0., top + 0.016, z),
            Vec3::new(w, 0.032, d),
            0.003,
        );
        doors(a, w, toe, top - toe - 0.02, z + d * 0.5, p, "cabinet");
        let upper = h * p.upper_bottom_fraction;
        let ud = d * p.upper_depth_fraction;
        let uz = z - d * 0.5 + ud * 0.5;
        a.box_part(
            Surface::Ceramic,
            "cabinet",
            Vec3::new(0., (upper + top) * 0.5, z - d * 0.5 + 0.012),
            Vec3::new(w, upper - top, 0.024),
            0.002,
        );
        let uw = w * p.upper_width_fraction;
        let shift = (w - uw) * p.upper_offset_fraction;
        let mut upper_assembly = Assembly::default();
        shell(
            &mut upper_assembly,
            Vec3::new(uw, h - upper, ud),
            upper,
            uz,
            p.finish,
            "cabinet",
        );
        shelves(
            &mut upper_assembly,
            o,
            p,
            Vec3::new(uw, h - upper, ud),
            upper,
            uz,
        );
        doors(
            &mut upper_assembly,
            uw,
            upper,
            h - upper,
            uz + ud * 0.5,
            p,
            "cabinet",
        );
        for ((surface, label), mut g) in upper_assembly.parts {
            for v in &mut g.positions {
                v[0] += shift;
            }
            a.part(surface, &label).append(g);
        }
    } else {
        shell(a, Vec3::new(w, h - toe, d), toe, z, p.finish, "cabinet");
        shelves(a, o, p, Vec3::new(w, h - toe, d), toe, z);
        doors(a, w, toe, h - toe, z + d * 0.5, p, "cabinet");
    }
}
fn shelves(
    a: &mut Assembly,
    o: &IndoorObject,
    p: &KitchenProgram,
    size: Vec3,
    bottom: f32,
    z: f32,
) {
    let Vec3 { x: w, y: h, z: d } = size;
    for level in 0..p.shelves {
        let y = bottom + 0.018 + level as f32 * (h - 0.05) / p.shelves as f32;
        if level > 0 {
            a.box_part(
                p.finish,
                "cabinet",
                Vec3::new(0., y - 0.008, z),
                Vec3::new(w - 0.04, 0.016, d - 0.025),
                0.001,
            );
        }
        if p.contents {
            let items = contents::shelf_contents(
                o,
                level,
                (
                    Vec2::new(-w * 0.5 + 0.025, z - d * 0.5 + 0.025),
                    Vec2::new(w * 0.5 - 0.025, z + d * 0.5 - 0.025),
                ),
                y,
                (h - 0.05) / p.shelves as f32 - 0.018,
            );
            contents::build(a, &items);
        }
    }
}

/// Cabinet fields occupy an actual aperture; raised/fluted fields never overlay
/// a coplanar full-size slab. Appliance glazing retains its appliance class.
fn front_panel(a: &mut Assembly, size: Vec2, tf: Transform, p: &KitchenProgram, label: &str) {
    if p.door_style == 1 {
        panel(a, size, tf, p.finish, label, true);
        return;
    }
    if p.front_profile == 0 {
        box_at(
            a,
            p.finish,
            label,
            Vec3::ZERO,
            Vec3::new(size.x, size.y, 0.022),
            tf,
        );
        return;
    }
    let edge = (size.min_element() * 0.13).clamp(0.018, 0.055);
    for side in [-1., 1.] {
        box_at(
            a,
            p.finish,
            label,
            Vec3::X * side * (size.x - edge) * 0.5,
            Vec3::new(edge, size.y, 0.022),
            tf,
        );
        box_at(
            a,
            p.finish,
            label,
            Vec3::Y * side * (size.y - edge) * 0.5,
            Vec3::new(size.x - 2. * edge, edge, 0.022),
            tf,
        );
    }
    let inner = size - Vec2::splat(2. * edge + 0.002);
    let t = if p.front_profile == 2 { 0.028 } else { 0.009 };
    box_at(
        a,
        p.finish,
        label,
        Vec3::ZERO,
        Vec3::new(inner.x, inner.y, t),
        tf,
    );
    if p.front_profile == 3 {
        let n = (inner.x / 0.028).floor().max(2.) as usize;
        for i in 0..n {
            box_at(
                a,
                p.finish,
                label,
                Vec3::new(
                    -inner.x * 0.5 + (i as f32 + 0.5) * inner.x / n as f32,
                    0.,
                    0.008,
                ),
                Vec3::new(inner.x / n as f32 * 0.55, inner.y, 0.006),
                tf,
            );
        }
    }
}
