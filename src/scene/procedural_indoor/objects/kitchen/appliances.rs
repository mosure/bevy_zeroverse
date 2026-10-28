//! Appliance cavities, doors, controls and supported interior contents.
use super::super::super::motion::{construction as motion, MotionKind};
use super::*;
pub mod cooktop;
pub mod interior;
pub mod notes;
pub mod program;
#[cfg(test)]
mod tests;
// The cooktop sheet is 0.07 * 0.24 of total height; its work plane is at 0.94.
const RANGE_BODY_TOP: f32 = 0.94 - 0.07 * 0.24;
#[allow(clippy::too_many_arguments)]
fn knobs(
    a: &mut Assembly,
    base: Vec3,
    step: Vec3,
    count: u32,
    radius: f32,
    label: &str,
    seed: u64,
) {
    let mut r = stream(seed, 0x4b4e4f4250414e45);
    let span = r.random_range(0.76..1.0);
    for i in 0..count {
        let radius = radius * r.random_range(0.72..1.08);
        let p = base
            + step
                * ((count - 1) as f32 * 0.5
                    + (i as f32 - (count - 1) as f32 * 0.5) * span
                    + r.random_range(-0.06..0.06));
        a.part(Surface::Metal, label).cylinder(
            radius,
            0.016,
            Transform::from_translation(p).with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
        );
        let angle = r.random_range(-2.7..2.7);
        let q = Quat::from_rotation_z(angle);
        a.part(Surface::Ink, label).cuboid(
            Vec3::new(radius * 0.12, radius * 0.65, 0.001),
            0.0001,
            Transform::from_translation(p + q * Vec3::Y * radius * 0.42 + Vec3::Z * 0.009)
                .with_rotation(q),
        );
    }
}
#[allow(clippy::too_many_arguments)]
fn oven_leaf(
    a: &mut Assembly,
    width: f32,
    height: f32,
    bottom: Vec2,
    front: f32,
    p: &KitchenProgram,
    label: &str,
    seed: u64,
    o: &IndoorObject,
) {
    let rotation = Quat::from_rotation_x(p.opening_radians);
    let tf = Transform::from_translation(
        Vec3::new(bottom.x, bottom.y, front + 0.014) + rotation * Vec3::Y * height * 0.5,
    )
    .with_rotation(rotation);
    let joint = || {
        let mut joint = motion::rigid(
            MotionKind::Appliance,
            seed,
            Vec3::new(bottom.x, bottom.y, front + 0.014),
            Vec3::X,
            motion::endpoint(seed, p.opening_radians, 1.35) - p.opening_radians,
            Vec3::ZERO,
        );
        joint.frame = tf;
        joint
    };
    motion::hinged(a, MotionKind::Appliance, seed, joint, |a| {
        cabinet::panel(
            a,
            Vec2::new(width, height),
            tf,
            program::finishes(o).1,
            label,
            p.viewport,
        );
        cabinet::handle(a, tf, 0., height * 0.36, p, true, label);
    });
}
fn side_leaf(
    a: &mut Assembly,
    center: Vec3,
    width: f32,
    height: f32,
    side: f32,
    p: &KitchenProgram,
    o: &IndoorObject,
) {
    let label = o.kind.class_name();
    let seed = o.seed ^ ((center.x.to_bits() as u64) << 32) ^ center.y.to_bits() as u64;
    let rotation = Quat::from_rotation_y(side * p.opening_radians);
    let tf = Transform::from_translation(
        center + Vec3::X * side * width * 0.5 + rotation * Vec3::X * (-side * width * 0.5),
    )
    .with_rotation(rotation);
    let joint = || {
        let mut joint = motion::rigid(
            MotionKind::Appliance,
            seed,
            center + Vec3::X * side * width * 0.5,
            Vec3::Y,
            side * (motion::endpoint(seed, p.opening_radians, 1.35) - p.opening_radians),
            Vec3::ZERO,
        );
        joint.frame = tf;
        joint
    };
    motion::hinged(a, MotionKind::Appliance, seed, joint, |a| {
        cabinet::panel(
            a,
            Vec2::new(width - 0.004, height),
            tf,
            program::finishes(o).1,
            label,
            p.viewport,
        );
        cabinet::handle(a, tf, -side * (width * 0.5 - 0.045), 0., p, false, label);

        if o.kind == ObjectKind::Refrigerator {
            let placements = notes::placements(seed, width, height, side, p);
            notes::build(a, &placements, tf, p);
        }
    });
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &KitchenProgram) {
    if o.kind == ObjectKind::Cooktop {
        cooktop::build(a, o.size, Vec3::ZERO, p, o.seed);
        return;
    }
    let s = o.size;
    let (d, z) = body(o, p);
    let w = s.x - 0.018;
    let front = z + d * 0.5;
    let label = o.kind.class_name();
    let product = program::parameters(o);
    let toe = product.toe_m;
    a.box_part(
        Surface::Rubber,
        label,
        Vec3::new(0., toe * 0.5, z),
        Vec3::new(w * 0.89, toe, d * 0.89),
        0.004,
    );
    cabinet::shell(
        a,
        Vec3::new(
            w,
            if o.kind == ObjectKind::Range {
                s.y * RANGE_BODY_TOP - toe
            } else {
                s.y - toe - 0.006
            },
            d,
        ),
        toe,
        z,
        product.shell,
        label,
    );
    // A distinct inner lining makes open doors reveal a cavity rather than a dark decal.
    a.box_part(
        if o.kind == ObjectKind::Refrigerator {
            Surface::Plastic
        } else {
            Surface::Metal
        },
        label,
        Vec3::new(0., s.y * 0.51, z - d * 0.5 + 0.025),
        Vec3::new(w * 0.91, s.y * 0.88, 0.008),
        0.002,
    );
    match o.kind {
        ObjectKind::Refrigerator => {
            for (i, &y) in product.shelf_levels_m.iter().enumerate() {
                let top = product
                    .shelf_levels_m
                    .get(i + 1)
                    .copied()
                    .unwrap_or(product.shelf_top_m);
                a.box_part(
                    Surface::ContainerGlass,
                    label,
                    Vec3::new(0., y, z),
                    Vec3::new(w * 0.92, 0.012, d * 0.88),
                    0.001,
                );
                if p.contents {
                    let items = contents::shelf_contents(
                        o,
                        i as u32,
                        (
                            Vec2::new(-w * 0.45, z - d * 0.42),
                            Vec2::new(w * 0.45, z + d * 0.42),
                        ),
                        y + 0.007,
                        top - y - 0.020,
                    );
                    contents::build(a, &items);
                }
            }
            for door in refrigerator_doors(o, p) {
                side_leaf(a, door.center, door.width, door.height, door.side, p, o);
            }
        }
        ObjectKind::Microwave => {
            let lower = p.controls == 2;
            let sign = if p.controls == 1 { -1. } else { 1. };
            let fraction = product.control_fraction;
            let door_w = program::microwave_door_width(o, p);
            let cx = if lower {
                0.
            } else {
                -sign * w * fraction * 0.5
            };
            side_leaf(
                a,
                Vec3::new(cx, s.y * if lower { 0.64 } else { 0.52 }, front + 0.014),
                door_w,
                s.y * if lower { 0.60 } else { 0.82 },
                -sign,
                p,
                o,
            );
            let center = if lower {
                Vec3::new(0., s.y * 0.17, front + 0.016)
            } else {
                Vec3::new(sign * w * (1. - fraction) * 0.5, s.y * 0.51, front + 0.016)
            };
            let panel_size = if lower {
                Vec3::new(w * 0.96, s.y * 0.21, 0.025)
            } else {
                Vec3::new(w * fraction - 0.006, s.y * 0.86, 0.025)
            };
            a.box_part(Surface::Plastic, label, center, panel_size, 0.003);
            if lower {
                knobs(
                    a,
                    center + Vec3::new(-w * 0.29, 0., 0.021),
                    Vec3::X * w * 0.18,
                    3,
                    0.012,
                    label,
                    o.seed,
                );
            } else {
                knobs(
                    a,
                    Vec3::new(sign * w * (1. - fraction) * 0.5, s.y * 0.24, front + 0.037),
                    Vec3::Y * s.y * 0.19,
                    2,
                    0.016,
                    label,
                    o.seed,
                );
            }
            a.box_part(
                Surface::Screen,
                label,
                if lower {
                    center + Vec3::new(w * 0.30, 0., 0.014)
                } else {
                    Vec3::new(sign * w * (1. - fraction) * 0.5, s.y * 0.77, front + 0.030)
                },
                Vec3::new(w * 0.12, s.y * 0.11, 0.002),
                0.001,
            );
            let tray_y = if lower {
                s.y * 0.34 + 0.01
            } else {
                toe + 0.025
            };
            if lower {
                a.box_part(
                    Surface::Metal,
                    label,
                    Vec3::new(0., tray_y - 0.013, z),
                    Vec3::new(w * 0.94, 0.018, d * 0.92),
                    0.001,
                );
            }
            a.part(Surface::ContainerGlass, "dishware").cylinder(
                (w * 0.30).min(d * 0.38),
                0.007,
                Transform::from_xyz(cx, tray_y, z),
            );
            if p.contents {
                let mut bowl = dishware::parameters(
                    Vec3::new(w * 0.30, s.y * 0.48, d * 0.38),
                    1,
                    1,
                    o.seed ^ 0x4d4943524f444953,
                );
                // Microwave contents use a microwave-safe substrate.
                bowl.surface = Surface::Ceramic;
                dishware::build(
                    a,
                    Transform::from_xyz(cx, tray_y + 0.0035, z),
                    &bowl,
                    o.seed,
                );
            }
        }
        ObjectKind::Dishwasher => {
            let bottom = toe + 0.024;
            // Fit below the control fascia even on short compact appliances.
            // A fixed fraction plus the toe height overlapped the fascia.
            let height = program::dishwasher_door_height(o);
            let fascia_bottom = s.y * (1. - product.fascia_fraction);
            let fascia_top = s.y - 0.01;
            let fascia_y = (fascia_bottom + fascia_top) * 0.5;
            oven_leaf(
                a,
                w * 0.96,
                height,
                Vec2::new(0., bottom),
                front,
                p,
                label,
                o.seed,
                o,
            );
            a.box_part(
                product.front,
                label,
                Vec3::new(0., fascia_y, front + 0.014),
                Vec3::new(w * 0.97, fascia_top - fascia_bottom, 0.022),
                0.003,
            );
            knobs(
                a,
                Vec3::new(-w * 0.30, fascia_y, front + 0.03),
                Vec3::X * w * 0.18,
                4,
                0.012,
                label,
                o.seed,
            );
            for basket in interior::dishwasher_baskets(o, p) {
                let mut rack = supplies::rack::parameters(
                    basket.size,
                    if p.contents && !basket.cutlery { 6 } else { 0 },
                    basket.seed,
                );
                // Stainless or coated-steel structure belongs to the appliance;
                // plates retain dishware semantics.
                rack.finish = Surface::Metal;
                rack.hardware = Surface::Metal;
                supplies::rack::build_with_label(
                    a,
                    Transform::from_translation(basket.position),
                    &rack,
                    basket.seed,
                    label,
                );
                for side in [-1., 1.] {
                    let pos = basket.position + Vec3::new(side * rack.size.x * 0.43, -0.006, 0.);
                    a.box_part(
                        Surface::Metal,
                        label,
                        pos,
                        Vec3::new(0.018, 0.012, rack.size.z * 0.96),
                        0.002,
                    );
                    let wall = side * (w * 0.5 - 0.018);
                    for depth_side in [-1., 1.] {
                        let end = pos + Vec3::Z * depth_side * rack.size.z * 0.38;
                        a.part(Surface::Metal, label)
                            .rod(end, end.with_x(wall), 0.004);
                    }
                }
            }
        }
        ObjectKind::Oven | ObjectKind::Range => {
            let tall = o.kind == ObjectKind::Oven && s.y > 1.3;
            let fronts = program::oven_fronts(o, p);
            let mut apertures = Vec::new();
            for (level, door) in fronts.iter().enumerate() {
                let bottom = door.bottom_m;
                let dh = door.height_m;
                let width = door.width_m;
                let offset = door.center_x;
                let mut q = p.clone();
                if level > 0 {
                    q.opening_radians = 0.;
                }
                apertures.push([
                    Vec2::new(offset - width * 0.5 - 0.003, bottom - 0.014),
                    Vec2::new(offset + width * 0.5 + 0.003, bottom + dh + 0.003),
                ]);
                oven_leaf(
                    a,
                    width,
                    dh,
                    Vec2::new(offset, bottom),
                    front,
                    &q,
                    label,
                    o.seed ^ level as u64,
                    o,
                );
                for y in [bottom + 0.008, bottom + dh - 0.008] {
                    a.box_part(
                        Surface::Metal,
                        label,
                        Vec3::new(offset, y, z),
                        Vec3::new(width, 0.016, d * 0.86),
                        0.002,
                    );
                }
                for side in [-1., 1.] {
                    a.box_part(
                        Surface::Metal,
                        label,
                        Vec3::new(offset + side * (width * 0.5 - 0.005), bottom + dh * 0.5, z),
                        Vec3::new(0.01, dh - 0.032, d * 0.86),
                        0.002,
                    );
                }
                for rack in interior::oven_racks(o, p, door, level) {
                    interior::build_oven_rack(a, &rack, Vec2::new(offset, z), label);
                    // Support ledges connect each rack to the actual cavity sides.
                    for side in [-1., 1.] {
                        let x = offset + side * (width * 0.5 - 0.010);
                        a.box_part(
                            Surface::Metal,
                            label,
                            Vec3::new(x - side * width * 0.038, rack.y_m - rack.wire_m - 0.004, z),
                            Vec3::new(width * 0.08, 0.008, rack.depth_m),
                            0.001,
                        );
                    }
                }
            }
            cabinet::aperture_front(
                a,
                if tall { p.finish } else { product.front },
                label,
                [
                    Vec2::new(-w * 0.485, toe),
                    Vec2::new(
                        w * 0.485,
                        s.y * if o.kind == ObjectKind::Range {
                            RANGE_BODY_TOP
                        } else {
                            0.99
                        },
                    ),
                ],
                &apertures,
                front + 0.014,
            );
            let y = if tall { s.y * 0.90 } else { s.y * 0.86 };
            if p.controls != 0 {
                let x = if p.controls == 1 { -w * 0.42 } else { w * 0.42 };
                let cy = if tall { s.y * 0.64 } else { s.y * 0.48 };
                knobs(
                    a,
                    Vec3::new(x, cy - s.y * 0.15, front + 0.031),
                    Vec3::Y * s.y * 0.14,
                    3,
                    0.015,
                    label,
                    o.seed,
                );
            } else {
                knobs(
                    a,
                    Vec3::new(-w * 0.30, y, front + 0.032),
                    Vec3::X * w * 0.20,
                    4,
                    0.016,
                    label,
                    o.seed,
                );
            }
            if o.kind == ObjectKind::Range {
                cooktop::build(
                    a,
                    Vec3::new(w, s.y * 0.07, d),
                    Vec3::new(0., s.y * RANGE_BODY_TOP, z),
                    p,
                    o.seed,
                );
            }
        }
        _ => unreachable!(),
    }
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct ApplianceDoor {
    pub center: Vec3,
    pub width: f32,
    pub height: f32,
    pub side: f32,
}
pub fn refrigerator_doors(o: &IndoorObject, p: &KitchenProgram) -> Vec<ApplianceDoor> {
    let split = program::proportions(o.seed).0;
    let toe = 0.075;
    let (d, z) = body(o, p);
    let total = o.size.x - 0.036;
    let widths = if p.paired {
        vec![total * split, total * (1. - split)]
    } else {
        vec![total]
    };
    let mut x = -total * 0.5;
    widths
        .into_iter()
        .enumerate()
        .map(|(i, width)| {
            let cx = x + width * 0.5;
            x += width;
            ApplianceDoor {
                center: Vec3::new(cx, (o.size.y + toe) * 0.5, z + d * 0.5 + 0.014),
                width,
                height: o.size.y - toe - 0.02,
                side: if i == 0 { -1. } else { 1. },
            }
        })
        .collect()
}
pub fn metadata(o: &IndoorObject, p: &KitchenProgram) -> serde_json::Value {
    let plan = program::parameters(o);
    let mut value = serde_json::json!({"shell":plan.shell,"front":plan.front});
    match o.kind {
        ObjectKind::Refrigerator => {
            let doors = refrigerator_doors(o, p);
            let notes: Vec<_> = doors
                .iter()
                .map(|d| {
                    let seed = o.seed
                        ^ ((d.center.x.to_bits() as u64) << 32)
                        ^ d.center.y.to_bits() as u64;
                    notes::placements(seed, d.width, d.height, d.side, p)
                })
                .collect();
            value["shelf_levels_m"] = serde_json::json!(plan.shelf_levels_m);
            value["doors"] = serde_json::json!(doors);
            value["notes"] = serde_json::json!(notes);
        }
        ObjectKind::Dishwasher => {
            value["fascia_fraction"] = serde_json::json!(plan.fascia_fraction);
            value["baskets"] = serde_json::json!(interior::dishwasher_baskets(o, p));
        }
        ObjectKind::Microwave => {
            value["control_fraction"] = serde_json::json!(plan.control_fraction);
        }
        ObjectKind::Cooktop | ObjectKind::Range => {
            let size = if o.kind == ObjectKind::Range {
                Vec3::new(o.size.x - 0.018, o.size.y * 0.07, body(o, p).0)
            } else {
                o.size
            };
            value["cooktop"] = serde_json::json!(cooktop::parameters(o.seed, size, p));
        }
        _ => {}
    }
    if matches!(o.kind, ObjectKind::Oven | ObjectKind::Range) {
        let fronts = program::oven_fronts(o, p);
        value["racks"] = serde_json::json!(fronts
            .iter()
            .enumerate()
            .map(|(i, door)| interior::oven_racks(o, p, door, i))
            .collect::<Vec<_>>());
        value["fronts"] = serde_json::json!(fronts);
    }
    value
}

/// Fixed chassis and closed doors isolate construction from size and opening.
/// An unexpanded occupied envelope cannot contain arbitrary open-door travel.
#[cfg(test)]
pub(crate) fn review_fixture(
    kind: ObjectKind,
    seed: u64,
) -> (IndoorObject, Assembly, KitchenProgram) {
    let mut o = super::super::audit::fixture(kind, seed);
    o.size = fixture_size(kind);
    let mut p = parameters(&o);
    p.opening_radians = 0.;
    let mut a = Assembly::default();
    build(&mut a, &o, &p);
    (o, a, p)
}
