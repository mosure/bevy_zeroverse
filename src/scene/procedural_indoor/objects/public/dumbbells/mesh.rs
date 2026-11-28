use super::*;
use crate::scene::procedural_indoor::{
    geometry::{ProfileJoints, RevolvedOutline},
    materials::strength::{end_marks, Finish},
};

fn stock(a: &mut Assembly, finish: Finish) -> &mut Geometry {
    let (surface, label) = finish.key();
    a.part(surface, label)
}

fn block(a: &mut Assembly, finish: Finish, center: Vec3, size: Vec3, bevel: f32) {
    stock(a, finish).cuboid(size, bevel, Transform::from_translation(center));
}

impl FreeWeightRack {
    pub(super) fn frame(&self, a: &mut Assembly, size: Vec3) {
        let foot_y = 0.040;
        let last = &self.tiers[self.tiers.len() - 1];
        let top_y = last.support_y - 0.047;
        let back = -size.z * 0.41;
        let top_z = if self.cantilever { back } else { last.center_z };
        for sign in [-1., 1.] {
            let x = sign * (size.x * 0.5 - 0.034);
            let toe = 0.055;
            block(
                a,
                Finish::Coat,
                Vec3::new(x, 0.025, 0.),
                Vec3::new(0.055, 0.032, size.z - 2. * toe),
                0.002,
            );
            for end in [-1., 1.] {
                block(
                    a,
                    Finish::Rubber,
                    Vec3::new(x, foot_y * 0.5, end * (size.z - toe) * 0.5),
                    Vec3::new(0.068, foot_y, toe),
                    0.004,
                );
            }
            if self.cantilever {
                block(
                    a,
                    Finish::Coat,
                    Vec3::new(x, 0.5 * (top_y + foot_y), back),
                    Vec3::new(0.047, top_y - foot_y, 0.057),
                    0.003,
                );
            } else {
                for base in [back, -back] {
                    stock(a, Finish::Coat).rod(
                        Vec3::new(x, foot_y - 0.004, base),
                        Vec3::new(x, top_y, top_z),
                        0.022,
                    );
                }
            }
            for tier in &self.tiers {
                let y = tier.support_y - 0.051;
                let t = ((y - foot_y) / (top_y - foot_y)).clamp(0., 1.);
                let post_z = back + (top_z - back) * t;
                let rear = post_z.min(tier.center_z - Self::tray_depth(self.half_grip));
                let front = tier.center_z + Self::tray_depth(self.half_grip);
                block(
                    a,
                    Finish::Coat,
                    Vec3::new(x, y, (rear + front) * 0.5),
                    Vec3::new(0.050, 0.051, front - rear),
                    0.003,
                );
            }
        }
        for tier in &self.tiers {
            self.shelf(a, tier, size.x - 0.068);
        }
    }

    fn shelf(&self, a: &mut Assembly, tier: &Tier, span: f32) {
        let y = tier.support_y;
        let z = tier.center_z;
        if self.shelf == Shelf::LinedTray {
            let half = Self::tray_depth(self.half_grip);
            block(
                a,
                Finish::Coat,
                Vec3::new(0., y - 0.017, z),
                Vec3::new(span, 0.024, 2. * half),
                0.002,
            );
            block(
                a,
                Finish::Rubber,
                Vec3::new(0., y - 0.0025, z),
                Vec3::new(span - 0.006, 0.005, 2. * half - 0.014),
                0.001,
            );
            for (edge, h) in [(-half, 0.027), (half, 0.012)] {
                block(
                    a,
                    Finish::Coat,
                    Vec3::new(0., y - 0.006 + h * 0.5, z + edge - edge.signum() * 0.003),
                    Vec3::new(span, h, 0.006),
                    0.001,
                );
            }
        } else {
            for side in [-1., 1.] {
                let rail = z + side * self.rail_offset();
                let gap = if self.shelf == Shelf::Saddles {
                    0.008
                } else {
                    0.004
                };
                block(
                    a,
                    Finish::Coat,
                    Vec3::new(0., y - gap - 0.017, rail),
                    Vec3::new(span, 0.034, 0.042),
                    0.002,
                );
                if self.shelf == Shelf::OpenRails {
                    block(
                        a,
                        Finish::Rubber,
                        Vec3::new(0., y - gap * 0.5, rail),
                        Vec3::new(span - 0.004, gap, 0.040),
                        0.0008,
                    );
                } else {
                    let g = stock(a, Finish::Rubber);
                    for pair in &tier.pairs {
                        for slot in pair.slots {
                            saddle(
                                g,
                                Vec3::new(slot.x, y, rail),
                                pair.radius * 0.80,
                                self.saddle_slope,
                            );
                        }
                    }
                }
            }
        }
    }

    pub(super) fn weights(
        &self,
        a: &mut Assembly,
        outline: &RevolvedOutline,
        tier: &Tier,
        pair: &WeightPair,
    ) {
        if !pair.slots.iter().any(|s| s.present) {
            return;
        }
        let radius = pair.radius;
        let w = pair.head_width;
        let g = self.half_grip;
        let body = match self.head_finish {
            HeadFinish::CastIron => Finish::Iron,
            _ => Finish::Rubber,
        };
        let cap = match self.head_finish {
            HeadFinish::ColourCodedEnds => {
                [Finish::Green, Finish::Yellow, Finish::Blue, Finish::Red]
                    [usize::from(pair.nominal_kg / 2 - 1) % 4]
            }
            _ => body,
        };
        let profile = [
            (0., 0.),
            (radius * 0.82, 0.),
            (radius, w * 0.16),
            (radius, w * 0.81),
            (radius * 0.86, w),
            (radius * 0.65, w),
            (radius * 0.62, w - 0.0016),
            (0., w - 0.0016),
        ];
        // One head blank supplies both ends of both matching weights. The
        // coloured insert occupies a recessed face with its own material.
        let mut head = Geometry::default();
        let mut insert = Geometry::default();
        let mut mark = Geometry::default();
        if body == cap {
            let plain = [
                profile[0],
                profile[1],
                profile[2],
                profile[3],
                profile[4],
                (0., w),
            ];
            outline.sweep(
                &mut head,
                &plain,
                ProfileJoints::Creased,
                Transform::IDENTITY,
            );
        } else {
            outline.sweep_range(
                &mut head,
                &profile,
                0..3,
                ProfileJoints::Creased,
                Transform::IDENTITY,
            );
            outline.sweep_range(
                &mut insert,
                &profile,
                3..6,
                ProfileJoints::Creased,
                Transform::IDENTITY,
            );
            outline.sweep_range(
                &mut mark,
                &profile,
                6..7,
                ProfileJoints::Creased,
                Transform::IDENTITY,
            );
            // In the display frame local +X is right and local +Z points down.
            let basis = Vec2::from_angle(self.head.phase());
            for (uv, p) in mark.uvs.iter_mut().zip(&mark.positions) {
                *uv = end_marks::uv(
                    pair.nominal_kg,
                    basis.rotate(Vec2::new(p[0], p[2]) / (0.62 * radius)),
                );
            }
        }
        let grip = pair.grip_radius;
        for slot in pair.slots.iter().filter(|s| s.present) {
            let center = Vec3::new(
                slot.x,
                tier.support_y + radius * self.contact,
                tier.center_z + slot.depth_offset,
            );
            let yaw = Quat::from_rotation_y(slot.yaw);
            let axis = yaw * Quat::from_rotation_x(FRAC_PI_2);
            stock(a, Finish::Grip).lathe(
                &[
                    (0.86 * grip, -g - 0.001),
                    (grip, -g * 0.66),
                    (grip, g * 0.66),
                    (0.86 * grip, g + 0.001),
                ],
                12,
                Transform::from_translation(center).with_rotation(axis),
            );
            let sides = self.head.sides() as f32;
            let clocking = (slot.face_turn * sides).floor() * TAU / sides;
            let phase = Quat::from_rotation_y(-self.head.phase() - clocking);
            for side in [-1., 1.] {
                let origin = center + yaw * Vec3::Z * (g * side);
                let facing = if side < 0. {
                    yaw * Quat::from_rotation_y(PI)
                } else {
                    yaw
                };
                let turn = facing * Quat::from_rotation_x(FRAC_PI_2) * phase;
                stock(a, body).append_rigid(&head, turn, origin);
                if body != cap {
                    stock(a, cap).append_rigid(&insert, turn, origin);
                    a.part(end_marks::KEY.0, end_marks::KEY.1)
                        .append_rigid(&mark, turn, origin);
                }
            }
        }
    }
}

/// A single moulded V under each head. The two halves share the valley edge.
fn saddle(g: &mut Geometry, valley: Vec3, half: f32, slope: f32) {
    let p = |x: f32, y: f32, z: f32| valley + Vec3::new(x, y, z);
    let d = 0.018;
    let base = -0.008;
    let rise = half * slope;
    g.metric_quad(
        [
            p(-half, base, -d),
            p(half, base, -d),
            p(half, base, d),
            p(-half, base, d),
        ],
        Vec3::NEG_Y,
    );
    for sign in [-1., 1.] {
        let x = sign * half;
        g.metric_quad(
            [p(0., 0., -d), p(x, rise, -d), p(x, rise, d), p(0., 0., d)],
            Vec3::Y,
        );
        g.metric_quad(
            [p(x, base, -d), p(x, rise, -d), p(x, rise, d), p(x, base, d)],
            Vec3::X * sign,
        );
        for end in [-1., 1.] {
            let z = d * end;
            g.metric_quad(
                [p(0., base, z), p(x, base, z), p(x, rise, z), p(0., 0., z)],
                Vec3::Z * end,
            );
        }
    }
}

pub(in super::super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let rack = FreeWeightRack::for_object(o);
    rack.frame(a, o.size);
    let outline = if rack.head == Head::Round {
        RevolvedOutline::round(rack.head.sides(), 0., 0)
    } else {
        RevolvedOutline::polygon(rack.head.sides())
    };
    for tier in &rack.tiers {
        for pair in &tier.pairs {
            rack.weights(a, &outline, tier, pair);
        }
    }
}
