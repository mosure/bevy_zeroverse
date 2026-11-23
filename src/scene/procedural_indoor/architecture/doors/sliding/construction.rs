use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;

// A single closed extrusion with an open slot. The two inner ledges carry
// the wheels; stitching the cut ends avoids overlapping faces between strips.
fn channel(g: &mut Geometry, track: &Track, span: Vec2, tf: Transform) {
    let section = |offset: f32| {
        let w = track.half_width - offset;
        let bottom = track.drop + offset;
        let top = track.top() - offset;
        [
            (-track.slot_half, bottom),
            (-w, bottom),
            (-w, top),
            (w, top),
            (w, bottom),
            (track.slot_half, bottom),
        ]
    };
    let outer = section(0.);
    let inner = section(track.gauge);
    let point = |x, (z, y)| tf.transform_point(Vec3::new(x, y, z));
    for (i, normal) in [Vec3::NEG_Y, Vec3::NEG_Z, Vec3::Y, Vec3::Z, Vec3::NEG_Y]
        .into_iter()
        .enumerate()
    {
        for (path, n) in [(&outer, normal), (&inner, -normal)] {
            g.metric_quad(
                [
                    point(span.x, path[i]),
                    point(span.y, path[i]),
                    point(span.y, path[i + 1]),
                    point(span.x, path[i + 1]),
                ],
                tf.rotation * n,
            );
        }
        for (x, n) in [(span.x, Vec3::NEG_X), (span.y, Vec3::X)] {
            g.metric_quad(
                [
                    point(x, outer[i]),
                    point(x, inner[i]),
                    point(x, inner[i + 1]),
                    point(x, outer[i + 1]),
                ],
                tf.rotation * n,
            );
        }
    }
    for (i, n) in [(0, Vec3::Z), (5, Vec3::NEG_Z)] {
        g.metric_quad(
            [
                point(span.x, outer[i]),
                point(span.y, outer[i]),
                point(span.y, inner[i]),
                point(span.x, inner[i]),
            ],
            tf.rotation * n,
        );
    }
}

impl Track {
    pub(in super::super) fn build_fixed(
        &self,
        a: &mut Assembly,
        d: &DoorInstallation,
        label: &str,
    ) {
        let p = &d.program;
        let span = self.span(d);
        let length = span.y - span.x;
        let middle = (span.x + span.y) * 0.5;
        let tf = d.transform();
        let head = Self::head(d);
        // Work in metric section coordinates: positive depth always points
        // out from the wall, even for a mirrored or rotated installation.
        let at = |x, y, z| tf.mul_transform(Transform::from_xyz(x, head + y, p.swing_side * z));
        let metal = a.part(p.hardware_finish, label);
        for lane in lanes(d) {
            channel(metal, self, span, at(0., 0., lane.depth));
        }
        let back = d.wall_depth * 0.5 + 0.012;
        let top = self.top() + CLEAT_HEIGHT;
        let bottom = BACKPLATE_BOTTOM;
        // A structural backplate bears on the jamb header, including where the
        // adjacent wall infill is glass. The top saddles support both runners.
        metal.cuboid(
            Vec3::new(length, top - bottom, 0.008),
            0.,
            at(middle, (bottom + top) * 0.5, back),
        );
        let outside = self.outer_depth(d);
        let front = outside - if self.fascia { 0.004 } else { 0.002 };
        let intervals = (length / 0.58).ceil().max(1.) as usize;
        for i in 0..=intervals {
            let x = span.x + 0.024 + (length - 0.048) * i as f32 / intervals as f32;
            metal.cuboid(
                Vec3::new(0.034, CLEAT_HEIGHT, front - back),
                0.,
                at(x, top - CLEAT_HEIGHT * 0.5 - 0.0004, (back + front) * 0.5),
            );
        }
        let bumpers = a.part(Surface::Rubber, label);
        for lane in lanes(d) {
            for x in self.stops(lane).to_array() {
                bumpers.cuboid(
                    Vec3::new(
                        STOP_LENGTH,
                        1.45 * self.radius,
                        2. * (self.half_width - self.gauge) - 0.001,
                    ),
                    0.,
                    at(x, self.drop + self.gauge + 0.72 * self.radius, lane.depth),
                );
            }
        }
        if self.fascia {
            // Removable pelmet sits proud of the outer rail, with a reveal
            // above the travelling leaf. Its finish follows the frame stock.
            let bottom = 0.005;
            a.part(p.frame_finish, label).cuboid(
                Vec3::new(length - 0.002, top - bottom, 0.006),
                0.001,
                at(middle, (top + bottom) * 0.5, outside - 0.003),
            );
        }
    }

    pub(in super::super) fn build_hangers(
        &self,
        a: &mut Assembly,
        d: &DoorInstallation,
        leaf: Leaf,
        label: &str,
    ) {
        let geom = a.part(d.program.hardware_finish, label);
        let at = |x, y, z| {
            leaf.transform
                .mul_transform(Transform::from_xyz(x, leaf.size.y * 0.5 + y, z))
        };
        let axle = self.axle_height();
        let ledge = self.half_width - self.gauge - self.slot_half;
        let wheel_depth = self.slot_half + ledge * 0.5;
        let axis = Quat::from_rotation_x(FRAC_PI_2);
        for side in [-1., 1.] {
            let x = side * self.hanger_offset(leaf.size.x);
            geom.cuboid(
                Vec3::new(0.052, 0.003, leaf.size.z * 0.72),
                0.,
                at(x, 0.001, 0.),
            );
            geom.cuboid(
                Vec3::new(0.018, axle, self.slot_half * 1.32),
                0.,
                at(x, axle * 0.5, 0.),
            );
            let pin = at(x, axle, 0.);
            geom.cylinder(
                self.radius * 0.25,
                2. * wheel_depth,
                pin.with_rotation(pin.rotation * axis),
            );
            for z in [-wheel_depth, wheel_depth] {
                let wheel = at(x, axle, z);
                geom.cylinder(
                    self.radius,
                    ledge * 0.76,
                    wheel.with_rotation(wheel.rotation * axis),
                );
            }
        }
    }
}
