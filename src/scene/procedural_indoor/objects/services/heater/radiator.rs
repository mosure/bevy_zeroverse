//! Hydronic emitter bodies with fitted floor saddles, headers and valve connections.
use super::*;
mod panel;
mod tubes;
pub use panel::PressedPanel;
pub use tubes::TubeBank;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Body {
    pub width_m: f32,
    pub bottom_m: f32,
    pub top_m: f32,
    pub end_clearance_m: f32,
    pub support_count: usize,
    pub support_span_m: f32,
    pub valve_left: bool,
}

impl Body {
    fn fit(o: &IndoorObject, r: &mut impl Rng) -> Self {
        let s = o.size;
        let end = r.random_range(0.048_f32..0.079).min(s.x * 0.105);
        let width = s.x - 2. * end;
        let inset = r.random_range(0.075_f32..0.20).min(0.20 * width);
        let span = width - 2. * inset;
        Self {
            width_m: width,
            bottom_m: (s.y * r.random_range(0.14..0.27)).clamp(0.023, 0.083),
            top_m: s.y - r.random_range(0.002..0.007),
            end_clearance_m: end,
            support_count: (span / r.random_range(0.80..1.25)).ceil().clamp(1., 4.) as usize + 1,
            support_span_m: span,
            valve_left: r.random_bool(0.5),
        }
    }

    fn height(self) -> f32 {
        self.top_m - self.bottom_m
    }

    // Each saddle reaches the underside of the first element, across both banks
    // when fitted. Its foot stays on the actual floor inside the reserved depth.
    fn feet(self, a: &mut Assembly, o: &IndoorObject, rear: f32, front: f32, attach_y: f32) {
        let width = self.end_clearance_m * 0.89;
        let depth = (front - rear + width).min(o.size.z - 0.004);
        let z = (rear + front) * 0.5;
        let travel = (o.size.z - depth) * 0.5;
        let foot_z = z.clamp(-travel, travel);
        let gauge = (self.bottom_m * 0.095).clamp(0.003, 0.006);
        let g = a.part(Surface::Metal, o.kind.class_name());
        for i in 0..self.support_count {
            let x = self.support_span_m * (i as f32 / (self.support_count - 1) as f32 - 0.5);
            g.cuboid(
                Vec3::new(width, gauge, depth),
                0.001,
                Transform::from_xyz(x, gauge * 0.5, foot_z),
            );
            g.cuboid(
                Vec3::new(width * 0.49, attach_y - gauge, front - rear + 0.008),
                0.,
                Transform::from_xyz(x, (attach_y + gauge) * 0.5, z),
            );
        }
    }

    // Run the supply to an upper side port, the return to the opposing lower
    // port. The thermostat sits on the exposed tee, along the valve spindle.
    fn plumbing(self, a: &mut Assembly, o: &IndoorObject, z: f32) {
        let depth_clearance = o.size.z * 0.5 - z.abs();
        let r = (o.size.z * 0.064)
            .clamp(0.005, 0.009)
            .min(depth_clearance * 0.63);
        let control = (0.21 * self.end_clearance_m)
            .min(self.height() * 0.15)
            .min(depth_clearance * 0.93);
        for sign in [-1., 1.] {
            let supply = (sign < 0.) == self.valve_left;
            let from_edge = if supply { control + 0.005 } else { r * 1.65 };
            let y = if supply {
                self.top_m - from_edge
            } else {
                self.bottom_m + from_edge
            };
            let end = self.width_m * 0.5;
            let pipe = end + self.end_clearance_m * 0.43;
            let center = Vec3::new(sign * pipe, y, z);
            let g = a.part(Surface::Chrome, o.kind.class_name());
            g.rod(center.with_y(0.), center, r);
            g.rod(center.with_x(sign * (end - 0.004)), center, r * 1.08);
            g.small_ellipsoid(Vec3::splat(r * 1.36), Transform::from_translation(center));
            let union = Transform::from_xyz(sign * (end + r * 0.36), y, z)
                .with_rotation(Quat::from_rotation_z(-sign * FRAC_PI_2));
            g.lathe(
                &[(r * 1.52, 0.), (r * 1.52, r * 1.18), (r, r * 1.28)],
                6,
                union,
            );
            if supply {
                let stock = a.part(Surface::Plastic, o.kind.class_name());
                let length = self.end_clearance_m * 0.50;
                stock.lathe(
                    &[
                        (control * 0.63, 0.),
                        (control * 0.64, length * 0.15),
                        (control, length * 0.27),
                        (control * 0.95, length * 0.87),
                        (control * 0.78, length),
                        (0., length),
                    ],
                    16,
                    Transform::from_translation(center)
                        .with_rotation(Quat::from_rotation_z(-sign * FRAC_PI_2)),
                );
            }
        }
    }
}
