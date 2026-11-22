//! Metric task heads, their visible exit aperture and their photometric frame.
use super::*;
use crate::scene::procedural_indoor::photometry::FixtureBeam;

pub(super) struct LampHead {
    pub pivot: Transform,
    radius: f32,
    bar: Vec2,
    depth_from_pivot: f32,
    lens_radius: f32,
    linear: bool,
}

impl LampHead {
    pub(super) fn new(o: &IndoorObject, p: &TaskLampProgram) -> Self {
        let s = o.size;
        let radius = s.x.min(s.z) * p.aperture_fraction;
        let bar = Vec2::new(s.x * p.bar_width, s.z * p.bar_depth);
        let dimensions = if p.head == 0 {
            bar * 0.5
        } else {
            Vec2::splat(radius)
        };
        let joint_radius = Self::joint_radius(o);
        let half = Vec3::new(
            dimensions.x.max(joint_radius * 2.),
            s.y * 0.075 + joint_radius * 2.4,
            dimensions.y + joint_radius * 2.5,
        );
        let axis = Vec3::new(
            -p.head_swivel_radians.cos(),
            0.,
            p.head_swivel_radians.sin(),
        );
        let proposed = Vec3::new(
            s.x * p.arm.tilt * 0.45,
            s.y * p.head_height,
            s.z * (p.arm.reach - 0.35).clamp(-0.15, 0.27),
        );
        let mut pitch = p.head_tilt_radians;
        // Short or anisotropic legacy reservations may only admit part of the
        // angular range. Fit the complete head before joining the arm to its pivot.
        let pivot = loop {
            let rotation = Quat::from_axis_angle(axis, pitch);
            let extent = (rotation * Vec3::X).abs() * half.x
                + (rotation * Vec3::Y).abs() * half.y
                + (rotation * Vec3::Z).abs() * half.z;
            let low = Vec3::new(-s.x * 0.5, 0., -s.z * 0.5) + extent;
            let high = Vec3::new(s.x * 0.5, s.y, s.z * 0.5) - extent;
            if low.cmple(high).all() || pitch <= 0.001 {
                break Transform::from_translation(proposed.clamp(low.min(high), high.max(low)))
                    .with_rotation(rotation);
            }
            pitch *= 0.5;
        };
        let depth_from_pivot = s.y * 0.065
            + match p.head {
                0 => 0.0015,
                3 => 0.,
                _ => -0.0015,
            };
        let lens_radius = if p.head == 3 {
            luminaires::shade_opening(&p.shade, radius, radius)
        } else {
            radius * 0.82
        };
        Self {
            pivot,
            radius,
            bar,
            depth_from_pivot,
            lens_radius,
            linear: p.head == 0,
        }
    }

    pub(super) fn joint_radius(o: &IndoorObject) -> f32 {
        (o.size.x.min(o.size.z) * 0.03).clamp(0.003, 0.009)
    }

    /// The pin clears the back rim. Its boss enters the side of the housing
    /// without crossing the luminous exit plane.
    pub(super) fn join_arm(&self, a: &mut Assembly, o: &IndoorObject, trim: Surface) -> Vec3 {
        let r = Self::joint_radius(o);
        let rear = if self.linear {
            0.5 * self.bar.y
        } else {
            self.radius
        };
        let bottom = -0.065 * o.size.y;
        let pin = self
            .pivot
            .transform_point(Vec3::new(0., bottom - r * 1.2, -rear - r * 1.15));
        let seat = self
            .pivot
            .transform_point(Vec3::new(0., bottom + r * 0.44, -rear + r * 0.38));
        a.part(trim, o.kind.class_name()).rod(pin, seat, r * 0.65);
        a.part(Surface::Chrome, o.kind.class_name()).cylinder(
            r,
            r * 2.7,
            Transform::from_translation(pin)
                .with_rotation(self.pivot.rotation * Quat::from_rotation_z(FRAC_PI_2)),
        );
        pin
    }

    fn at(&self, y: f32) -> Transform {
        self.pivot
            .with_translation(self.pivot.transform_point(Vec3::Y * y))
    }

    pub(super) fn area(&self) -> f32 {
        if self.linear {
            self.bar.x * 0.88 * self.bar.y * 0.83
        } else {
            // A regular 32-sided exit aperture, matching the cut diffuser.
            16. * self.lens_radius.powi(2) * (TAU / 32.).sin()
        }
    }

    pub(super) fn beam(&self, o: &IndoorObject, p: &TaskLampProgram) -> FixtureBeam {
        let world = o.transform().mul_transform(self.pivot);
        let mut r = stream(o.seed, 5320);
        let (inner, outer) = match p.head {
            2 => (r.random_range(0.32..0.51), r.random_range(0.75..0.98)),
            3 => (r.random_range(0.48..0.65), r.random_range(1.04..1.30)),
            _ => (r.random_range(0.38..0.61), r.random_range(0.88..1.17)),
        };
        FixtureBeam {
            position: world.transform_point(Vec3::NEG_Y * (self.depth_from_pivot + 0.003)),
            radius: (self.area() / PI).sqrt() * 0.5,
            direction: world.rotation * Vec3::NEG_Y,
            inner,
            outer,
        }
    }

    fn corners(&self) -> [Vec3; 4] {
        let half = self.bar * Vec2::new(0.88, 0.83) * 0.5;
        let lens = self.at(-self.depth_from_pivot);
        [
            (-half.x, -half.y),
            (half.x, -half.y),
            (half.x, half.y),
            (-half.x, half.y),
        ]
        .map(|(x, z)| lens.transform_point(Vec3::new(x, 0., z)))
    }

    pub(super) fn aperture(&self, g: &mut Geometry) {
        if self.linear {
            g.metric_quad(self.corners(), self.pivot.rotation * Vec3::NEG_Y);
        } else {
            let y = -self.depth_from_pivot;
            g.lathe(&[(0., y), (self.lens_radius, y)], 32, self.pivot);
        }
    }

    pub(super) fn build(&self, a: &mut Assembly, o: &IndoorObject, p: &TaskLampProgram) {
        let label = o.kind.class_name();
        let h = o.size.y;
        let bottom = -h * 0.065;
        if self.linear {
            a.part(p.arm.finish, label).cuboid(
                Vec3::new(self.bar.x, h * 0.08, self.bar.y),
                0.008,
                self.at(bottom + h * 0.04),
            );
            // Opal cover is seated against the metal back. Only its outward
            // face emits; the exact same exit plane feeds the light renderer.
            let corners = self.corners();
            let up = self.pivot.rotation * Vec3::Y;
            for i in 0..4 {
                let from = corners[i];
                let to = corners[(i + 1) % 4];
                a.part(p.arm.finish, label).metric_quad(
                    [from, to, to + up * 0.0015, from + up * 0.0015],
                    up.cross(to - from),
                );
            }
            self.aperture(a.part(Surface::Light, label));
            return;
        }
        let r = self.radius;
        let top = if p.head == 3 { h * 0.075 } else { h * 0.05 };
        if p.head == 3 {
            // Upright shaded downlight with a bottom lens and a capped driver.
            // Its rim and upper cap carry the lens, without a second glowing bulb.
            let taper = 0.65 + p.shade.shoulder * 0.3;
            luminaires::shade(
                a,
                label,
                &p.shade,
                self.at(bottom).translation,
                r,
                h * 0.14,
                taper,
            );
            let upper = r * if matches!(p.shade.shape, 1 | 3) {
                taper.clamp(0.88, 1.)
            } else {
                taper
            };
            let wall = (r * 0.018).clamp(0.001, 0.004);
            let open = luminaires::shade_opening(&p.shade, r, upper);
            a.part(p.shade.trim, label).lathe(
                &[(0., top - wall), (open, top - wall), (open, top), (0., top)],
                32,
                self.pivot,
            );
        } else {
            let upper = r * if p.head == 2 { 0.45 } else { 1. };
            let wall = (r * 0.045).clamp(0.0015, 0.003);
            a.part(p.shade.trim, label).lathe(
                &[
                    (r, bottom),
                    (upper, top),
                    (0., top),
                    (0., top - wall),
                    (upper - wall, top - wall),
                    (r - wall, bottom + 0.0035),
                    (self.lens_radius, bottom + 0.0035),
                    (self.lens_radius, -self.depth_from_pivot),
                    (r - wall, bottom),
                    (r, bottom),
                ],
                32,
                self.pivot,
            );
        }
        // The existing hem or reflector ends at the cut lens circumference.
        self.aperture(a.part(Surface::Light, label));
    }
}
