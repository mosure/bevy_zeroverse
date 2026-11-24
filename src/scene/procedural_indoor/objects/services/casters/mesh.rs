use super::*;

impl CasterSet {
    pub(in super::super) fn build(&self, a: &mut Assembly, label: &'static str) {
        let r = self.radius_m;
        let w = self.tread_width_m;
        for m in self.mounts {
            let swivel = Quat::from_rotation_y(m.heading_rad);
            let pivot = Vec3::new(m.pivot_xz_m.x, 0., m.pivot_xz_m.y);
            let shift = if m.fixed { 0. } else { self.swivel_trail_m };
            let at = |x, y, z| pivot + swivel * Vec3::new(x, y, z);
            let axle = at(0., r, -shift);
            let axis = swivel * Vec3::X;
            let wheel_rotation = swivel * Quat::from_rotation_z(FRAC_PI_2);
            if self.wheel == Wheel::Twin {
                for side in [-1., 1.] {
                    self.wheel_mesh(
                        a,
                        label,
                        axle + axis * (side * (0.19 * r + w * 0.5)),
                        wheel_rotation,
                        0.,
                    );
                }
            } else {
                self.wheel_mesh(
                    a,
                    label,
                    axle,
                    wheel_rotation,
                    m.spoke_phase_rad.unwrap_or(0.),
                );
            }

            let frame = a.part(self.frame, label);
            let crown_y = 2.20 * r;
            let cap = Self::axle_half_width(self.wheel, w / r) * r;
            frame.lathe(
                &[(0., -cap), (0.19 * r, -cap), (0.19 * r, cap), (0., cap)],
                8,
                Transform::from_translation(axle).with_rotation(wheel_rotation),
            );
            let cheeks = if self.wheel == Wheel::Twin {
                0.
            } else {
                w * 0.5 + 0.11 * r
            };
            let crown = at(0., crown_y, 0.);
            frame.cuboid(
                Vec3::new(cheeks * 2. + 0.22 * r, 0.16 * r, 0.54 * r),
                0.,
                Transform::from_translation(crown).with_rotation(swivel),
            );

            // The load passes through a continuous crown, fork and cross axle.
            // A pressed fork straddles a single wheel; the twin has one central yoke.
            let arm = Vec3::new(0., crown_y - r, shift);
            let rot = swivel * Quat::from_rotation_x(arm.z.atan2(arm.y));
            for side in [-1., 1.].into_iter().take(if cheeks == 0. { 1 } else { 2 }) {
                frame.cuboid(
                    Vec3::new(
                        if cheeks == 0. { 0.27 * r } else { 0.14 * r },
                        arm.length(),
                        0.32 * r,
                    ),
                    0.,
                    Transform::from_translation((axle + crown) * 0.5 + axis * (side * cheeks))
                        .with_rotation(rot),
                );
            }
            // The top mates with the floor/post. Leave the buried top uncapped
            // to avoid coplanar faces on a pedestal's lower board.
            frame.lathe(
                &[
                    (0., crown_y),
                    (0.25 * r, crown_y),
                    (0.25 * r, self.mount_height_m - 0.10 * r),
                    (0.43 * r, self.mount_height_m - 0.10 * r),
                    (0.43 * r, self.mount_height_m),
                ],
                12,
                Transform::from_translation(pivot),
            );
            if self.post_mount {
                // A complete cap avoids a vanishing annulus when a post and
                // flange have nearly equal radii or different radial tessellation.
                frame.lathe(
                    &[(0.43 * r, self.mount_height_m), (0., self.mount_height_m)],
                    12,
                    Transform::from_translation(pivot),
                );
            }
        }
    }

    fn wheel_mesh(
        &self,
        a: &mut Assembly,
        label: &'static str,
        center: Vec3,
        rotation: Quat,
        phase: f32,
    ) {
        let r = self.radius_m;
        let h = self.tread_width_m * 0.5;
        let rim = r * self.core_fraction;
        let tf = Transform::from_translation(center).with_rotation(rotation);
        let tread = a.part(Surface::Rubber, label);
        let profile = [
            (rim, -h),
            (r * 0.91, -h),
            (r, -h * 0.63),
            (r, h * 0.63),
            (r * 0.91, h),
            (rim, h),
            (rim, -h),
        ];
        // Only the open-spoke design exposes the inner wall of the tyre.
        let n = if matches!(self.wheel, Wheel::Spoked(_)) {
            7
        } else {
            6
        };
        tread.lathe(&profile[..n], 16, tf);
        let core = a.part(self.core, label);
        if let Wheel::Spoked(count) = self.wheel {
            core.lathe(
                &[
                    (0., -h * 0.92),
                    (r * 0.27, -h * 0.92),
                    (r * 0.27, h * 0.92),
                    (0., h * 0.92),
                ],
                12,
                tf,
            );
            for i in 0..count {
                let rot = rotation
                    * Quat::from_rotation_y(phase + 2. * PI * f32::from(i) / f32::from(count));
                let inner = r * 0.23;
                let outer = rim + r * 0.035;
                core.cuboid(
                    Vec3::new(outer - inner, h * 1.28, r * 0.16),
                    0.,
                    Transform::from_translation(center + rot * Vec3::X * ((outer + inner) * 0.5))
                        .with_rotation(rot),
                );
            }
        } else {
            // The dished hub meets the tyre at its side-wall seam. It needs no
            // invisible cylinder buried inside the rubber.
            core.lathe(&[(0., -h * 0.65), (rim, -h)], 16, tf);
            core.lathe(&[(rim, h), (0., h * 0.65)], 16, tf);
        }
    }
}
