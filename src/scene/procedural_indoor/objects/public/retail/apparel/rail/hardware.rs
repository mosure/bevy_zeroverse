use super::*;

pub(super) const STOP_RADIUS: f32 = 0.0045;
const LOOP_RADIUS: f32 = ARM_RADIUS + WIRE + 0.0045;

impl ApparelRail {
    pub(super) fn frame(&self, a: &mut Assembly, s: Vec3) {
        let post = self.post_half_span_m;
        let top = self.spine_y_m;
        let back = self.spine_z_m;
        for x in [-post, post] {
            board(
                a,
                self.frame,
                FIXTURE,
                Vec3::new(x, 0.017, 0.),
                Vec3::new(0.060, 0.034, s.z * 0.94),
            );
            a.part(self.frame, FIXTURE).tube(
                &[Vec3::new(x, 0.034, back), Vec3::new(x, top, back)],
                0.018,
                12,
            );
        }
        let metal = a.part(self.hardware, FIXTURE);
        metal.tube(
            &[Vec3::new(-post, top, back), Vec3::new(post, top, back)],
            ARM_RADIUS,
            12,
        );
        if self.form == RailForm::SideHang {
            return;
        }
        for arm in &self.arms {
            metal.tube(&[arm.start_m, arm.end_m], ARM_RADIUS, 12);
            metal.tube(
                &[arm.point(0.006), arm.point(0.033)],
                ARM_RADIUS + 0.004,
                12,
            );
            if arm.stops_m.is_empty() {
                metal.tube(
                    &[arm.end_m - Vec3::Z * 0.005, arm.end_m + Vec3::Z * 0.003],
                    ARM_RADIUS + 0.005,
                    10,
                );
            }
            let (_, up) = arm.hook_frame();
            for &stop in &arm.stops_m {
                let base = arm.point(stop) + up * (ARM_RADIUS - 0.002);
                metal.tube(&[base, base + up * 0.018], STOP_RADIUS, 8);
            }
        }
    }

    /// The hook stays perpendicular to its rail even when the hanger swivels.
    /// The top of its inside bend bears on the tube. Its shank joins the neck.
    pub(super) fn hook(&self, h: &HungItem) -> [Vec3; 8] {
        let arm = &self.arms[h.arm];
        let (across, up) = arm.hook_frame();
        let center = arm.point(h.along_m) - up * (LOOP_RADIUS - ARM_RADIUS - WIRE);
        std::array::from_fn(|i| {
            if i == 0 {
                return center - across * LOOP_RADIUS - Vec3::Y * 0.063;
            }
            let angle = (i - 1) as f32 * PI / 6.;
            center + LOOP_RADIUS * (up * angle.sin() - across * angle.cos())
        })
    }

    pub(super) fn hanger(&self, a: &mut Assembly, h: &HungItem) -> Vec3 {
        let points = self.hook(h);
        a.part(Surface::Chrome, FIXTURE).tube(&points, WIRE, 8);
        let neck = points[0];
        let shoulder = self.collections[h.collection].cut.shoulder_width_m * h.size_grade * 0.5;
        let axis = Quat::from_rotation_y(h.yaw_rad) * Vec3::Z;
        let left = neck - shoulder * axis - Vec3::Y * (0.51 * shoulder);
        let right = left + axis * (2. * shoulder);
        let body = a.part(self.hanger_stock, FIXTURE);
        if self.hanger_stock == Surface::Metal {
            body.tube(&bent_wire([neck, left, right]), 0.0032, 8);
        } else {
            body.tube(&[left, neck, right], 0.0065, 10);
            body.tube(&[left, right], 0.0035, 8);
        }
        neck
    }
}

/// Tangent circular bends keep the inside of the wire open at the two acute
/// shoulder returns. Sweeping a thick tube through a sharp reversal inverts
/// its inner triangles.
fn bent_wire(corners: [Vec3; 3]) -> [Vec3; 16] {
    let mut path = [Vec3::ZERO; 16];
    let radius = 0.009;
    for (i, &corner) in corners.iter().enumerate() {
        let incoming = (corners[(i + 2) % 3] - corner).normalize();
        let outgoing = (corners[(i + 1) % 3] - corner).normalize();
        let cosine = incoming.dot(outgoing);
        let half_sine = ((1. - cosine) * 0.5).sqrt();
        let center = corner + (incoming + outgoing).normalize() * (radius / half_sine);
        let setback = radius * ((1. + cosine) / (1. - cosine)).sqrt();
        let first = (corner - center + incoming * setback).normalize();
        let last = (corner - center + outgoing * setback).normalize();
        for step in 0..=4 {
            path[5 * i + step] = center + radius * first.slerp(last, step as f32 * 0.25);
        }
    }
    path[15] = path[0];
    path
}
