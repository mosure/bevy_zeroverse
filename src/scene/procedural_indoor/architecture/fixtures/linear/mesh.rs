use super::*;

impl LinearFixture {
    fn frame(&self, center: Vec3) -> Transform {
        Transform::from_translation(center).with_rotation(if self.along_z {
            Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2)
        } else {
            Quat::IDENTITY
        })
    }

    fn tube_offset(&self, i: usize) -> Vec3 {
        Vec3::new(
            0.,
            self.light_bottom_y + self.strip_width_m * 0.5,
            (i as f32 - (self.strips - 1) as f32 * 0.5) * self.pitch_m,
        )
    }

    // (transverse position, height), from the two latching rails through a
    // flat central web and two radiused shoulders. End caps use this same cut.
    fn lens_section(&self) -> [Vec2; 10] {
        const ARC: [(f32, f32); 5] = [
            (1., 0.),
            (0.9238795, 0.38268343),
            (
                std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2,
            ),
            (0.38268343, 0.9238795),
            (0., 1.),
        ];
        let half = self.strip_width_m * 0.5;
        let flat = half * self.flat_fraction;
        std::array::from_fn(|i| {
            let (sign, j) = if i < 5 { (-1., i) } else { (1., 9 - i) };
            let (c, s) = ARC[j];
            Vec2::new(
                sign * (flat + (half - flat) * c),
                self.pan_bottom_y - (self.pan_bottom_y - self.light_bottom_y) * s,
            )
        })
    }

    fn lens_strip(&self, g: &mut Geometry, tf: Transform, lo: f32, hi: f32, section: &[Vec2; 10]) {
        let vertex = |x, p: Vec2| tf.transform_point(Vec3::new(x, p.y, p.x));
        let width = self.strip_width_m * 0.5;
        let flat = width * self.flat_fraction;
        let height = self.pan_bottom_y - self.light_bottom_y;
        let normal = |p: Vec2| {
            tf.rotation
                * Vec3::new(
                    0.,
                    (p.y - self.pan_bottom_y) / (height * height),
                    p.x.signum() * (p.x.abs() - flat).max(0.) / (width - flat).powi(2),
                )
                .normalize()
        };
        let mut arc = 0.;
        for edge in section.windows(2) {
            let [a, b] = [edge[0], edge[1]];
            let next = arc + a.distance(b);
            g.vertex_quad(
                [vertex(lo, a), vertex(hi, a), vertex(hi, b), vertex(lo, b)],
                [normal(a), normal(a), normal(b), normal(b)],
                [[lo, arc], [hi, arc], [hi, next], [lo, next]],
            );
            arc = next;
        }
    }

    pub(in super::super) fn aperture(&self, g: &mut Geometry, center: Vec3) {
        let frame = self.frame(center);
        let half = self.lit_length_m * 0.5;
        if self.form == LinearForm::OpalWrap {
            self.lens_strip(g, frame, -half, half, &self.lens_section());
        } else {
            let radius = self.strip_width_m * 0.5;
            let turns = frame.rotation * Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2);
            for i in 0..self.strips {
                let tf = Transform::from_translation(frame.transform_point(self.tube_offset(i)))
                    .with_rotation(turns);
                // Opaque holders close both ends. Only the glass envelope emits.
                g.lathe(&[(radius, -half), (radius, half)], 20, tf);
            }
        }
    }

    pub(in super::super) fn body(
        &self,
        a: &mut Assembly,
        label: &str,
        trim_label: &str,
        center: Vec3,
    ) {
        let frame = self.frame(center);
        let at = |p| frame.with_translation(frame.transform_point(p));
        let top = self.top_y;
        let bottom = self.pan_bottom_y;
        let len = self.body_length_m;
        a.part(self.housing, trim_label).cuboid(
            Vec3::new(len, top - bottom, self.pan_width_m),
            0.0015,
            at(Vec3::Y * ((top + bottom) * 0.5)),
        );
        let half = self.lit_length_m * 0.5;
        if self.form == LinearForm::OpalWrap {
            let cut = self.lens_section();
            let g = a.part(self.trim, label);
            for sign in [-1., 1.] {
                let end = sign * len * 0.5;
                let tip = sign * half;
                self.lens_strip(g, frame, tip.min(end), tip.max(end), &cut);
                // End closure is a fan from the mounting edge to the convex lens.
                let root = frame.transform_point(Vec3::new(end, bottom, 0.));
                for edge in cut.windows(2) {
                    let v = |p: Vec2| frame.transform_point(Vec3::new(end, p.y, p.x));
                    crate::scene::procedural_indoor::envelope::construction::triangle(
                        g,
                        [root, v(edge[0]), v(edge[1])],
                        self.along() * sign,
                    );
                }
            }
            return;
        }
        if self.form == LinearForm::WingReflector {
            let g = a.part(self.trim, label);
            for sign in [-1., 1.] {
                let upper = Vec2::new(sign * self.pan_width_m * 0.5, bottom + 0.0005);
                let lip = Vec2::new(sign * self.reflector_half_m, self.reflector_edge_y);
                let span = lip - upper;
                let mid = (upper + lip) * 0.5;
                let fold = frame.rotation * Quat::from_rotation_x((-span.y).atan2(span.x));
                g.cuboid(
                    Vec3::new(len, 0.001, span.length()),
                    0.,
                    at(Vec3::new(0., mid.y, mid.x)).with_rotation(fold),
                );
            }
        }
        let socket = if self.form == LinearForm::WingReflector {
            Surface::Ceramic
        } else {
            self.trim
        };
        let g = a.part(socket, label);
        for i in 0..self.strips {
            let tube = self.tube_offset(i);
            let low = tube.y - self.strip_width_m * 0.57;
            for side in [-1., 1.] {
                let tf = at(Vec3::new(
                    side * (half + self.end_m * 0.5),
                    (bottom + 0.0006 + low) * 0.5,
                    tube.z,
                ));
                g.cuboid(
                    Vec3::new(self.end_m, bottom + 0.0006 - low, self.strip_width_m * 1.34),
                    0.0012,
                    tf,
                );
            }
        }
    }
}
