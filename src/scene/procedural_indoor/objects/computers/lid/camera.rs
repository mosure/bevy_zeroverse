use super::*;

impl CameraModule {
    pub(super) fn sample(o: &IndoorObject, top: f32, header_span: f32) -> Option<Self> {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"LidOptic"));
        let window = match r.random_range(0..100) {
            0..=5 => return None,
            6..=49 => CameraWindow::LensOnly,
            50..=76 => CameraWindow::WideWindow,
            77..=92 => CameraWindow::ShutterOpen,
            _ => CameraWindow::ShutterClosed,
        };
        // Leave room for the outer edge bevel and the screen reveal on both sides.
        let height_m = (top * r.random_range(0.33..0.54))
            .min(0.0070)
            .min(header_span * 0.76);
        let width_m = match window {
            CameraWindow::LensOnly => height_m * r.random_range(1.0..1.34),
            CameraWindow::WideWindow => height_m * r.random_range(2.3..5.6),
            _ => height_m * r.random_range(2.7..4.7),
        };
        let aperture_offset_x_m = match window {
            CameraWindow::LensOnly => 0.,
            CameraWindow::WideWindow => width_m * r.random_range(-0.19..0.19),
            _ => width_m * if r.random_bool(0.5) { -0.23 } else { 0.23 },
        };
        Some(Self {
            window,
            // Integrated cameras usually sit on the centreline. Some share a
            // wider sensor strip or are mounted slightly to one side.
            offset_x_m: if r.random_bool(0.67) {
                0.
            } else {
                o.size.x * r.random_range(-0.082..0.082)
            },
            width_m,
            height_m,
            aperture_radius_m: height_m * r.random_range(0.24..0.38),
            aperture_offset_x_m,
        })
    }

    pub(super) fn surround(self) -> LidStock {
        match self.window {
            CameraWindow::LensOnly => LidStock::Molded,
            _ => LidStock::Bonded,
        }
    }

    pub(super) fn aperture_stock(self) -> LidStock {
        if self.window == CameraWindow::ShutterClosed {
            LidStock::Shutter
        } else {
            LidStock::Lens
        }
    }

    /// Replace the top bezel strip with an opening. The window, optical seat
    /// and lens (or closed privacy shutter) meet at shared edges.
    pub(super) fn build(
        self,
        a: &mut Assembly,
        header: [Vec2; 4],
        tf: &Transform,
        z: f32,
        bezel: (Surface, &str),
    ) {
        let center = Vec2::new(self.offset_x_m, (header[0].y + header[3].y) * 0.5);
        let half = Vec2::new(self.width_m, self.height_m) * 0.5;
        let corners = [
            Vec2::NEG_ONE,
            Vec2::new(1., -1.),
            Vec2::ONE,
            Vec2::new(-1., 1.),
        ];
        let rect = corners.map(|sign| center + sign * half);
        let at = |p: Vec2, z| tf.transform_point(Vec3::new(p.x, z, p.y));
        let forward = tf.rotation * Vec3::Y;
        let depth = (self.height_m * 0.08).min(0.00046);
        let seat_z = z - depth;
        let collar = corners.map(|s| center + s * (half + depth * 0.65));
        let face = a.part(bezel.0, bezel.1);
        for i in 0..4 {
            let j = (i + 1) % 4;
            face.metric_quad(
                [
                    at(header[i], z),
                    at(header[j], z),
                    at(collar[j], z),
                    at(collar[i], z),
                ],
                forward,
            );
            face.metric_quad(
                [
                    at(collar[i], z),
                    at(collar[j], z),
                    at(rect[j], seat_z),
                    at(rect[i], seat_z),
                ],
                forward,
            );
        }
        let center = center + Vec2::X * self.aperture_offset_x_m;
        let d = std::f32::consts::FRAC_1_SQRT_2;
        let ring = [
            Vec2::NEG_X,
            Vec2::splat(-d),
            Vec2::NEG_Y,
            Vec2::new(d, -d),
            Vec2::X,
            Vec2::splat(d),
            Vec2::Y,
            Vec2::new(-d, d),
        ]
        .map(|v| center + v * self.aperture_radius_m);
        let window = a.part(Surface::Plastic, self.surround().label());
        let base = window.positions.len() as u32;
        for &p in rect.iter().chain(&ring) {
            window.positions.push(at(p, seat_z).to_array());
            window.normals.push(forward.to_array());
            window.uvs.push((p - center).to_array());
        }
        for i in 0..4_u32 {
            let j = (i + 1) % 4;
            let arc = 4 + 2 * i;
            let next = 4 + 2 * j;
            window
                .indices
                .extend([i, next, j, i, arc, arc + 1, i, arc + 1, next].map(|k| base + k));
        }
        let closed = self.window == CameraWindow::ShutterClosed;
        let inset = if closed { 0.92 } else { 0.80 };
        let aperture_z = seat_z - depth * if closed { 0.32 } else { 1.1 };
        let aperture = ring.map(|p| center + (p - center) * inset);
        for i in 0..ring.len() {
            let j = (i + 1) % ring.len();
            window.metric_quad(
                [
                    at(ring[i], seat_z),
                    at(ring[j], seat_z),
                    at(aperture[j], aperture_z),
                    at(aperture[i], aperture_z),
                ],
                forward,
            );
        }
        a.part(Surface::Plastic, self.aperture_stock().label())
            .slab_cap(&aperture, 1., center, aperture_z, tf, |p| p - center);
    }
}
