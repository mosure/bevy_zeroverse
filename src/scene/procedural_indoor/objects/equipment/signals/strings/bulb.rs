use super::*;

pub(super) struct Bulb {
    pub tap: Vec3,
    pub socket: Vec3,
    pub axis: Vec3,
    pub radius: f32,
    pub channel: usize,
    pub shape: u32,
    pub reflector: Option<Reflector>,
}

const PEAR: [(f32, f32); 7] = [
    (0., -2.5),
    (0.55, -2.35),
    (0.90, -1.9),
    (1., -1.3),
    (0.75, -0.8),
    (0.38, -0.5),
    (0., -0.5),
];

impl StringProgram {
    pub(super) fn sockets(&self, o: &IndoorObject) -> impl Iterator<Item = Bulb> + '_ {
        self.sockets_on(o, self.line(o.size))
    }

    // Factory pitch follows wire length around the sag. Blend in the minimum
    // horizontal spacing needed for tilted globes at the steepest part of a span.
    pub(super) fn sockets_on(
        &self,
        o: &IndoorObject,
        line: Vec<Vec3>,
    ) -> impl Iterator<Item = Bulb> + '_ {
        let mut lengths = Vec::with_capacity(line.len());
        let mut total = 0.;
        lengths.push(total);
        for seg in line.windows(2) {
            total += seg[0].distance(seg[1]);
            lengths.push(total);
        }
        let width = 0.96 * o.size.x;
        let clearance = (self.bulbs as f32 * (2.28 * self.radius_m + 0.008)
            / (width * (1. - 2. * self.spacing_jitter)))
            .min(1.);
        let station_x = move |t: f32| {
            let arc = t * total;
            let k = lengths
                .partition_point(|&s| s < arc)
                .saturating_sub(1)
                .min(lengths.len() - 2);
            let x = line[k].x
                + (line[k + 1].x - line[k].x) * (arc - lengths[k]) / (lengths[k + 1] - lengths[k]);
            x + (width * (t - 0.5) - x) * clearance
        };
        let mut rng = stream(o.seed, 0x42554c4253);
        let mut palette_rng = stream(o.seed, 0x42554c4250414c);
        let mut channels: [usize; 8] = std::array::from_fn(|i| i);
        let size = o.size;
        let n = self.colors_srgb.len();
        let n_bulbs = self.bulbs as f32;
        let mut left = -width * 0.5;
        (0..self.bulbs).map(move |i| {
            if i % n == 0 && !self.ordered {
                channels[..n].shuffle(&mut palette_rng);
            }
            let offset = rng.random_range(-self.spacing_jitter..self.spacing_jitter);
            let t = (i as f32 + 0.5 + offset) / self.bulbs as f32;
            let tap_x = station_x(t);
            let right = station_x((i + 1) as f32 / n_bulbs);
            let room_x = (tap_x - left).min(right - tap_x);
            left = right;
            let x = tap_x / size.x;
            let bay = self
                .supports
                .partition_point(|p| p.x < x)
                .saturating_sub(1)
                .min(self.bays - 1);
            let u = (x - self.supports[bay].x) / (self.supports[bay + 1].x - self.supports[bay].x);
            // Attach to the rendered cable segment, not a separately evaluated curve.
            let k = (u * SEGMENTS as f32).clamp(0., SEGMENTS as f32);
            let seg = (k as usize).min(SEGMENTS - 1);
            let a = self.in_bay(size, bay, seg as f32 / SEGMENTS as f32);
            let b = self.in_bay(size, bay, (seg + 1) as f32 / SEGMENTS as f32);
            let tap = a.lerp(b, k - seg as f32);
            let tangent = (b - a).normalize();
            let phase = self.phase + TAU * t * self.twist;
            let angle = self.lean * (0.68 * phase.sin() + rng.random_range(-0.32..0.32));
            let dir = if self.clipped {
                // A short wired socket can rotate around its cable. Long,
                // gravity-loaded drops instead keep their axes near plumb.
                let side = tangent.cross(Vec3::NEG_Y).normalize();
                side.cross(tangent) * angle.cos() + side * angle.sin()
            } else {
                let bearing = phase + rng.random_range(-1.2..1.2);
                Vec3::new(
                    bearing.cos() * angle.sin(),
                    -angle.cos(),
                    bearing.sin() * angle.sin(),
                )
            };
            let radius = self.radius_m * rng.random_range(0.89..1.06);
            let drop =
                self.drop_m * (0.87 + 0.18 * (phase * 1.3).sin() + rng.random_range(-0.15..0.18));
            // Each lamp owns a disjoint longitudinal cell. Bound horizontal
            // reach also by the wall/outer envelope before normalizing its axis.
            let reach = drop + 3.0 * radius;
            let dx = (room_x - 1.07 * radius - 0.003).max(0.) / reach;
            let dz =
                ((0.5 * size.z - tap.z * dir.z.signum()) - 1.07 * radius - 0.003).max(0.) / reach;
            let x = dir.x.clamp(-dx, dx);
            let z = dir.z.clamp(-dz, dz);
            let axis = Vec3::new(x, -(1. - x * x - z * z).max(0.).sqrt(), z);
            let lead = if self.clipped {
                axis
            } else {
                Vec3::NEG_Y.lerp(axis, 0.35).normalize()
            };
            Bulb {
                tap,
                socket: tap + lead * drop,
                axis,
                radius,
                channel: channels[i % n],
                shape: self.shape,
                reflector: self.reflector,
            }
        })
    }
}

impl Bulb {
    fn tf(&self) -> Transform {
        Transform::from_translation(self.socket)
            .with_rotation(Quat::from_rotation_arc(Vec3::NEG_Y, self.axis))
    }
    fn relative_radii(&self) -> Vec3 {
        match self.shape {
            2 => Vec3::new(0.48, 1.22, 0.48),
            3 => Vec3::new(0.65, 0.82, 0.65),
            _ => Vec3::ONE,
        }
    }
    pub(super) fn center(&self) -> Vec3 {
        let offset = match self.shape {
            1 => 1.52,
            4 => Reflector::LENS_DEPTH,
            _ => 0.48 + self.relative_radii().y,
        };
        self.socket + self.axis * (offset * self.radius)
    }
    pub(super) fn housing(&self, a: &mut Assembly, label: &str) {
        let r = self.radius;
        let tf = self.tf();
        a.part(Surface::Plastic, label).cylinder(
            r * 0.44,
            r * 0.68,
            tf * Transform::from_xyz(0., -r * 0.22, 0.),
        );
        if let Some(shade) = self.reflector {
            // Recess the lens behind a ring. The aperture normal points out of
            // the housing along the same axis as the analytic beam.
            let lip = Reflector::LENS_DEPTH + shade.recess;
            let profile = [
                (shade.lens_radius - 0.006, -Reflector::LENS_DEPTH + 0.015),
                (shade.mouth_radius, -lip),
                (1., -lip),
                (1., -lip + 0.08),
                (0.42, -0.46),
                (0.35, -0.48),
            ]
            .map(|(x, y)| (x * r, y * r));
            a.part(Surface::Metal, label).lathe(&profile, 16, tf);
        }
    }
    pub(super) fn aperture(&self, g: &mut Geometry) {
        let r = self.radius;
        let tf = self.tf();
        if let Some(shade) = self.reflector {
            let y = -Reflector::LENS_DEPTH * r;
            g.lathe(&[(0., y), (shade.lens_radius * r, y)], 16, tf);
            return;
        }
        match self.shape {
            1 => g.lathe(&PEAR.map(|(x, y)| (x * r, y * r)), 12, tf),
            _ => g.ellipsoid(
                r * self.relative_radii(),
                Transform::from_translation(self.center()).with_rotation(tf.rotation),
            ),
        }
    }
    pub(super) fn bounds(&self) -> (Vec3, Vec3) {
        let a = self.socket - self.axis * (0.12 * self.radius);
        let b = self.socket + self.axis * (3. * self.radius);
        (
            a.min(b) - Vec3::splat(self.radius),
            a.max(b) + Vec3::splat(self.radius),
        )
    }
    /// Projected emitting area for mean omnidirectional intensity. For a
    /// Lambertian closed surface, mean projected area is surface area / 4.
    /// The reflector instead uses its actual frontal aperture on axis.
    pub(super) fn projected_area(&self) -> f32 {
        let r2 = self.radius * self.radius;
        if let Some(shade) = self.reflector {
            return PI * shade.lens_radius.powi(2) * r2;
        }
        if self.shape == 1 {
            let area: f32 = PEAR
                .windows(2)
                .map(|p| {
                    let [(a, y), (b, z)] = [p[0], p[1]];
                    PI * (a + b) * (a - b).hypot(y - z)
                })
                .sum();
            return 0.25 * area * r2;
        }
        let radii = self.relative_radii();
        if radii.x == radii.y {
            return PI * r2;
        }
        let e = (1. - (radii.x / radii.y).powi(2)).sqrt();
        PI * 0.5 * radii.x * (radii.x + radii.y * e.asin() / e) * r2
    }
}
