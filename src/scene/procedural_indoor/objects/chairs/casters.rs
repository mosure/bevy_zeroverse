//! Twin-wheel swivel casters. The kingpin sits ahead of the axle; each wheel
//! rolls on the floor and the carrier fits between its two tyres.
use super::*;

pub(super) struct CasterSet {
    radius: f32,
    trail: f32,
    tyre_width: f32,
    track: f32,
    heading: f32,
}

impl CasterSet {
    pub(super) fn sample(r: &mut impl Rng) -> Self {
        let radius = r.random_range(0.024..0.036);
        let tyre_width = radius * r.random_range(0.39..0.54);
        Self {
            radius,
            tyre_width,
            trail: radius * r.random_range(0.34..0.67),
            track: tyre_width + radius * 0.64,
            heading: r.random_range(-PI..PI),
        }
    }
    pub(super) fn mount_height(&self) -> f32 {
        2. * self.radius + 0.010
    }

    pub(super) fn sweep_radius(&self) -> f32 {
        (self.radius + self.trail).hypot((self.track + self.tyre_width) * 0.5 + 0.001)
    }

    pub(super) fn build(&self, a: &mut Assembly, mount: Vec3, r: &mut impl Rng) {
        // Most wheels trail the chair's last roll. An unloaded wheel can be
        // left turned independently when the occupant swivels or lifts a leg.
        let yaw = self.heading
            + if r.random_bool(0.22) {
                r.random_range(-PI..PI)
            } else {
                r.random_range(-0.49..0.49)
            };
        let swivel = Quat::from_rotation_y(yaw);
        let axis = swivel * Vec3::X;
        let axle = mount.with_y(self.radius) - swivel * Vec3::Z * self.trail;
        let wheel = swivel * Quat::from_rotation_z(FRAC_PI_2);
        let h = self.tyre_width * 0.5;
        let radius = self.radius;
        let profile = [
            (0., -h),
            (radius * 0.86, -h),
            (radius, -h * 0.58),
            (radius, h * 0.58),
            (radius * 0.86, h),
            (0., h),
        ];
        for side in [-1., 1.] {
            a.part(Surface::Rubber, "chair").lathe(
                &profile,
                12,
                Transform::from_translation(axle + axis * (side * self.track * 0.5))
                    .with_rotation(wheel),
            );
        }
        let carrier = a.part(Surface::Plastic, "chair");
        carrier.tube(&[mount, axle], radius * 0.27, 8);
        let end = (self.track + self.tyre_width) * 0.5 + 0.001;
        carrier.tube(&[axle - axis * end, axle + axis * end], radius * 0.37, 10);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_tyres_ground_and_the_whole_swivel_fits_its_reserved_sweep() {
        let mut headings = std::collections::BTreeSet::new();
        for seed in 0..96 {
            let mut r = stream(seed, 31);
            let set = CasterSet::sample(&mut r);
            let mut a = Assembly::default();
            let mount = Vec3::Y * set.mount_height();
            set.build(&mut a, mount, &mut r);
            let o = super::super::super::audit::fixture(ObjectKind::Chair, seed);
            super::super::super::audit::check(&o, &a).unwrap();
            let tyres = &a.parts[&(Surface::Rubber, "chair".into())];
            let mut centers = [Vec3::ZERO; 2];
            for (i, t) in tyres
                .positions
                .chunks_exact(tyres.positions.len() / 2)
                .enumerate()
            {
                let mut lo = Vec3::splat(f32::INFINITY);
                let mut hi = -lo;
                for pos in t {
                    let v = Vec3::from_array(*pos);
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
                assert!(lo.y.abs() < 1e-6);
                centers[i] = (lo + hi) * 0.5;
            }
            assert!((centers[0].distance(centers[1]) - set.track).abs() < 1e-5);
            let axle = (centers[0] + centers[1]) * 0.5;
            assert!((axle.xz().length() - set.trail).abs() < 1e-5);
            headings.insert((axle.x.atan2(axle.z) * 4.).round() as i32);
            assert!(a
                .parts
                .values()
                .flat_map(|g| &g.positions)
                .all(|v| Vec3::from_array(*v).xz().length() <= set.sweep_radius() + 1e-5));
        }
        assert!(headings.len() > 20);
    }
}
