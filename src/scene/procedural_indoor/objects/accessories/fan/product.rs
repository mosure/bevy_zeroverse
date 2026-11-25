//! A stationary guard and motor surround a balanced, pitched impeller. All
//! dimensions below are in the head frame, with positive Z pointing at the user.
use super::*;
use crate::scene::procedural_indoor::geometry::rotor::RotorBlades;

#[derive(Debug, Clone, serde::Serialize)]
pub struct FanHead {
    pub guard_radius: f32,
    pub rim_half_depth: f32,
    pub front_crown: f32,
    pub rear_crown: f32,
    pub grille_sweep: f32,
    pub rear_phase: f32,
    pub badge_radius: f32,
    pub motor_radius: f32,
    pub housing: Surface,
    pub guard: Surface,
    pub blades: Surface,
    pub(super) rotor: RotorBlades,
}

impl FanHead {
    pub fn sample(o: &IndoorObject, p: &FanProgram) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"DeskWind"));
        let guard_radius = p.radius * 0.94;
        let half = p.depth * 0.5;
        let rim_half_depth = half * r.random_range(0.22..0.42);
        let front_crown = half * r.random_range(0.72..0.92);
        let rear_crown = half * r.random_range(0.64..0.75);
        let tip = guard_radius * r.random_range(0.82..0.91);
        let hub = guard_radius * r.random_range(0.16..0.26);
        let width = tip * p.blade_chord * r.random_range(2.18..3.0);
        // Bound both skins across the *entire* swept disk, including the
        // inward polygon chord of a wire hoop and the wire thickness.
        let outer_wire = (tip + 2. * p.wire) / guard_radius;
        let clear = rim_half_depth
            + (front_crown.min(rear_crown) - rim_half_depth) * (1. - outer_wire.min(1.).powi(2))
            - 1.8 * p.wire;
        let pitch = r
            .random_range(0.42_f32..0.77)
            .min((1.17 * clear / width).min(1.).asin());
        let rotor = RotorBlades {
            blades: p.blades as u8,
            phase: p.blade_phase,
            hub_radius: hub,
            radius_m: tip,
            rotor_y: 0.,
            blade_width_m: width,
            root_chord_ratio: r.random_range(0.29..0.55),
            shoulder_span: r.random_range(0.38..0.71),
            tip_chord_ratio: r.random_range(0.20..0.80),
            pitch_radians: pitch,
            tip_pitch_radians: pitch * r.random_range(0.38..0.76),
            sweep_m: tip * p.blade_sweep.sin() * if r.random_bool(0.22) { -0.65 } else { 0.92 },
            thickness_m: (tip * r.random_range(0.012..0.027)).min(clear * 0.14),
            camber_m: clear * r.random_range(0.08..0.21),
            tip_rise_m: clear * r.random_range(-0.07..0.09),
        };
        Self {
            guard_radius,
            rim_half_depth,
            front_crown,
            rear_crown,
            grille_sweep: r.random_range(-0.37..0.37),
            rear_phase: r.random_range(-PI..PI),
            badge_radius: guard_radius * r.random_range(0.10..0.20),
            motor_radius: hub * r.random_range(1.07..1.33),
            housing: p.finish,
            guard: p.guard_finish,
            blades: p.blade_finish,
            rotor,
        }
    }

    pub(super) fn build(&self, a: &mut Assembly, o: &IndoorObject, p: &FanProgram) {
        let label = o.kind.class_name();
        a.part(self.housing, label).housing(
            &[(0., p.base.xz() * 0.5), (p.base.y, p.base.xz() * 0.39)],
            if p.yoke { 3.5 } else { 2.2 },
            Transform::IDENTITY,
        );
        let mut head = Assembly::default();
        self.grille(head.part(self.guard, label), p);
        let axial = Transform::from_rotation(Quat::from_rotation_x(FRAC_PI_2));
        let hub = self.rotor.hub_radius;
        let half = p.depth * 0.5;
        // Independent spinning cap and fixed motor with one shaft between them.
        // The emboss and rear case close the otherwise exposed wire roots.
        head.part(self.housing, label).lathe(
            &[
                (0., -half * 0.98),
                (self.motor_radius * 0.72, -half * 0.98),
                (self.motor_radius, -self.rear_crown),
                (self.motor_radius * 0.93, -half * 0.32),
                (hub * 0.29, -half * 0.32),
                (hub * 0.29, -half * 0.17),
            ],
            24,
            axial,
        );
        head.part(self.blades, label).lathe(
            &[
                (0., -half * 0.17),
                (hub * 0.94, -half * 0.17),
                (hub, -half * 0.11),
                (hub, half * 0.12),
                (hub * 0.84, half * 0.20),
                (0., half * 0.22),
            ],
            24,
            axial,
        );
        self.rotor
            .append(head.part(self.blades, label), o.seed, |[x, y, z]| {
                [x, -z, y]
            });
        head.part(self.housing, label).lathe(
            &[
                (0., self.front_crown - p.wire),
                (self.badge_radius, self.front_crown - p.wire),
                (self.badge_radius, self.front_crown + p.wire * 0.5),
                (self.badge_radius * 0.77, self.front_crown + p.wire),
                (0., self.front_crown + p.wire),
            ],
            24,
            axial,
        );
        let anchor = self.mount(head.part(self.housing, label), p);
        let tf = p.head_transform();
        let base = Vec3::Y * (p.base.y * 0.78);
        let end = tf.transform_point(anchor);
        let neck = (end - base).normalize();
        // The stem ends in the cradle or on the rear motor bracket.
        a.part(self.housing, label).lathe(
            &[
                (o.size.x * 0.030, 0.),
                (Self::joint_radius(p) * 0.76, end.distance(base)),
            ],
            12,
            Transform::from_translation(base).with_rotation(Quat::from_rotation_arc(Vec3::Y, neck)),
        );
        public::append_at(a, head, tf);
    }

    fn joint_radius(p: &FanProgram) -> f32 {
        (p.radius * 0.082).min(p.depth * 0.075)
    }

    /// The surrounding yoke bears on two rim pivots. The single-post design
    /// instead carries the rear of the motor, clear of the blade disk and cage.
    pub(super) fn mount(&self, g: &mut Geometry, p: &FanProgram) -> Vec3 {
        let joint = Self::joint_radius(p);
        let anchor = if p.yoke {
            let seat = p.radius * 0.984;
            let width = p.radius * 0.016;
            let points: [Vec3; 17] = std::array::from_fn(|i| {
                let (y, x) = (PI + PI * i as f32 / 16.).sin_cos();
                Vec3::new(x * seat, y * seat, 0.)
            });
            g.tube(&points, width, 6);
            for side in [-1., 1.] {
                let axis = Vec3::X * side;
                let start = self.guard_radius - p.wire * 0.6;
                let stop = seat + width * 0.75;
                let bearing = p.radius * 0.045;
                g.lathe(
                    &[(bearing, 0.), (bearing, stop - start), (0., stop - start)],
                    12,
                    Transform::from_translation(axis * start)
                        .with_rotation(Quat::from_rotation_arc(Vec3::Y, axis)),
                );
            }
            points[8]
        } else {
            let tail = -p.depth * 0.41;
            let foot = Vec3::new(0., -p.radius * 0.915, tail);
            let end = Vec3::new(0., -self.motor_radius * 0.72, tail);
            let width = joint * 0.88;
            g.lathe(
                &[(joint * 0.72, 0.), (width, end.y - foot.y)],
                10,
                Transform::from_translation(foot),
            );
            foot
        };
        // A rounded joint encloses the angled ends of both members.
        g.lathe(
            &[
                (0., -joint),
                (joint * 0.74, -joint * 0.72),
                (joint, -joint * 0.18),
                (joint, joint * 0.18),
                (joint * 0.74, joint * 0.72),
                (0., joint),
            ],
            12,
            Transform::from_translation(anchor),
        );
        anchor
    }
}
