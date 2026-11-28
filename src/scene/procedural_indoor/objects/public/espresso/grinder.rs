use super::*;
use crate::scene::procedural_indoor::materials::coffee_beans;

#[derive(Clone, Debug, Serialize)]
pub struct Grinder {
    pub fill_fraction: f32,
    pub hopper_radius_m: f32,
    pub throat_radius_m: f32,
    pub throat_height_m: f32,
    pub shoulder_height_m: f32,
    pub rim_height_m: f32,
    pub body: Surface,
    pub fork_spacing_m: f32,
    pub roundedness: f32,
}

impl Grinder {
    pub fn for_object(o: &IndoorObject) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"BurrMill"));
        let mut finish = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"MillCase"),
        );
        let radius = o.size.x.min(o.size.z * 0.72) * r.random_range(0.407..0.486);
        let throat = o.size.y * r.random_range(0.45..0.58);
        Self {
            fill_fraction: if r.random_bool(0.095) {
                0.
            } else {
                r.random_range(0.075..0.94)
            },
            hopper_radius_m: radius,
            throat_radius_m: radius * r.random_range(0.254..0.395),
            throat_height_m: throat,
            shoulder_height_m: throat + (o.size.y - throat) * r.random_range(0.27..0.55),
            rim_height_m: o.size.y - r.random_range(0.012..0.022),
            body: [
                Surface::Ink,
                Surface::Metal,
                Surface::Plastic,
                Surface::Accent,
            ][finish.random_range(0..4)],
            fork_spacing_m: r.random_range(0.031..0.045),
            roundedness: r.random_range(2.1..6.2),
        }
    }

    fn radius_at(&self, y: f32) -> f32 {
        let t = ((y - self.throat_height_m) / (self.shoulder_height_m - self.throat_height_m))
            .clamp(0., 1.);
        self.throat_radius_m + t * (self.hopper_radius_m - self.throat_radius_m)
    }

    fn fill_profile(&self) -> Vec<(f32, f32)> {
        if self.fill_fraction == 0. {
            return Vec::new();
        }
        let bottom = self.throat_height_m + 0.003;
        let top = bottom + (self.rim_height_m - 0.010 - bottom) * self.fill_fraction;
        let radius = |y| self.radius_at(y) - 0.0035;
        let mut rings = vec![(0., bottom), (radius(bottom), bottom)];
        if top > self.shoulder_height_m + 0.001 {
            rings.push((radius(self.shoulder_height_m), self.shoulder_height_m));
        }
        rings.extend([
            (radius(top), top),
            (0.42 * radius(top), top + 0.0035),
            (0., top + 0.0045),
        ]);
        rings
    }
}

pub(in super::super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = Grinder::for_object(o);
    let s = o.size;
    let half = s.xz() * 0.48;
    let deck = s.y * 0.088;
    let z = -s.z * 0.105;
    let axis = Transform::from_xyz(0., 0., z);
    a.part(Surface::Rubber, CLASS).housing(
        &[(0., half * 0.90), (0.007, half * 0.96)],
        5.,
        Transform::IDENTITY,
    );
    a.part(p.body, CLASS).housing(
        &[
            (0.007, half * 0.96),
            (0.014, half),
            (deck - 0.006, half),
            (deck, half * 0.93),
        ],
        5.,
        Transform::IDENTITY,
    );
    let pillar = Vec2::new(s.x * 0.40, s.z * 0.297);
    a.part(p.body, CLASS).housing(
        &[
            (deck, pillar * 0.86),
            (deck + 0.023, pillar * 0.91),
            (p.throat_height_m - 0.037, pillar),
            (p.throat_height_m - 0.019, pillar * 0.89),
        ],
        p.roundedness,
        axis,
    );
    a.part(Surface::Metal, STEEL).lathe(
        &[
            (0., p.throat_height_m - 0.020),
            (p.throat_radius_m + 0.013, p.throat_height_m - 0.020),
            (p.throat_radius_m + 0.012, p.throat_height_m + 0.001),
            (p.throat_radius_m, p.throat_height_m + 0.001),
        ],
        24,
        axis,
    );
    let ro = p.hopper_radius_m;
    let rn = p.throat_radius_m;
    let wall = 0.0018;
    a.part(Surface::ContainerGlass, CLASS).lathe(
        &[
            (rn, p.throat_height_m),
            (ro, p.shoulder_height_m),
            (ro, p.rim_height_m),
            (ro - wall, p.rim_height_m),
            (ro - wall, p.shoulder_height_m + wall),
            (rn - wall, p.throat_height_m),
        ],
        32,
        axis,
    );
    let fill = p.fill_profile();
    if !fill.is_empty() {
        a.part(Surface::Terracotta, coffee_beans::LABEL)
            .lathe(&fill, 32, axis);
    }
    a.part(Surface::Ink, CLASS).lathe(
        &[
            (0., p.rim_height_m),
            (ro + 0.0015, p.rim_height_m),
            (ro + 0.0015, s.y - 0.006),
            (ro * 0.93, s.y),
            (0., s.y),
        ],
        32,
        axis,
    );
    let chute = Vec3::new(0., p.throat_height_m - 0.058, s.z * 0.163);
    let nose = chute + Vec3::new(0., -0.032, s.z * 0.153);
    a.part(Surface::Metal, STEEL).rod(chute, nose, 0.014);
    let fork_y = (deck + nose.y) * 0.55;
    for sign in [-1., 1.] {
        let x = sign * p.fork_spacing_m * 0.5;
        a.part(Surface::Chrome, CLASS).tube(
            &[
                Vec3::new(x, fork_y, s.z * 0.129),
                Vec3::new(x, fork_y, s.z * 0.30),
                Vec3::new(x + sign * 0.005, fork_y + 0.006, s.z * 0.376),
            ],
            0.0035,
            8,
        );
    }
    button(
        a,
        Vec3::new(0., p.throat_height_m - 0.081, s.z * 0.174),
        0.008,
        Surface::Ink,
    );
    a.box_part(
        Surface::Rubber,
        CLASS,
        power_port(o) + Vec3::Z * 0.0035,
        Vec3::new(0.016, 0.012, 0.009),
        0.002,
    );
}

pub(crate) fn power_port(o: &IndoorObject) -> Vec3 {
    Vec3::new(0., o.size.y * 0.045, -o.size.z * 0.482)
}
