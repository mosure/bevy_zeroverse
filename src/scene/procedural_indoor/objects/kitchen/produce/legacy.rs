//! Replay for bowls through generator 229.
use super::*;

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    let p = props::program(o);
    let mut bowl = dishware::parameters_at(Vec3::new(s.x, s.y * 0.52, s.z), 1, 1, o.seed, 229);
    // A broad, flat interior floor supports the fruit arrangement.
    bowl.foot = 0.78;
    bowl.aspect = Vec2::ONE;
    if !p.basket {
        dishware::build(a, Transform::IDENTITY, &bowl, o.seed);
    } else {
        a.part(Surface::Wood, "dishware").cylinder(
            s.x.min(s.z) * 0.27,
            0.008,
            Transform::from_xyz(0., 0.004, 0.),
        );
        for height in [0.18, 0.30, 0.38] {
            let radius = 0.26 + height * 0.46;
            let points: Vec<_> = (0..=32)
                .map(|i| {
                    let t = TAU * i as f32 / 32.;
                    Vec3::new(t.cos() * s.x * radius, s.y * height, t.sin() * s.z * radius)
                })
                .collect();
            pipe(a, Surface::WoodEdge, "dishware", &points, 0.003);
        }
    }
    if p.basket {
        for i in 0..16 {
            let t = TAU * i as f32 / 16.;
            pipe(
                a,
                Surface::WoodEdge,
                "dishware",
                &[
                    Vec3::new(t.cos() * s.x * 0.26, 0.006, t.sin() * s.z * 0.26),
                    Vec3::new(t.cos() * s.x * 0.43, s.y * 0.38, t.sin() * s.z * 0.43),
                ],
                0.003,
            );
        }
    }
    for i in 0..p.count {
        let t = TAU * i as f32 / p.count as f32;
        // All fruit rests on the actual interior floor; crowded baskets
        // shrink fruit radii to retain separation between neighbours.
        let floor_radius = if p.basket {
            s.x.min(s.z) * 0.255
        } else {
            bowl.radius * bowl.foot - bowl.wall
        };
        let orbit = floor_radius * 0.50;
        let r = (orbit * (PI / p.count as f32).sin() * 0.93).min(s.y * 0.35);
        let floor = if p.basket { 0.008 } else { bowl.wall };
        let c = Vec3::new(t.cos() * orbit, floor + r * 1.10, t.sin() * orbit);
        a.part(Surface::Ceramic, &format!("food#finish{}", 100 + i % 3))
            .ellipsoid(Vec3::new(r, r * 1.10, r), Transform::from_translation(c));
        pipe(
            a,
            Surface::Wood,
            "food",
            &[c + Vec3::Y * r, c + Vec3::new(0.003, r * 1.32, 0.)],
            0.002,
        );
    }
}
