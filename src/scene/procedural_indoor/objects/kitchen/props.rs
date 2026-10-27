//! Hollow dishware and supported kitchen supplies; bounded mesh counts per item.
use super::*;
#[derive(Serialize)]
pub struct PropProgram {
    pub style: u8,
    pub count: u32,
    pub horizontal: bool,
    pub pump: bool,
    pub basket: bool,
}
pub fn program(o: &IndoorObject) -> PropProgram {
    let mut r = stream(o.seed, 8030);
    PropProgram {
        style: r.random_range(0..5),
        count: r.random_range(2..=5),
        horizontal: r.random_bool(0.5),
        pump: r.random_bool(0.60),
        basket: r.random_bool(0.5),
    }
}

pub(super) fn dish(a: &mut Assembly, base: Vec3, size: Vec3, style: u8, count: u32, seed: u64) {
    let p = dishware::parameters(size, style, count, seed);
    dishware::build(a, Transform::from_translation(base), &p, seed);
}
pub(super) fn rack(a: &mut Assembly, base: Vec3, size: Vec3, plates: u32, seed: u64) {
    let p = supplies::rack::parameters(size, plates, seed);
    supplies::rack::build(a, Transform::from_translation(base), &p, seed);
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    let p = program(o);
    let label = o.kind.class_name();
    match o.kind {
        ObjectKind::SoapDispenser => {
            super::supplies::dispenser::build(a, Vec3::ZERO, s, p.pump, o.seed)
        }
        ObjectKind::Sponge => {
            a.box_part(
                Surface::Fabric,
                label,
                Vec3::Y * s.y * 0.38,
                Vec3::new(s.x, s.y * 0.76, s.z),
                0.009,
            );
            a.box_part(
                Surface::FabricAlt,
                label,
                Vec3::Y * s.y * 0.88,
                Vec3::new(s.x * 0.98, s.y * 0.23, s.z * 0.98),
                0.002,
            );
        }
        ObjectKind::DryingRack => {
            let program = supplies::rack::parameters(s, p.count, o.seed);
            supplies::rack::build(
                a,
                Transform::IDENTITY,
                &program,
                o.finish_seed.unwrap_or(o.seed),
            );
        }
        ObjectKind::Dishware => {
            let program = dishware::parameters(s, p.style, p.count, o.seed);
            dishware::build(
                a,
                Transform::IDENTITY,
                &program,
                o.finish_seed.unwrap_or(o.seed),
            );
        }
        ObjectKind::CuttingBoard => super::supplies::board(a, o),
        ObjectKind::KitchenTowel => super::supplies::towel(a, o),
        ObjectKind::KnifeBlock => super::supplies::knives::build(a, o),
        ObjectKind::UtensilHolder => supplies::utensils::build(a, o),
        ObjectKind::PaperTowel => {
            if p.horizontal {
                a.box_part(
                    Surface::Metal,
                    label,
                    Vec3::new(0., 0.004, -s.z * 0.35),
                    Vec3::new(s.x * 0.94, 0.008, s.z * 0.18),
                    0.002,
                );
                let r = (s.z * 0.43).min(s.y * 0.36);
                let len = s.x * 0.80;
                let y = r + 0.045;
                a.part(Surface::Paper, label).lathe(
                    &[(r, 0.), (r, len), (r * 0.24, len), (r * 0.24, 0.), (r, 0.)],
                    32,
                    Transform::from_xyz(-len * 0.5, y, 0.)
                        .with_rotation(Quat::from_rotation_z(-FRAC_PI_2)),
                );
                for x in [-1., 1.] {
                    pipe(
                        a,
                        Surface::Metal,
                        label,
                        &[
                            Vec3::new(x * s.x * 0.46, 0.006, -s.z * 0.35),
                            Vec3::new(x * s.x * 0.46, y, 0.),
                            Vec3::new(x * s.x * 0.36, y, 0.),
                        ],
                        0.006,
                    );
                }
                a.box_part(
                    Surface::Paper,
                    label,
                    Vec3::new(0., y * 0.65, s.z * 0.41),
                    Vec3::new(len * 0.98, y * 0.55, 0.001),
                    0.0001,
                );
            } else {
                let r = s.x.min(s.z) * 0.43;
                let h = s.y * 0.88;
                a.part(Surface::Wood, label)
                    .cylinder(r, 0.018, Transform::from_xyz(0., 0.009, 0.));
                a.part(Surface::Paper, label).lathe(
                    &[(r, 0.), (r, h), (r * 0.24, h), (r * 0.24, 0.), (r, 0.)],
                    32,
                    Transform::from_xyz(0., 0.019, 0.),
                );
                a.part(Surface::Chrome, label).cylinder(
                    r * 0.14,
                    s.y * 0.96,
                    Transform::from_xyz(0., s.y * 0.48, 0.),
                );
            }
        }
        ObjectKind::FruitBowl => {
            let mut bowl = dishware::parameters(Vec3::new(s.x, s.y * 0.52, s.z), 1, 1, o.seed);
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
        ObjectKind::Bread => super::supplies::bread::build(a, o),
        _ => unreachable!("non kitchen prop"),
    }
}

/// Folded around an actual handle: two fabric faces, narrow thickness and open hem.
pub(super) fn hanging_towel(a: &mut Assembly, center: Vec3, width: f32, length: f32) {
    let g = a.part(Surface::FabricAlt, "towel");
    for side in [-1., 1.] {
        for row in 0..20 {
            for col in 0..8 {
                let at = |x: usize, y: usize| {
                    let u = x as f32 / 8.;
                    let v = y as f32 / 20.;
                    center
                        + Vec3::new(
                            (u - 0.5) * width,
                            -v * length,
                            side * (0.006 + 0.003 * (u * TAU * 3.).sin() * v),
                        )
                };
                let points = [
                    at(col, row),
                    at(col + 1, row),
                    at(col + 1, row + 1),
                    at(col, row + 1),
                ];
                let n = (points[1] - points[0])
                    .cross(points[3] - points[0])
                    .normalize()
                    * side;
                let start = g.positions.len() as u32;
                for (i, p) in points.iter().enumerate() {
                    g.positions.push(p.to_array());
                    g.normals.push(n.to_array());
                    g.uvs.push([
                        (col + usize::from(i == 1 || i == 2)) as f32 * width / 8.,
                        (row + usize::from(i >= 2)) as f32 * length / 20.,
                    ]);
                }
                g.indices.extend(if side > 0. {
                    [start, start + 1, start + 2, start, start + 2, start + 3]
                } else {
                    [start, start + 2, start + 1, start, start + 3, start + 2]
                });
            }
        }
    }
}
