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

pub(super) fn pantry_item(a: &mut Assembly, tf: Transform, size: Vec3, style: u8) {
    let h = size.y * 0.92;
    let r = size.x.min(size.z) * 0.28;
    if style == 0 {
        a.part(Surface::PrintedPaper, "food").cuboid(
            Vec3::new(size.x * 0.67, h, size.z * 0.60),
            0.002,
            tf.mul_transform(Transform::from_translation(Vec3::Y * (h * 0.5 + 0.001))),
        );
    } else if style == 1 {
        a.part(Surface::ContainerGlass, "food").lathe(
            &[
                (0., 0.),
                (r, 0.),
                (r, h * 0.87),
                (r * 0.85, h * 0.93),
                (r * 0.74, h * 0.93),
                (r * 0.88, 0.008),
                (0., 0.008),
            ],
            24,
            tf.mul_transform(Transform::from_translation(Vec3::Y * 0.001)),
        );
        a.part(Surface::Metal, "food").cylinder(
            r * 0.91,
            h * 0.06,
            tf.mul_transform(Transform::from_translation(Vec3::Y * (h * 0.96 + 0.001))),
        );
        a.part(Surface::Terracotta, "food#finish103").cylinder(
            r * 0.85,
            h * 0.66,
            tf.mul_transform(Transform::from_translation(Vec3::Y * (h * 0.33 + 0.009))),
        );
    } else {
        a.part(Surface::Metal, "food").cylinder(
            r,
            h,
            tf.mul_transform(Transform::from_translation(Vec3::Y * (h * 0.5 + 0.001))),
        );
        a.part(Surface::PrintedPaper, "food").lathe(
            &[(r + 0.001, h * 0.12), (r + 0.001, h * 0.88)],
            24,
            tf.mul_transform(Transform::from_translation(Vec3::Y * 0.001)),
        );
    }
}
pub(super) fn dish(a: &mut Assembly, base: Vec3, size: Vec3, style: u8, count: u32) {
    dish_at(a, Transform::from_translation(base), size, style, count);
}
pub(super) fn dish_at(a: &mut Assembly, tf: Transform, size: Vec3, style: u8, count: u32) {
    let radius = size.x.min(size.z) * 0.43;
    if style <= 1 {
        let n = if style == 0 { count } else { 1 };
        let rise = if style == 0 {
            (size.y / n as f32).min(0.018)
        } else {
            size.y * 0.72
        };
        for i in 0..n {
            let profile = [
                (0., 0.),
                (radius * 0.70, 0.),
                (radius, rise * 0.80),
                (radius, rise),
                (radius * 0.94, rise),
                (radius * 0.65, rise * 0.20),
                (0., rise * 0.20),
            ];
            a.part(Surface::Ceramic, "dishware").lathe(
                &profile,
                24,
                tf.mul_transform(Transform::from_translation(
                    Vec3::Y * (i as f32 * rise + 0.002),
                )),
            );
        }
    } else {
        let r = radius * 0.62;
        let h = size.y * 0.87;
        let profile = [
            (0., 0.),
            (r * 0.84, 0.),
            (r, h),
            (r * 0.91, h),
            (r * 0.75, 0.012),
            (0., 0.012),
        ];
        let surface = if style == 4 {
            Surface::ContainerGlass
        } else {
            Surface::Ceramic
        };
        a.part(surface, "dishware").lathe(
            &profile,
            24,
            tf.mul_transform(Transform::from_translation(Vec3::Y * 0.002)),
        );
        if style == 2 {
            let points: Vec<_> = (0..=12)
                .map(|i| {
                    let t = PI * i as f32 / 12.;
                    tf.transform_point(Vec3::new(
                        r * 0.96 + (t.sin() * radius * 0.32),
                        h * (0.5 + 0.31 * t.cos()),
                        0.,
                    ))
                })
                .collect();
            pipe(a, surface, "dishware", &points, 0.005);
        }
    }
}
fn utensil(a: &mut Assembly, base: Vec3, height: f32, style: u8, lean: f32) {
    let tip = base + Vec3::new(lean, height, 0.);
    let neck = base.lerp(tip, 0.72);
    pipe(a, Surface::Wood, "kitchen_utensil", &[base, neck], 0.006);
    if style.is_multiple_of(2) {
        a.part(Surface::Metal, "kitchen_utensil").ellipsoid(
            Vec3::new(height * 0.065, height * 0.14, 0.004),
            Transform::from_translation(neck.lerp(tip, 0.55)),
        );
    } else {
        a.box_part(
            Surface::Plastic,
            "kitchen_utensil",
            neck.lerp(tip, 0.55),
            Vec3::new(height * 0.19, height * 0.22, 0.006),
            0.004,
        );
    }
}
pub(super) fn soap(a: &mut Assembly, base: Vec3, size: Vec3, pump: bool) {
    let h = size.y * 0.73;
    a.part(Surface::Plastic, "soap_dispenser").ellipsoid(
        Vec3::new(size.x * 0.43, h * 0.5, size.z * 0.38),
        Transform::from_translation(base + Vec3::Y * h * 0.5),
    );
    a.box_part(
        Surface::PrintedPaper,
        "soap_dispenser",
        base + Vec3::new(0., h * 0.51, size.z * 0.36),
        Vec3::new(size.x * 0.57, h * 0.35, 0.003),
        0.001,
    );
    if pump {
        pipe(
            a,
            Surface::Chrome,
            "soap_dispenser",
            &[
                base + Vec3::Y * h,
                base + Vec3::Y * size.y * 0.94,
                base + Vec3::new(0., size.y * 0.94, size.z * 0.43),
            ],
            size.x * 0.065,
        );
    } else {
        a.part(Surface::Plastic, "soap_dispenser").cylinder(
            size.x * 0.15,
            size.y * 0.13,
            Transform::from_translation(base + Vec3::Y * (h + size.y * 0.065)),
        );
    }
}
pub(super) fn rack(a: &mut Assembly, base: Vec3, size: Vec3, plates: u32) {
    a.box_part(
        Surface::Plastic,
        "dishware",
        base + Vec3::Y * 0.012,
        Vec3::new(size.x, 0.024, size.z),
        0.004,
    );
    for i in 0..=6 {
        let x = (i as f32 / 6. - 0.5) * size.x * 0.86;
        pipe(
            a,
            Surface::Metal,
            "dishware",
            &[
                base + Vec3::new(x, 0.032, -size.z * 0.40),
                base + Vec3::new(x, size.y * 0.65, -size.z * 0.30),
                base + Vec3::new(x, size.y * 0.65, size.z * 0.30),
                base + Vec3::new(x, 0.032, size.z * 0.40),
            ],
            0.003,
        );
    }
    for z in [-0.38, 0.38] {
        pipe(
            a,
            Surface::Metal,
            "dishware",
            &[
                base + Vec3::new(-size.x * 0.46, 0.035, size.z * z),
                base + Vec3::new(size.x * 0.46, 0.035, size.z * z),
            ],
            0.003,
        );
    }
    let r = ((size.y - 0.030) * 0.48).min(size.z * 0.40);
    for i in 0..plates.min(5) {
        a.part(Surface::Ceramic, "dishware").cylinder(
            r,
            0.007,
            Transform::from_translation(
                base + Vec3::new((i as f32 - 2.) * size.x * 0.13, r + 0.027, 0.),
            )
            .with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
        );
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    let p = program(o);
    let label = o.kind.class_name();
    match o.kind {
        ObjectKind::SoapDispenser => soap(a, Vec3::ZERO, s, p.pump),
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
        ObjectKind::DryingRack => rack(a, Vec3::ZERO, s, p.count),
        ObjectKind::Dishware => dish(a, Vec3::ZERO, s, p.style, p.count),
        ObjectKind::CuttingBoard => {
            a.box_part(Surface::Wood, label, Vec3::Y * s.y * 0.5, s, 0.008);
            for z in [-0.34, 0.34] {
                a.box_part(
                    Surface::WoodEdge,
                    label,
                    Vec3::new(0., s.y + 0.0005, s.z * z),
                    Vec3::new(s.x * 0.88, 0.001, s.z * 0.012),
                    0.0001,
                );
            }
        }
        ObjectKind::KitchenTowel => {
            // Several folded layers, with an offset hem and gentle height variation.
            for i in 0..4 {
                a.box_part(
                    Surface::FabricAlt,
                    label,
                    Vec3::new(i as f32 * s.x * 0.012, s.y * (i as f32 + 0.5) / 4., 0.),
                    Vec3::new(s.x * 0.93, s.y * 0.245, s.z * (1. - 0.015 * i as f32)),
                    0.002,
                );
            }
        }
        ObjectKind::KnifeBlock => {
            a.box_part(
                Surface::Wood,
                label,
                Vec3::Y * s.y * 0.28,
                Vec3::new(s.x * 0.88, s.y * 0.56, s.z * 0.88),
                0.008,
            );
            for i in 0..p.count {
                let x = (i as f32 - (p.count - 1) as f32 * 0.5) * s.x * 0.75 / p.count as f32;
                a.box_part(
                    Surface::Ink,
                    label,
                    Vec3::new(x, s.y * 0.563, 0.),
                    Vec3::new(s.x * 0.07, 0.002, s.z * 0.40),
                    0.0002,
                );
                a.box_part(
                    Surface::Metal,
                    label,
                    Vec3::new(x, s.y * 0.66, 0.),
                    Vec3::new(s.x * 0.025, s.y * 0.20, s.z * 0.22),
                    0.001,
                );
                a.box_part(
                    Surface::Plastic,
                    label,
                    Vec3::new(x, s.y * 0.86, 0.),
                    Vec3::new(s.x * 0.08, s.y * 0.26, s.z * 0.24),
                    0.003,
                );
            }
        }
        ObjectKind::UtensilHolder => {
            let r = s.x.min(s.z) * 0.34;
            a.part(Surface::Ceramic, label).lathe(
                &[
                    (0., 0.),
                    (r, 0.),
                    (r, s.y * 0.48),
                    (r * 0.91, s.y * 0.48),
                    (r * 0.91, 0.008),
                    (0., 0.008),
                ],
                24,
                Transform::IDENTITY,
            );
            for i in 0..p.count {
                let t = TAU * i as f32 / p.count as f32;
                utensil(
                    a,
                    Vec3::new(t.cos() * r * 0.4, 0.013, t.sin() * r * 0.4),
                    s.y * (0.83 + 0.02 * i as f32),
                    i as u8,
                    t.cos() * r * 0.22,
                );
            }
        }
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
            if !p.basket {
                dish(a, Vec3::ZERO, Vec3::new(s.x, s.y * 0.52, s.z), 1, 1);
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
                let r = s.x.min(s.z) * (0.19 * (PI / p.count as f32).sin()).min(0.13);
                let floor = if p.basket {
                    0.008
                } else {
                    0.002 + s.y * 0.52 * 0.72 * 0.20
                };
                let c = Vec3::new(t.cos() * s.x * 0.20, floor + r * 1.10, t.sin() * s.z * 0.20);
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
        ObjectKind::Bread => {
            let loaf = a.part(Surface::Terracotta, "food#finish103");
            loaf.ellipsoid(
                Vec3::new(s.x * 0.49, s.y * 0.49, s.z * 0.49),
                Transform::from_xyz(0., s.y * 0.49, 0.),
            );
            let scoring = bread_scoring(loaf, s, p.count);
            a.part(Surface::Paper, "food#finish104").append(scoring);
        }
        _ => unreachable!("non kitchen prop"),
    }
}

/// Clip thin crust markings from the real loaf triangles. Analytic ellipsoid
/// heights or flat strips can float above its tessellated surface. The 0.3 mm
/// crust layer follows each triangle exactly and avoids coincident surfaces.
fn bread_scoring(loaf: &Geometry, size: Vec3, count: u32) -> Geometry {
    let mut out = Geometry::default();
    for triangle in loaf.indices.as_chunks::<3>().0 {
        let vertices: Vec<_> = triangle
            .iter()
            .map(|&i| Vec3::from_array(loaf.positions[i as usize]))
            .collect();
        if vertices.iter().any(|v| v.y < size.y * 0.60) {
            continue;
        }
        let normal = (vertices[1] - vertices[0])
            .cross(vertices[2] - vertices[0])
            .normalize();
        for i in 0..count {
            let x = (i as f32 - (count - 1) as f32 * 0.5) * size.x * 0.12;
            let mut polygon = vertices.clone();
            for (axis, bound, sign) in [
                (0, x - size.x * 0.0075, 1.),
                (0, x + size.x * 0.0075, -1.),
                (2, -size.z * 0.22, 1.),
                (2, size.z * 0.22, -1.),
            ] {
                if polygon.is_empty() {
                    break;
                }
                let mut clipped = Vec::with_capacity(polygon.len() + 1);
                let mut previous = *polygon.last().unwrap();
                let mut previous_distance = (previous[axis] - bound) * sign;
                for &current in &polygon {
                    let distance = (current[axis] - bound) * sign;
                    if (distance >= 0.) != (previous_distance >= 0.) {
                        clipped.push(
                            previous
                                .lerp(current, previous_distance / (previous_distance - distance)),
                        );
                    }
                    if distance >= 0. {
                        clipped.push(current);
                    }
                    previous = current;
                    previous_distance = distance;
                }
                polygon = clipped;
            }
            for j in 1..polygon.len().saturating_sub(1) {
                let points = [polygon[0], polygon[j], polygon[j + 1]];
                if (points[1] - points[0])
                    .cross(points[2] - points[0])
                    .length_squared()
                    < 1e-16
                {
                    continue;
                }
                let start = out.positions.len() as u32;
                for point in points {
                    out.positions.push((point + normal * 0.0003).to_array());
                    out.normals.push(normal.to_array());
                    out.uvs.push(point.xz().to_array());
                }
                out.indices.extend([start, start + 1, start + 2]);
            }
        }
    }
    out
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
