//! Service furniture construction shares the exact admitted metric envelope.
//! Counter tops remain fully usable rectangles; details never occupy prop space.
use super::*;

#[derive(Clone, Debug, serde::Serialize)]
pub struct CounterProgram {
    pub structure: u8,
    pub toe_height: f32,
    pub body_depth: f32,
    pub body_width: f32,
    pub slab_thickness: f32,
    pub leg_width: f32,
    pub shelf_height: f32,
    pub taper: f32,
    pub bay_weights: Vec<f32>,
    pub slats: usize,
    pub footrail: bool,
    pub finish: Surface,
    pub top: Surface,
}

pub fn counter_parameters(o: &IndoorObject) -> CounterProgram {
    let mut r = stream(o.seed, 0x434f_554e_5445_5247);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x434f_554e_5445_5246);
    let n = (o.size.x / r.random_range(0.40..0.72))
        .round()
        .clamp(2., 6.) as usize;
    let structure = r.random_range(0..6);
    CounterProgram {
        structure,
        toe_height: r.random_range(0.075..0.18),
        body_depth: r.random_range(0.67..0.87),
        body_width: r.random_range(0.82..0.94),
        slab_thickness: r.random_range(0.026..0.064),
        leg_width: r.random_range(0.028..0.060),
        shelf_height: r.random_range(0.14..0.35),
        taper: r.random_range(0.66..0.94),
        bay_weights: (0..n).map(|_| r.random_range(0.65..1.45)).collect(),
        slats: r.random_range(8..=22),
        footrail: r.random_bool(0.42) && !matches!(structure, 1 | 5),
        finish: [Surface::Wood, Surface::Plastic, Surface::Metal][f.random_range(0..3)],
        top: [
            Surface::Wood,
            Surface::Ceramic,
            Surface::Concrete,
            Surface::Metal,
        ][f.random_range(0..4)],
    }
}

pub(super) fn counter(a: &mut Assembly, o: &IndoorObject) {
    let p = counter_parameters(o);
    let s = o.size;
    let label = o.kind.class_name();
    let upper = s.y - p.slab_thickness;
    let width = s.x * p.body_width;
    let depth = s.z * p.body_depth;
    let t = 0.018;
    if p.structure == 1 {
        // Open preparation unit: posts carry the slab, with an actual low shelf.
        for x in [-0.48, 0.48] {
            for z in [-0.46, 0.46] {
                a.box_part(
                    p.finish,
                    label,
                    Vec3::new(x * width, upper * 0.5, z * depth),
                    Vec3::new(p.leg_width, upper, p.leg_width),
                    0.004,
                );
            }
        }
        a.box_part(
            p.finish,
            label,
            Vec3::Y * (upper * p.shelf_height),
            Vec3::new(width, 0.025, depth),
            0.003,
        );
        a.box_part(
            p.finish,
            label,
            Vec3::new(0., upper - 0.07, depth * 0.46),
            Vec3::new(width, 0.14, t),
            0.003,
        );
    } else if p.structure == 5 {
        // Trestle preparation bench: two braced frames rather than a solid
        // cabinet hidden beneath every beverage station.
        for side in [-1., 1.] {
            let x = side * width * 0.36;
            for z in [-1., 1.] {
                a.box_part(
                    Surface::Rubber,
                    label,
                    Vec3::new(x, p.leg_width * 0.25, z * depth * 0.42),
                    Vec3::new(p.leg_width * 1.1, p.leg_width * 0.5, p.leg_width * 1.1),
                    0.002,
                );
                a.part(p.finish, label).rod(
                    Vec3::new(x, p.leg_width * 0.5, z * depth * 0.42),
                    Vec3::new(x, upper - p.leg_width * 0.5, z * depth * 0.26),
                    p.leg_width * 0.5,
                );
            }
            a.box_part(
                p.finish,
                label,
                Vec3::new(x, upper - p.leg_width * 0.5, 0.),
                Vec3::new(p.leg_width, p.leg_width, depth * 0.90),
                0.003,
            );
        }
        a.box_part(
            p.finish,
            label,
            Vec3::Y * (upper * p.shelf_height),
            Vec3::new(width * 0.80, p.leg_width, depth * 0.44),
            0.004,
        );
    } else if p.structure == 3 {
        // Curved/tapered bar carcass underneath a rectangular supported worktop.
        a.part(p.finish, label).housing(
            &[
                (0., Vec2::new(width * 0.46, depth * 0.43)),
                (p.toe_height, Vec2::new(width * 0.46, depth * 0.43)),
                (upper, Vec2::new(width * 0.50, depth * 0.50 * p.taper)),
            ],
            3.5 + p.taper * 5.,
            Transform::IDENTITY,
        );
    } else {
        a.box_part(
            Surface::Rubber,
            label,
            Vec3::Y * p.toe_height * 0.5,
            Vec3::new(width * 0.95, p.toe_height, depth * 0.89),
            0.003,
        );
        a.box_part(
            p.finish,
            label,
            Vec3::Y * (upper + p.toe_height) * 0.5,
            Vec3::new(width, upper - p.toe_height, depth),
            0.004,
        );
        let total = p.bay_weights.iter().sum::<f32>();
        let mut cursor = -width * 0.5;
        for &weight in &p.bay_weights {
            let w = width * weight / total;
            let x = cursor + w * 0.5;
            cursor += w;
            // Storage and handles face the reserved staff aisle (-Z).
            a.box_part(
                p.finish,
                label,
                Vec3::new(x, (upper + p.toe_height) * 0.5, -depth * 0.5 - 0.006),
                Vec3::new(w - 0.008, upper - p.toe_height - 0.014, 0.016),
                0.003,
            );
            a.box_part(
                Surface::Metal,
                label,
                Vec3::new(x + w * 0.22, upper - 0.09, -depth * 0.5 - 0.024),
                Vec3::new((w * 0.24).min(0.14), 0.012, 0.022),
                0.003,
            );
        }
        if p.structure == 2 {
            let pitch = width / p.slats as f32;
            for i in 0..p.slats {
                a.box_part(
                    p.finish,
                    label,
                    Vec3::new(
                        -width * 0.5 + pitch * (i as f32 + 0.5),
                        (upper + p.toe_height) * 0.5,
                        depth * 0.5 + 0.007,
                    ),
                    Vec3::new(pitch * 0.64, upper - p.toe_height - 0.012, 0.020),
                    0.003,
                );
            }
        }
        if p.structure == 4 {
            // Continuous stone/timber returns, with daylight between the
            // returns and the narrower storage carcass.
            let thickness = (s.x - width) * 0.32;
            for side in [-1., 1.] {
                a.box_part(
                    p.top,
                    label,
                    Vec3::new(side * (s.x - thickness) * 0.5, upper * 0.5, 0.),
                    Vec3::new(thickness, upper, s.z),
                    0.003,
                );
            }
        }
    }
    a.box_part(
        p.top,
        label,
        Vec3::Y * (s.y - p.slab_thickness * 0.5),
        Vec3::new(s.x, p.slab_thickness, s.z),
        0.003,
    );
    if p.footrail && p.structure != 1 {
        let rail_y = p.toe_height + 0.07;
        let rail_z = s.z * 0.475;
        for x in [-0.36, 0.36] {
            a.part(Surface::Metal, label).rod(
                Vec3::new(x * s.x, rail_y, depth * 0.20),
                Vec3::new(x * s.x, rail_y, rail_z),
                0.011,
            );
        }
        a.part(Surface::Metal, label).rod(
            Vec3::new(-s.x * 0.42, rail_y, rail_z),
            Vec3::new(s.x * 0.42, rail_y, rail_z),
            0.014_f32.min(s.z * 0.023),
        );
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct HoodProgram {
    pub construction: u8,
    pub canopy_fraction: f32,
    pub chimney_fraction: Vec2,
    pub chimney_x: f32,
    pub taper: f32,
    pub filters: usize,
    pub lights: usize,
    pub finish: Surface,
}

pub fn hood_parameters(o: &IndoorObject) -> HoodProgram {
    let mut r = stream(o.seed, 0x484f_4f44_5348_4150);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x484f_4f44_4649_4e49);
    HoodProgram {
        construction: r.random_range(0..4),
        canopy_fraction: r.random_range(0.19..0.48),
        chimney_fraction: Vec2::new(r.random_range(0.20..0.48), r.random_range(0.28..0.52)),
        chimney_x: r.random_range(-0.12..0.12),
        taper: r.random_range(0.45..0.78),
        filters: r.random_range(1..=3),
        lights: r.random_range(1..=3),
        finish: [Surface::Metal, Surface::Plastic, Surface::Chrome][f.random_range(0..3)],
    }
}

pub(super) fn hood(a: &mut Assembly, o: &IndoorObject) {
    let p = hood_parameters(o);
    let s = o.size;
    let label = o.kind.class_name();
    let base = 0.018;
    let top = base + s.y * p.canopy_fraction;
    let center = Vec3::new(s.x * p.chimney_x, 0., -s.z * 0.10);
    let chimney = Vec2::new(s.x * p.chimney_fraction.x, s.z * p.chimney_fraction.y);
    match p.construction {
        0 => a.box_part(
            p.finish,
            label,
            Vec3::Y * (base + top) * 0.5,
            Vec3::new(s.x * 0.98, top - base, s.z * 0.98),
            0.010,
        ),
        1 => a.part(p.finish, label).housing(
            &[
                (base, s.xz() * 0.49),
                (base + 0.025, s.xz() * 0.49),
                (top, s.xz() * (p.taper * 0.49)),
            ],
            10.,
            Transform::IDENTITY,
        ),
        2 => {
            // Elliptical glass canopy attached to the filter/motor housing.
            a.part(Surface::Glass, label).housing(
                &[(base, s.xz() * 0.49), (base + 0.014, s.xz() * 0.49)],
                2.6,
                Transform::IDENTITY,
            );
            a.box_part(
                p.finish,
                label,
                center.with_y((top + base) * 0.5),
                Vec3::new(chimney.x * 1.2, top - base, chimney.y * 1.2),
                0.008,
            );
        }
        _ => a.part(p.finish, label).housing(
            &[
                (base, s.xz() * 0.39),
                (base + (top - base) * 0.3, s.xz() * 0.48),
                (top, s.xz() * (p.taper * 0.45)),
            ],
            2.2,
            Transform::IDENTITY,
        ),
    }
    // Overlap the real canopy volume, never leave the flue floating above it.
    a.box_part(
        p.finish,
        label,
        center.with_y((s.y + base) * 0.5),
        Vec3::new(chimney.x, s.y - base, chimney.y),
        0.005,
    );
    let filter_width = if p.construction == 2 {
        chimney.x
    } else {
        s.x * 0.72
    };
    let filter_depth = if p.construction == 2 {
        chimney.y
    } else {
        s.z * 0.55
    };
    let filter_center = if p.construction == 2 {
        center
    } else {
        Vec3::ZERO
    };
    for i in 0..p.filters {
        let w = filter_width / p.filters as f32;
        let x = filter_center.x + w * (i as f32 + 0.5) - filter_width * 0.5;
        a.box_part(
            Surface::Metal,
            label,
            Vec3::new(x, base - 0.003, filter_center.z),
            Vec3::new(w * 0.94, 0.006, filter_depth),
            0.002,
        );
        for j in 0..6 {
            a.box_part(
                Surface::Rubber,
                label,
                Vec3::new(
                    x + w * (j as f32 - 2.5) * 0.12,
                    base - 0.0065,
                    filter_center.z,
                ),
                Vec3::new(w * 0.035, 0.002, filter_depth * 0.84),
                0.,
            );
        }
    }
    for i in 0..p.lights {
        let x = filter_center.x + filter_width * ((i as f32 + 0.5) / p.lights as f32 - 0.5);
        a.part(Surface::Light, "lamp").cylinder(
            0.018,
            0.004,
            Transform::from_xyz(x, base - 0.003, filter_center.z + filter_depth * 0.36),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::objects::{audit, build_object};

    #[test]
    fn service_shapes_vary_at_fixed_envelope_and_keep_worktops_clear() {
        let root = std::path::Path::new("out/regularity_review");
        std::fs::create_dir_all(root).unwrap();
        let mut report = Vec::new();
        let kinds = [
            ObjectKind::ServiceCounter,
            ObjectKind::ExtractionHood,
            ObjectKind::FloorLamp,
        ];
        let mut sheet = image::RgbImage::from_pixel(8 * 168, 3 * 168, image::Rgb([241, 243, 246]));
        for (row, kind) in kinds.into_iter().enumerate() {
            let size = audit::fixture(kind, 0).size;
            let mut descriptors = Vec::new();
            let mut mesh_ms = Vec::new();
            let mut triangles = Vec::new();
            for seed in 0..64 {
                let mut o = audit::fixture(kind, seed);
                o.size = size;
                let now = std::time::Instant::now();
                let a = build_object(&o);
                mesh_ms.push(now.elapsed().as_secs_f64() * 1000.);
                triangles.push(audit::check(&o, &a).unwrap().triangles);
                if kind == ObjectKind::ServiceCounter {
                    let top_area: f32 =
                        a.parts
                            .values()
                            .flat_map(|g| {
                                g.indices.as_chunks::<3>().0.iter().map(move |t| {
                                    t.map(|i| Vec3::from_array(g.positions[i as usize]))
                                })
                            })
                            .filter(|p| p.iter().all(|v| (v.y - size.y).abs() < 1e-5))
                            .map(|p| ((p[1] - p[0]).cross(p[2] - p[0]).y).max(0.) * 0.5)
                            .sum();
                    assert!(top_area >= (size.x - 0.006) * (size.z - 0.006) * 0.999);
                    assert!(a.bounds().0.y.abs() < 1e-5);
                }
                if seed < 8 {
                    let thumb = crate::scene::procedural_indoor::geometry::preview::orthographic(
                        &a,
                        Vec3::Y * size.y * 0.5,
                        size.length() * 1.02,
                        168,
                        Vec3::new(0.7, 0.45, 2.),
                    );
                    image::imageops::replace(
                        &mut sheet,
                        &thumb,
                        seed as i64 * 168,
                        row as i64 * 168,
                    );
                }
                descriptors.push(audit::descriptor(&o, &a));
            }
            let representatives = audit::representatives(&descriptors, 0.015);
            assert!(
                representatives.len() >= 8,
                "{kind:?}: silhouette variety collapsed"
            );
            report.push(serde_json::json!({"kind":kind,"size_m":size,"samples":64,
                "coverage_threshold":0.015,"representatives":representatives,
                "mesh_ms":mesh_ms,"triangles":triangles}));
        }
        sheet.save(root.join("service-geometry.png")).unwrap();
        std::fs::write(root.join("service-geometry.json"), serde_json::to_vec_pretty(
            &serde_json::json!({"generator":crate::CAPTURE_ENGINE_IDENTITY,
            "scope":"fixed-envelope six-view CPU depth-map coverage, not rendered appearance or learning utility",
            "categories":report})).unwrap()).unwrap();
    }
}
