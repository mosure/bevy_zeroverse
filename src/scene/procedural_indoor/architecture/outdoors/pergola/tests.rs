use super::*;

fn site(seed: u64) -> Parcel {
    Parcel {
        center: Vec2::new(14., -11.),
        size: Vec2::new(
            2.2 + (seed % 15) as f32 * 0.30,
            2. + (seed % 9) as f32 * 0.42,
        ),
        yaw: seed as f32 * 0.23,
        kind: 3,
        seed,
    }
}

// Sample the built bearing face along both cross-beams, including between posts.
fn has_bearing(g: &Geometry, inverse: &bevy::math::Affine3A, point: Vec2, height: f32) -> bool {
    g.indices.as_chunks::<3>().0.iter().any(|tri| {
        let v = tri.map(|i| inverse.transform_point3(Vec3::from_array(g.positions[i as usize])));
        if v.iter().any(|p| (p.y - height).abs() > 0.0001) {
            return false;
        }
        let a = v[1].xz() - v[0].xz();
        let b = v[2].xz() - v[0].xz();
        let q = point - v[0].xz();
        let det = a.perp_dot(b);
        if det.abs() < 1e-8 {
            return false;
        }
        let u = q.perp_dot(b) / det;
        let w = a.perp_dot(q) / det;
        u >= -1e-4 && w >= -1e-4 && u + w <= 1.0001
    })
}

#[test]
fn roofs_land_on_beams_and_posts_and_plants_fit_at_all_installation_angles() {
    let scene = IndoorManifest::generate_with_humans(
        6,
        crate::scene::procedural_indoor::layout::IndoorLayout::Mixed,
        0.65,
        0,
        0.,
    )
    .unwrap();
    let mut types = [0; 3];
    let mut screens = 0;
    for seed in 0..72 {
        let p = site(seed);
        let plan = Program::sample(&p);
        let tf = Transform::from_xyz(p.center.x, 0., p.center.y)
            .with_rotation(Quat::from_rotation_y(p.yaw));
        let inverse = tf.compute_affine().inverse();
        let mut frame = Assembly::default();
        build(&mut frame, &p);
        let triangles = super::super::qualification::check(&frame);
        assert!((100..=3500).contains(&triangles), "{seed} {triangles}");
        assert!(frame.parts.values().flat_map(|g| &g.positions).all(|v| {
            let w = inverse.transform_point3(Vec3::from_array(*v));
            w.x.abs() <= p.size.x * 0.5 && w.z.abs() <= p.size.y * 0.5 && w.y >= 0.0279
        }));
        let label = super::super::super::super::materials::exterior::label(
            plan.stock,
            "other_structure",
            p.seed,
        );
        let beams = &frame.parts[&(plan.stock, label)];
        for z in [-plan.supports.y, plan.supports.y] {
            for i in 0..=plan.bays {
                let x = -plan.supports.x + i as f32 * plan.pitch();
                assert!(
                    has_bearing(beams, &inverse, Vec2::new(x, z), plan.beam_bottom),
                    "seed {seed}, {x} {z}"
                );
            }
        }
        let mut planting = Assembly::default();
        let installed = super::super::garden_plants(&scene, &p);
        super::super::garden(&mut planting, &p, &installed, scene.generator_version);
        for pnt in planting.parts.values().flat_map(|g| &g.positions) {
            let v = inverse.transform_point3(Vec3::from_array(*pnt));
            assert!(
                v.x.abs() < p.size.x * 0.45
                    && v.z.abs() < p.size.y * 0.22
                    && v.y < plan.headroom() - 0.02
            );
        }
        types[match plan.roof {
            Roof::Rafters => 0,
            Roof::Louvres { .. } => 1,
            Roof::Canvas { .. } => 2,
        }] += 1;
        screens += usize::from(plan.screen.is_some());
    }
    assert!(types.iter().all(|n| *n >= 8) && (12..58).contains(&screens));
}

#[test]
fn fabric_skins_follow_the_sag_contact_both_rafters_and_close_with_metric_uvs() {
    use std::collections::BTreeMap;
    let mut g = Geometry::default();
    canopy::panel(&mut g, &Transform::IDENTITY, -0.36, 0.44, 1.32, 2.74, 0.092);
    let mut edges = BTreeMap::new();
    for tri in g.indices.as_chunks::<3>().0 {
        let points = tri.map(|i| g.positions[i as usize].map(f32::to_bits));
        for j in 0..3 {
            let (a, b) = (points[j], points[(j + 1) % 3]);
            let (key, sign) = if a < b { ((a, b), 1) } else { ((b, a), -1) };
            let v = edges.entry(key).or_insert((0, 0));
            v.0 += 1;
            v.1 += sign;
        }
    }
    assert!(edges.values().all(|v| *v == (2, 0)));
    assert!((g.positions[12][1] - (2.74 - 0.092)).abs() < 1e-6);
    assert!((g.positions[0][1] - 2.74).abs() < 1e-6 && (g.positions[24][1] - 2.74).abs() < 1e-6);
    assert!(g.uvs[24][0] > 0.80 && (g.uvs[1][1] - 2.64).abs() < 1e-6);
}
