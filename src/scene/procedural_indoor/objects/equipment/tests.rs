use super::*;

fn object(kind: ObjectKind, size: Vec3, seed: u64) -> IndoorObject {
    IndoorObject {
        id: 0,
        kind,
        position: Vec3::ZERO,
        size,
        yaw: 0.,
        variant: 0,
        seed,
        solid: false,
        support: None,
        neighbor: false,
        interaction_target: None,
        cable_path: None,
        cable_target: None,
    }
}

#[test]
fn equipment_meshes_retain_metric_envelopes_and_finite_normals() {
    let cases = [
        (ObjectKind::Projector, Vec3::new(0.23, 0.09, 0.21)),
        (ObjectKind::PowerStrip, Vec3::new(0.22, 0.03, 0.043)),
        (ObjectKind::WallShelf, Vec3::new(0.60, 0.17, 0.20)),
        (ObjectKind::NoticeBoard, Vec3::new(0.65, 0.55, 0.020)),
        (ObjectKind::ProjectionScreen, Vec3::new(1.2, 0.8, 0.06)),
        (ObjectKind::WallSconce, Vec3::new(0.12, 0.19, 0.08)),
        (ObjectKind::CeilingPendant, Vec3::new(0.22, 0.30, 0.22)),
    ];
    for (kind, size) in cases {
        for seed in 0..96 {
            let mut o = object(kind, size, seed);
            o.variant = (seed % 4) as u32;
            let mut assembly = Assembly::default();
            build(&mut assembly, &o);
            let (lo, hi) = assembly.bounds();
            assert!(
                lo.cmpge(Vec3::new(-size.x * 0.5, 0., -size.z * 0.5) - Vec3::splat(0.001))
                    .all(),
                "{kind:?} {seed} lo={lo:?}"
            );
            assert!(
                hi.cmple(Vec3::new(size.x * 0.5, size.y, size.z * 0.5) + Vec3::splat(0.001))
                    .all(),
                "{kind:?} {seed} hi={hi:?}"
            );
            for geometry in assembly.parts.values() {
                assert!(!geometry.indices.is_empty());
                assert!(geometry
                    .positions
                    .iter()
                    .all(|&p| Vec3::from_array(p).is_finite()));
                assert!(geometry
                    .normals
                    .iter()
                    .all(|&n| (Vec3::from_array(n).length_squared() - 1.).abs() < 0.001));
                for triangle in geometry.indices.as_chunks::<3>().0 {
                    let p: [Vec3; 3] = std::array::from_fn(|i| {
                        Vec3::from_array(geometry.positions[triangle[i] as usize])
                    });
                    assert!(
                        (p[1] - p[0]).cross(p[2] - p[0]).length_squared() > 1e-20,
                        "degenerate {kind:?} {seed}"
                    );
                }
            }
        }
    }
}

#[test]
fn equipment_connections_and_emitters_use_geometry_coordinates() {
    for seed in 0..64 {
        for kind in [ObjectKind::WallSconce, ObjectKind::CeilingPendant] {
            let o = object(kind, Vec3::new(0.24, 0.40, 0.24), seed);
            let (position, color, flux, inner, outer) = fixture_photometry(&o);
            assert!(color.is_finite() && color.min_element() > 0. && flux > 300.);
            assert!(inner > 0. && outer > inner && outer < FRAC_PI_2);
            let mut a = Assembly::default();
            build(&mut a, &o);
            let emit = a
                .parts
                .iter()
                .find(|((s, _), _)| *s == Surface::Light)
                .unwrap()
                .1;
            let bottom = emit
                .positions
                .iter()
                .map(|p| p[1])
                .min_by(f32::total_cmp)
                .unwrap();
            assert!(position.y < bottom, "emitter must lie below its diffuser");
        }
        let p = object(ObjectKind::Projector, Vec3::new(0.30, 0.12, 0.25), seed);
        let mut a = Assembly::default();
        build(&mut a, &p);
        assert!(!a.parts.keys().any(|(s, _)| *s == Surface::Light));
        assert!((power_port(&p, false).unwrap().z + p.size.z * 0.48 + 0.001).abs() < 1e-6);
    }
}

#[test]
fn cable_mesh_uses_archived_centerline_and_strain_relief_bounds() {
    for seed in 0..64 {
        let mut o = object(ObjectKind::Cable, Vec3::ONE, seed);
        o.cable_path = Some(vec![
            Vec3::new(-0.20, 0.025, 0.),
            Vec3::new(-0.18, 0.023, 0.008),
            Vec3::new(0., 0.012, 0.016),
            Vec3::new(0.18, 0.023, 0.008),
            Vec3::new(0.20, 0.025, 0.),
        ]);
        let controls = o.cable_path.take().unwrap();
        o.cable_path = Some(wires::fillet_path(&controls, wires::radius(&o)).unwrap());
        let mut a = Assembly::default();
        build(&mut a, &o);
        let points = o.cable_path.as_ref().unwrap();
        let r = wires::bound_radius(&o);
        let expected_lo = points
            .iter()
            .fold(Vec3::splat(f32::INFINITY), |a, &b| a.min(b))
            - Vec3::splat(r);
        let expected_hi = points
            .iter()
            .fold(Vec3::splat(f32::NEG_INFINITY), |a, &b| a.max(b))
            + Vec3::splat(r);
        let (lo, hi) = a.bounds();
        assert!(lo.cmpge(expected_lo - Vec3::splat(1e-5)).all());
        assert!(hi.cmple(expected_hi + Vec3::splat(1e-5)).all());
    }
}

#[test]
fn monitor_mains_port_is_on_a_connected_housing_with_shared_departure_normal() {
    for seed in 0..32 {
        for size in [Vec3::new(0.29, 0.56, 0.19), Vec3::new(0.83, 0.33, 0.28)] {
            let mut o = object(ObjectKind::Monitor, size, seed);
            o.position = Vec3::new(0.7, 0.74, -0.3);
            o.yaw = 0.47;
            let port = power_port(&o, false).unwrap();
            let normal = power_port_normal(&o, false).unwrap();
            let local = o
                .transform()
                .compute_affine()
                .inverse()
                .transform_point3(port);
            assert!((local.z + size.z * 0.5 + 0.001).abs() < 1e-6);
            assert!(normal.dot(o.transform().rotation * Vec3::NEG_Z) > 0.9999);
            let assembly = super::super::build_object(&o);
            let socket = assembly
                .parts
                .iter()
                .find(|((surface, _), _)| *surface == Surface::Ink)
                .expect("monitor has a rendered mains socket")
                .1;
            let lo = socket
                .positions
                .iter()
                .map(|p| Vec3::from_array(*p))
                .fold(Vec3::splat(f32::INFINITY), Vec3::min);
            let hi = socket
                .positions
                .iter()
                .map(|p| Vec3::from_array(*p))
                .fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max);
            assert!(local.x >= lo.x && local.x <= hi.x);
            assert!(local.y >= lo.y && local.y <= hi.y);
            assert!(local.z < lo.z && lo.z - local.z < 0.002);
            let (bounds_lo, bounds_hi) = assembly.bounds();
            assert!(bounds_lo
                .cmpge(Vec3::new(-size.x * 0.5, 0., -size.z * 0.5) - Vec3::splat(0.003))
                .all());
            assert!(bounds_hi
                .cmple(Vec3::new(size.x * 0.5, size.y, size.z * 0.5) + Vec3::splat(0.003))
                .all());
        }
    }
}
