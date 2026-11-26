use super::*;
use crate::scene::procedural_indoor::{
    motion::IndoorMotionConfig,
    placement::{rotated_half, Footprint},
};
use std::collections::BTreeSet;

#[test]
fn portable_products_and_partitions_fit_size_extremes_and_replay() {
    use ObjectKind::*;
    let cases = [
        (
            MobileCart,
            [[0.45, 0.58, 0.32], [0.91, 1.02, 0.59], [0.60, 0.76, 0.43]],
        ),
        (
            RoomDivider,
            [[0.90, 1.05, 0.35], [1.60, 1.70, 0.49], [0.65, 0.90, 0.22]],
        ),
        (
            Kettle,
            [[0.15, 0.19, 0.22], [0.22, 0.28, 0.31], [0.24, 0.20, 0.18]],
        ),
        (
            DeskFan,
            [[0.18, 0.27, 0.15], [0.28, 0.42, 0.23], [0.18, 0.40, 0.15]],
        ),
        (
            Printer,
            [[0.36, 0.19, 0.30], [0.53, 0.29, 0.40], [0.36, 0.32, 0.30]],
        ),
    ];
    let mut errors = Vec::new();
    for (kind, sizes) in cases {
        let mut shapes = BTreeSet::new();
        for seed in 0..96 {
            for size in sizes {
                let mut o = fixture(kind, seed);
                o.size = Vec3::from_array(size);
                let a = build_object(&o);
                shapes.insert(serde_json::to_string(&program(&o)).unwrap());
                match check(&o, &a) {
                    Err(e) => errors.push(format!("{kind:?}/{seed}/{size:?}: {e}")),
                    Ok(report) => {
                        if report.max_envelope_overrun_m >= 0.0005
                            || a.bounds().0.y.abs() >= 0.0001
                            || report.triangles >= 30000
                        {
                            errors.push(format!("{kind:?}/{seed}/{size:?}: {report:?}"));
                        }
                    }
                }
                let replay = build_object(&o);
                for (key, g) in &a.parts {
                    let b = &replay.parts[key];
                    assert_eq!(g.positions, b.positions);
                    assert_eq!(g.normals, b.normals);
                    assert_eq!(g.uvs, b.uvs);
                    assert_eq!(g.indices, b.indices);
                }
            }
        }
        assert!(
            shapes.len() > 200,
            "{kind:?}: insufficient dimension-conditioned programs"
        );
    }
    assert!(
        errors.is_empty(),
        "{} invalid meshes: {:?}",
        errors.len(),
        errors.iter().take(16).collect::<Vec<_>>()
    );
}

#[test]
fn cart_contents_are_supported_separated_and_spatially_varied() {
    let mut counts = [0; 4];
    let mut poses = BTreeSet::new();
    let mut centered = 0;
    let mut total = 0;
    for seed in 0..128 {
        let o = fixture(ObjectKind::MobileCart, seed);
        let p = services::cart::parameters(&o);
        let a = build_object(&o);
        assert!(
            (a.bounds().1.y - o.size.y).abs() < 1e-6,
            "top support contract"
        );
        for level in 0..p.shelf_tops.len() - 1 {
            let (lo, hi) = p.bay(o.size, level);
            let items = services::cart::cargo(&o, &p, level);
            counts[items.len()] += 1;
            let mut occupied: Vec<Footprint> = Vec::new();
            for child in items {
                let a = build_object(&child);
                check(&child, &a).unwrap();
                assert!(a.bounds().0.y.abs() < 0.0001);
                for v in a.parts.values().flat_map(|g| &g.positions) {
                    let v = child.transform().transform_point(Vec3::from_array(*v));
                    assert!(
                        v.cmpge(lo - Vec3::splat(0.0001)).all()
                            && v.cmple(hi + Vec3::splat(0.0001)).all(),
                        "{seed}/{level}: {v:?} outside {lo:?}..{hi:?}"
                    );
                }
                let fp = Footprint {
                    center: child.position.xz(),
                    half: rotated_half(child.size.xz(), child.yaw),
                    yaw: child.yaw,
                };
                assert!(!occupied.iter().any(|v| v.overlaps(fp, 0.0079)));
                occupied.push(fp);
                total += 1;
                centered +=
                    usize::from((child.position.xz() / (hi - lo).xz()).abs().max_element() < 0.02);
                poses.insert((
                    (child.position.x * 100.).round() as i32,
                    (child.position.z * 100.).round() as i32,
                    (child.yaw * 10.).round() as i32,
                ));
            }
        }
    }
    assert!(
        counts[0] > 15 && counts[1] > 15 && counts[2] > 5,
        "occupancy {counts:?}"
    );
    assert!(
        poses.len() > 60 && centered * 10 < total,
        "poses={}, centered={centered}/{total}",
        poses.len()
    );
    if let Some(path) = std::env::var_os("ZEROVERSE_OBJECT_REPORT").map(std::path::PathBuf::from) {
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(
            path.join("cart-placement.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "provenance":crate::provenance::capture_provenance(),"carts":128,
                "bays_by_item_count":counts,"items":total,"centered_within_two_percent":centered,
                "distinct_rounded_poses":poses.len(),"pose_rounding":"1 cm X/Z and 0.1 rad yaw",
                "all_vertices_contained_and_grounded":true,"conservative_footprints_separated":true
            }))
            .unwrap(),
        )
        .unwrap();
    }
}

#[test]
fn rigid_motion_retains_cart_cargo_and_equipment_rest_meshes() {
    let policy = IndoorMotionConfig {
        cart_fraction: 1.,
        prop_fraction: 1.,
        ..Default::default()
    };
    let vertices = |a: &Assembly| {
        let mut v: Vec<_> = a
            .parts
            .values()
            .flat_map(|g| &g.positions)
            .map(|p| p.map(f32::to_bits))
            .collect();
        v.sort();
        v
    };
    for kind in [
        ObjectKind::StorageBox,
        ObjectKind::TrashCan,
        ObjectKind::MobileCart,
        ObjectKind::Kettle,
        ObjectKind::DeskFan,
        ObjectKind::Printer,
    ] {
        for seed in 0..32 {
            let mut o = fixture(kind, seed);
            if kind != ObjectKind::MobileCart {
                o.support = Some(0);
            }
            let still = build_object(&o);
            let moving = super::super::build_object_with_motion(&o, Some(&policy));
            assert!(still.joints.is_empty());
            assert_eq!(moving.joints.len(), 1, "{kind:?}/{seed}");
            assert_eq!(vertices(&still), vertices(&moving));
            assert_eq!(
                moving.joints[0].parts.len(),
                moving.parts.len(),
                "unowned cargo"
            );
        }
    }
}

#[test]
fn archived_kettle_handle_roots_and_spout_follow_body_profile() {
    let mut overhead = 0;
    for seed in 0..128 {
        let o = fixture(ObjectKind::Kettle, seed);
        let version = hospitality::kettle::FORMED_KETTLE_VERSION - 1;
        let p = hospitality::kettle::parameters_at(&o, version);
        assert!(p.hardware.is_none());
        let handle = p.handle(o.size);
        overhead += usize::from(p.overhead_handle);
        for root in [handle[0], *handle.last().unwrap()] {
            let radius = p.radius_at(root.y - p.base);
            assert!(
                root.xz().length() < radius && root.xz().length() > radius * 0.90,
                "unattached handle {seed}"
            );
        }
        let (root, tip) = p.spout();
        assert!(root.z < p.radius_at(root.y - p.base));
        assert!(tip.z > p.radius_at(tip.y - p.base) + p.spout_radius);
        // Check the actual open spout mouth: a ray from just beyond its tip must
        // travel into the bore before hitting any geometry.
        let a = build_object_at_version(&o, None, version);
        let axis = (tip - root).normalize();
        let origin = tip + axis * 0.002;
        let mut nearest = f32::INFINITY;
        for g in a.parts.values() {
            for tri in g.indices.as_chunks::<3>().0 {
                let v = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
                if let Some(t) = ray_triangle(origin, -axis, v) {
                    nearest = nearest.min(t);
                }
            }
        }
        assert!(
            nearest > root.distance(tip) * 0.25,
            "spout is capped {seed}: {nearest}"
        );
    }
    assert!((20..70).contains(&overhead));
}

#[test]
fn kettle_pouring_openings_connect_and_hardware_fits_the_installed_envelope() {
    use hospitality::kettle::{parameters, Outlet};
    let mut variants = std::collections::BTreeSet::new();
    for seed in 0..384 {
        let mut o = fixture(ObjectKind::Kettle, seed);
        // Include tall, short, broad and narrow footprints admitted by placement.
        o.size = Vec3::new(
            [0.16, 0.23][seed as usize % 2],
            [0.19, 0.30][(seed as usize / 2) % 2],
            [0.23, 0.31][(seed as usize / 4) % 2],
        );
        let p = parameters(&o);
        let hw = p.hardware.unwrap();
        variants.insert(format!("{:?}/{:?}", hw.outlet, hw.lid_grip));
        let a = build_object(&o);
        let check = check(&o, &a).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        assert!(
            check.max_envelope_overrun_m < 1e-5 && check.min_y_m.abs() < 1e-6,
            "{seed}: {check:?}"
        );
        assert!(a
            .parts
            .keys()
            .all(|(_, tag)| part_label(tag) == o.kind.class_name()));
        let first_hit = |eye: Vec3, dir: Vec3| {
            a.parts
                .values()
                .flat_map(|g| {
                    g.indices.as_chunks::<3>().0.iter().filter_map(move |t| {
                        ray_triangle(
                            eye,
                            dir,
                            t.map(|i| Vec3::from_array(g.positions[i as usize])),
                        )
                    })
                })
                .fold(f32::INFINITY, f32::min)
        };
        let top = p.base + p.height;
        if hw.outlet == Outlet::FormedLip {
            let opening = (p.radius * p.neck + p.spout_reach.min(p.radius * 1.37)) * 0.5;
            let d = first_hit(Vec3::new(0., top + 0.02, opening), Vec3::NEG_Y) - 0.02;
            assert!(
                d > p.height * 0.025 && d < p.height,
                "closed or missing jug cavity {seed}: {d}"
            );
        } else {
            let level = p.height * p.spout_root;
            let root = p.radius_at(level);
            let from = Vec3::new(0., p.base + level, root - 0.012);
            let throat = first_hit(from, Vec3::Z);
            assert!(
                throat.is_finite() && throat > 0.013,
                "blocked spout root {seed}: {throat}"
            );
            let tip = Vec3::new(0., p.base + p.height * p.spout_tip, p.spout_reach);
            let exit = hw.exit_elevation((tip.y - from.y).atan2(tip.z - p.socket_z(hw)));
            let axis = Vec3::new(0., exit.sin(), exit.cos());
            let rim_radius = a
                .parts
                .values()
                .flat_map(|g| &g.positions)
                .map(|&v| Vec3::from(v) - tip)
                .filter(|v| v.dot(axis).abs() < 1e-6 && v.length() < p.spout_radius * 1.7)
                .map(|v| v.length())
                .fold(f32::INFINITY, f32::min);
            assert!(
                rim_radius.is_finite() && rim_radius > 0.001,
                "missing open rim {seed}: {rim_radius}"
            );
            let depth = first_hit(tip + axis * 0.002, -axis) - 0.002;
            assert!(
                depth.is_finite() && depth > rim_radius,
                "obstructed lip {seed}: bore {rim_radius}, depth {depth}"
            );
        }
        // The rest of the neck is covered and the lid is seated against the rim.
        let offcenter = p.radius * p.neck * 0.62;
        let cover = first_hit(Vec3::new(offcenter, top + 0.05, 0.), Vec3::NEG_Y);
        assert!(cover < 0.05, "missing lid {seed}");
    }
    assert_eq!(variants.len(), 9);
}

#[test]
fn kettle_ports_fit_continuous_stock_sizes() {
    let mut r = stream(574, 0x4b4554544c455141);
    for _ in 0..4096 {
        let mut o = fixture(ObjectKind::Kettle, r.random());
        o.size = Vec3::new(
            r.random_range(0.16..0.23),
            r.random_range(0.19..0.30),
            r.random_range(0.23..0.31),
        );
        check(&o, &build_object(&o)).unwrap_or_else(|e| panic!("{:?}: {e}", o.size));
    }
}

#[test]
fn kettle_finish_resampling_preserves_construction_and_archived_meshes() {
    for seed in 0..16 {
        let o = fixture(ObjectKind::Kettle, seed);
        let mut alternate = o.clone();
        alternate.finish_seed = Some(seed ^ 0xb5ea);
        let p = |o| serde_json::to_value(hospitality::kettle::parameters(o).hardware).unwrap();
        assert_eq!(p(&o), p(&alternate));
        let version = hospitality::kettle::FORMED_KETTLE_VERSION;
        // Finish registration is shared by all objects and may retag batches.
        // The archived kettle itself had the same mesh on both earlier versions.
        let mesh = |v| {
            let a = build_object_at_version(&o, None, v);
            a.parts
                .into_iter()
                .map(|(k, g)| (k, g.positions, g.indices))
                .collect::<Vec<_>>()
        };
        assert_eq!(mesh(version - 1), mesh(version - 2));
    }
}

fn ray_triangle(origin: Vec3, direction: Vec3, p: [Vec3; 3]) -> Option<f32> {
    let o = origin.as_dvec3();
    let d = direction.as_dvec3();
    let p = p.map(|v| v.as_dvec3());
    let e = p[1] - p[0];
    let f = p[2] - p[0];
    let h = d.cross(f);
    let determinant = e.dot(h);
    if determinant.abs() < 1e-12 {
        return None;
    }
    let s = o - p[0];
    let q = s.cross(e);
    let u = s.dot(h) / determinant;
    let v = d.dot(q) / determinant;
    let t = f.dot(q) / determinant;
    (u >= -1e-8 && v >= -1e-8 && u + v <= 1.00000001 && t > 0.).then_some(t as f32)
}
