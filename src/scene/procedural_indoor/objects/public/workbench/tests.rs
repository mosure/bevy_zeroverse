use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object, part_label};

/// Intersect the top mesh in plan without assuming which diagonal the tessellator emits.
fn covers(g: &Geometry, q: Vec2) -> bool {
    g.indices.as_chunks::<3>().0.iter().any(|tri| {
        let [a, b, c] = tri.map(|i| Vec3::from_array(g.positions[i as usize]).xz());
        let area = (b - a).perp_dot(c - a);
        if area.abs() < 1e-10 {
            return false;
        }
        let u = (q - a).perp_dot(c - a) / area;
        let v = (b - a).perp_dot(q - a) / area;
        u >= -1e-6 && v >= -1e-6 && u + v <= 1. + 1e-6
    })
}

#[test]
fn workbench_mesh_fits_measured_fixture_and_has_open_bores() {
    let mut surfaces = [0; 3];
    let mut rows = [0; 2];
    let mut hands = [0; 2];
    for seed in 0..128 {
        let o = audit::fixture(ObjectKind::Workbench, seed);
        let p = Bench::sample(&o);
        surfaces[p.surface as usize] += 1;
        let a = build_object(&o);
        let review = audit::check(&o, &a).unwrap_or_else(|e| panic!("bench {seed}: {e}"));
        assert!(review.triangles < 9000);
        assert!(a
            .parts
            .keys()
            .all(|(_, label)| part_label(label) == "workbench"));
        let stock = audit::fixture(ObjectKind::TimberOffcut, seed);
        audit::check(&stock, &build_object(&stock)).unwrap();
        let Some(v) = p.vise else {
            continue;
        };
        hands[usize::from(v.center_x_m > 0.)] += 1;
        rows[usize::from(p.two_bore_rows)] += 1;
        let mut slab = Assembly::default();
        top::build(&mut slab, &o, &p, &mut TimberCuts::new(seed, 0));
        let g = slab.part(Surface::Wood, "workbench");
        let d = p.deck(&o);
        let count = p.bores_per_row(&o);
        for strip in [1, p.laminations - 2] {
            let z = d.min.y + d.height() * (strip as f32 + 0.5) / p.laminations as f32;
            for j in [0, count / 2, count - 1] {
                let x = (j as f32 + 0.5 - count as f32 * 0.5) * p.bore_pitch_m;
                assert_eq!(covers(g, Vec2::new(x, z)), !p.bored_strip(strip));
                assert!(covers(g, Vec2::new(x + p.bore_radius_m + 0.003, z)));
            }
        }
    }
    assert!(
        surfaces.iter().all(|n| *n >= 12)
            && rows.iter().all(|n| *n >= 12)
            && hands.iter().all(|n| *n >= 10)
    );
}

#[test]
fn workbench_jaws_require_two_sided_contact_and_deck_rejects_empty_space() {
    let o = (0..32)
        .map(|s| audit::fixture(ObjectKind::Workbench, s))
        .find(|o| Bench::sample(o).vise.is_some())
        .unwrap();
    let p = Bench::sample(&o);
    let v = p.vise.unwrap();
    let mut piece = audit::fixture(ObjectKind::TimberOffcut, 0);
    piece.size = Vec3::new(0.5, 0.14, v.opening_m);
    piece.yaw = o.yaw;
    piece.position = o.transform().transform_point(Vec3::new(
        v.center_x_m,
        Bench::clamped_bottom(&o),
        p.front_z_m + v.opening_m * 0.5,
    ));
    assert!(p.grips(&o, &piece));
    for bad in [
        IndoorObject {
            size: piece.size.with_z(v.opening_m - 0.007),
            ..piece.clone()
        },
        IndoorObject {
            position: piece.position + Vec3::Y * 0.08,
            ..piece.clone()
        },
        IndoorObject {
            yaw: piece.yaw + 0.13,
            ..piece.clone()
        },
        IndoorObject {
            kind: ObjectKind::Notebook,
            ..piece.clone()
        },
    ] {
        assert!(!p.grips(&o, &bad));
    }
    let n = p.bores_per_row(&o);
    let d = p.deck(&o);
    let hole = Vec2::new(
        (0.5 - n as f32 * 0.5) * p.bore_pitch_m,
        d.min.y + d.height() * (p.laminations as f32 - 1.5) / p.laminations as f32,
    );
    assert!(!p.supports_top(&o, hole - 0.002, hole + 0.002, 0.));
    assert!(p.supports_top(&o, hole - 0.035, hole + 0.035, 0.));
    assert!(!p.supports_top(
        &o,
        Vec2::new(-0.05, p.front_z_m + 0.005),
        Vec2::new(0.05, p.front_z_m + 0.015),
        0.
    ));
}

#[test]
fn workshop_layout_keeps_operator_space_and_independent_supported_stock() {
    use crate::scene::procedural_indoor::{
        layout::{IndoorLayout, IndoorManifest, SupportPlane},
        validation::validate_layout,
    };
    let (mut shops, mut multi, mut clamped, mut loose) = (0, 0, 0, 0);
    for seed in 0..80 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Workshop, 0.65, 0, 0.25)
                .unwrap();
        validate_layout(&scene).unwrap();
        if seed < 3 {
            assert_eq!(
                scene,
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Workshop, 0.65, 0, 0.25)
                    .unwrap()
            );
        }
        let stations = scene
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::Workbench)
            .collect::<Vec<_>>();
        shops += usize::from(!stations.is_empty());
        multi += usize::from(stations.len() > 1);
        let (lo, hi) = scene.primary_room_bounds();
        for station in stations {
            let at = station.position.xz();
            assert!(at.cmpgt(lo).all() && at.cmplt(hi).all());
            assert!(scene.placement_clear(station, 0.));
            for obj in scene
                .objects
                .iter()
                .filter(|o| o.support == Some(station.id))
            {
                assert!(scene.prop_clear(obj, station, 0.));
                assert!(
                    scene.clutter.active(station.seed),
                    "clear workbench received {:?}",
                    obj.kind
                );
                if obj.kind == ObjectKind::TimberOffcut {
                    match scene.support_plane(obj) {
                        Some(SupportPlane::Vise) => {
                            clamped += 1;
                            assert!(Bench::sample(station).grips(station, obj));
                        }
                        Some(SupportPlane::Top) => loose += 1,
                        other => panic!("invalid stock support {other:?}"),
                    }
                }
            }
        }
    }
    assert!(
        shops > 30 && shops < 78 && multi > 5 && clamped > 4 && loose > 10,
        "{shops} shops, {multi} multiple, {clamped} gripped and {loose} loose workpieces"
    );
}
