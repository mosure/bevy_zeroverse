use super::*;
use crate::scene::procedural_indoor::{envelope::tests::ray_triangle, layout::IndoorLayout};

fn hits(mesh: &Geometry, origin: Vec3, direction: Vec3) -> Vec<f32> {
    let mut hits: Vec<_> = mesh
        .indices
        .as_chunks::<3>()
        .0
        .iter()
        .filter_map(|t| {
            ray_triangle(
                origin,
                direction,
                t.map(|i| Vec3::from(mesh.positions[i as usize])),
            )
        })
        .collect();
    hits.sort_by(f32::total_cmp);
    hits.dedup_by(|a, b| (*a - *b).abs() < 1e-5);
    hits
}

#[test]
fn adjoining_panes_replace_wall_stock_and_close_their_openings_at_the_archived_recess() {
    let mut apertures = 0;
    let mut high_sills = 0;
    let mut closed = 0;
    let mut edges = std::collections::BTreeSet::new();
    for seed in 44300..44396 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let e = scene.envelope.as_ref().unwrap();
        let elevation = scene.adjoining_elevation.as_ref().unwrap();
        let Some(w) = &elevation.windows else {
            closed += 1;
            continue;
        };
        w.validate(&scene).unwrap();
        let (p, q) = w.endpoints(&scene).unwrap();
        let frame = polygon::wall_frame(p, q);
        let outward = frame.rotation * Vec3::NEG_Z;
        edges.insert((outward.x.round() as i8, outward.z.round() as i8));
        let mut a = Assembly::default();
        perimeter(
            &mut a,
            &scene,
            e,
            &scene.adjoining_space().footprint(scene.room_size),
        );
        let plaster = &a.parts[&(Surface::Paint, "wall#exterior".into())];
        for o in &w.facade.openings {
            apertures += 1;
            high_sills += usize::from(o.min.y > 1.5);
            let label = format!(
                "window#glazing{}#exterior",
                o.glazing.as_ref().unwrap().identity
            );
            let pane = &a.parts[&(Surface::Glass, label)];
            for (lo, hi) in w.facade.pane_rectangles(o) {
                assert!(hi.y - lo.y < 3.6, "{seed}: oversize undivided glass unit");
                let x = lo.lerp(hi, 0.43);
                let from = frame.transform_point(x.extend(0.23));
                assert!(
                    hits(plaster, from, outward).is_empty(),
                    "{seed}: uncut opening"
                );
                let t = hits(pane, from, outward);
                assert_eq!(t.len(), 2, "{seed}: missing glass face");
                assert!((t[1] - t[0] - o.glazing.as_ref().unwrap().thickness_m).abs() < 1e-5);
                assert!(t[0] > 0.23 && t[1] < 0.23 + e.wall_thickness());
            }
            let y = (o.min.y + o.max.y) * 0.5;
            let jamb =
                frame.transform_point(Vec3::new(o.min.x + 0.09, y, -0.5 * e.wall_thickness()));
            let side = hits(plaster, jamb, frame.rotation * Vec3::NEG_X);
            assert!((side[0] - 0.09).abs() < 1e-5, "{seed}: open jamb");
            let below = frame.transform_point(Vec3::new((o.min.x + o.max.x) * 0.5, 0.21, 0.23));
            let t = hits(plaster, below, outward);
            assert_eq!(t.len(), 2);
            assert!((t[0] - 0.23).abs() < 1e-5 && (t[1] - t[0] - e.wall_thickness()).abs() < 1e-5);
        }
    }
    assert!(
        (20..64).contains(&closed) && apertures >= 45 && high_sills > 4 && edges.len() >= 3,
        "{closed} sealed, {apertures} openings, {high_sills} clerestories, {edges:?}"
    );
}

#[test]
fn the_skirting_is_one_closed_run_at_both_kinds_of_corner() {
    use crate::scene::procedural_indoor::layout::{AdjoiningReturn, AdjoiningSpace};
    use std::collections::BTreeMap;
    let trim = (0..100)
        .map(TrimProgram::sample)
        .find(|t| t.base.is_some())
        .unwrap();
    for positive_x in [false, true] {
        let bay = AdjoiningSpace {
            depth_m: 2.7,
            side_return: Some(AdjoiningReturn {
                positive_x,
                split_x_m: 0.7,
                extra_depth_m: 1.4,
            }),
            ..default()
        };
        let size = Vec3::new(8., 3., 7.);
        let mut a = Assembly::default();
        skirting::build(
            &mut a,
            &bay.footprint(size),
            trim.joint_m,
            trim.base.as_ref().unwrap(),
            None,
            0,
        );
        let mesh = a.parts.values().next().unwrap();
        let nearest = mesh
            .positions
            .iter()
            .map(|p| p[2])
            .fold(f32::INFINITY, f32::min);
        assert!((nearest - size.z * 0.5 - SHARED_WALL_THICKNESS_M - 0.002).abs() < 1e-6);
        // Spatial welding ignores the necessary normal and UV splits along a
        // mitre. Every edge must occur twice with opposite orientation.
        let mut incidence = BTreeMap::new();
        for t in mesh.indices.as_chunks::<3>().0 {
            let p = t.map(|i| Vec3::from(mesh.positions[i as usize]));
            let cross = (p[1] - p[0]).cross(p[2] - p[0]);
            assert!(cross.is_finite() && cross.length_squared() > 1e-14);
            assert!(cross.dot(Vec3::from(mesh.normals[t[0] as usize])) > 0.);
            let pos = p.map(|v| v.to_array().map(|x| x.to_bits()));
            for i in 0..3 {
                let (a, b) = (pos[i], pos[(i + 1) % 3]);
                let (pair, sign) = if a < b { ((a, b), 1) } else { ((b, a), -1) };
                let counter = incidence.entry(pair).or_insert((0, 0));
                counter.0 += 1;
                counter.1 += sign;
            }
        }
        assert!(incidence.values().all(|v| *v == (2, 0)));
    }
}
