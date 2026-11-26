use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object, part_label};
use std::collections::BTreeSet;

// Nearest intersection with the actual emitted triangles, double sided.
fn distance(g: &Geometry, origin: Vec3, dir: Vec3) -> Option<f32> {
    g.indices
        .as_chunks::<3>()
        .0
        .iter()
        .filter_map(|t| {
            let [a, b, c] = t.map(|i| Vec3::from_array(g.positions[i as usize]));
            let e = b - a;
            let f = c - a;
            let h = dir.cross(f);
            let det = e.dot(h);
            if det.abs() < 1e-9 {
                return None;
            }
            let v = origin - a;
            let u = v.dot(h) / det;
            let cross = v.cross(e);
            let w = dir.dot(cross) / det;
            let hit = f.dot(cross) / det;
            (u >= 0. && w >= 0. && u + w <= 1. && hit > 1e-5).then_some(hit)
        })
        .min_by(f32::total_cmp)
}

#[test]
fn strength_racks_keep_open_sightlines_and_support_every_bar_and_pair() {
    let mut forms = BTreeSet::new();
    let mut stocks = BTreeSet::new();
    let mut loads = BTreeSet::new();
    let mut widths = BTreeSet::new();
    for seed in 1200..1296 {
        let mut o = audit::fixture(ObjectKind::PowerRack, seed);
        o.finish_seed = Some(seed + 91);
        let q = PowerRack::for_object(&o);
        let mesh = build_object(&o);
        let check = audit::check(&o, &mesh).unwrap();
        assert!(
            check.triangles < 9000 && check.batches <= 8 && check.max_envelope_overrun_m < 0.0002
        );
        assert!(mesh
            .parts
            .keys()
            .all(|(_, l)| part_label(l) == "fitness_equipment"));
        forms.insert(q.full_cage);
        stocks.insert(format!("{:?}", q.plate_stock));
        widths.insert((q.size.x * 100.).round() as i32);
        loads.insert(q.plate_pairs_kg.iter().filter(|&&kg| kg > 0).count());
        assert!(q.safety_pin < q.cup_pin && q.cup_pin < q.holes);
        assert!(q.plate_pairs_kg.windows(2).all(|p| p[0] >= p[1]));
        let outside = q.discs(|x, r, w, _| {
            assert!(
                x - w * 0.5 >= INNER_SLEEVE + 0.029
                    && x + w * 0.5 < q.size.x * 0.5
                    && (0.115..=0.225).contains(&r)
            );
        });
        assert!(outside + 0.033 < q.size.x * 0.5);
        let bar = q.bar_center();
        let (rubber, l) = Finish::Rubber.key();
        for side in [-1., 1.] {
            let p = bar + Vec3::X * (side * q.upright_span * 0.5);
            let contact = distance(&mesh.parts[&(rubber, l.to_owned())], p, -Vec3::Y).unwrap();
            assert!(
                (contact - q.shaft_radius()).abs() < 1e-5,
                "unsupported shaft for {seed}"
            );
        }
        let y = (bar.y + q.size.y - 0.12) * 0.5;
        let ray = Vec3::new(0., y, q.size.z + 0.5);
        assert!(mesh
            .parts
            .values()
            .all(|g| distance(g, ray, -Vec3::Z).is_none()));
        let mut sees_bar = false;
        q.view_boxes(|p, h| {
            assert!(!(p.x.abs() < h.x && (p.y - y).abs() < h.y));
            sees_bar |= p.x.abs() < h.x && (p.y - bar.y).abs() < h.y;
        });
        assert!(sees_bar);
        // An unused adjustment hole sees through both faces of the steel post.
        let mut post = Geometry::default();
        mesh::upright(&mut post, q, 0.);
        let p = Vec3::new(0., q.pin_y(1), q.work_z() + 0.25);
        assert!(distance(&post, p, -Vec3::Z).is_none());
        assert!(distance(&post, p + Vec3::X * 0.016, -Vec3::Z).is_some());
    }
    assert_eq!(
        (forms.len(), stocks.len(), loads.len(), widths.len()),
        (2, 3, 4, 2)
    );
}

#[test]
fn strength_rack_archive_replays_hardware_and_material_slots() {
    let o = audit::fixture(ObjectKind::PowerRack, 356);
    let replay: IndoorObject = serde_json::from_value(serde_json::to_value(&o).unwrap()).unwrap();
    let original = build_object(&o);
    let again = build_object(&replay);
    assert_eq!(original.parts.len(), again.parts.len());
    for (k, g) in &original.parts {
        let b = &again.parts[k];
        assert_eq!(g.positions, b.positions);
        assert_eq!(g.indices, b.indices);
        assert_eq!(g.normals, b.normals);
        assert_eq!(g.uvs, b.uvs);
    }
}
