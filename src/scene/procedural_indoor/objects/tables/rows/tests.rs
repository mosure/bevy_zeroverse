use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object};

#[test]
fn one_order_shares_its_construction_but_takes_individual_surface_cuts() {
    let mut substrates = std::collections::BTreeSet::new();
    for product in 0..96 {
        let mut a = audit::fixture(ObjectKind::Table, product + 1000);
        a.variant = SHARED_ROW_VARIANT;
        a.finish_seed = Some(product);
        let mut b = a.clone();
        b.seed += 2000;
        let (pa, pb) = (parameters(&a), parameters(&b));
        assert_eq!(
            serde_json::to_value(&pa).unwrap(),
            serde_json::to_value(&pb).unwrap()
        );
        substrates.insert(pa.top_surface);
        if let (
            top::TopStock::Sheet { label, cut: x, .. },
            top::TopStock::Sheet {
                label: other,
                cut: y,
                ..
            },
        ) = (
            top::TopStock::sample(&a, &pa, REGISTERED_SHEET_VERSION),
            top::TopStock::sample(&b, &pb, REGISTERED_SHEET_VERSION),
        ) {
            assert_eq!(label, other);
            assert_ne!(x.face_origin_m, y.face_origin_m);
            assert_eq!((x.edge, x.heading_rad), (y.edge, y.heading_rad));
        }
        a.variant = ORIGINAL_SHARED_ROW_VARIANT;
        a.kind = ObjectKind::Desk;
        a.size.x = 5.;
        let old = serde_json::to_value(parameters(&a)).unwrap();
        a.finish_seed = None;
        assert_eq!(old, serde_json::to_value(parameters(&a)).unwrap());
        assert_eq!(support_bay_count(&a), 0);
    }
    assert_eq!(substrates.len(), 4);
}

/// Probe the actual horizontal faces of a batch in table-local metres.
fn face_at(a: &Assembly, y: f32, point: Vec2) -> bool {
    a.parts.values().any(|g| {
        g.indices.as_chunks::<3>().0.iter().any(|tri| {
            let vertices = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
            if vertices.iter().any(|p| (p.y - y).abs() > 0.00001) {
                return false;
            }
            let [v, w, z] = vertices.map(|p| p.xz());
            if (w - v).perp_dot(z - v).abs() < 1e-10 {
                return false;
            }
            let edges = [(v, w), (w, z), (z, v)].map(|(a, b)| (b - a).perp_dot(point - a));
            edges.iter().all(|x| *x >= -1e-7) || edges.iter().all(|x| *x <= 1e-7)
        })
    })
}

#[test]
fn splices_land_on_real_crossrails_and_the_whole_frame_reaches_the_floor() {
    let mut styles = std::collections::BTreeSet::new();
    for product in (0..64).chain([6461441224796484456]) {
        let kind = if product < 64 && product % 2 == 0 {
            ObjectKind::Desk
        } else {
            ObjectKind::Table
        };
        let mut o = audit::fixture(kind, product + 100);
        o.variant = SHARED_ROW_VARIANT;
        o.finish_seed = Some(product);
        for (x, y, z) in [
            (1.7, 0.76, 0.53),
            (0.72, 0.76, 4.7),
            (7.2, 0.76, 0.60),
            (2.33453, 0.74646, 0.90805), // cafeteria 50820, instance 3
        ] {
            o.size = Vec3::new(x, y, z);
            let p = parameters(&o);
            let count = support_bay_count(&o);
            let (pitch, pieces) = row_stock_dimensions(&o, &p, count);
            assert!(p.workstation.is_none() && count >= 2 && pitch <= p.span_pitch + 1e-6);
            if x.max(z) < 2.42 {
                assert_eq!(pieces, 1);
            } else {
                assert!(pieces >= 2 && count >= 3);
            }
            styles.insert(p.bay_style);
            let outline = outline(&o);
            let mut frame = Assembly::default();
            build_structure(
                &mut frame,
                &o,
                &p,
                &outline,
                bevy_zeroverse_capture::GENERATOR_VERSION,
            );
            let bottom = o.size.y - p.top_thickness - 0.012;
            assert!(frame.bounds().0.y.abs() < 1e-5);
            let (axis, across) = if x > z {
                (Vec2::X, Vec2::Y)
            } else {
                (Vec2::Y, Vec2::X)
            };
            let mut leaf = Assembly::default();
            top(
                &mut leaf,
                &o,
                &p,
                &outline,
                &top::TopStock::sample(&o, &p, REGISTERED_SHEET_VERSION),
            );
            for joint in joints(&o, &p) {
                for side in [-0.13, 0.13] {
                    let middle = axis * joint + across * (x.min(z) * side);
                    assert!(
                        face_at(&frame, bottom, middle),
                        "unsupported seam {product}/{joint}"
                    );
                    assert!(
                        !face_at(&leaf, o.size.y, middle),
                        "covered seam {product}/{joint}"
                    );
                    // At most a 1 mm half-gap and the selected edge bevel.
                    let limit = 0.001 + p.edge_bevel + 0.0001;
                    for dx in [-limit, limit] {
                        assert!(
                            face_at(&leaf, o.size.y, middle + axis * dx),
                            "missing worktop at {product}/{joint}"
                        );
                    }
                }
            }
            let built = build_object(&o);
            audit::check(&o, &built).unwrap();
            let clearance = Clearance::new(&o);
            for g in frame.parts.values() {
                for point in &g.positions {
                    let point = Vec3::from_array(*point);
                    assert!(clearance.hits_capsule(point, point, 0.00005));
                }
            }
        }
    }
    assert_eq!(styles.len(), 6);
}
