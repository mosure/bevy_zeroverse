use super::*;
use crate::scene::procedural_indoor::{
    objects::{audit, build_object_at_version},
    placement::rotated_half,
};
use std::collections::BTreeSet;

#[test]
fn finished_media_fits_the_real_bed_and_preserves_document_identity() {
    let mut media = BTreeSet::new();
    let mut counts = BTreeSet::new();
    let mut print_sides = BTreeSet::new();
    for seed in 0..256 {
        let mut o = audit::fixture(ObjectKind::Printer, seed);
        let mut r = stream(seed, 121);
        o.size = crate::scene::procedural_indoor::items::sizing::loose(ObjectKind::Printer, &mut r)
            .unwrap();
        let p = parameters(&o);
        let out = OutputProgram::sample(&o, &p);
        let mesh = build_object_at_version(&o, None, OUTPUT_VERSION);
        counts.insert(out.count);
        assert_eq!(
            mesh.parts
                .keys()
                .any(|(_, k)| k.split('#').next() == Some("paper")),
            out.count > 0
        );
        assert!(!mesh.parts.keys().any(|(s, _)| *s == Surface::Screen));
        assert_eq!(
            mesh.parts
                .keys()
                .any(|(s, k)| *s == Surface::Ink && k.starts_with("printer#")),
            p.screen
        );
        let Some(stock) = out.media else { continue };
        media.insert(stock);
        let half = rotated_half(out.sheet_m, out.yaw);
        let cx = p.body.x * p.slot_center;
        assert!((out.center.x - cx).abs() + half.x <= p.body.x * p.slot_fraction * 0.455);
        assert!(out.center.y - half.y >= out.tray_start_z + 2. * out.roller_radius_m);
        assert!(out.center.y + half.y <= out.tray_tip_z && out.tray_tip_z <= o.size.z * 0.5);
        let top_y = out.bed_y + out.caliper_m * out.count as f32;
        assert!(top_y < p.lower_top + p.slot_height);
        let top = mesh
            .parts
            .iter()
            .filter(|((_, k), _)| k.split('#').next() == Some("paper"))
            .flat_map(|((s, _), g)| {
                g.indices.as_chunks::<3>().0.iter().filter_map(move |tri| {
                    let v = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
                    ((v[1] - v[0]).cross(v[2] - v[0]).normalize().y > 0.99).then_some((*s, v))
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(top.len(), 2);
        let mut area = 0.;
        for (s, v) in top {
            assert_eq!(s == Surface::PrintedPaper, out.face_up);
            assert!(v
                .iter()
                .all(|p| (p.y - out.bed_y - out.caliper_m * out.count as f32).abs() < 1e-6));
            area += (v[1] - v[0]).cross(v[2] - v[0]).length() * 0.5;
        }
        assert!((area - out.sheet_m.element_product()).abs() < 1e-6);
        print_sides.insert(out.face_up);
    }
    assert_eq!(media.len(), 4);
    assert_eq!(counts.len(), 6);
    assert_eq!(print_sides.len(), 2);
}

#[test]
fn rear_mains_and_panel_replay_with_the_installed_revision() {
    use crate::scene::procedural_indoor::objects::equipment;
    let mut tested_lcd = false;
    for seed in 0..24 {
        let mut o = audit::fixture(ObjectKind::Printer, seed);
        o.position = Vec3::new(1.4, 0.83, -0.27);
        o.yaw = 1.18;
        let port = power_port(&o);
        assert_eq!(
            equipment::power_port(&o, false),
            Some(o.transform().transform_point(port))
        );
        assert_eq!(equipment::power_port(&o, true), None);
        assert_eq!(
            equipment::power_port_normal(&o, false),
            Some(o.transform().rotation * Vec3::NEG_Z)
        );
        let current = build_object_at_version(&o, None, OUTPUT_VERSION);
        let fitting = &current.parts[&(Surface::Rubber, "printer".to_owned())];
        assert!(fitting.positions.iter().any(|&v| {
            let d = Vec3::from_array(v) - port;
            d.z.abs() < 1e-6 && d.x.abs() < 0.013 && d.y.abs() < 0.008
        }));
        let previous = build_object_at_version(&o, None, OUTPUT_VERSION - 1);
        assert!(!previous.parts.keys().any(|(s, _)| *s == Surface::Ink));
        if parameters(&o).screen {
            assert!(previous.parts.keys().any(|(s, _)| *s == Surface::Screen));
            tested_lcd = true;
        }
    }
    assert!(tested_lcd);
}

#[test]
fn requested_jobs_reserve_space_for_their_original_paper_format() {
    let mut large = 0;
    let mut loaded = 0;
    for seed in 0..256 {
        let mut o = audit::fixture(ObjectKind::Printer, seed);
        o.size = Vec3::new(0.36, 0.22, 0.30);
        reserve_media(&mut o);
        let p = parameters(&o);
        let out = OutputProgram::sample(&o, &p);
        assert_eq!(out.count, p.sheet_count);
        assert!(o.size.x <= 0.56 && o.size.z <= 0.48);
        if out.count > 0 {
            loaded += 1;
            if matches!(out.media, Some("Letter" | "A4")) {
                large += 1
            }
        }
    }
    assert!(large * 3 > loaded * 2);
}
