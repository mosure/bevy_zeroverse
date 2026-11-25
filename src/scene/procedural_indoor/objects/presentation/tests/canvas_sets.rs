use super::*;
use crate::scene::procedural_indoor::materials::art::frame_parameters;

#[test]
fn canvas_sets_fit_the_mount_and_expose_real_wall_between_panels() {
    use artwork::PanelAxis;
    let mut counts = BTreeSet::new();
    let mut axes = [0; 2];
    for seed in 0..96 {
        for size in [
            Vec3::new(1.85, 0.90, 0.035),
            Vec3::new(0.82, 1.90, 0.062),
            Vec3::new(0.55, 0.22, 0.020),
        ] {
            let mut o = audit::fixture(ObjectKind::WallArt, seed);
            o.size = size;
            o.variant = if seed % 2 == 0 { 6 } else { 10 };
            let set = artwork::canvas_set(&o).unwrap();
            counts.insert(set.panels.len());
            let horizontal = set.axis == PanelAxis::Horizontal;
            axes[usize::from(horizontal)] += 1;
            let axis = if horizontal { Vec2::X } else { Vec2::Y };
            let mesh = build_object(&o);
            audit::check(&o, &mesh).unwrap();
            let (lo, hi) = mesh.bounds();
            assert!(lo.abs_diff_eq(Vec3::new(-0.5 * size.x, 0., -0.5 * size.z), 2e-6));
            assert!(hi.abs_diff_eq(Vec3::new(0.5 * size.x, size.y, 0.5 * size.z), 2e-6));
            for panel in &set.panels {
                assert!((panel.max_m - panel.min_m).min_element() >= 0.22 - 2e-6);
                let center = (panel.max_m + panel.min_m) * 0.5;
                let (s, t) = ray_hit(&mesh, center.extend(1.), Vec3::NEG_Z).unwrap();
                assert_eq!(s, Surface::Art);
                assert!(
                    (1. - t - (size.z * 0.5 - frame_parameters(&o).picture_inset)).abs() < 2e-6
                );
            }
            for pair in set.panels.windows(2) {
                let start = axis.dot(pair[0].max_m);
                let end = axis.dot(pair[1].min_m);
                assert!((end - start - set.gap_m).abs() < 2e-6);
                let mid = (start + end) * 0.5;
                let gap = if horizontal {
                    Vec2::new(mid, size.y * 0.5)
                } else {
                    Vec2::new(0., mid)
                };
                let at_front = gap.extend(size.z * 0.5);
                for shear in [-0.25, 0., 0.25] {
                    let towards_eye =
                        (Vec3::Z + (axis * (shear * set.gap_m / size.z)).extend(0.)).normalize();
                    assert!(ray_hit(&mesh, at_front + towards_eye, -towards_eye).is_none());
                }
            }
        }
    }
    assert_eq!(counts, BTreeSet::from([2, 3, 4]));
    assert!(axes.iter().all(|&n| n > 40));
}

#[test]
fn all_panel_fronts_and_returns_register_to_one_master_image() {
    let mut wrapped = 0;
    for seed in 0..96 {
        let mut o = audit::fixture(ObjectKind::WallArt, seed);
        o.size = Vec3::new(1.9, 1.4, 0.05);
        o.variant = if seed % 2 == 0 { 10 } else { 6 };
        let p = frame_parameters(&o);
        let set = artwork::canvas_set(&o).unwrap();
        let a = build_object(&o);
        let (_, art) = a
            .parts
            .iter()
            .find(|((s, _), _)| *s == Surface::Art)
            .unwrap();
        let quads = if p.image_wrap { 5 } else { 1 };
        assert_eq!(art.positions.len(), set.panels.len() * quads * 4);
        let inset = p.border_width + p.shadow_gap;
        let min = Vec2::new(-0.5 * o.size.x, 0.) + inset;
        let span = o.size.xy() - 2. * inset;
        for (i, panel) in set.panels.iter().enumerate() {
            let base = (i * quads + quads - 1) * 4;
            for j in 0..4 {
                let pos = Vec3::from_array(art.positions[base + j]).xy();
                assert!(pos.cmpge(panel.min_m).all() && pos.cmple(panel.max_m).all());
                let at = (pos - min) / span;
                let uv = Vec2::from_array(art.uvs[base + j]);
                assert!(uv.abs_diff_eq(Vec2::new(at.x, 1. - at.y), 2e-6));
                if p.image_wrap {
                    let edge = (i * quads + j) * 4;
                    assert_eq!(art.uvs[edge + 3], art.uvs[base + j]);
                    assert_eq!(art.uvs[edge + 2], art.uvs[base + (j + 1) % 4]);
                    assert_ne!(art.uvs[edge], art.uvs[edge + 3]);
                }
            }
        }
        wrapped += usize::from(p.image_wrap);
    }
    assert!(wrapped > 15 && wrapped < 85);
}

#[test]
fn archived_single_frames_stay_single_and_small_mounts_do_not_split() {
    let mut sets = 0;
    for seed in 0..512 {
        let mut o = audit::fixture(ObjectKind::WallArt, seed);
        let installed = artwork::canvas_set(&o);
        let restored = serde_json::from_slice(&serde_json::to_vec(&o).unwrap()).unwrap();
        assert_eq!(installed, artwork::canvas_set(&restored));
        o.finish_seed = Some(555);
        assert_eq!(installed, artwork::canvas_set(&o));
        sets += usize::from(installed.is_some());
        o.variant = (seed % 6) as u32;
        assert!(artwork::canvas_set(&o).is_none());
        o.size = Vec3::new(0.27, 0.34, 0.025);
        artwork::select_installation(&mut o);
        assert!(o.variant < 6);
        o.variant = 10;
        assert!(artwork::canvas_set(&o).is_none());
        audit::check(&o, &build_object(&o)).unwrap();
    }
    assert!((32..120).contains(&sets), "split installations {sets}/512");
}
