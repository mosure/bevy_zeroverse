use super::*;
use crate::scene::procedural_indoor::{envelope::ray_triangle, objects::audit};

/// Front/back intersections with the built triangles at a point on the elevation.
fn face_hits(g: &Geometry, x: f32, y: f32) -> Vec<f32> {
    let eye = Vec3::new(x, y, 1.);
    g.indices
        .as_chunks::<3>()
        .0
        .iter()
        .filter_map(|t| {
            let p = t.map(|i| Vec3::from_array(g.positions[i as usize]));
            ray_triangle(eye, Vec3::NEG_Z, p).map(|d| 1. - d)
        })
        .collect()
}

#[test]
fn four_live_clamps_hold_the_panel_faces_and_join_the_posts() {
    let mut total = 0;
    for seed in 0..128 {
        let o = audit::fixture(ObjectKind::QueueBarrier, seed);
        let product = parameters(&o);
        let mut install = Installation::sample(&o, &product);
        let Infill::Panel {
            bottom_m,
            thickness_m,
            side_gap_m,
            clamp_inset_m,
            jaw_reach_m,
            finish,
        } = install.infill
        else {
            continue;
        };
        install.hardware = Surface::Chrome;
        let mut a = Assembly::default();
        let l = o.kind.class_name();
        spans::build(&mut a, l, &product, &install);
        let hardware = &a.parts[&(install.hardware, l.into())];
        let sheet = &a.parts[&(finish, l.into())];
        let half = install.post_half_span_m - install.post_radius_m - side_gap_m;
        let mid = (bottom_m + install.connection_y_m) * 0.5;
        let face = thickness_m * 0.5;
        for side in [-1., 1.] {
            let gap_x = side * (half + side_gap_m * 0.65);
            assert!(face_hits(hardware, gap_x, mid).is_empty());
            assert!(face_hits(sheet, gap_x, mid).is_empty());
            for y in [
                bottom_m + clamp_inset_m,
                install.connection_y_m - clamp_inset_m,
            ] {
                let grip = side * (half - jaw_reach_m * 0.53);
                for z in [-face, face] {
                    assert!(face_hits(sheet, grip, y)
                        .iter()
                        .any(|h| (z - h).abs() < 2e-6));
                    let hits = face_hits(hardware, grip, y);
                    // Both jaws overlap the sheet by the submillimetre liner bite.
                    assert!(hits
                        .iter()
                        .any(|h| (h.abs() - (face - 0.0004)).abs() < 2e-6
                            && h.signum() == z.signum()));
                    assert!(hits.iter().any(|h| h * z.signum() > face + 0.001));
                }
                assert!(
                    !face_hits(hardware, gap_x, y).is_empty(),
                    "missing clamp shank {seed}"
                );
                let in_post = side * (install.post_half_span_m - install.post_radius_m * 0.78);
                assert!(
                    !face_hits(hardware, in_post, y).is_empty(),
                    "disconnected from post {seed}"
                );
            }
        }
        total += 1;
    }
    assert!(total >= 16);
}

#[test]
fn flexible_spans_sag_from_their_installed_end_fittings() {
    let mut kinds = [0; 2];
    let mut mirrored = [false; 2];
    for seed in 0..128 {
        let o = audit::fixture(ObjectKind::QueueBarrier, seed);
        let p = parameters(&o);
        let f = Installation::sample(&o, &p);
        let (radius, mid_x) = match f.infill {
            Infill::Webbing {
                width_m,
                reel_radius_m,
                reel_left,
                ..
            } => {
                kinds[0] += 1;
                mirrored[usize::from(reel_left)] = true;
                let sign = if reel_left { 1. } else { -1. };
                (
                    width_m * 0.5,
                    sign * (reel_radius_m - 0.002 - f.post_radius_m - 0.011 - 0.008) * 0.5,
                )
            }
            Infill::Rope { radius_m, .. } => {
                kinds[1] += 1;
                (radius_m, 0.)
            }
            _ => continue,
        };
        let l = o.kind.class_name();
        let mut a = Assembly::default();
        spans::build(&mut a, l, &p, &f);
        let fabric = if p.rail == Surface::Fabric {
            Surface::Fabric
        } else {
            Surface::FabricAlt
        };
        let web = &a.parts[&(fabric, l.into())];
        let mid: Vec<_> = web
            .positions
            .iter()
            .filter(|v| (v[0] - mid_x).abs() < 2e-6)
            .map(|v| v[1])
            .collect();
        let top = mid.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let bottom = mid.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            ((top + bottom) * 0.5 - (f.connection_y_m - f.sag_m().unwrap())).abs() < 2e-5,
            "sag {seed}"
        );
        assert!(top - bottom > radius * 1.88 && bottom > 0.);
        for t in web.indices.as_chunks::<3>().0 {
            let [a, b, c] = t.map(|i| Vec3::from_array(web.positions[i as usize]));
            let n = t
                .iter()
                .map(|&i| Vec3::from_array(web.normals[i as usize]))
                .sum::<Vec3>();
            assert!((b - a).cross(c - a).dot(n) > 0., "inward span {seed}");
        }
    }
    assert!(kinds.iter().all(|&v| v > 15) && mirrored.iter().all(|&v| v));
}
