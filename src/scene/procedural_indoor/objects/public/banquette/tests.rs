use super::*;
use crate::scene::procedural_indoor::objects::audit;
use std::collections::BTreeSet;

// Cast into the emitted front from +Z. Includes both the cloth and any covered
// buttons. Interpolate the actual texture chart at the visible intersection.
fn front(g: &Geometry, xy: Vec2) -> (f32, Vec2) {
    g.indices
        .as_chunks::<3>()
        .0
        .iter()
        .filter_map(|tri| {
            let [a, b, c] = tri.map(|i| Vec3::from(g.positions[i as usize]));
            let area = (b - a).xy().perp_dot((c - a).xy());
            if area < 1e-9 {
                return None;
            }
            let delta = xy - a.xy();
            let u = delta.perp_dot((c - a).xy()) / area;
            let v = (b - a).xy().perp_dot(delta) / area;
            if u < -1e-5 || v < -1e-5 || u + v > 1.00001 {
                return None;
            }
            let [sa, sb, sc] = tri.map(|i| Vec2::from(g.uvs[i as usize]));
            Some((
                a.z + (b.z - a.z) * u + (c.z - a.z) * v,
                sa + (sb - sa) * u + (sc - sa) * v,
            ))
        })
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .expect("gap in the upholstered skin")
}

#[test]
fn sewn_channels_keep_cloth_in_the_valley_and_a_continuous_metric_weave() {
    let mut directions = BTreeSet::new();
    for seed in 0..96 {
        let o = audit::fixture(ObjectKind::Banquette, seed);
        let p = parameters(&o);
        let Some(c) = p.sewn_back(&o) else {
            continue;
        };
        let Stitch::Channels {
            vertical,
            pitch_m,
            offset_m,
        } = c.stitch
        else {
            continue;
        };
        if !directions.insert(vertical) {
            continue;
        }
        let mut g = Geometry::default();
        c.build(&mut g, Transform::IDENTITY);
        let axis = if vertical { Vec2::X } else { Vec2::Y };
        let coord = |d| axis * (offset_m + pitch_m * d);
        let (valley, start) = front(&g, coord(0.));
        let (crown, _) = front(&g, coord(0.5));
        let (_, end) = front(&g, coord(1.));
        assert!(valley > c.foam_m * 0.22 && (crown - valley - c.tuck_m).abs() < 0.004);
        let cloth = (end - start).dot(axis);
        assert!(cloth > pitch_m + 0.0008 && cloth < pitch_m + 2. * c.tuck_m);
        let mut previous = start;
        for i in 1..=32 {
            let (_, uv) = front(&g, coord(i as f32 / 32.));
            assert!((uv - previous).dot(axis) > 0., "weave reset at a seam");
            previous = uv;
        }
    }
    assert_eq!(directions.len(), 2);
}

#[test]
fn tuft_anchors_recess_the_cloth_and_buttons_contact_their_fold() {
    let mut seen = BTreeSet::new();
    // Installed stock from mixed rooms 93400 and 93454 includes both types
    // of tufting even if the frequencies of seeded fixtures change.
    for seed in [14_426_512_498_346_065_685, 3_067_231_949_363_011_937] {
        let o = audit::fixture(ObjectKind::Banquette, seed);
        let Some(c) = parameters(&o).sewn_back(&o) else {
            continue;
        };
        let Stitch::Diamonds {
            pitch_m,
            offset_m,
            button_radius_m,
        } = c.stitch
        else {
            continue;
        };
        if !seen.insert(button_radius_m.is_some()) {
            continue;
        }
        let smooth = SewnBack {
            stitch: Stitch::Diamonds {
                pitch_m,
                offset_m,
                button_radius_m: None,
            },
            ..c
        };
        let mut fabric = Geometry::default();
        smooth.build(&mut fabric, Transform::IDENTITY);
        // Successive rows must stagger the pull points by half the repeat.
        for row in [0., 1.] {
            let xy = offset_m + Vec2::new(row * 0.5 * pitch_m.x, row * pitch_m.y);
            if xy.y.abs() + 0.04 >= c.size_m.y * 0.5 {
                continue;
            }
            let anchor = front(&fabric, xy).0;
            let crown = front(&fabric, xy + Vec2::X * pitch_m.x * 0.5).0;
            assert!((crown - anchor - c.tuck_m).abs() < 0.0025);
        }
        if button_radius_m.is_some() {
            let mut complete = Geometry::default();
            c.build(&mut complete, Transform::IDENTITY);
            let mut count = 0;
            c.buttons(|xy, r| {
                let cloth = front(&fabric, xy).0;
                let head = front(&complete, xy).0;
                assert!(head > cloth + 0.3 * r);
                for axis in [Vec2::X, Vec2::Y, -Vec2::X, -Vec2::Y] {
                    assert!(head > front(&fabric, xy + axis * r).0);
                }
                assert!(
                    (front(&complete, xy + Vec2::X * (1.09 * r)).0
                        - front(&fabric, xy + Vec2::X * (1.09 * r)).0)
                        .abs()
                        < 1e-6
                );
                count += 1;
            });
            assert!(count >= 2);
        }
    }
    assert_eq!(seen.len(), 2);
}

#[test]
fn supports_land_in_the_deck_and_board_across_product_sizes_and_finishes() {
    let mut forms = BTreeSet::new();
    let mut missed_before = 0;
    for seed in (0..96).chain([3_067_231_949_363_011_937, 14_347_151_663_193_390_405]) {
        for size in [
            Vec3::new(1.15, 0.88, 0.60),
            Vec3::new(2., 1.03, 0.73),
            Vec3::new(3., 1.25, 0.85),
        ] {
            let mut o = audit::fixture(ObjectKind::Banquette, seed);
            o.size = size;
            let p = parameters(&o);
            let fit = BackFit::new(&o, &p);
            let cover = p.sewn_back(&o);
            forms.insert(
                cover
                    .as_ref()
                    .map_or_else(|| p.stock_name(), SewnBack::construction),
            );
            let old = fit.tf.transform_point(Vec3::new(
                0.,
                -fit.height * 0.5 + 0.016,
                (FRAME_DEPTH - p.back_thickness_m) * 0.5,
            ));
            missed_before += usize::from(old.z + 0.009 < -size.z * 0.44);
            for (base, tip, section) in fit.stays(&o, &p) {
                let deck_top = p.seat_height_m - p.cushion_m;
                assert!(base.y < deck_top && base.y > deck_top - 0.026);
                assert!(base.x.abs() + section.x * 0.5 < fit.width * 0.5);
                assert!(base.z.abs() + section.y * 0.5 < size.z * 0.44);
                let local = fit.tf.rotation.inverse() * (tip - fit.tf.translation);
                assert!(
                    local.y.abs() < fit.height * 0.5
                        && (local.z + (p.back_thickness_m - FRAME_DEPTH) * 0.5).abs() < 1e-6
                );
                assert!(tip.y > base.y);
            }
            let a = super::super::super::build_object(&o);
            let check = audit::check(&o, &a).unwrap();
            assert!(check.max_envelope_overrun_m < 1e-5 && check.min_y_m.abs() < 1e-6);
            o.finish_seed = Some(seed ^ 0x4d41544c);
            let alternative = parameters(&o).sewn_back(&o);
            assert_eq!(
                serde_json::to_value(cover).unwrap(),
                serde_json::to_value(alternative).unwrap()
            );
        }
    }
    assert_eq!(forms.len(), 8);
    assert!(missed_before > 5);
}

#[test]
fn rolled_corner_faces_and_normals_point_outward_even_at_cropped_stitches() {
    // The narrow end flute on the first banquette in room 93471 crossed a
    // corner roll. Fixture 79 also exposed the issue at other product sizes.
    let seeds = [7_864_261_404_211_537_703, 79].into_iter().chain(0..192);
    let mut checked = 0;
    for seed in seeds {
        let o = audit::fixture(ObjectKind::Banquette, seed);
        let Some(c) = parameters(&o).sewn_back(&o) else {
            continue;
        };
        let mut g = Geometry::default();
        c.build(&mut g, Transform::IDENTITY);
        for tri in g.indices.as_chunks::<3>().0 {
            let [a, b, c] = tri.map(|i| Vec3::from(g.positions[i as usize]));
            if a.z < 0. || b.z < 0. || c.z < 0. {
                break; // the next strips are the sides of the wrapped hem
            }
            let face = (b - a).cross(c - a);
            let smooth: Vec3 = tri.iter().map(|&i| Vec3::from(g.normals[i as usize])).sum();
            assert!(
                face.z > 0. && face.dot(smooth) > 0.,
                "stock {seed} triangle {tri:?}"
            );
        }
        checked += 1;
    }
    assert!(checked > 50);
}
