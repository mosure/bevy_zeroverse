use super::*;
use crate::scene::procedural_indoor::{envelope::ray_triangle, objects::audit};

fn side_depth(g: &Geometry, y: f32) -> Option<f32> {
    let eye = Vec3::new(1., y, 0.);
    g.indices
        .as_chunks::<3>()
        .0
        .iter()
        .filter_map(|tri| {
            ray_triangle(
                eye,
                Vec3::NEG_X,
                tri.map(|i| Vec3::from_array(g.positions[i as usize])),
            )
        })
        .min_by(f32::total_cmp)
        .map(|d| 1. - d)
}

#[test]
fn installed_lifts_seat_in_the_bearing_and_leave_the_pan_interior_clear() {
    let mut controls = [0; 5];
    for seed in 0..180 {
        let mut o = audit::fixture(ObjectKind::Chair, seed);
        o.variant = [0, 1, 5, 9][seed as usize % 4];
        let p = parameters(&o);
        assert!(p.swivel_pedestal(&o, 259).is_none());
        let fit = p.swivel_pedestal(&o, 260).unwrap();
        for pan_bottom in [0.372, 0.403] {
            let mut lower = Assembly::default();
            let mut upper = Assembly::default();
            let pan = Vec2::new(0.44, 0.47);
            fit.column(&mut lower, pan_bottom);
            fit.upper(&mut upper, pan, pan_bottom);
            let bearing = fit.bearing_y(pan_bottom);
            assert!((lower.bounds().1.y - bearing - 0.0015).abs() < 2e-6);
            assert!((upper.bounds().1.y - pan_bottom - 0.0012).abs() < 2e-6);
            // Both surfaces occupy the axial joint; the column ends beneath the pan.
            let joint_y = bearing + 0.0007;
            assert!(
                side_depth(&upper.parts[&(Surface::Metal, "chair".into())], joint_y).unwrap()
                    > 0.045
            );
            let radius_at = |y: f32| {
                lower
                    .parts
                    .values()
                    .filter_map(|g| side_depth(g, y))
                    .max_by(f32::total_cmp)
                    .unwrap()
            };
            let expected = fit
                .gas_lift
                .map_or(fit.barrel_radius_m * 0.94, |l| l.piston_radius_m);
            assert!((radius_at(joint_y) - expected).abs() < 0.002);
            assert!(fit.hub_height_m < bearing - 0.12);
            if let Some(lift) = fit.gas_lift {
                let exposed = lift.exposed_fraction * (bearing - fit.hub_height_m - 0.024);
                assert!((radius_at(bearing - exposed * 0.5) - lift.piston_radius_m).abs() < 2e-6);
                let grip = &upper.parts[&(Surface::Plastic, "chair".into())];
                let sign = if lift.lever.left { -1. } else { 1. };
                let furthest = grip
                    .positions
                    .iter()
                    .map(|v| v[0] * sign)
                    .fold(0., f32::max);
                assert!(furthest > pan.x * 0.39 && furthest < pan.x * 0.57);
                assert!(grip
                    .positions
                    .iter()
                    .all(|v| v[0] * sign > 0. && v[1] < pan_bottom && v[1] > 0.26));
            } else {
                assert_eq!(upper.parts.len(), 1);
                assert_eq!(lower.parts.len(), 1);
            }
        }
        let group = fit.gas_lift.map_or(0, |lift| {
            1 + usize::from(lift.lever.grip == Grip::Sleeve) * 2 + usize::from(lift.lever.left)
        });
        controls[group] += 1;
        o.variant = 6;
        assert!(p.swivel_pedestal(&o, 260).is_none());
    }
    assert!(controls.iter().all(|&n| n >= 4), "{controls:?}");
}

#[test]
fn cast_spokes_rotate_their_real_cross_section_and_uvs_together() {
    let o = (0..128)
        .find_map(|seed| {
            let mut o = audit::fixture(ObjectKind::Chair, seed);
            o.variant = 0;
            (parameters(&o).base_style == 2).then_some(o)
        })
        .unwrap();
    let fit = parameters(&o).swivel_pedestal(&o, 260).unwrap();
    let blade = fit.cast_spoke.unwrap();
    let mesh = |radial| {
        let mut g = Geometry::default();
        blade.build(
            &mut g,
            fit.spoke_root(radial),
            radial * 0.25 + Vec3::Y * 0.072,
            fit.hub_radius(),
        );
        g
    };
    let first = mesh(Vec3::Z);
    assert_eq!(first.indices.len() / 3, 60);
    for angle in [0.33, 1.56, 2.7, 3.91, 5.13] {
        let q = Quat::from_rotation_y(angle);
        let next = mesh(q * Vec3::Z);
        assert_eq!(first.indices, next.indices);
        for (a, b) in first.positions.iter().zip(&next.positions) {
            assert!((q * Vec3::from_array(*a)).distance(Vec3::from_array(*b)) < 1e-6);
        }
        for (a, b) in first.normals.iter().zip(&next.normals) {
            assert!((q * Vec3::from_array(*a)).dot(Vec3::from_array(*b)) > 0.9999);
        }
        for t in next.indices.as_chunks::<3>().0 {
            let [a, b, c] = t.map(|i| Vec3::from_array(next.positions[i as usize]));
            let normal = Vec3::from_array(next.normals[t[0] as usize]);
            assert!((b - a).cross(c - a).dot(normal) > 1e-8);
            let [a, b, c] = t.map(|i| Vec2::from_array(next.uvs[i as usize]));
            assert!((b - a).perp_dot(c - a).abs() > 1e-8);
        }
        for (a, b) in first.uvs.iter().zip(&next.uvs) {
            assert!(Vec2::from_array(*a).distance(Vec2::from_array(*b)) < 1e-5);
        }
    }
}
