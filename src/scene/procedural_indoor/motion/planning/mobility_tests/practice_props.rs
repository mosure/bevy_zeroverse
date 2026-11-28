use super::*;
use crate::scene::procedural_indoor::objects::public::mats::PracticeMat;

#[test]
fn practice_props_move_on_their_own_end_panel_and_return_to_contact() {
    for kind in [ObjectKind::YogaBlock, ObjectKind::YogaBolster] {
        let mut admitted = 0;
        let mut pickups = 0;
        let mut lateral = 0;
        for seed in 0..48 {
            let (mut scene, mut geom) = room();
            scene.indoor_motion = Some(IndoorMotionReport::new(IndoorMotionConfig {
                prop_pickup_fraction: 1.,
                prop_lift_m: 0.19,
                prop_travel_m: if seed % 2 == 0 { 0. } else { 0.028 },
                prop_rotation_degrees: if seed % 2 == 0 { 0. } else { 6. },
                ..policy()
            }));
            let mut mat = fixture(ObjectKind::ExerciseMat, seed);
            mat.size = Vec3::new(0.91, 0.009, 2.13);
            mat.yaw = 0.83;
            let area = PracticeMat::for_object(&mat).prop_area(mat.size);
            let inv = mat.transform().compute_affine().inverse();
            let mut prop = fixture(kind, seed + 231);
            prop.support = Some(0);
            prop.solid = false;
            prop.position = mat.transform().transform_point(Vec3::new(
                0.,
                mat.size.y,
                (area.0.y + area.1.y) * 0.5,
            ));
            prop.yaw = mat.yaw + std::f32::consts::FRAC_PI_2;
            add(&mut scene, &mut geom, mat);
            if !scene.prop_clear(&prop, &scene.objects[0], 0.015) {
                continue;
            }
            add(&mut scene, &mut geom, prop);
            prepare(&mut scene, &mut geom);
            let motion = &scene.indoor_motion.as_ref().unwrap().mechanisms[0];
            if motion.rejection.is_some() {
                continue;
            }
            assert!(motion.requested_lift_m > 0.);
            pickups += usize::from(motion.lift_m > 0.01);
            lateral += usize::from(motion.translation_m.length() > 0.001);
            let surface = scene.objects[0].size.y;
            for tick in 0..=32 {
                let tf = inv
                    * motion
                        .rigid_transform_at(tick as f32 / 32.)
                        .unwrap()
                        .compute_affine()
                    * scene.objects[1].transform().compute_affine();
                let mut lowest = f32::INFINITY;
                for p in geom.objects[1].parts.values().flat_map(|g| &g.positions) {
                    let p = tf.transform_point3(Vec3::from(*p));
                    assert!(
                        p.xz().cmpge(area.0 - 1e-5).all()
                            && p.xz().cmple(area.1 + 1e-5).all()
                            && p.y >= surface - 1e-5
                    );
                    lowest = lowest.min(p.y);
                }
                if tick == 0 || tick == 32 {
                    assert!((lowest - surface).abs() < 1e-5);
                }
            }
            admitted += 1;
            if admitted >= 3 && pickups >= 2 && lateral >= 1 {
                break;
            }
        }
        assert!(
            admitted >= 3 && pickups >= 2 && lateral >= 1,
            "{kind:?}: {admitted} accepted, {pickups} lifted, {lateral} translated"
        );
    }
}
