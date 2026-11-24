use super::*;
use crate::scene::procedural_indoor::{layout::floor::detail_clear, objects::audit::fixture};

#[test]
fn seated_clearance_encloses_rendered_soles_on_floors_and_footrests() {
    use super::super::objects::chairs::stool;
    for seed in 0..24 {
        let mut h = sample_person(
            seed,
            1,
            Vec3::ZERO,
            0.,
            HumanPoseKind::SeatedWorking,
            Some(0),
            false,
        );
        if seed % 2 == 1 {
            let mut chair = fixture(ObjectKind::Chair, seed);
            chair.variant = 6;
            chair.size = Vec3::new(0.61, 0.76, 0.59);
            let pose = h.pose_program.as_mut().unwrap();
            pose.stool_support = Some(stool::parameters(&chair).support(&chair));
            h.joints = pose.solve(h.stature, h.build, h.shoulder_width, true);
            assert!(h.foot_support_height() > 0.);
        }
        let soles = footwear::seated_soles(&h);
        let mesh = build_human(&h);
        let vertices = &mesh.parts[&HumanSurface::Sole].positions;
        assert!(!vertices.is_empty());
        for &vertex in vertices {
            let p = Vec3::from_array(vertex);
            assert!(
                soles.iter().any(|&(a, b, r)| {
                    let ab = b - a;
                    let t = ((p - a).dot(ab) / ab.length_squared()).clamp(0., 1.);
                    p.distance_squared(a + t * ab) <= r * r
                }),
                "seed {seed}: sole vertex {p} escapes table clearance"
            );
        }
    }
}

#[test]
fn people_respect_the_same_wall_service_gap_as_fixture_admission() {
    let mut scene = IndoorManifest::generate_with_humans(
        87,
        super::super::layout::IndoorLayout::Mixed,
        0.75,
        0,
        0.,
    )
    .unwrap();
    scene.room_size = Vec3::new(10., 4., 10.);
    scene.envelope = None;
    scene.program = None;
    scene.ceiling_installation = None;
    scene.objects.clear();
    scene.rug_supports.clear();
    let person = sample_person(
        42,
        0,
        Vec3::ZERO,
        0.,
        HumanPoseKind::StandingPresenting,
        None,
        false,
    );
    assert!(placement_clear(&scene, &person));
    let (lo, hi) = person.bounds();
    let mut speaker = fixture(ObjectKind::Speaker, 73);
    speaker.size = Vec3::new(0.2, 0.2, 0.06);
    speaker.yaw = 0.;
    speaker.solid = false;
    speaker.support = None;
    speaker.position = Vec3::new(
        hi.x + speaker.size.x * 0.5 + 0.020,
        (lo.y + hi.y) * 0.5,
        (lo.z + hi.z) * 0.5,
    );
    scene.objects.push(speaker);
    assert!(
        !placement_clear(&scene, &person),
        "20 mm must not bypass the 25 mm service reservation"
    );
    scene.objects[0].position.x += 0.010;
    assert!(
        placement_clear(&scene, &person),
        "30 mm clears the service reservation"
    );
}

#[test]
fn raised_gesture_does_not_project_a_collision_onto_floor_detail() {
    // Explicit support/pose fixture: the test must not depend on a particular
    // generated room continuing to contain a rug and a standing person.
    let mut scene = IndoorManifest::generate_with_humans(
        87,
        super::super::layout::IndoorLayout::Mixed,
        0.75,
        0,
        0.,
    )
    .unwrap();
    scene.room_size = Vec3::new(10., 4., 10.);
    scene.envelope = None;
    scene.program = None;
    scene.ceiling_installation = None;
    scene.objects.clear();
    scene.rug_supports.clear();
    let mut person = sample_person(
        42,
        0,
        Vec3::ZERO,
        0.,
        HumanPoseKind::StandingPresenting,
        None,
        false,
    );
    let pose = person.pose_program.as_mut().unwrap();
    pose.lean = Vec2::ZERO;
    pose.torso_twist = 0.;
    pose.arm_elevation = [0., 0.];
    pose.arm_sweep = [std::f32::consts::FRAC_PI_2; 2];
    pose.arm_reach = [0.9; 2];
    person.joints = pose.solve(person.stature, person.build, person.shoulder_width, false);
    update_bounds(&mut person);
    let mut mat = fixture(ObjectKind::Rug, 42);
    mat.solid = false;
    mat.size = Vec3::new(0.08, 0.03, 0.08);
    mat.position = person.joints[7].with_y(0.);
    let mut at_foot = mat.clone();
    at_foot.position = person.joints[16].with_y(0.);
    let (lo, hi) = mat.bounds();
    let (a, b) = person.bounds();
    assert!(
        lo.cmplt(b).all() && hi.cmpgt(a).all(),
        "fixture must overlap the whole-body AABB"
    );
    scene.humans = vec![person];
    assert!(
        detail_clear(&scene, &mat),
        "raised arm leaves the floor clear"
    );
    assert!(
        !detail_clear(&scene, &at_foot),
        "actual foot contact must remain blocked"
    );
    scene.humans.clear();
    assert!(
        detail_clear(&scene, &at_foot),
        "only the foot blocks the floor fixture"
    );
}
