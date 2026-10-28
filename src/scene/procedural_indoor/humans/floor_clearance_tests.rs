use super::*;
use crate::scene::procedural_indoor::{layout::floor::detail_clear, objects::audit::fixture};

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
