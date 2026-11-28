use bevy::prelude::*;
use bevy_zeroverse::scene::procedural_indoor::{
    humans::HumanPoseKind,
    layout::{segment_hits_box, IndoorManifest, IndoorObject, ObjectKind},
    objects::{
        build_manifest_object,
        public::furnishing::{lectern, parameters},
    },
};

pub(super) fn observe(room: &IndoorManifest, desk: &IndoorObject) -> serde_json::Value {
    let zone = room
        .functional_programs()
        .iter()
        .find(|p| p.members.contains(&desk.id))
        .unwrap();
    let standing = lectern::presenter_position(desk);
    let mesh_bounds = build_manifest_object(room, desk).bounds();
    let forward = desk.transform().rotation * Vec3::NEG_Z;
    let eye = standing + Vec3::Y * 1.55;
    let local = desk.transform().compute_affine().inverse();
    let presenters_at_desk: Vec<_> = room
        .humans
        .iter()
        .filter(|person| {
            let offset = local.transform_point3(person.position);
            !person.neighbor
                && person.pose == HumanPoseKind::StandingPresenting
                && offset.x.abs() < (desk.size.x * 0.5).max(0.35)
                && (desk.size.z * 0.5..=desk.size.z * 0.5 + lectern::standing_depth(desk))
                    .contains(&offset.z)
                && forward.dot(person.transform().rotation * Vec3::NEG_Z) > 0.9
        })
        .map(|person| person.id)
        .collect();
    let members = || zone.members.iter().map(|&id| &room.objects[id]);
    let seats: Vec<_> = members().filter(|o| o.kind == ObjectKind::Chair).map(|chair| {
        let seated_eye = chair.position + Vec3::Y * 1.16;
        let direction = (seated_eye - eye).with_y(0.).normalize_or_zero();
        let occluded = members().filter(|o| o.solid && o.id != chair.id && o.id != desk.id).any(|o| {
            let (lo, hi) = o.bounds();
            segment_hits_box(eye, seated_eye, lo, hi)
        });
        serde_json::json!({"position":chair.position,"cos_heading":forward.dot(direction),"blocked_by_furniture_bounds":occluded})
    }).collect();
    serde_json::json!({"seed":room.seed,"activity":zone.activity,"zone_min":zone.min,"zone_max":zone.max,
        "desk":desk,"local_mesh_bounds":mesh_bounds,"standing":standing,"forward":forward,"standing_depth_m":lectern::standing_depth(desk),
        "reading_desk":parameters(desk).reading_desk,"presenters_at_desk":presenters_at_desk,"seats":seats})
}
