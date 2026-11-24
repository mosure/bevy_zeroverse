use super::*;
#[test]
fn lockers_face_open_floor_from_solid_wall_sections_across_classrooms() {
    let mut normals = std::collections::BTreeSet::new();
    let mut count = 0;
    let mut settings = std::collections::BTreeSet::new();
    for seed in 11400..11464 {
        let s = IndoorManifest::generate_with_humans(seed, IndoorLayout::Classroom, 0.65, 0, 0.)
            .unwrap();
        let e = s.envelope.as_ref().unwrap();
        for program in s.functional_programs() {
            for o in program
                .members
                .iter()
                .map(|&i| &s.objects[i])
                .filter(|o| o.kind == ObjectKind::Locker)
            {
                count += 1;
                settings.insert(o.variant);
                let normal = o.transform().rotation * Vec3::Z;
                let (lo, hi) = o.bounds();
                assert!(s.placement_clear(o, 0.035));
                assert!(!facade::overlaps_opening(&s, lo, hi, 0.045));
                assert!(
                    (o.variant == geometry::education::FREESTANDING_BANK)
                        || (0..e.footprint.len()).any(|edge| {
                            let tf = e.wall_transform(edge);
                            let local = tf.rotation.inverse() * (o.position - tf.translation);
                            let span = e.footprint[edge]
                                .distance(e.footprint[(edge + 1) % e.footprint.len()]);
                            let backed = normal.dot(tf.rotation * Vec3::Z) > 0.99999
                                && (0.044..0.14).contains(&(local.z - o.size.z * 0.5))
                                && local.x.abs() + o.size.x * 0.5 < span * 0.5;
                            if backed && e.shared_edge(edge, s.room_size) {
                                let f = interior_glazing::shared_facade(
                                    &s,
                                    interior_glazing::partition_height(&s),
                                );
                                return !f.openings.iter().any(|w| {
                                    let x = -o.position.x;
                                    x + o.size.x * 0.5 > w.min.x
                                        && x - o.size.x * 0.5 < w.max.x
                                        && hi.y > w.min.y
                                        && lo.y < w.max.y
                                });
                            }
                            backed
                        }),
                    "missing solid wall in {seed}"
                );
                let stood = o
                    .transform()
                    .transform_point(Vec3::Z * (o.size.z * 0.5 + 0.24))
                    .xz();
                assert!(program
                    .reservations
                    .iter()
                    .any(|r| r.role == ReservationRole::DoorApproach
                        && stood.cmpge(r.min).all()
                        && stood.cmple(r.max).all()));
                if o.variant == geometry::education::FREESTANDING_BANK {
                    let behind = o
                        .transform()
                        .transform_point(Vec3::NEG_Z * (o.size.z * 0.5 + 0.27))
                        .xz();
                    assert!(program.reservations.iter().any(|r| r.role
                        == ReservationRole::Circulation
                        && behind.cmpge(r.min).all()
                        && behind.cmple(r.max).all()
                        && !r.overlaps(lo, hi)));
                }
                normals.insert((o.yaw / std::f32::consts::FRAC_PI_2).round() as i32);
            }
        }
    }
    assert_eq!(settings.len(), 2);
    assert!(
        count >= 40 && normals.len() >= 3,
        "{count} banks, {normals:?}"
    );
}
