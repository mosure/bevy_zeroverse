use super::*;
use crate::scene::procedural_indoor::{
    materials,
    objects::{self, audit},
};
use std::collections::BTreeSet;

#[test]
fn dishwasher_prints_follow_the_installed_mount_through_the_door_sweep() {
    use crate::scene::procedural_indoor::motion::IndoorMotionConfig;
    let motion = IndoorMotionConfig {
        appliance_fraction: 1.,
        ..default()
    };
    let mut cases = BTreeSet::new();
    for seed in 0..64 {
        let o = audit::fixture(ObjectKind::Dishwasher, seed);
        let recipe = parameters(&o);
        let panel = for_object(&o).unwrap();
        let mounted = panel.fascia.mount == Mount::DropDoor;
        cases.insert((mounted, recipe.opening_radians > 0.02));
        let built = objects::build_object_at_version(&o, Some(&motion), INTRODUCED);
        let [door] = built.joints.as_slice() else {
            panic!("expected one dishwasher hinge for {seed}");
        };
        let mut closed = recipe.clone();
        closed.opening_radians = 0.;
        // body() reserves the installed door clearance; both poses share that chassis.
        let (depth, z) = body(&o, &recipe);
        let product = program::parameters(&o);
        let rest = Fascia::fitted(&o, &closed, &product, z + depth * 0.5);
        let face = panel.fascia.face;
        assert_eq!(door.axis, Vec3::X);
        if mounted {
            assert!(super::super::interior::dishwasher_baskets(&o, &recipe)
                .iter()
                .all(|b| b.position.y >= door.pivot.y + 0.018));
        } else {
            // Rays through the closed reveal must meet stationary rubber before the rack.
            let low = o.size.y * (1. - product.fascia_fraction);
            let front = z + depth * 0.5;
            let gasket = &built.parts[&(Surface::Rubber, "dishwasher".to_owned())];
            let points: Vec<_> = gasket
                .positions
                .iter()
                .copied()
                .map(Vec3::from_array)
                .filter(|p| (p.y - low).abs() < 0.011 && (front - p.z).abs() < 0.007)
                .collect();
            let lo = points
                .iter()
                .copied()
                .fold(Vec3::splat(f32::INFINITY), Vec3::min);
            let hi = points
                .iter()
                .copied()
                .fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max);
            assert!(lo.x < -rest.size.x * 0.49 && hi.x > rest.size.x * 0.49);
            assert!(lo.y < low - 0.004 && hi.y >= low);
            assert!((hi.z - front).abs() < 1e-5 && lo.z < hi.z);
        }
        let slots: Vec<_> = [Some(panel.key_film), panel.readout.map(|d| d.film)]
            .into_iter()
            .flatten()
            .filter_map(|f| materials::finish_slot(Surface::Ink, &f.label("dishwasher")))
            .collect();
        let mut printed = 0;
        for (part, geometry) in &built.parts {
            let (surface, label) = part;
            if *surface != Surface::Ink
                || !materials::finish_slot(*surface, label).is_some_and(|s| slots.contains(&s))
            {
                continue;
            }
            printed += 1;
            assert_eq!(door.parts.contains(part), mounted);
            assert_eq!(label.split('#').next(), Some("dishwasher"));
            for t in [0., 0.25, 0.5, 0.75, 1.] {
                let transform = if mounted {
                    door.transform(t)
                } else {
                    Transform::IDENTITY
                };
                let angle = if mounted {
                    recipe.opening_radians + t * door.angle
                } else {
                    0.
                };
                let orientation = Quat::from_rotation_x(angle);
                let expected = door.pivot + orientation * (rest.face - door.pivot);
                assert!(transform.transform_point(face).distance(expected) < 1e-5);
                for (&p, &n) in geometry.positions.iter().zip(&geometry.normals) {
                    let v = transform.transform_point(Vec3::from_array(p));
                    assert!(v.y > 0. && v.x.abs() < o.size.x * 0.5);
                    assert!(
                        (transform.rotation * Vec3::from_array(n)).distance(orientation * Vec3::Z)
                            < 1e-5
                    );
                }
            }
        }
        assert!(printed > 0);
    }
    assert_eq!(cases.len(), 4);
}

#[test]
fn actual_controls_fit_both_fascia_orientations_and_replay_across_finishes() {
    for kind in [ObjectKind::Microwave, ObjectKind::Dishwasher] {
        let mut families = BTreeSet::new();
        let mut positions = BTreeSet::new();
        let mut states = BTreeSet::new();
        for seed in 0..256 {
            let mut o = audit::fixture(kind, seed);
            o.finish_seed = Some(seed ^ 0x8300);
            let p = for_object(&o).unwrap();
            let f = p.fascia;
            let mut boxes: Vec<(Vec2, Vec2)> = Vec::new();
            let mut place = |xy: Vec2, size: Vec2| {
                assert!((size.cmpgt(Vec2::ZERO)).all());
                assert!(
                    (xy.abs() + size * 0.5).cmplt(f.size * 0.5).all(),
                    "{kind:?}/{seed} outside fascia"
                );
                for (q, extent) in &boxes {
                    assert!(
                        ((*q - xy).abs() * 2.).cmpge(*extent + size).any(),
                        "{kind:?}/{seed} overlapping controls {xy:?}/{q:?}"
                    );
                }
                boxes.push((xy, size));
            };
            for d in &p.dials {
                place(d.center, Vec2::splat(d.radius_m * 2.76 + 0.001));
            }
            for k in &p.keys {
                place(k.center, k.size + Vec2::splat(0.0013));
            }
            if let Some(d) = p.readout {
                place(d.center, d.size + Vec2::splat(0.0016));
                states.insert(format!("{:?}", d.film));
                if parameters(&o).opening_radians > 0.02 {
                    assert_eq!(d.timer_seconds, 0);
                }
                assert!(d
                    .glyphs()
                    .iter()
                    .all(|g| !matches!(g,Glyph::Number(v) if *v>9)));
            }
            let mut recolored = o.clone();
            recolored.finish_seed = Some(seed ^ 0x00ff_55aa);
            let q = for_object(&recolored).unwrap();
            assert_eq!(p.family, q.family);
            assert_eq!(
                serde_json::to_value(&p.dials).unwrap(),
                serde_json::to_value(&q.dials).unwrap()
            );
            assert_eq!(
                serde_json::to_value(&p.keys).unwrap(),
                serde_json::to_value(&q.keys).unwrap()
            );
            assert_eq!(
                p.readout.map(|d| (d.center, d.timer_seconds, d.blank)),
                q.readout.map(|d| (d.center, d.timer_seconds, d.blank))
            );
            families.insert(format!("{:?}", p.family));
            positions.insert((f.face.x.signum() as i32, (f.size.y > f.size.x)));
            if seed < 32 {
                let mesh = objects::build_object_at_version(&o, None, INTRODUCED);
                audit::check(&o, &mesh).unwrap_or_else(|e| panic!("{kind:?}/{seed}: {e}"));
                let previous = objects::build_object_at_version(&o, None, INTRODUCED - 1);
                if kind == ObjectKind::Microwave {
                    assert!(previous.parts.keys().any(|(s, _)| *s == Surface::Screen));
                    assert!(!mesh.parts.keys().any(|(s, _)| *s == Surface::Screen));
                }
                let print_label = p.key_film.label(o.kind.class_name());
                let key = (Surface::Ink, print_label);
                assert!(!previous.parts.contains_key(&key));
                let g = &mesh.parts[&key];
                assert!(g
                    .normals
                    .iter()
                    .all(|&n| Vec3::from_array(n).distance(f.orientation * Vec3::Z) < 1e-5));
                assert!(g.uvs.iter().flatten().all(|u| (0.0..1.0).contains(u)));
                assert!(materials::registered_finish(
                    key.0,
                    materials::finish_slot(key.0, &key.1).unwrap()
                ));
                assert!(
                    super::super::metadata_for_version(&o, &parameters(&o), INTRODUCED - 1)
                        .get("operator_panel")
                        .is_none()
                );
                assert_eq!(
                    super::super::metadata(&o, &parameters(&o))["operator_panel"],
                    serde_json::to_value(&p).unwrap()
                );
            }
        }
        assert_eq!(families.len(), 3);
        assert_eq!(states.len(), 4);
        if kind == ObjectKind::Microwave {
            assert!(positions.len() >= 3);
        }
    }
}
