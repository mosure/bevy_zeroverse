//! Floor detail is placed after people and furniture settle, before cable routing.
//! Low fixtures have explicit floor support, portal and occupied-volume checks.
use super::*;

pub(crate) fn detail_clear(scene: &IndoorManifest, o: &IndoorObject) -> bool {
    if o.solid || o.support.is_some() || o.neighbor || !scene.placement_clear(o, 0.025) {
        return false;
    }
    let (lo, hi) = o.bounds();
    if !scene.floor_support_clear(lo, hi) {
        return false;
    }
    let pad = Vec3::new(0.03, 0.005, 0.03);
    if scene
        .objects
        .iter()
        .filter(|p| p.id != o.id && matches!(p.kind, ObjectKind::Rug | ObjectKind::FloorOutlet))
        .any(|p| {
            let (a, b) = p.bounds();
            lo.cmplt(b + pad).all() && hi.cmpgt(a - pad).all()
        })
    {
        return false;
    }
    !scene.humans.iter().any(|h| {
        let (a, b) = h.bounds();
        lo.cmplt(b + pad).all() && hi.cmpgt(a - pad).all()
    })
}

impl IndoorManifest {
    pub(super) fn finish_floor_outlets(&mut self) {
        let used: std::collections::BTreeSet<_> = self
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::Cable)
            .filter_map(|o| o.cable_target)
            .collect();
        let candidates: Vec<_> = self
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::FloorOutlet && !used.contains(&o.id))
            .cloned()
            .collect();
        for mut o in candidates {
            o.variant |= 8;
            o.size.y = 0.006;
            // A nearby lead may cross the recessed, open service box. Closing
            // that cover must not invalidate an already accepted wire route.
            if equipment::cables_clear_proposed_prop(self, &o) {
                let id = o.id;
                self.objects[id] = o;
            }
        }
    }
    pub(super) fn furnish_floor_details(&mut self) {
        let mut rng = stream(self.seed, 7330);
        let anchors: Vec<_> = self
            .objects
            .iter()
            .filter(|o| {
                !o.neighbor
                    && matches!(
                        o.kind,
                        ObjectKind::Sofa
                            | ObjectKind::Bench
                            | ObjectKind::CoffeeTable
                            | ObjectKind::Desk
                            | ObjectKind::Table
                    )
            })
            .cloned()
            .collect();
        let mut rugs = 0;
        for anchor in &anchors {
            if rugs >= 3
                || !rng.random_bool(
                    if matches!(anchor.kind, ObjectKind::Sofa | ObjectKind::CoffeeTable) {
                        0.80
                    } else {
                        0.12
                    },
                )
            {
                continue;
            }
            for _ in 0..10 {
                let size = Vec3::new(
                    rng.random_range(0.85..2.7),
                    rng.random_range(0.004..0.011),
                    rng.random_range(0.65..1.8),
                );
                let side = if rng.random_bool(0.5) { 1. } else { -1. };
                let p = anchor
                    .transform()
                    .transform_point(Vec3::new(
                        rng.random_range(-0.45..0.45) * anchor.size.x,
                        0.,
                        side * ((anchor.size.z + size.z) * 0.5 + rng.random_range(0.05..0.35)),
                    ))
                    .with_y(0.);
                let yaw = anchor.yaw + rng.random_range(-0.25..0.25);
                let mut rug = self.floor_candidate(ObjectKind::Rug, p, size, yaw, &mut rng);
                rug.solid = false;
                if detail_clear(self, &rug) {
                    self.objects.push(rug);
                    rugs += 1;
                    break;
                }
            }
        }
        // Only powered work surfaces request service outlets; no random sockets
        // in circulation paths. Each can feed one routed mains plug.
        let powered: Vec<_> = self
            .objects
            .iter()
            .filter(|o| {
                !o.neighbor
                    && o.support.is_some()
                    && matches!(
                        o.kind,
                        ObjectKind::Monitor | ObjectKind::PowerStrip | ObjectKind::Projector
                    )
            })
            .map(|o| o.support.unwrap())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut installed = 0;
        for id in powered {
            if installed >= 4 || !rng.random_bool(0.66) {
                continue;
            }
            let parent = self.objects[id].clone();
            for attempt in 0..16 {
                let side = attempt % 4;
                let size = Vec3::new(
                    rng.random_range(0.16..0.24),
                    rng.random_range(0.07..0.12),
                    rng.random_range(0.16..0.24),
                );
                let gap = rng.random_range(0.14..0.30);
                let along = rng.random_range(-0.42..0.42);
                let local = match side {
                    0 => Vec3::new(along * parent.size.x, 0., -parent.size.z * 0.5 - gap),
                    1 => Vec3::new(-parent.size.x * 0.5 - gap, 0., along * parent.size.z),
                    2 => Vec3::new(parent.size.x * 0.5 + gap, 0., along * parent.size.z),
                    _ => Vec3::new(along * parent.size.x, 0., parent.size.z * 0.5 + gap),
                };
                let p = parent.transform().transform_point(local).with_y(0.);
                let mut o = self.floor_candidate(
                    ObjectKind::FloorOutlet,
                    p,
                    size,
                    parent.yaw + rng.random_range(-0.18..0.18),
                    &mut rng,
                );
                o.solid = false;
                o.variant = rng.random_range(0..3);
                if detail_clear(self, &o) {
                    self.objects.push(o);
                    installed += 1;
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn floor_details_have_support_clearance_and_routable_ports() {
        let mut rugs = 0;
        let mut outlets = 0;
        let mut wired = 0;
        for seed in 0..96 {
            let scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.75, 0, 0.35)
                    .unwrap();
            for o in &scene.objects {
                if !matches!(o.kind, ObjectKind::Rug | ObjectKind::FloorOutlet) {
                    continue;
                }
                if o.kind == ObjectKind::Rug {
                    rugs += 1;
                } else {
                    outlets += 1;
                }
                assert!(detail_clear(&scene, o), "seed {seed} {:?} {}", o.kind, o.id);
                let a = super::super::super::objects::build_object(o);
                let (lo, hi) = a.bounds();
                assert!(
                    lo.y.abs() < 0.001 && hi.y <= o.size.y + 0.001,
                    "mesh/support mismatch {lo}..{hi}"
                );
            }
            for cable in scene.objects.iter().filter(|o| o.kind == ObjectKind::Cable) {
                if scene.objects[cable.cable_target.unwrap()].kind == ObjectKind::FloorOutlet {
                    wired += 1;
                    assert!(equipment::cable_route_clear(&scene, cable));
                    let mut invalid = scene.clone();
                    invalid.objects[cable.cable_target.unwrap()].variant |= 8;
                    assert!(
                        !equipment::cable_route_clear(&invalid, cable),
                        "a closed lid cannot accept a plug"
                    );
                }
            }
        }
        assert!(
            rugs >= 15 && outlets >= 35,
            "sparse floor detail: rugs={rugs} outlets={outlets}"
        );
        assert!(
            wired >= 10,
            "floor cables are not reaching their real sockets: {wired}"
        );
    }
}

#[cfg(test)]
mod closure_regression {
    use super::*;
    #[test]
    fn closing_unused_floor_boxes_preserves_adjacent_wire_clearance() {
        let scene = IndoorManifest::generate(98, IndoorLayout::Conference, 1., 0).unwrap();
        super::super::super::validation::validate_layout(&scene).unwrap();
        for wire in scene.objects.iter().filter(|o| o.kind == ObjectKind::Cable) {
            assert!(equipment::cable_route_clear(&scene, wire));
        }
    }
}
