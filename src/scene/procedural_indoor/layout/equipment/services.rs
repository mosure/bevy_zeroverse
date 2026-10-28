//! Room installations attached to real backing. Electrical/life-safety fixtures
//! are fitted after furniture, before occupants; nothing is placed across glazing.
use super::*;

mod exits;
mod suspended;
use crate::scene::procedural_indoor::{architecture, objects::equipment::signals};

pub(crate) fn mounted_clear(scene: &IndoorManifest, o: &IndoorObject) -> bool {
    if o.solid || o.neighbor || o.support.is_some() {
        return false;
    }
    let mounted = match o.kind {
        ObjectKind::ExitSign => exits::mounted(scene, o),
        ObjectKind::FireDetector => suspended::detector_mounted(scene, o),
        ObjectKind::FirePullStation => pull_mounted(scene, o),
        ObjectKind::StringLights if signals::string_program(o).suspended => {
            suspended::string_mounted(scene, o)
        }
        _ => scene.wall_attachment_clear(o),
    };
    if !mounted {
        return false;
    }
    let structure: Vec<_> = scene
        .camera_obstacles()
        .into_iter()
        .chain(scene.humans.iter().map(|h| h.bounds()))
        .collect();
    // Match the existing fan/pendant reservation margin in both directions.
    // A late service must not invalidate an already accepted fixture (seed 66).
    let overlaps = |lo: Vec3, hi: Vec3, a: Vec3, b: Vec3| {
        lo.cmplt(b + Vec3::splat(0.025)).all() && hi.cmpgt(a - Vec3::splat(0.025)).all()
    };
    let reservations: Vec<_> = o.obstacle_bounds().collect();
    if reservations
        .iter()
        .any(|&(lo, hi)| structure.iter().any(|&(a, b)| overlaps(lo, hi, a, b)))
    {
        return false;
    }
    let (lo, hi) = o.bounds();
    !scene.objects.iter().filter(|p| p.id != o.id).any(|p| {
        let (a, b) = p.bounds();
        overlaps(lo, hi, a, b)
            && p.obstacle_bounds()
                .any(|(a, b)| reservations.iter().any(|&(lo, hi)| overlaps(lo, hi, a, b)))
    })
}

/// Manual controls belong beside the actual exit at reachable height. The rear
/// partition has independent glazing, so test its real aperture/frame extents.
fn pull_mounted(scene: &IndoorManifest, o: &IndoorObject) -> bool {
    let Some(door) = &scene.entry_door else {
        return false;
    };
    if o.position.xz().distance(door.center.xz()) > 2.2 || !(0.95..1.4).contains(&o.position.y) {
        return false;
    }
    if scene.wall_attachment_clear(o) {
        return true;
    }
    if (o.position.z - (door.center.z - o.size.z * 0.5 - 0.001)).abs() > 0.001
        || (o.transform().rotation * Vec3::Z).dot(Vec3::NEG_Z) < 0.99999
    {
        return false;
    }
    let x = -o.position.x;
    if x.abs() + o.size.x * 0.5 >= scene.room_size.x * 0.5 - 0.08
        || (o.position.x - door.center.x).abs() - o.size.x * 0.5 <= door.width * 0.5 + 0.09
    {
        return false;
    }
    let f =
        architecture::interior_glazing::shared_facade(scene, scene.ceiling_height(o.position.xz()));
    !f.openings.iter().any(|w| {
        x + o.size.x * 0.5 > w.min.x - 0.065
            && x - o.size.x * 0.5 < w.max.x + 0.065
            && o.position.y + o.size.y > w.min.y - 0.065
            && o.position.y < w.max.y + 0.065
    })
}

impl IndoorManifest {
    pub(in super::super::super) fn furnish_room_services(&mut self) {
        let mut r = stream(self.seed, 0x524f4f4d535256);
        exits::furnish(self, &mut r);
        // Speakers come in matched installations but do not force symmetrical
        // room-axis placement. Broad dimensions cover soundbars/column speakers.
        for kind in [ObjectKind::FireAlarm, ObjectKind::Speaker] {
            if kind == ObjectKind::Speaker && !r.random_bool(0.72) {
                continue;
            }
            let count = if kind == ObjectKind::Speaker {
                r.random_range(1..=3)
            } else {
                1
            };
            let size = match kind {
                ObjectKind::FireAlarm => Vec3::new(
                    r.random_range(0.10..0.18),
                    r.random_range(0.13..0.24),
                    r.random_range(0.035..0.07),
                ),
                ObjectKind::Speaker => {
                    if r.random_bool(0.35) {
                        Vec3::new(
                            r.random_range(0.50..1.10),
                            r.random_range(0.09..0.18),
                            r.random_range(0.07..0.13),
                        )
                    } else {
                        Vec3::new(
                            r.random_range(0.14..0.32),
                            r.random_range(0.23..0.65),
                            r.random_range(0.10..0.25),
                        )
                    }
                }
                _ => unreachable!(),
            };
            let product_seed = r.random();
            for _ in 0..count {
                for _ in 0..16 {
                    let (tf, span) = self.attachment_wall(&mut r);
                    if size.x + 0.25 >= span {
                        continue;
                    }
                    let x =
                        r.random_range((-span + size.x) * 0.5 + 0.10..(span - size.x) * 0.5 - 0.10);
                    let offset = if kind == ObjectKind::Speaker {
                        0.46
                    } else {
                        0.5
                    };
                    let mut p = tf.transform_point(Vec3::new(x, 0., size.z * offset + 0.001));
                    let roof = self.ceiling_height(p.xz());
                    p.y = r.random_range(1.80_f32..2.15).min(roof - size.y - 0.28);
                    if p.y < 1.55 {
                        continue;
                    }
                    let mut o = self.candidate(
                        kind,
                        p,
                        size,
                        tf.rotation.to_euler(EulerRot::YXZ).0,
                        &mut r,
                    );
                    o.solid = false;
                    if kind == ObjectKind::Speaker {
                        o.seed = product_seed;
                    } else {
                        o.variant = r.random_range(0..4);
                    }
                    if kind == ObjectKind::FireAlarm && o.variant == 3 {
                        o.size.y = o.size.x;
                    }
                    // A bell's rear face is shallower than a rectangular alarm
                    // enclosure. Fit the sampled geometry, not the nominal box.
                    let mounting = super::super::super::objects::wall::mount_offset(&o)
                        .expect("wall service has a mounting plane");
                    o.position += tf.rotation * Vec3::Z * (mounting - size.z * offset - 0.001);
                    if mounted_clear(self, &o) {
                        self.objects.push(o);
                        break;
                    }
                }
            }
        }
        self.furnish_suspended_services();
        self.furnish_pull_station();
    }
    fn furnish_pull_station(&mut self) {
        let Some(door) = self.entry_door.clone() else {
            return;
        };
        let mut r = stream(self.seed, 0x50554c4c535441);
        for i in 0..32 {
            let size = Vec3::new(
                r.random_range(0.085..0.14),
                r.random_range(0.105..0.18),
                r.random_range(0.032..0.065),
            );
            let (position, yaw) = if i < 20 {
                (
                    Vec3::new(
                        door.center.x
                            + if i % 2 == 0 { 1. } else { -1. }
                                * (door.width * 0.5 + r.random_range(0.18..0.95)),
                        r.random_range(1.0..1.28),
                        door.center.z - size.z * 0.5 - 0.001,
                    ),
                    PI,
                )
            } else {
                let (tf, span) = self.attachment_wall(&mut r);
                (
                    tf.transform_point(Vec3::new(
                        r.random_range(-0.40..0.40) * span,
                        r.random_range(1.0..1.28),
                        size.z * 0.5 + 0.001,
                    )),
                    tf.rotation.to_euler(EulerRot::YXZ).0,
                )
            };
            let mut o = self.candidate(ObjectKind::FirePullStation, position, size, yaw, &mut r);
            o.variant = r.random_range(0..4);
            o.solid = false;
            if mounted_clear(self, &o) {
                self.objects.push(o);
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn room_services_and_area_rugs_have_real_mounts_support_and_diverse_coverage() {
        let started = std::time::Instant::now();
        let mut distribution = std::collections::BTreeMap::<String, usize>::new();
        let mut counts = std::collections::BTreeMap::<String, usize>::new();
        let mut supported = 0;
        let mut supported_kinds = std::collections::BTreeMap::<String, usize>::new();
        let mut rugs = 0;
        let mut max_rug_area = 0_f32;
        let mut sample_seeds = std::collections::BTreeMap::<String, u64>::new();
        for seed in 0..128 {
            let scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.7, 0, 0.25)
                    .unwrap();
            crate::scene::procedural_indoor::validation::validate_layout(&scene)
                .unwrap_or_else(|e| panic!("{seed}: {e}"));
            for o in &scene.objects {
                if matches!(
                    o.kind,
                    ObjectKind::ExitSign
                        | ObjectKind::FireAlarm
                        | ObjectKind::FireDetector
                        | ObjectKind::FirePullStation
                        | ObjectKind::StringLights
                ) || (o.kind == ObjectKind::Speaker && o.support.is_none() && !o.solid)
                {
                    assert!(mounted_clear(&scene, o), "{seed} {:?}", o.kind);
                    let name = format!("{:?}", o.kind);
                    *counts.entry(name.clone()).or_default() += 1;
                    sample_seeds.entry(name).or_insert(seed);
                    let a = super::super::super::super::objects::build_object(o);
                    super::super::super::super::objects::audit::check(o, &a).unwrap();
                    if o.kind == ObjectKind::ExitSign {
                        let p = signals::exit::parameters(o);
                        for value in [
                            format!("exit_mount_{:?}", p.mount),
                            format!("exit_legend_{:?}", p.legend),
                        ] {
                            *distribution.entry(value.clone()).or_default() += 1;
                            sample_seeds.entry(value).or_insert(seed);
                        }
                        let mut floating = o.clone();
                        if p.mount == signals::exit::ExitMount::Wall {
                            floating.position.z -= 0.1;
                        } else {
                            floating.position.y -= 0.1;
                        }
                        assert!(!mounted_clear(&scene, &floating));
                    }
                    if o.kind == ObjectKind::StringLights {
                        let p = signals::string_program(o);
                        let mode = if p.suspended { "suspended" } else { "wall" };
                        *distribution.entry(mode.into()).or_default() += 1;
                        sample_seeds.entry(mode.into()).or_insert(seed);
                        *distribution
                            .entry(format!("palette_{}_colors", p.colors_srgb.len()))
                            .or_default() += 1;
                        *distribution.entry(format!("{}_bays", p.bays)).or_default() += 1;
                        *distribution
                            .entry(format!("bulb_shape_{}", p.shape))
                            .or_default() += 1;
                        if p.suspended {
                            let mut floating = o.clone();
                            floating.position.y -= 0.1;
                            assert!(!mounted_clear(&scene, &floating));
                            // A new object on an actual cable segment must fail,
                            // while empty air under the strand remains available.
                            let (lo, hi) = o.obstacle_bounds().next().unwrap();
                            let mut blocked = scene.clone();
                            let mut blocker = super::super::super::super::objects::audit::fixture(
                                ObjectKind::StorageBox,
                                0,
                            );
                            blocker.id = blocked.objects.len();
                            blocker.size = Vec3::splat(0.05);
                            blocker.position = (lo + hi) * 0.5 - Vec3::Y * 0.025;
                            blocked.objects.push(blocker);
                            assert!(!mounted_clear(&blocked, o));
                        }
                    }
                    if matches!(
                        o.kind,
                        ObjectKind::FireAlarm
                            | ObjectKind::FireDetector
                            | ObjectKind::FirePullStation
                    ) {
                        let p = signals::fire::program(o);
                        *distribution
                            .entry(format!("{:?}_style_{}", o.kind, p.style))
                            .or_default() += 1;
                        *distribution
                            .entry(format!("body_finish_{}", p.finish))
                            .or_default() += 1;
                    }
                    if o.kind == ObjectKind::FireDetector {
                        let mut floating = o.clone();
                        floating.position.y -= 0.1;
                        assert!(!mounted_clear(&scene, &floating));
                    }
                }
            }
            for rug in &scene.rug_supports {
                rugs += 1;
                let o = &scene.objects[rug.object];
                max_rug_area = max_rug_area.max(o.size.x * o.size.z);
                sample_seeds.entry("AreaRug".into()).or_insert(seed);
                assert!(super::super::super::floor::detail_clear(&scene, o));
                for other in scene.objects.iter().filter(|o| {
                    o.solid
                        && (o.position.y - rug.top).abs() < 0.001
                        && rug.contains(o.position.xz())
                }) {
                    assert!(rug.supports(other.bounds().0, other.bounds().1));
                    assert!(scene.floor_support_clear(other.bounds().0, other.bounds().1));
                    supported += 1;
                    *supported_kinds
                        .entry(format!("{:?}", other.kind))
                        .or_default() += 1;
                }
            }
            if seed < 4 {
                assert_eq!(
                    scene,
                    IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.7, 0, 0.25)
                        .unwrap()
                );
            }
        }
        println!(
            "room services audit: {}",
            serde_json::json!({"rooms":128,"counts":counts,"area_rugs":rugs,"supported_furniture":supported,"supported_kinds":supported_kinds,"max_rug_area_m2":max_rug_area,"sample_seeds":sample_seeds,"distributions":distribution,"elapsed_s":started.elapsed().as_secs_f64()})
        );
        for kind in [
            "ExitSign",
            "FireAlarm",
            "FireDetector",
            "FirePullStation",
            "Speaker",
            "StringLights",
        ] {
            assert!(
                counts.get(kind).copied().unwrap_or(0) > 3,
                "{kind}: {counts:?}"
            );
        }
        for mount in ["Wall", "CeilingCanopy", "PendantRods", "PendantCables"] {
            assert!(
                distribution
                    .get(&format!("exit_mount_{mount}"))
                    .copied()
                    .unwrap_or(0)
                    >= 2,
                "{distribution:?}"
            );
        }
        for legend in ["EuropeanPictogram", "AmericanRed", "AmericanGreen"] {
            assert!(
                distribution
                    .get(&format!("exit_legend_{legend}"))
                    .copied()
                    .unwrap_or(0)
                    >= 5,
                "{distribution:?}"
            );
        }
        assert!(supported > 3 && max_rug_area > 5.);
        assert!(distribution.get("suspended").copied().unwrap_or(0) > 3);
        assert!(distribution.get("wall").copied().unwrap_or(0) > 3);
        assert!(distribution
            .keys()
            .any(|k| k.starts_with("palette_") && k != "palette_1_colors"));
    }
}
