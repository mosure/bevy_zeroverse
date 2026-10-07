//! Rare completion of otherwise valid groups in compact or concave rooms.
//! The ordinary seeded search always runs first and retains its exact results.
use super::*;

// Occupied seats can expose narrow, distinct viewing pockets at a worktop.
// Compact stepped lounges also need distinct anchors for landscape frusta.
// Preserve enough alternate anchors for the rare recovery path; ordinary
// successful groups keep their exact search, cost and acceptance predicates.
pub(super) const SAVED_GROUPS: usize = 12;
const NEARBY_ATTEMPTS: usize = 24;

pub(super) fn retain_diverse(groups: &mut Vec<Vec<IndoorCamera>>, cameras: &[IndoorCamera]) {
    let mut incoming = cameras.to_vec();
    if groups.is_empty() {
        groups.push(incoming);
        return;
    }
    // Reserve the longest partial group, while retaining distinct reference
    // domains independently of cardinality. Sparse all-singleton retention
    // keeps its earlier ordering and diversity replacement exactly.
    if incoming.len() > groups[0].len() {
        std::mem::swap(&mut incoming, &mut groups[0]);
    }
    if groups.len() < SAVED_GROUPS {
        groups.push(incoming);
        return;
    }
    let protected = usize::from(groups[0].len() > 1);
    let (mut replace, mut separation) = (protected, f32::INFINITY);
    for (i, group) in groups.iter().enumerate().skip(protected) {
        for other in &groups[i + 1..] {
            let distance = group[0].start.distance(other[0].start);
            if distance < separation {
                (replace, separation) = (i, distance);
            }
        }
    }
    let novelty = groups
        .iter()
        .map(|g| g[0].start.distance(incoming[0].start))
        .fold(f32::INFINITY, f32::min);
    if novelty > separation {
        groups[replace] = incoming;
    }
}

struct Prefix {
    cameras: Vec<IndoorCamera>,
    tracks: Vec<Track>,
}
impl Prefix {
    fn new(cameras: &[IndoorCamera]) -> Self {
        Self {
            cameras: cameras.to_vec(),
            tracks: cameras.iter().map(Track::new).collect(),
        }
    }
}
fn retain_prefix(prefixes: &mut Vec<Prefix>, incoming: Prefix) {
    let origin = incoming.cameras.last().unwrap().start.xz();
    if prefixes.iter().any(|p| p.cameras == incoming.cameras) {
        return;
    }
    if prefixes.len() < SAVED_GROUPS {
        prefixes.push(incoming);
        return;
    }
    let (mut replace, mut separation) = (0, f32::INFINITY);
    for (i, prefix) in prefixes.iter().enumerate() {
        for other in &prefixes[i + 1..] {
            let distance = prefix
                .cameras
                .last()
                .unwrap()
                .start
                .xz()
                .distance(other.cameras.last().unwrap().start.xz());
            if distance < separation {
                (replace, separation) = (i, distance);
            }
        }
    }
    let novelty = prefixes
        .iter()
        .map(|p| p.cameras.last().unwrap().start.xz().distance(origin))
        .fold(f32::INFINITY, f32::min);
    if novelty > separation {
        prefixes[replace] = incoming;
    }
}

fn alternate_reference(
    scene: &IndoorManifest,
    source: &IndoorCamera,
    rng: &mut rand_chacha::ChaCha8Rng,
    coverage: &Coverage,
    retain_source_frame: bool,
) -> (Option<IndoorCamera>, usize) {
    let Some(human) = scene
        .humans
        .iter()
        .filter(|h| {
            !h.neighbor
                && (!scene.camera_settings.primary_room || scene.in_primary_room(h.position, 0.))
        })
        .min_by(|a, b| {
            a.transform()
                .transform_point(a.joints[2])
                .distance_squared(source.target)
                .total_cmp(
                    &b.transform()
                        .transform_point(b.joints[2])
                        .distance_squared(source.target),
                )
        })
    else {
        return (None, 0);
    };
    let head = human.transform().transform_point(human.joints[4]);
    let body_delta = human.transform().transform_point(human.joints[2]) - source.start;
    let body_forward = if body_delta.x.abs() > body_delta.z.abs() {
        Vec3::X * body_delta.x.signum()
    } else {
        Vec3::Z * body_delta.z.signum()
    };
    let forward = (head - source.start).with_y(0.).normalize_or_zero();
    let right = forward.cross(Vec3::Y);
    let aim_axis = if right.x.abs() >= right.z.abs() {
        Vec3::X * right.x.signum()
    } else {
        Vec3::Z * right.z.signum()
    };
    let floor = scene.envelope.as_ref().map_or_else(
        || scene.floor_height(source.start.xz()),
        |e| e.support_height(source.start),
    );
    let length = if scene.camera_settings.path_length_max == 0. {
        0.
    } else {
        scene
            .camera_settings
            .path_length_min
            .max(0.001)
            .min(scene.camera_settings.path_length_max)
    };
    // Nine actual-body framings, each with eight short directions. This comes
    // from the existing per-view budget. High room views can see above a guard;
    // the small horizontal aim offsets keep the person on two proxy columns.
    for attempt in 0..72 {
        let frame = attempt / 8;
        let nearby_body = retain_source_frame && frame < 3;
        let original_frame = retain_source_frame && frame == 5;
        let height = if original_frame {
            source.start.y
        } else if nearby_body {
            floor + 3.5
        } else {
            floor + [3.5, 2.8, 2.2][frame / 3]
        };
        let aim = if original_frame {
            source.target
        } else if nearby_body {
            head.with_y(height - 0.85) + aim_axis * [-0.4, 0.4, -0.2][frame]
        } else {
            head.with_y(height - 0.6) + aim_axis * [0.4, -0.4, 0.2][frame % 3]
        };
        let origin =
            (source.start + body_forward * if nearby_body { 1.2 } else { 0. }).with_y(height);
        let optical = (aim - origin).normalize_or_zero();
        let direction = [
            Vec3::X,
            -Vec3::X,
            Vec3::Z,
            -Vec3::Z,
            Vec3::Y,
            -Vec3::Y,
            optical,
            optical.cross(Vec3::Y).normalize_or_zero(),
        ][attempt % 8];
        let end = origin + direction * length;
        let mut camera = IndoorCamera {
            start: origin,
            end,
            target: aim,
            fov_degrees: if original_frame {
                source.fov_degrees
            } else {
                106.
            },
            motion: (length > 0.).then_some(super::super::CameraMotion {
                orientations: None,
                route: Vec::new(),
                control: [origin.lerp(end, 1. / 3.), origin.lerp(end, 2. / 3.)],
                target_end: aim,
                roll: [0.; 2],
            }),
        };
        if let Some(handheld) = &scene.camera_settings.handheld {
            handheld.apply(&mut camera, rng);
        }
        let path_length = camera.path_length();
        if !scene.camera_clear(camera.start)
            || !scene.camera_view_clear(camera.start, camera.target)
            || path_length + 1e-4 < scene.camera_settings.path_length_min
            || path_length > scene.camera_settings.path_length_max + 1e-4
            || !scene.camera_curve_clear(&camera)
            || !coverage.suitable(&camera, true)
        {
            continue;
        }
        return (Some(camera), attempt + 1);
    }
    (None, 72)
}

pub(super) fn complete(
    scene: &mut IndoorManifest,
    groups: &[Vec<IndoorCamera>],
    count: usize,
    policy: &MultiViewSettings,
    rng: &mut rand_chacha::ChaCha8Rng,
    coverage: &Coverage,
    rejected: &mut [usize; 6],
) -> usize {
    let mut most_placed = 0;
    let (lo, hi) = if scene.camera_settings.primary_room {
        scene.primary_room_bounds()
    } else {
        let half = scene.room_size.xz() * 0.5;
        (-half, half)
    };
    let fixture_margin = scene.program.as_ref().map_or(0.5, |p| p.light_drop)
        + 0.04
        + super::super::super::layout::CAMERA_CLEARANCE;
    let content: Vec<_> = scene
        .objects
        .iter()
        .filter(|o| {
            o.solid
                && !o.neighbor
                && (!scene.camera_settings.primary_room || scene.in_primary_room(o.position, 0.))
        })
        .take(32)
        .map(|o| o.position + Vec3::Y * (o.size.y * 0.85).clamp(0.75, 1.55))
        .collect();
    let saved_views: Vec<_> = groups.iter().flat_map(|group| group.iter()).collect();
    let beam = groups
        .iter()
        .any(|group| group.len() > 1 && group.len() < count);
    for group in groups {
        // A legal pair can complete in its original reference domain even when
        // the first independently valid alternate cannot. Explore that intact
        // pair first, then reserve its spent ceiling from the alternate branch.
        let branches = if group.len() == 2 { 2 } else { 1 };
        for branch in 0..branches {
            scene.cameras.clone_from(group);
            let mut attempts = if branches == 2 && branch == 0 {
                64
            } else {
                VIEW_ATTEMPTS
            };
            if branches == 2 && branch == 1 {
                let reserved = 64 * (count - group.len());
                attempts -= reserved.div_ceil(count - 1);
            }
            let mut replaced_reference = false;
            if group.len() <= 2 && (branches == 1 || branch == 1) {
                // One accepted torso view or a cramped pair can prevent completion:
                // adding the third view introduces the full horizontal spread gate.
                // Restart from a validated anatomical aim/eye-height when available,
                // charging its proposals to the same per-view recovery budget.
                let (alternate, spent) =
                    alternate_reference(scene, &group[0], rng, coverage, group.len() > 1);
                attempts -= spent;
                if let Some(alternate) = alternate {
                    // A replacement reference need not retain the failed anchor.
                    // Co-located horizontal origins would prevent a spread rig.
                    scene.cameras.clear();
                    scene.cameras.push(alternate);
                    replaced_reference = true;
                }
            }
            let head_target = scene
                .humans
                .iter()
                .filter(|h| !h.neighbor)
                .min_by(|a, b| {
                    a.transform()
                        .transform_point(a.joints[2])
                        .distance_squared(group[0].target)
                        .total_cmp(
                            &b.transform()
                                .transform_point(b.joints[2])
                                .distance_squared(group[0].target),
                        )
                })
                .map_or(group[0].target, |h| {
                    h.transform().transform_point(h.joints[4])
                });
            let reference = scene.cameras[0].clone();
            let views: Vec<_> = CHECK_TIMES
                .into_iter()
                .map(|t| coverage.view(&reference, scene.camera_aspect_ratio, t))
                .collect();
            let mut tracks: Vec<_> = scene.cameras.iter().map(Track::new).collect();
            let reachable = [lo, hi, Vec2::new(lo.x, hi.y), Vec2::new(hi.x, lo.y)]
                .into_iter()
                .map(|p| p.distance(reference.start.xz()))
                .fold(0.0_f32, f32::max)
                .hypot(0.9)
                .min(policy.max_baseline);
            if reachable < policy.min_baseline.max(policy.min_reference_baseline) {
                continue;
            }
            // Preserve a small frontier instead of locking on the first legal pair.
            // Parent and child frontiers each have a bounded set of accepted prefixes;
            // cached tracks avoid resampling retained paths. The loop still spends
            // at most one candidate-parent consideration per original proposal slot.
            let mut parents = vec![Prefix::new(&scene.cameras)];
            while scene.cameras.len() < count {
                let mut found = None;
                let mut children = Vec::new();
                for attempt in 0..attempts {
                    if beam {
                        let parent = &parents[attempt % parents.len()];
                        scene.cameras.clone_from(&parent.cameras);
                        tracks = parent.tracks.iter().map(|t| Track(t.0)).collect();
                    }
                    let saved_index = if beam {
                        attempt / parents.len()
                    } else {
                        attempt
                    };
                    let camera = if (replaced_reference || beam) && saved_index < saved_views.len()
                    {
                        // The saved person-facing paths already proved clearance
                        // and full framing at low eye heights. Reuse them as rare
                        // counterparts before proposing new paths. Co-located
                        // horizontal origins cannot form a spread rig later.
                        let camera = (*saved_views[saved_index]).clone();
                        if !beam
                            && camera.start.xz().distance(reference.start.xz())
                                < policy.min_baseline
                        {
                            continue;
                        }
                        camera
                    } else if beam && attempt < saved_views.len() * parents.len() + NEARBY_ATTEMPTS
                    {
                        // Valid counterpart paths reveal small navigable pockets in
                        // furnished rooms. Preserve their optical framing while the
                        // ordinary deformation varies motion independently. Every
                        // candidate remains subject to the complete group gates.
                        let index = (attempt - saved_views.len() * parents.len()) / parents.len();
                        let template = if scene.cameras.len() > 1 {
                            &scene.cameras[1 + (index / 6) % (scene.cameras.len() - 1)]
                        } else {
                            saved_views[(1 + index / 6) % saved_views.len()]
                        };
                        let (direction, radius) = if scene.cameras.len() == 2 {
                            // A third origin needs lateral area relative to the
                            // existing pair. Vertical offsets have zero horizontal
                            // spread at t=0; a fixed pairwise-clearance offset also
                            // becomes too small as the reference baseline grows.
                            let pair = scene.cameras[1].start - reference.start;
                            let lateral = pair.with_y(0.).normalize_or_zero().cross(Vec3::Y);
                            (
                                [Vec3::X, -Vec3::X, Vec3::Z, -Vec3::Z, lateral, -lateral]
                                    [index % 6],
                                (policy.min_baseline * 1.15).max(pair.xz().length() * 0.35),
                            )
                        } else {
                            (
                                [Vec3::X, -Vec3::X, Vec3::Z, -Vec3::Z, Vec3::Y, -Vec3::Y]
                                    [index % 6],
                                policy.min_baseline * 1.15,
                            )
                        };
                        let origin = template.start + direction * radius;
                        if !scene.camera_clear(origin) {
                            rejected[0] += 1;
                            continue;
                        }
                        let mode = (attempt - saved_views.len() * parents.len()) / 6;
                        let mut camera = if mode == 0 {
                            let mut camera = propose(
                                template,
                                template
                                    .start
                                    .distance(template.end)
                                    .max(template.path_length() * 0.5)
                                    .max(0.001),
                                reachable,
                                Vec2::ZERO,
                                policy,
                                rng,
                            );
                            let offset = origin - camera.start;
                            let aim_offset = template.target - camera.target;
                            camera.start += offset;
                            camera.end += offset;
                            camera.target = template.target;
                            camera.fov_degrees = template.fov_degrees;
                            if let Some(motion) = &mut camera.motion {
                                for point in motion.control.iter_mut().chain(&mut motion.route) {
                                    *point += offset;
                                }
                                motion.target_end += aim_offset;
                            }
                            camera
                        } else {
                            // Short authored-range paths keep the proxy framing
                            // stable inside a verified compact viewing pocket.
                            // Lens/motion alternatives spend these same 24 slots.
                            let length = if scene.camera_settings.path_length_max == 0. {
                                0.
                            } else {
                                scene
                                    .camera_settings
                                    .path_length_min
                                    .max(0.001)
                                    .min(scene.camera_settings.path_length_max)
                            };
                            let optical = (template.target - origin).normalize_or_zero();
                            let direction = [optical, -optical, Vec3::Y][mode - 1];
                            let end = origin + direction * length;
                            IndoorCamera {
                                start: origin,
                                end,
                                target: template.target,
                                fov_degrees: [template.fov_degrees, 96., 106.][mode - 1],
                                motion: (length > 0.).then_some(super::super::CameraMotion {
                                    orientations: None,
                                    route: Vec::new(),
                                    control: [origin.lerp(end, 1. / 3.), origin.lerp(end, 2. / 3.)],
                                    target_end: template.target,
                                    roll: [0.; 2],
                                }),
                            }
                        };
                        if let Some(handheld) = &scene.camera_settings.handheld {
                            handheld.apply(&mut camera, rng);
                        }
                        camera
                    } else if beam
                        && replaced_reference
                        && !content.is_empty()
                        && saved_index - saved_views.len() - NEARBY_ATTEMPTS / parents.len()
                            < 24 * content.len()
                    {
                        // High ceiling-relative views can share the reference's
                        // content from above a mezzanine guard. Enumerate both sides
                        // and short longitudinal offsets, retaining all acceptance
                        // gates below. Each parent/candidate consumes one view slot.
                        let index =
                            saved_index - saved_views.len() - NEARBY_ATTEMPTS / parents.len();
                        let spatial = (index / content.len()) % 12;
                        let body_delta = group[0].target - reference.start;
                        let forward = if body_delta.x.abs() > body_delta.z.abs() {
                            Vec3::X * body_delta.x.signum()
                        } else {
                            Vec3::Z * body_delta.z.signum()
                        };
                        let right = forward.cross(Vec3::Y);
                        let side = if spatial / 6 == 0 { -1. } else { 1. };
                        let along = [-0.3, 0., 0.3][(spatial / 2) % 3];
                        let xz = (reference.start
                            + forward * along
                            + right * (side * policy.min_baseline * 1.6))
                            .xz();
                        let mut origin = Vec3::new(
                            xz.x,
                            scene.ceiling_height(xz) - fixture_margin - [0.7, 0.3][spatial % 2],
                            xz.y,
                        );
                        // Small optical changes can move a surface between the
                        // coverage proxy's rows. Try a decimeter lens grid first,
                        // then the continuous roof-relative poses, within this same
                        // fixed budget; clearance and visibility still decide.
                        if index < 12 * content.len() {
                            origin = (origin * 10.).round() / 10.;
                        }
                        let length = if scene.camera_settings.path_length_max == 0. {
                            0.
                        } else {
                            scene
                                .camera_settings
                                .path_length_min
                                .max(0.001)
                                .min(scene.camera_settings.path_length_max)
                        };
                        let direction =
                            [Vec3::Z, -Vec3::X, Vec3::X, -Vec3::Y, -Vec3::Z, Vec3::Y][spatial % 6];
                        let end = origin + direction * length;
                        let aim = content[index % content.len()];
                        let mut camera = IndoorCamera {
                            start: origin,
                            end,
                            target: aim,
                            fov_degrees: 106.,
                            motion: (length > 0.).then_some(super::super::CameraMotion {
                                orientations: None,
                                route: Vec::new(),
                                control: [origin.lerp(end, 1. / 3.), origin.lerp(end, 2. / 3.)],
                                target_end: aim,
                                roll: [0.; 2],
                            }),
                        };
                        if let Some(handheld) = &scene.camera_settings.handheld {
                            handheld.apply(&mut camera, rng);
                        }
                        camera
                    } else {
                        // Radial offsets spend almost all proposals outside a narrow
                        // usable room. Draw the real origin first, including its local
                        // floor/roof interval, while retaining independent deformation.
                        // Sparse rooms need the person to supply their fourth label;
                        // include origins near verified views and their shared content.
                        let template = if attempt % 3 == 0 && scene.cameras.len() > 1 {
                            &scene.cameras[rng.random_range(1..scene.cameras.len())]
                        } else {
                            &reference
                        };
                        let xz = if attempt % 3 == 2 {
                            Vec2::new(
                                rng.random_range(lo.x + 0.5..hi.x - 0.5),
                                rng.random_range(lo.y + 0.5..hi.y - 0.5),
                            )
                        } else {
                            let angle = rng.random_range(0.0..std::f32::consts::TAU);
                            let (center, radius) = if attempt % 3 == 0 {
                                (template.start.xz(), rng.random_range(0.3..1.2))
                            } else {
                                (reference.target.xz(), rng.random_range(1.65..3.4))
                            };
                            center + Vec2::new(angle.cos(), angle.sin()) * radius
                        };
                        let floor = scene
                            .envelope
                            .as_ref()
                            .and_then(|e| e.mezzanine.as_ref())
                            .filter(|m| {
                                m.deck.contains(xz) && (!replaced_reference || attempt % 2 == 0)
                            })
                            .map_or_else(|| scene.floor_height(xz), |m| m.deck.height);
                        let low = floor + 0.70;
                        // Eye height is relative to the supporting floor. An absolute
                        // 3.25 m cap excludes valid origins above a mezzanine outright.
                        let high = (scene.ceiling_height(xz) - fixture_margin).min(floor + 3.25);
                        if low > high {
                            rejected[0] += 1;
                            continue;
                        }
                        let height = if attempt % 3 == 2 {
                            rng.random_range(low..=high)
                        } else {
                            (template.start.y + rng.random_range(-0.3..0.3)).clamp(low, high)
                        };
                        let origin = Vec3::new(xz.x, height, xz.y);
                        if !scene.camera_clear(origin) {
                            rejected[0] += 1;
                            continue;
                        }
                        let mut camera = propose(
                            template,
                            template
                                .start
                                .distance(template.end)
                                .max(template.path_length() * 0.5)
                                .max(0.001),
                            reachable,
                            // The origin already supplies the height. A second radial
                            // height draw could have no interval under a sloping roof.
                            Vec2::ZERO,
                            policy,
                            rng,
                        );
                        let offset = origin - camera.start;
                        camera.start += offset;
                        camera.end += offset;
                        // Keep verified content, original aim drift, and other actual
                        // room content in the proposal mix. A tall mezzanine room may
                        // have very few separated views that all frame the person.
                        let aim = match attempt % 4 {
                            0 => template.target,
                            1 if replaced_reference => group[0].target,
                            2 if replaced_reference => head_target,
                            1 => template.target,
                            2 => camera.target,
                            _ if !content.is_empty() => content[rng.random_range(0..content.len())],
                            _ => Vec3::new(
                                (lo.x + hi.x) * 0.5,
                                (low + high) * 0.5,
                                (lo.y + hi.y) * 0.5,
                            ),
                        };
                        let aim_offset = aim - camera.target;
                        camera.target = aim;
                        // Wide alternatives stay in the original focal-length range.
                        camera.fov_degrees = [template.fov_degrees, 84., 96., 106.][attempt % 4];
                        if let Some(motion) = &mut camera.motion {
                            for point in motion.control.iter_mut().chain(&mut motion.route) {
                                *point += offset;
                            }
                            motion.target_end += aim_offset;
                        }
                        if let Some(handheld) = &scene.camera_settings.handheld {
                            handheld.apply(&mut camera, rng);
                        }
                        camera
                    };
                    let track = Track::new(&camera);
                    let rejection = if !scene.camera_clear(camera.start) {
                        Some(0)
                    } else if !geometry(&tracks, Some(&track)).unwrap().accepts(policy) {
                        Some(1)
                    } else if camera.path_length() + 1e-4 < scene.camera_settings.path_length_min
                        || camera.path_length() > scene.camera_settings.path_length_max + 1e-4
                    {
                        Some(2)
                    } else if !scene.camera_curve_clear(&camera) {
                        Some(3)
                    } else if !coverage.suitable(&camera, false) {
                        Some(4)
                    } else if !policy.accepts(&pair_samples(
                        coverage,
                        &reference,
                        &views,
                        &camera,
                        scene.camera_aspect_ratio,
                    )) {
                        Some(5)
                    } else {
                        None
                    };
                    if let Some(reason) = rejection {
                        rejected[reason] += 1;
                        continue;
                    }
                    if beam {
                        let mut child = Prefix {
                            cameras: scene.cameras.clone(),
                            tracks: tracks.iter().map(|t| Track(t.0)).collect(),
                        };
                        child.cameras.push(camera);
                        child.tracks.push(track);
                        most_placed = most_placed.max(child.cameras.len());
                        if child.cameras.len() == count {
                            scene.cameras = child.cameras;
                            return count;
                        }
                        retain_prefix(&mut children, child);
                    } else {
                        found = Some((camera, track));
                        break;
                    }
                }
                if beam {
                    if children.is_empty() {
                        break;
                    }
                    parents = children;
                    scene.cameras.clone_from(&parents[0].cameras);
                    tracks = parents[0].tracks.iter().map(|t| Track(t.0)).collect();
                    continue;
                }
                let Some((camera, track)) = found else {
                    break;
                };
                scene.cameras.push(camera);
                tracks.push(track);
            }
            most_placed = most_placed.max(scene.cameras.len());
            if scene.cameras.len() == count {
                return count;
            }
        }
    }
    most_placed
}
