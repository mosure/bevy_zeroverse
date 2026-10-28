//! Seeded work-area transforms and mixed clustered/uniform surface proposals.
//! All proposals still pass the same true-outline and peer-overlap checks.
use super::*;

/// Preserve a user's comfortable viewing cone without imposing exact optical
/// convergence on their seat. Reuse the same adjustment when seating settles.
pub(super) fn screen_facing_yaw(screen: &IndoorObject, viewer: Vec3) -> f32 {
    let delta = viewer - screen.position;
    let range = if screen.kind == ObjectKind::Laptop {
        0.18
    } else {
        0.19
    };
    delta.x.atan2(delta.z) + stream(screen.seed, 0x5343_5245_454e_5941).random_range(-range..range)
}
/// A desk's equipment and paper use vary continuously before any item is
/// proposed. Display geometry and every accepted footprint remain independent.
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct WorkspaceProgram {
    pub screen_affinity: f32,
    pub monitor_affinity: f32,
    pub companion_affinity: f32,
    pub docking_affinity: f32,
    pub paper_affinity: f32,
    pub display_span_fraction: f32,
    pub rear_fraction: f32,
    pub asymmetry_fraction: f32,
    pub accessory_affinity: f32,
    pub monitor_gap_m: f32,
    pub monitor_depth_scatter_m: f32,
    pub monitor_size_scatter: f32,
    pub workstation_heading: f32,
    pub laptop_depth_fraction: f32,
    pub docking_side: f32,
}

impl WorkspaceProgram {
    fn monitor_width(&self, width: f32, rng: &mut ChaCha8Rng) -> f32 {
        // Zero scatter is a valid coherent workstation, including when a
        // tiny sampled spread rounds both floating-point endpoints to 1.
        width * rng.random_range(1. - self.monitor_size_scatter..=1. + self.monitor_size_scatter)
    }
}

pub(crate) fn workspace_parameters(support: &IndoorObject) -> WorkspaceProgram {
    let mut rng = stream(support.seed, 907);
    let mut arrangement = stream(support.seed, 0x574f524b504f5345);
    WorkspaceProgram {
        screen_affinity: rng.random_range(0.38..0.98),
        monitor_affinity: rng.random_range(0.14..0.92),
        companion_affinity: rng.random_range(0.04..0.82),
        docking_affinity: rng.random_range(0.06..0.76),
        paper_affinity: rng.random_range(0.18..0.98),
        display_span_fraction: rng.random_range(0.34..0.66),
        rear_fraction: rng.random_range(0.11..0.27),
        asymmetry_fraction: rng.random_range(-0.09..0.09),
        accessory_affinity: rng.random_range(0.20..0.95),
        monitor_gap_m: arrangement.random_range(0.025..0.16),
        monitor_depth_scatter_m: arrangement.random_range(0.0..0.065),
        monitor_size_scatter: arrangement.random_range(0.0..0.16),
        workstation_heading: arrangement.random_range(-0.16..0.16),
        laptop_depth_fraction: arrangement.random_range(-0.20..0.13),
        docking_side: if arrangement.random_bool(0.5) {
            -1.
        } else {
            1.
        },
    }
}

impl IndoorManifest {
    /// Functional work areas vary within every room activity, rather than
    /// restricting desktop displays to one named office profile. Proposals
    /// reference the actual chair and all use the unchanged support/peer gate.
    pub(super) fn decorate_workspace(&mut self, support: &IndoorObject) {
        let p = workspace_parameters(support);
        let mut rng = stream(support.seed, 908);
        let screen = rng.random_bool(p.screen_affinity as f64);
        let monitor = screen
            && support.kind == ObjectKind::Desk
            && support.size.y >= 0.65
            && rng.random_bool(p.monitor_affinity as f64);
        let twin = monitor
            && support.size.x >= 1.18
            && support.size.z >= 0.62
            && rng.random_bool(p.companion_affinity as f64);
        let docked = monitor && !twin && rng.random_bool(p.docking_affinity as f64);
        let rear = -support.size.z * p.rear_fraction;
        let centre = p.asymmetry_fraction * support.size.x;
        if monitor {
            let count = if twin { 2 } else { 1 };
            let span = (support.size.x * p.display_span_fraction).min(1.12);
            let width = if twin {
                ((span - p.monitor_gap_m) * 0.5).clamp(0.34, 0.53)
            } else {
                rng.random_range(0.39_f32..0.83).min(support.size.x * 0.58)
            };
            for i in 0..count {
                let side = if twin { i as f32 * 2. - 1. } else { 0. };
                let width = p.monitor_width(width, &mut rng);
                let x = centre + side * (width * 0.5 + p.monitor_gap_m * 0.5)
                    - if docked {
                        p.docking_side * support.size.x * 0.12
                    } else {
                        0.
                    };
                // Portrait and broad displays reserve their true dimensions,
                // including stand and tilt; there is no post-placement scale.
                let portrait = rng.random_bool(0.17);
                let width = if portrait { width.min(0.36) } else { width };
                let height = if portrait {
                    rng.random_range(0.44..0.60)
                } else {
                    rng.random_range(0.33..0.49)
                };
                self.supported_prop(
                    support,
                    ObjectKind::Monitor,
                    Vec3::new(
                        x,
                        0.,
                        rear + rng
                            .random_range(-p.monitor_depth_scatter_m..=p.monitor_depth_scatter_m),
                    ),
                    Vec3::new(width, height, rng.random_range(0.19..0.28)),
                    p.workstation_heading + side * rng.random_range(0.08..0.25),
                    &mut rng,
                );
            }
        }
        if screen && (!monitor || docked) {
            let x = if docked {
                p.docking_side * support.size.x * rng.random_range(0.24..0.32)
            } else {
                centre
            };
            self.supported_prop(
                support,
                ObjectKind::Laptop,
                Vec3::new(x, 0., support.size.z * p.laptop_depth_fraction),
                Vec3::new(
                    rng.random_range(0.27..0.40),
                    rng.random_range(0.23..0.32),
                    rng.random_range(0.30..0.39),
                ),
                p.workstation_heading + rng.random_range(-0.16..0.16),
                &mut rng,
            );
        }
        if self
            .objects
            .iter()
            .any(|o| o.support == Some(support.id) && o.kind == ObjectKind::Monitor)
        {
            self.place_prop(
                support,
                ObjectKind::Keyboard,
                Vec3::ZERO,
                super::super::items::sizing::loose(ObjectKind::Keyboard, &mut rng).unwrap(),
                0.,
                &mut rng,
            );
            self.place_prop(
                support,
                ObjectKind::Mouse,
                Vec3::ZERO,
                super::super::items::sizing::loose(ObjectKind::Mouse, &mut rng).unwrap(),
                0.,
                &mut rng,
            );
        }
        // A personal device is a workstation companion, not only a rare draw
        // from the general accessory catalog. It uses the same support gate.
        if rng.random_bool((0.42 + p.accessory_affinity * 0.42) as f64) {
            self.place_prop(
                support,
                ObjectKind::Phone,
                Vec3::new(
                    support.size.x * rng.random_range(-0.32..0.32),
                    0.,
                    support.size.z * rng.random_range(0.05..0.30),
                ),
                super::super::items::sizing::loose(ObjectKind::Phone, &mut rng).unwrap(),
                rng.random_range(-1.1..1.1),
                &mut rng,
            );
        }
        if rng.random_bool(p.paper_affinity as f64) {
            let paper_size =
                super::super::items::sizing::loose(ObjectKind::Notepad, &mut rng).unwrap();
            let side = if rng.random_bool(0.5) { -1. } else { 1. };
            self.place_prop(
                support,
                if rng.random_bool(0.42) {
                    ObjectKind::Notepad
                } else {
                    ObjectKind::Notebook
                },
                Vec3::new(side * support.size.x * 0.27, 0., support.size.z * 0.10),
                paper_size,
                rng.random_range(-0.30..0.30),
                &mut rng,
            );
        }
        if rng.random_bool(p.accessory_affinity as f64) {
            let (kind, size) = super::decor::drink(&mut rng);
            self.place_prop(
                support,
                kind,
                Vec3::new(support.size.x * 0.32, 0., support.size.z * 0.12),
                size,
                rng.random_range(-1.2..1.2),
                &mut rng,
            );
        }
        if rng.random_bool((p.accessory_affinity * 0.55) as f64) {
            self.place_prop(
                support,
                ObjectKind::PenHolder,
                Vec3::new(-support.size.x * 0.34, 0., -support.size.z * 0.24),
                Vec3::new(0.08, rng.random_range(0.13..0.20), 0.08),
                rng.random_range(-0.3..0.3),
                &mut rng,
            );
        }
    }
}

impl IndoorManifest {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn place_prop(
        &mut self,
        support: &IndoorObject,
        kind: ObjectKind,
        offset: Vec3,
        size: Vec3,
        yaw: f32,
        rng: &mut ChaCha8Rng,
    ) {
        let size = super::super::items::sizing::loose(kind, rng).unwrap_or(size);
        let mut program = stream(support.seed, 906);
        let handed = if program.random_bool(0.5) { -1. } else { 1. };
        let angle = program.random_range(-0.32..0.32);
        let spread = program.random_range(0.78..1.05);
        let disorder = program.random_range(0.025..0.16);
        let shift = Vec3::new(
            program.random_range(-0.12..0.12) * support.size.x.min(1.8),
            0.,
            program.random_range(-0.12..0.10) * support.size.z.min(1.0),
        );
        let mut offset = shift
            + Quat::from_rotation_y(angle)
                * Vec3::new(offset.x * handed * spread, 0., offset.z * spread);
        offset += Vec3::new(
            rng.random_range(-disorder..disorder),
            0.,
            rng.random_range(-disorder..disorder),
        );
        let mut yaw = angle + yaw * handed + rng.random_range(-disorder * 2.5..disorder * 2.5);
        // Handles, bottle labels and loose personal items should not all face
        // the same seat/room axis. Keep screens and inputs coupled to the user.
        if matches!(
            kind,
            ObjectKind::Mug
                | ObjectKind::CoffeeCup
                | ObjectKind::WaterBottle
                | ObjectKind::SodaCan
                | ObjectKind::PenHolder
                | ObjectKind::Dishware
        ) {
            yaw = rng.random_range(-std::f32::consts::PI..std::f32::consts::PI);
        } else if matches!(
            kind,
            ObjectKind::Notebook | ObjectKind::Notepad | ObjectKind::Phone
        ) {
            yaw += rng.random_range(-0.55..0.55);
        }
        if matches!(kind, ObjectKind::Keyboard | ObjectKind::Mouse) {
            let (position, facing, count) = self
                .objects
                .iter()
                .filter(|o| o.support == Some(support.id) && o.kind == ObjectKind::Monitor)
                .fold(
                    (Vec3::ZERO, Vec3::ZERO, 0_u32),
                    |(position, facing, count), monitor| {
                        (
                            position + monitor.position,
                            facing + Quat::from_rotation_y(monitor.yaw) * Vec3::Z,
                            count + 1,
                        )
                    },
                );
            if count > 0 {
                // A dual-screen workstation has one user's keyboard between
                // the screens, rather than one input device at each monitor.
                let facing_yaw = facing.x.atan2(facing.z);
                // A shared user's reach couples input devices to the screen,
                // without erasing all the previously sampled placement noise.
                let mut reach = stream(support.seed, 0x494e_5055_5450_4f53);
                let reach_z = reach.random_range(0.22..0.36);
                let lateral = reach.random_range(-0.065..0.065);
                let mouse_spacing = reach.random_range(0.22..0.34);
                let local = Vec3::new(
                    lateral
                        + if kind == ObjectKind::Mouse {
                            handed * mouse_spacing
                        } else {
                            0.
                        }
                        + rng.random_range(-0.025..0.025),
                    0.,
                    reach_z + rng.random_range(-0.035..0.035),
                );
                let world = position / count as f32 + Quat::from_rotation_y(facing_yaw) * local;
                offset = support
                    .transform()
                    .compute_affine()
                    .inverse()
                    .transform_point3(world)
                    .with_y(0.);
                yaw = facing_yaw - support.yaw + rng.random_range(-0.18..0.18);
            }
        }

        self.prop(support, kind, offset, size, yaw, rng);
    }
}

pub(super) fn scatter_proposal(
    objects: &[IndoorObject],
    support: &IndoorObject,
    kind: ObjectKind,
    size: Vec3,
    rng: &mut ChaCha8Rng,
) -> (Vec3, f32) {
    use rand::seq::IteratorRandom;
    let mut offset = Vec3::new(
        rng.random_range(-0.47..0.47) * support.size.x,
        0.,
        rng.random_range(-0.47..0.47) * support.size.z,
    );
    let mut yaw = rng.random_range(-std::f32::consts::PI..std::f32::consts::PI);
    // Accessories form loose, variably rotated clusters around an existing
    // work item; the uniform component still explores the entire usable top.
    if rng.random_bool(0.48) {
        if let Some(anchor) = objects
            .iter()
            .filter(|o| {
                o.support == Some(support.id)
                    && matches!(
                        o.kind,
                        ObjectKind::Notebook
                            | ObjectKind::Notepad
                            | ObjectKind::Laptop
                            | ObjectKind::Books
                    )
            })
            .choose(rng)
        {
            let angle = rng.random_range(0.0..std::f32::consts::TAU);
            let r = 0.5 * (anchor.size.xz().length() + size.xz().length())
                + rng.random_range(0.025..0.15);
            let local = support
                .transform()
                .compute_affine()
                .inverse()
                .transform_point3(anchor.position);
            offset = (local + Vec3::new(angle.cos() * r, 0., angle.sin() * r)).with_y(0.);
            if matches!(
                kind,
                ObjectKind::Pencil | ObjectKind::Phone | ObjectKind::Notepad
            ) {
                yaw = anchor.yaw - support.yaw + rng.random_range(-0.45..0.45);
            }
        }
    }
    (offset, yaw)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn coherent_monitor_widths_accept_zero_and_sub_ulp_spread() {
        let support = super::super::super::objects::audit::fixture(ObjectKind::Desk, 7);
        let mut p = workspace_parameters(&support);
        for spread in [0., f32::EPSILON * 0.1, 0.05, 0.16] {
            p.monitor_size_scatter = spread;
            for seed in 0..32 {
                let actual = p.monitor_width(0.5, &mut stream(seed, 42));
                assert_eq!(actual, p.monitor_width(0.5, &mut stream(seed, 42)));
                assert!(actual.is_finite());
                assert!((actual - 0.5).abs() <= spread * 0.5 + f32::EPSILON);
                if spread < f32::EPSILON * 0.25 {
                    assert_eq!(actual, 0.5);
                }
            }
        }
    }
    use crate::scene::procedural_indoor::validation::validate_layout;
    use std::collections::BTreeSet;

    #[test]
    fn functional_screens_do_not_depend_on_loose_clutter_thinning() {
        let mut checked = 0;
        for seed in 0..32 {
            let mut base =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::OpenOffice, 0.65, 0, 0.)
                    .unwrap();
            let Some(mut support) = base
                .objects
                .iter()
                .find(|o| o.kind == ObjectKind::Desk)
                .cloned()
            else {
                continue;
            };
            support.id = 0;
            base.objects = vec![support.clone()];
            let screens = |clutter| {
                let mut scene = base.clone();
                scene
                    .program
                    .as_mut()
                    .unwrap()
                    .domain
                    .as_mut()
                    .unwrap()
                    .clutter = clutter;
                scene.decorate_workspace(&support);
                let screens: Vec<_> = scene
                    .objects
                    .iter()
                    .filter(|o| matches!(o.kind, ObjectKind::Monitor | ObjectKind::Laptop))
                    .cloned()
                    .collect();
                for screen in &screens {
                    assert!(scene.prop_clear(screen, &support, 0.012));
                }
                screens
            };
            let sparse = screens(0.);
            checked += sparse.len();
            assert_eq!(sparse, screens(1.), "workspace seed={seed}");
        }
        assert!(checked >= 8);
    }

    #[test]
    fn workspaces_accept_multiple_screen_arrangements_without_support_or_facing_regressions() {
        let mut programme_cells = BTreeSet::new();
        let mut arrangements = BTreeSet::new();
        let mut desks_with_monitors = 0;
        let mut dual_desks = 0;
        let mut facing_angles = Vec::new();
        for seed in 0..96 {
            let scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.)
                    .unwrap();
            validate_layout(&scene).unwrap();
            for support in scene.objects.iter().filter(|s| s.kind == ObjectKind::Desk) {
                let p = workspace_parameters(support);
                assert_eq!(
                    serde_json::to_value(&p).unwrap(),
                    serde_json::to_value(workspace_parameters(support)).unwrap()
                );
                programme_cells.insert([
                    (p.monitor_affinity * 20.) as u8,
                    (p.companion_affinity * 20.) as u8,
                    (p.paper_affinity * 20.) as u8,
                ]);
                let screens: Vec<_> = scene
                    .objects
                    .iter()
                    .filter(|o| {
                        o.support == Some(support.id)
                            && matches!(o.kind, ObjectKind::Laptop | ObjectKind::Monitor)
                    })
                    .collect();
                let monitors = screens
                    .iter()
                    .filter(|s| s.kind == ObjectKind::Monitor)
                    .count();
                let laptops = screens
                    .iter()
                    .filter(|s| s.kind == ObjectKind::Laptop)
                    .count();
                arrangements.insert((monitors, laptops));
                desks_with_monitors += usize::from(monitors > 0);
                dual_desks += usize::from(monitors >= 2);
                for screen in screens {
                    assert!(scene.prop_clear(screen, support, 0.012));
                    if let Some(chair) = screen.interaction_target.map(|i| &scene.objects[i]) {
                        let forward = Quat::from_rotation_y(screen.yaw) * Vec3::Z;
                        let delta = (chair.position - screen.position).with_y(0.).normalize();
                        assert!(forward.dot(delta) >= 0.19_f32.cos() - 1e-5);
                        facing_angles.push(forward.dot(delta).clamp(-1., 1.).acos());
                    }
                }
            }
        }
        assert!(programme_cells.len() > 96);
        assert!(
            desks_with_monitors >= 24,
            "accepted monitor desks: {desks_with_monitors}"
        );
        assert!(dual_desks >= 3, "accepted dual desks: {dual_desks}");
        assert!(facing_angles.len() > 32);
        assert!(facing_angles.iter().filter(|&&a| a > 0.04).count() > facing_angles.len() / 2);
        assert!(arrangements.contains(&(0, 0)) && arrangements.contains(&(0, 1)));
        assert!(arrangements.contains(&(1, 0)) && arrangements.contains(&(2, 0)));
    }
}
