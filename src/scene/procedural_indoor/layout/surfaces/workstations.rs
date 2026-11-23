//! Fit a separate operator position on each side of a bench. Equipment is
//! admitted against the real worktop outline and installed peers in order.
use super::*;
use crate::scene::procedural_indoor::{items::sizing, objects::desktop::tower};
#[cfg(test)]
mod tests;

/// Personal laptops and employer-installed machines occur in both offices
/// and coworking suites. Nearby desks share a procurement tendency while
/// retaining independent equipment, dimensions, finishes and operator choices.
#[derive(Clone, Copy)]
pub(in crate::scene::procedural_indoor::layout) struct OfficeFitout {
    fixed_computers: f32,
}
impl OfficeFitout {
    pub(in crate::scene::procedural_indoor::layout) fn sample(
        scene: &IndoorManifest,
    ) -> Option<Self> {
        let range = match scene.layout {
            IndoorLayout::Coworking => 0.15..0.76,
            IndoorLayout::OpenOffice => 0.32..0.92,
            _ => return None,
        };
        Some(Self {
            fixed_computers: stream(scene.seed, 0x4f46464943454954).random_range(range),
        })
    }
}

struct Station<'a> {
    desk: &'a IndoorObject,
    seat: Option<usize>,
    sign: f32,
    center: f32,
    rear: f32,
    front: f32,
}
impl Station<'_> {
    fn fit(
        &self,
        scene: &mut IndoorManifest,
        kind: ObjectKind,
        size: Vec3,
        placement: (Vec2, f32),
        target: Option<usize>,
        r: &mut ChaCha8Rng,
    ) -> Option<usize> {
        let (point, yaw) = placement;
        let desk = self.desk;
        let pos = desk.transform().transform_point(Vec3::new(
            point.x * self.sign,
            desk.size.y,
            point.y * self.sign,
        ));
        let angle = desk.yaw
            + if self.sign < 0. {
                std::f32::consts::PI
            } else {
                0.
            }
            + yaw;
        let mut o = scene.candidate(kind, pos, size, angle, r);
        o.interaction_target = target;
        let id = o.id;
        scene.admit_supported_prop(desk, o).then_some(id)
    }
    fn screens(
        &self,
        scene: &mut IndoorManifest,
        p: &WorkspaceProgram,
        r: &mut ChaCha8Rng,
    ) -> Vec<usize> {
        let desk = self.desk;
        let dual = desk.size.x > 1.17 && r.random_bool((0.22 + 0.64 * p.companion_affinity) as f64);
        let portrait = dual && r.random_bool(0.45);
        let main_width: f32 = if dual {
            r.random_range(0.40..0.55)
        } else {
            r.random_range(0.42..0.76)
        };
        let other = if portrait {
            r.random_range(0.27..0.34)
        } else {
            main_width * r.random_range(0.83..1.07)
        };
        let gap = r.random_range(0.035..0.095);
        let side = if r.random_bool(0.5) { -1. } else { 1. };
        let mut screens = Vec::with_capacity(2);
        for i in 0..if dual { 2 } else { 1 } {
            let width = if i == 0 { main_width } else { other };
            let height = if i == 1 && portrait {
                r.random_range(0.46..0.58)
            } else {
                (width / r.random_range(1.55..2.36) + r.random_range(0.06..0.11)).clamp(0.28, 0.51)
            };
            let x = self.center
                + if !dual {
                    0.
                } else if i == 0 {
                    -side * (other + gap) * 0.5
                } else {
                    side * (main_width + gap) * 0.5
                };
            let size = Vec3::new(width, height, r.random_range(0.157..0.224));
            let yaw = if dual {
                -x * r.random_range(0.30..0.62)
            } else {
                p.workstation_heading
            };
            // Bring curved/tapered-edge displays inward before giving up a useful station.
            for (dx, dz) in [(x, 0.), (x * 0.92, 0.032), (x * 0.90, -0.027)] {
                if let Some(id) = self.fit(
                    scene,
                    ObjectKind::Monitor,
                    size,
                    (Vec2::new(dx, self.rear + dz), yaw),
                    self.seat,
                    r,
                ) {
                    screens.push(id);
                    break;
                }
            }
        }
        screens
    }
    fn laptop(&self, scene: &mut IndoorManifest, x: f32, z: f32, r: &mut ChaCha8Rng) {
        let size = Vec3::new(
            r.random_range(0.27..0.397),
            r.random_range(0.235..0.31),
            r.random_range(0.28..0.35),
        );
        for dz in [0., 0.036, -0.044] {
            if self
                .fit(
                    scene,
                    ObjectKind::Laptop,
                    size,
                    (Vec2::new(x, z + dz), r.random_range(-0.13..0.13)),
                    self.seat,
                    r,
                )
                .is_some()
            {
                break;
            }
        }
    }
    fn inputs(&self, scene: &mut IndoorManifest, screen: usize, r: &mut ChaCha8Rng) {
        let w = self.desk.size.x;
        let side = if r.random_bool(0.83) { 1. } else { -1. };
        let mut x = self.center;
        let mut kb_width = 0.36;
        let size = sizing::loose(ObjectKind::Keyboard, r).unwrap();
        let yaw = r.random_range(-0.14..0.14);
        for (dx, dz) in [
            (0., 0.),
            (-side * 0.06, 0.025),
            (0.035, -0.036),
            (-side * 0.10, -0.02),
        ] {
            let kx = self.center + dx;
            if self
                .fit(
                    scene,
                    ObjectKind::Keyboard,
                    size,
                    (Vec2::new(kx, self.front + dz), yaw),
                    Some(screen),
                    r,
                )
                .is_some()
            {
                x = kx;
                kb_width = size.x;
                break;
            }
        }
        let mouse = sizing::loose(ObjectKind::Mouse, r).unwrap();
        let gap = r.random_range(0.022..0.092);
        let advance = r.random_range(-0.075..0.055);
        for handed in [side, -side] {
            for dz in [advance, 0.035, -0.061] {
                let mx = x + handed * (kb_width * 0.5 + mouse.x * 0.5 + gap);
                if mx.abs() + mouse.x * 0.5 < w * 0.5
                    && self
                        .fit(
                            scene,
                            ObjectKind::Mouse,
                            mouse,
                            (
                                Vec2::new(mx, self.front + dz),
                                yaw + r.random_range(-0.24..0.24),
                            ),
                            Some(screen),
                            r,
                        )
                        .is_some()
                {
                    return;
                }
            }
        }
    }
    fn computer(&self, scene: &mut IndoorManifest, monitor: usize, r: &mut ChaCha8Rng) {
        let d = self.desk;
        let side = if r.random_bool(0.5) { 1. } else { -1. };
        let floor_first = r.random_bool(0.46);
        for floor in [floor_first, !floor_first] {
            let size = tower::dimensions(r, floor);
            for turn in [side, -side] {
                if floor {
                    // Beside the casework, outside the knee well and chair approach.
                    let x = turn * (d.size.x * 0.5 + size.x * 0.5 + r.random_range(0.09..0.16));
                    let pos = d.transform().transform_point(Vec3::new(
                        x,
                        0.,
                        self.sign * d.size.z * r.random_range(-0.17..0.13),
                    ));
                    let y = scene.floor_height(pos.xz());
                    if (y - d.position.y).abs() > 0.012 {
                        continue;
                    }
                    let mut o = scene.candidate(
                        ObjectKind::PcTower,
                        pos.with_y(y),
                        size,
                        d.yaw
                            + if self.sign < 0. {
                                std::f32::consts::PI
                            } else {
                                0.
                            }
                            + r.random_range(-0.10..0.10),
                        r,
                    );
                    o.interaction_target = Some(monitor);
                    if scene.placement_clear(&o, 0.06) {
                        scene.objects.push(o);
                        return;
                    }
                } else {
                    let x = turn * (d.size.x * 0.5 - size.x * 0.5 - r.random_range(0.046..0.105));
                    for z in [self.rear + 0.035, self.front - 0.04] {
                        if self
                            .fit(
                                scene,
                                ObjectKind::PcTower,
                                size,
                                (Vec2::new(x, z), r.random_range(-0.10..0.10)),
                                Some(monitor),
                                r,
                            )
                            .is_some()
                        {
                            return;
                        }
                    }
                }
            }
        }
    }
}

impl IndoorManifest {
    pub(super) fn fit_office_stations(
        &mut self,
        desk: &IndoorObject,
        p: &WorkspaceProgram,
        office: OfficeFitout,
    ) {
        let seats: Vec<_> = self
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::Chair && o.interaction_target == Some(desk.id))
            .map(|o| o.id)
            .collect();
        let shared = seats.len() > 1 && desk.size.z >= 1.17;
        let count = if shared { 2 } else { 1 };
        for i in 0..count {
            let seat = seats.get(i).copied();
            let mut r = stream(desk.seed, 0x4f50455241544f52 + i as u64);
            if !r.random_bool((0.66 + 0.31 * p.screen_affinity) as f64) {
                continue;
            }
            let local = seat.map(|id| {
                desk.transform().rotation.inverse() * (self.objects[id].position - desk.position)
            });
            let sign = if local.is_some_and(|p| p.z < 0.) {
                -1.
            } else {
                1.
            };
            let pos = Station {
                desk,
                seat,
                sign,
                center: local.map_or(0., |p| sign * p.x * 0.30).clamp(-0.13, 0.13)
                    + r.random_range(-0.058..0.058),
                rear: if shared {
                    desk.size.z * r.random_range(0.106..0.143)
                } else {
                    -desk.size.z * r.random_range(0.17..0.28)
                },
                front: desk.size.z
                    * if shared {
                        r.random_range(0.32..0.39)
                    } else {
                        r.random_range(0.12..0.25)
                    },
            };
            if r.random_bool(
                (0.49 + 0.27 * p.monitor_affinity + 0.24 * office.fixed_computers) as f64,
            ) {
                let screens = pos.screens(self, p, &mut r);
                if let Some(&main) = screens.first() {
                    pos.inputs(self, main, &mut r);
                    let docked = screens.len() == 1
                        && r.random_bool(
                            (0.08
                                + 0.48 * (1. - office.fixed_computers)
                                + 0.36 * p.docking_affinity) as f64,
                        );
                    if docked {
                        pos.laptop(
                            self,
                            desk.size.x * p.docking_side * 0.32,
                            pos.rear + 0.08,
                            &mut r,
                        );
                    }
                    if (!docked
                        || !self.objects.iter().any(|o| {
                            o.kind == ObjectKind::Laptop
                                && o.support == Some(desk.id)
                                && o.interaction_target == seat
                        }))
                        && r.random_bool((0.32 + 0.64 * office.fixed_computers) as f64)
                    {
                        pos.computer(self, main, &mut r);
                    }
                    continue;
                }
            }
            pos.laptop(self, pos.center, (pos.rear + pos.front) * 0.5, &mut r);
        }
    }
}
