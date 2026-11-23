//! One metric door program drives rendering, clearance, annotations and audits.
//! Programs describe rest states; optional room motion supplies kinematic joints.
mod construction;
mod sliding;
#[cfg(test)]
mod tests;
use super::{Assembly, IndoorManifest, Surface};
use crate::scene::procedural_indoor::layout::stream;
use bevy::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::f32::consts::{FRAC_PI_2, PI, TAU};

pub(crate) const ENTRY_APPROACH_DEPTH_M: f32 = 1.35;
/// Width of the square stock centered on each entry jamb and head.
pub(crate) const FRAME_WIDTH_M: f32 = 0.035;
/// Distance from the revolving axis to the inner end of each wing.
const ROTOR_HEEL_RADIUS_M: f32 = 0.052;
const LEAF_WIDTH_ALLOWANCE: f32 = 0.044;
const LEAF_HEIGHT_ALLOWANCE: f32 = 0.035;
const FLOOR_GAP: f32 = 0.012;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DoorMechanism {
    Hinged,
    Paired,
    Sliding,
    Telescoping,
    Revolving,
}
impl DoorMechanism {
    pub const ALL: [Self; 5] = [
        Self::Hinged,
        Self::Paired,
        Self::Sliding,
        Self::Telescoping,
        Self::Revolving,
    ];
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DoorPanel {
    Flush,
    Recessed,
    Glazed,
    VisionLite,
    Louvered,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DoorHandle {
    Lever,
    Knob,
    Pull,
    PanicBar,
    Recessed,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DoorProgram {
    pub mechanism: DoorMechanism,
    pub panel: DoorPanel,
    pub handle: DoorHandle,
    /// 0 = shut, 1 = fully open (hinged: 90 degrees). Revolving uses rotor_phase.
    pub opening: f32,
    pub rotor_phase: f32,
    pub handedness: f32,
    pub swing_side: f32,
    pub thickness: f32,
    pub stile: f32,
    pub handle_height: f32,
    pub finish: Surface,
    pub frame_finish: Surface,
    pub hardware_finish: Surface,
    pub finish_variant: u8,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DoorInstallation {
    pub id: usize,
    pub center: Vec3,
    pub yaw: f32,
    /// Width between jamb centre lines. Actual clear aperture is 36 mm narrower.
    pub width: f32,
    pub height: f32,
    pub wall_depth: f32,
    pub program: DoorProgram,
}
#[derive(Clone, Copy)]
pub(crate) struct Leaf {
    pub transform: Transform,
    pub size: Vec3,
}
impl Leaf {
    pub fn bounds(self, padding: f32) -> (Vec3, Vec3) {
        let r = Mat3::from_quat(self.transform.rotation).abs()
            * (self.size * 0.5 + Vec3::splat(padding));
        (
            self.transform.translation - r,
            self.transform.translation + r,
        )
    }
}
impl DoorProgram {
    pub fn sample(seed: u64, mechanism: DoorMechanism) -> Self {
        let mut r = stream(seed, 9200);
        let opening = match r.random_range(0..10) {
            0..=2 => 0.,
            3..=6 => r.random_range(0.10..0.83),
            _ => r.random_range(0.88..1.0),
        };
        let panel = if mechanism == DoorMechanism::Revolving {
            DoorPanel::Glazed
        } else {
            [
                DoorPanel::Flush,
                DoorPanel::Recessed,
                DoorPanel::Glazed,
                DoorPanel::VisionLite,
                DoorPanel::Louvered,
            ][r.random_range(0..5)]
        };
        let handle = match mechanism {
            DoorMechanism::Sliding | DoorMechanism::Telescoping => {
                if r.random_bool(0.5) {
                    DoorHandle::Pull
                } else {
                    DoorHandle::Recessed
                }
            }
            DoorMechanism::Revolving => DoorHandle::PanicBar,
            _ => [
                DoorHandle::Lever,
                DoorHandle::Knob,
                DoorHandle::Pull,
                DoorHandle::PanicBar,
            ][r.random_range(0..4)],
        };
        Self {
            mechanism,
            panel,
            handle,
            opening: if mechanism == DoorMechanism::Revolving {
                0.
            } else {
                opening
            },
            rotor_phase: r.random_range(0.0..TAU),
            handedness: if r.random_bool(0.5) { -1. } else { 1. },
            swing_side: if r.random_bool(0.5) { -1. } else { 1. },
            thickness: r.random_range(0.033..0.054),
            stile: r.random_range(0.045..0.095),
            handle_height: r.random_range(0.91..1.09),
            finish: [
                Surface::Wood,
                Surface::WoodEdge,
                Surface::Plastic,
                Surface::Metal,
            ][r.random_range(0..4)],
            frame_finish: [Surface::WoodEdge, Surface::Metal, Surface::Plastic]
                [r.random_range(0..3)],
            hardware_finish: [Surface::Chrome, Surface::Metal, Surface::Ink][r.random_range(0..3)],
            finish_variant: r.random_range(0..super::super::materials::variants::COUNT) as u8,
        }
    }
    pub fn validate(&self) -> bool {
        [
            self.opening,
            self.rotor_phase,
            self.handedness,
            self.swing_side,
            self.thickness,
            self.stile,
            self.handle_height,
        ]
        .into_iter()
        .all(f32::is_finite)
            && (0.0..=1.).contains(&self.opening)
            && self.handedness.abs() == 1.
            && self.swing_side.abs() == 1.
            && (0.025..=0.065).contains(&self.thickness)
            && (0.03..=0.12).contains(&self.stile)
            && (0.85..=1.2).contains(&self.handle_height)
            && (self.finish_variant as usize) < super::super::materials::variants::COUNT
    }
}
impl DoorInstallation {
    pub fn transform(&self) -> Transform {
        Transform::from_translation(self.center).with_rotation(Quat::from_rotation_y(self.yaw))
    }
    pub fn build(&self, a: &mut Assembly, frame: bool) {
        construction::build(self, a, frame);
    }
    pub(crate) fn leaves(&self) -> Vec<Leaf> {
        let p = &self.program;
        let width = self.width - LEAF_WIDTH_ALLOWANCE;
        let height = if matches!(
            p.mechanism,
            DoorMechanism::Sliding | DoorMechanism::Telescoping
        ) {
            // The head of an overlay leaf laps the frame. An inset leaf fits below it.
            self.height
        } else {
            self.height - LEAF_HEIGHT_ALLOWANCE
        };
        let mut leaves = Vec::with_capacity(4);
        let mut leaf = |center: Vec3, rotation: Quat, w: f32| {
            let tf = self.transform();
            leaves.push(Leaf {
                transform: Transform::from_translation(tf.transform_point(center))
                    .with_rotation(tf.rotation * rotation),
                size: Vec3::new(w, height, p.thickness),
            });
        };
        match p.mechanism {
            DoorMechanism::Hinged | DoorMechanism::Paired => {
                let count = if p.mechanism == DoorMechanism::Paired {
                    2
                } else {
                    1
                };
                let w = width / count as f32;
                for i in 0..count {
                    let side = if count == 1 {
                        p.handedness
                    } else if i == 0 {
                        -1.
                    } else {
                        1.
                    };
                    let rotation =
                        Quat::from_rotation_y(side * p.swing_side * p.opening * FRAC_PI_2);
                    // Pivot on the opening face; the heel cannot enter the wall.
                    let hinge = Vec3::new(
                        side * width * 0.5,
                        height * 0.5 + FLOOR_GAP,
                        p.swing_side * (self.wall_depth * 0.5 + p.thickness * 0.5 + 0.008),
                    );
                    leaf(
                        hinge + rotation * Vec3::new(-side * w * 0.5, 0., 0.),
                        rotation,
                        w - 0.004,
                    );
                }
            }
            DoorMechanism::Sliding | DoorMechanism::Telescoping => {
                for lane in sliding::lanes(self) {
                    leaf(
                        Vec3::new(
                            lane.x_at(p.opening),
                            height * 0.5 + FLOOR_GAP,
                            p.swing_side * lane.depth,
                        ),
                        Quat::IDENTITY,
                        lane.leaf_width,
                    );
                }
            }
            DoorMechanism::Revolving => {
                let r = width * 0.5 - 0.04;
                for i in 0..4 {
                    let rotation = Quat::from_rotation_y(p.rotor_phase + i as f32 * FRAC_PI_2);
                    leaf(
                        rotation * Vec3::X * ((r + ROTOR_HEEL_RADIUS_M) * 0.5)
                            + Vec3::Y * (height * 0.5 + FLOOR_GAP),
                        rotation,
                        r - ROTOR_HEEL_RADIUS_M,
                    );
                }
            }
        }
        leaves
    }
    /// Conservative static collision volumes, split along the leaf to avoid one
    /// huge diagonal AABB blocking an otherwise usable passage. No mesh build.
    pub fn obstacles(&self) -> Vec<(Vec3, Vec3)> {
        let mut result = Vec::with_capacity(20);
        for leaf in self.leaves() {
            for i in 0..4 {
                let center = leaf
                    .transform
                    .transform_point(Vec3::X * leaf.size.x * ((i as f32 + 0.5) / 4. - 0.5));
                result.push(
                    Leaf {
                        transform: Transform::from_translation(center)
                            .with_rotation(leaf.transform.rotation),
                        size: Vec3::new(leaf.size.x / 4., leaf.size.y, leaf.size.z),
                    }
                    .bounds(0.085),
                );
            }
        }
        if let Some(track) = sliding::Track::for_program(&self.program) {
            result.push(track.overhead_bounds(self));
        } else if self.program.mechanism == DoorMechanism::Revolving {
            // No dynamic door-opening controller: reserve the rotor drum for
            // walking/camera paths, rather than send a person through a blade.
            let tf = self.transform();
            result.push(
                Leaf {
                    transform: Transform::from_translation(
                        tf.transform_point(Vec3::Y * self.height * 0.5),
                    )
                    .with_rotation(tf.rotation),
                    size: Vec3::new(self.width, self.height, self.width),
                }
                .bounds(0.025),
            );
        }
        result
    }
    pub fn reservation(&self) -> (Vec3, Vec3) {
        let depth = if self.program.mechanism == DoorMechanism::Revolving {
            self.width * 0.5 + 0.30
        } else {
            self.width.min(1.7) + 0.25
        };
        let span = if matches!(
            self.program.mechanism,
            DoorMechanism::Sliding | DoorMechanism::Telescoping
        ) {
            sliding::parking_span(self)
        } else {
            let half = self.width * 0.5 + 0.15;
            Vec2::new(-half, half)
        };
        let tf = self.transform();
        Leaf {
            transform: tf.with_translation(tf.transform_point(Vec3::X * ((span.x + span.y) * 0.5))),
            size: Vec3::new(span.y - span.x, 0., 2. * depth),
        }
        .bounds(0.)
    }
}
impl IndoorManifest {
    pub fn validate_doors(&self) -> Result<(), String> {
        if let Some(d) = &self.entry_door {
            if d.id != 0
                || !d.center.is_finite()
                || !d.yaw.is_finite()
                || d.yaw != 0.
                || d.center.y != 0.
                || (d.center.x - self.door_x).abs() > 1e-5
                || (d.center.z - self.room_size.z * 0.5).abs() > 1e-5
                || !(0.9..=3.0).contains(&d.width)
                || !(2.0..=2.6).contains(&d.height)
                || !(0.1..=0.3).contains(&d.wall_depth)
                || !d.program.validate()
                || d.center.x.abs() + d.width * 0.5 > self.room_size.x * 0.5 - 0.15
            {
                return Err("invalid entry door construction".into());
            }
        }
        if self.program.as_ref().is_some_and(|p| {
            p.partitions.iter().any(|wall| {
                wall.door.as_ref().is_some_and(|d| {
                    !d.validate() || wall.arch_rise > 0. || d.mechanism == DoorMechanism::Revolving
                })
            })
        }) {
            return Err("invalid partition door construction".into());
        }
        Ok(())
    }
    pub fn door_installations(&self) -> Vec<DoorInstallation> {
        self.entry_door
            .iter()
            .cloned()
            .chain(
                self.program
                    .iter()
                    .flat_map(|p| p.partitions.iter().enumerate())
                    .filter_map(|(i, p)| {
                        p.door.as_ref().map(|door| DoorInstallation {
                            id: i + 1,
                            center: p.position(p.door_center, 0.),
                            yaw: if p.axis == 0 { -FRAC_PI_2 } else { 0. },
                            width: p.door_width,
                            height: p.door_height,
                            wall_depth: p.thickness,
                            program: door.clone(),
                        })
                    }),
            )
            .collect()
    }
    pub fn entry_width(&self) -> f32 {
        self.entry_door.as_ref().map_or(1.1, |d| d.width)
    }
    pub fn entry_height(&self) -> f32 {
        self.entry_door.as_ref().map_or(2.27, |d| d.height)
    }
    pub fn entry_approach(&self) -> (Vec3, Vec3) {
        let half = self.entry_width() * 0.5 + 0.15;
        let depth = self.entry_door.as_ref().map_or(1.45, |d| {
            if d.program.mechanism == DoorMechanism::Revolving {
                d.width * 0.5 + 0.30
            } else {
                ENTRY_APPROACH_DEPTH_M
            }
        });
        (
            Vec3::new(self.door_x - half, 0., self.room_size.z * 0.5 - depth),
            Vec3::new(self.door_x + half, 0., self.room_size.z * 0.5 + depth),
        )
    }
    pub fn entry_approach_overlaps(&self, lo: Vec3, hi: Vec3) -> bool {
        let (a, b) = self.entry_approach();
        if lo.x < b.x && hi.x > a.x && lo.z < b.z && hi.z > a.z {
            return true;
        }
        // Sliding travel is on the neighboring side. Reserving it across both
        // rooms removed valid primary-room furniture along an unrelated wall.
        self.entry_door.as_ref().is_some_and(|d| {
            let (a, b) = d.reservation();
            let a = a.with_z(a.z.max(d.center.z));
            lo.x < b.x && hi.x > a.x && lo.z < b.z && hi.z > a.z
        })
    }
}
pub(crate) fn sample_entry(scene: &mut IndoorManifest) -> Result<(), String> {
    let mut r = stream(scene.seed, 9210);
    use crate::scene::procedural_indoor::layout::IndoorLayout;
    let lobby = matches!(
        scene.layout,
        IndoorLayout::ElevatorLanding
            | IndoorLayout::Reception
            | IndoorLayout::Showroom
            | IndoorLayout::Gallery
            | IndoorLayout::PublicSuite
    );
    let mechanism = if lobby && r.random_bool(0.16) {
        DoorMechanism::Revolving
    } else {
        DoorMechanism::ALL[r.random_range(0..4)]
    };
    let mut width = match mechanism {
        DoorMechanism::Revolving => r.random_range(2.35..2.80),
        DoorMechanism::Paired => r.random_range(1.55..2.20),
        _ => r.random_range(0.96..1.35),
    };
    let mut program = DoorProgram::sample(scene.seed, mechanism);
    // The exterior face is the neighboring volume. This keeps entry hardware
    // away from nearby ceiling details and primary-room circulation.
    program.swing_side = 1.;
    let spans_for = |p: &DoorProgram, width: f32| {
        let sliding = matches!(
            p.mechanism,
            DoorMechanism::Sliding | DoorMechanism::Telescoping
        );
        let margin = width * 0.5 + 0.65;
        let lo = -scene.room_size.x * 0.5
            + margin
            + if sliding && p.handedness < 0. {
                width
            } else {
                0.
            };
        let hi = scene.room_size.x * 0.5
            - margin
            - if sliding && p.handedness > 0. {
                width
            } else {
                0.
            };
        let mut spans = if hi - lo > 0.02 {
            vec![(lo, hi)]
        } else {
            vec![]
        };
        if let Some(walls) = &scene.program {
            for wall in walls
                .partitions
                .iter()
                .filter(|p| p.axis == 0 && p.end >= scene.room_size.z * 0.5 - 0.1)
            {
                let pad = width * 0.5 + wall.thickness * 0.5 + 0.20;
                spans = spans
                    .into_iter()
                    .flat_map(|(lo, hi)| {
                        let a = wall.coordinate - pad;
                        let b = wall.coordinate + pad;
                        if hi <= a || lo >= b {
                            vec![(lo, hi)]
                        } else {
                            [(lo, hi.min(a)), (lo.max(b), hi)]
                                .into_iter()
                                .filter(|(a, b)| b - a > 0.02)
                                .collect()
                        }
                    })
                    .collect();
            }
        }
        spans
    };
    let mut spans = spans_for(&program, width);
    if spans.is_empty() {
        program = DoorProgram::sample(scene.seed, DoorMechanism::Hinged);
        program.swing_side = 1.;
        width = r.random_range(0.96..1.15);
        spans = spans_for(&program, width);
    }
    // Recursive zones are at least 3.25 m wide; the narrow fallback always fits.
    // A deterministic longest interval avoids rejection loops in dense programs.
    let (lo, hi) = spans
        .into_iter()
        .max_by(|a, b| (a.1 - a.0).total_cmp(&(b.1 - b.0)))
        .ok_or_else(|| "rear partitions leave no valid entrance interval".to_owned())?;
    scene.door_x = r.random_range(lo..hi);
    scene.entry_door = Some(DoorInstallation {
        id: 0,
        center: Vec3::new(scene.door_x, 0., scene.room_size.z * 0.5),
        yaw: 0.,
        width,
        height: r.random_range(2.10_f32..2.34).min(scene.room_size.y - 0.25),
        wall_depth: 0.20,
        program,
    });
    Ok(())
}
pub(crate) fn sample_partitions(scene: &mut IndoorManifest) {
    let Some(program) = &mut scene.program else {
        return;
    };
    for (i, p) in program.partitions.iter_mut().enumerate() {
        let mut r = stream(scene.seed, 9220 + i as u64);
        if p.arch_rise > 0. || r.random_bool(0.12) {
            continue;
        }
        let available = [
            p.door_center - p.door_width * 0.5 - p.start,
            p.end - p.door_center - p.door_width * 0.5,
        ];
        let mut mechanism = DoorMechanism::ALL[r.random_range(0..4)];
        let mut door = DoorProgram::sample(scene.seed.wrapping_add(i as u64 * 3571 + 1), mechanism);
        if matches!(
            mechanism,
            DoorMechanism::Sliding | DoorMechanism::Telescoping
        ) {
            let need = p.door_width + 0.15;
            if available[0].max(available[1]) < need {
                mechanism = DoorMechanism::Hinged;
            }
            door.handedness = if available[0] > available[1] { -1. } else { 1. };
        }
        if mechanism == DoorMechanism::Paired && p.door_width < 1.30 {
            mechanism = DoorMechanism::Hinged;
        }
        door.mechanism = mechanism;
        p.door = Some(door);
    }
}

/// Raised floors are authored before doors. Retain a collision-free static state
/// rather than let a wide swung leaf cut into a seating tier or raised landing.
pub(crate) fn fit_floor_states(scene: &mut IndoorManifest) {
    let fits = |d: &DoorInstallation| {
        d.leaves().iter().all(|leaf| {
            (0..=16).all(|i| {
                let p = leaf.transform.transform_point(Vec3::new(
                    leaf.size.x * (i as f32 / 16. - 0.5),
                    -leaf.size.y * 0.5,
                    0.,
                ));
                scene.floor_height(p.xz()) <= p.y + 0.001
            })
        })
    };
    let repairs: Vec<_> = scene
        .door_installations()
        .into_iter()
        .filter(|d| !fits(d))
        .map(|mut d| {
            if d.program.mechanism == DoorMechanism::Revolving {
                d.program.mechanism = DoorMechanism::Paired;
            }
            d.program.opening = 0.;
            d
        })
        .collect();
    for d in repairs {
        if d.id == 0 {
            scene.entry_door = Some(d);
        } else if let Some(p) = scene
            .program
            .as_mut()
            .and_then(|p| p.partitions.get_mut(d.id - 1))
        {
            p.door = Some(d.program);
        }
    }
}
