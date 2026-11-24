//! Fitted inset doors. The chassis sets the allowed opening and each leaf
//! samples a hand, rest angle and grip within its own bay.
use super::*;
use crate::scene::procedural_indoor::motion::{construction as motion, MotionKind};
use serde::Serialize;

pub(crate) const INTRODUCED: u32 = 233;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Hand {
    Left,
    Right,
}
impl Hand {
    pub(super) fn sign(self) -> f32 {
        if self == Self::Left {
            -1.
        } else {
            1.
        }
    }
}
#[derive(Debug, Copy, Clone, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Face {
    Slab,
    Recessed,
    Raised,
    Fluted,
    Reeded,
}
#[derive(Debug, Copy, Clone, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Pull {
    Bow,
    Knob,
    Tab,
    Touch,
}
#[derive(Copy, Clone)]
enum Hanging {
    Pair,
    Same,
    Independent,
}

pub(crate) struct Bank {
    hanging: Hanging,
    first: Hand,
    face: Face,
    pull: Pull,
    pull_vertical: bool,
    stile: f32,
    rail: f32,
    thickness: f32,
    relief: f32,
    reach: f32,
    grip_span: f32,
    grip_radius: f32,
    grip_inset: f32,
    grip_level: f32,
    standing_reach_m: f32,
    pull_finish: Surface,
    panel_finish: Surface,
    horizontal_grain: bool,
}

#[derive(Debug, Copy, Clone, Serialize)]
pub(crate) struct Leaf {
    pub bay: usize,
    pub hand: Hand,
    pub face: Face,
    pub pull: Pull,
    pub size_m: Vec2,
    pub center_m: Vec3,
    pub thickness_m: f32,
    pub stile_m: f32,
    pub rail_m: f32,
    pub relief_m: f32,
    pub rest_radians: f32,
    pub grip_center_m: Vec2,
    pub vertical_pull: bool,
    pub grip_span_m: f32,
    pub grip_radius_m: f32,
    pub reach_m: f32,
    pub pull_finish: Surface,
    pub panel_finish: Surface,
    pub horizontal_grain: bool,
}

fn opening(p: &ShelfProgram, i: usize) -> Option<(f32, f32)> {
    let b = &p.bays[i];
    if b.drawers || p.front == 4 {
        return None;
    }
    let top = b.levels.last()? - p.panel_m * 0.5 - 0.004;
    let bottom = if p.front == 3 {
        b.levels[program::open_levels(p, b)] + p.panel_m * 0.5
    } else {
        p.bottom_m + p.panel_m * 0.5 + 0.006
    };
    (top > bottom + 0.012).then_some((bottom, top))
}

impl Bank {
    pub(crate) fn new(o: &IndoorObject, p: &ShelfProgram) -> Option<Self> {
        if o.kind != ObjectKind::Cabinet || matches!(p.front, 1 | 4) {
            return None;
        }
        let mut r = stream(o.seed, u64::from_le_bytes(*b"DoorBank"));
        let mut f = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"DoorPull"),
        );
        let thickness = r.random_range(0.014..0.021);
        Some(Self {
            hanging: match r.random_range(0..100) {
                0..=55 => Hanging::Pair,
                56..=83 => Hanging::Same,
                _ => Hanging::Independent,
            },
            first: if r.random_bool(0.5) {
                Hand::Left
            } else {
                Hand::Right
            },
            face: if p.front == 5 {
                if r.random_bool(0.52) {
                    Face::Fluted
                } else {
                    Face::Reeded
                }
            } else {
                match r.random_range(0..100) {
                    0..=48 => Face::Slab,
                    49..=80 => Face::Recessed,
                    _ => Face::Raised,
                }
            },
            pull: [Pull::Bow, Pull::Knob, Pull::Tab, Pull::Touch][p.handle as usize],
            pull_vertical: r.random_bool(0.58),
            stile: r.random_range(0.032..0.084),
            rail: r.random_range(0.036..0.105),
            thickness,
            relief: r.random_range(0.003..0.009),
            reach: r.random_range(0.014_f32..0.031).min(0.042 - thickness),
            grip_span: r.random_range(0.070..0.185),
            grip_radius: r.random_range(0.0030..0.0058),
            grip_inset: r.random_range(0.33..0.69),
            grip_level: r.random_range(0.54..0.88),
            pull_finish: [
                Surface::Chrome,
                Surface::Metal,
                Surface::Plastic,
                Surface::Wood,
            ][f.random_range(0..4)],
            panel_finish: if f.random_bool(0.23) {
                p.shelves
            } else {
                p.fronts
            },
            horizontal_grain: r.random_bool(0.27),
            standing_reach_m: r.random_range(0.88..1.27),
        })
    }
    pub(crate) fn leaf(&self, o: &IndoorObject, p: &ShelfProgram, i: usize) -> Option<Leaf> {
        let (lo, hi) = opening(p, i)?;
        let b = &p.bays[i];
        let mut r = stream(o.seed, u64::from_le_bytes(*b"LeafFit!") ^ i as u64);
        let hand = match self.hanging {
            Hanging::Pair => {
                // Pairs restart after drawers and unfronted bays.
                let run_start = (0..i)
                    .rev()
                    .find(|&j| opening(p, j).is_none())
                    .map_or(0, |j| j + 1);
                if (i - run_start) % 2 == 1 {
                    Hand::Right
                } else if i + 1 < p.bays.len() && opening(p, i + 1).is_some() {
                    Hand::Left
                } else {
                    self.first
                }
            }
            Hanging::Same => self.first,
            Hanging::Independent => {
                if r.random_bool(0.5) {
                    Hand::Left
                } else {
                    Hand::Right
                }
            }
        };
        let size = Vec2::new(b.right - b.left - 0.006, hi - lo);
        let half = size * 0.5;
        let stile = self.stile.min(size.x * 0.22);
        let rail = self.rail.min(size.y * 0.22);
        let pull = if size.min_element() < 0.055 {
            Pull::Touch
        } else {
            self.pull
        };
        let vertical = self.pull_vertical
            || matches!(self.face, Face::Fluted | Face::Reeded)
            || (hi > 1.52 && lo < 0.55 && self.face != Face::Slab);
        let span = self.grip_span.min(if vertical {
            size.y * 0.48
        } else {
            size.x * 0.49
        });
        // A formed bar needs room for the inside of both bends, including
        // on short pulls mounted to a shallow inset door.
        let radius = if pull == Pull::Bow {
            self.grip_radius.min(span * 0.060).min(self.reach * 0.19)
        } else {
            self.grip_radius.min(span * 0.18)
        };
        let latch = -hand.sign();
        let reach = if pull == Pull::Touch { 0. } else { self.reach };
        // The pull must fit the placement envelope, and the back of the
        // closed leaf must clear shelf noses at body_front - 45 mm.
        let z = p.body_depth_m * 0.5 - (reach + 0.0015).max(0.012);
        let center = Vec3::new((b.left + b.right) * 0.5, (lo + hi) * 0.5, z);
        let bar = matches!(pull, Pull::Tab | Pull::Bow);
        let inset = stile * self.grip_inset;
        let margin = if vertical && bar {
            span * 0.5 + radius * 2.
        } else if pull == Pull::Knob {
            (radius * 2.5).min(stile * 0.25) + 0.001
        } else {
            0.012
        };
        let reach_y = (lo + (hi - lo) * self.grip_level).min(self.standing_reach_m);
        let y =
            (reach_y - center.y).clamp(-half.y + margin.min(half.y), half.y - margin.min(half.y));
        let grip_center = if !vertical && bar && self.face != Face::Slab {
            // Drill horizontal pulls through a solid rail, away from the thin field.
            let bottom = (lo - self.standing_reach_m).abs() < (hi - self.standing_reach_m).abs();
            Vec2::new(
                latch * (half.x - inset - span * 0.5),
                if bottom {
                    -half.y + rail * 0.5
                } else {
                    half.y - rail * 0.5
                },
            )
        } else {
            Vec2::new(
                latch * (half.x - inset - if !vertical && bar { span * 0.5 } else { 0. }),
                y,
            )
        };
        Some(Leaf {
            bay: i,
            hand,
            face: self.face,
            pull,
            size_m: size,
            center_m: center,
            thickness_m: self.thickness,
            stile_m: stile,
            rail_m: rail,
            relief_m: self.relief.min(self.thickness * 0.48),
            rest_radians: if r.random_bool(0.38) {
                0.
            } else {
                p.opening_radians * r.random_range(0.20..1.)
            },
            grip_center_m: grip_center,
            vertical_pull: vertical,
            grip_span_m: span,
            grip_radius_m: radius,
            reach_m: reach,
            pull_finish: self.pull_finish,
            panel_finish: match self.face {
                Face::Recessed | Face::Raised => self.panel_finish,
                _ => p.fronts,
            },
            horizontal_grain: self.horizontal_grain,
        })
    }
}

impl Leaf {
    pub(super) fn pivot(self) -> Vec3 {
        self.center_m + Vec3::X * (self.hand.sign() * self.size_m.x * 0.5)
    }
    pub(super) fn build(self, a: &mut Assembly, o: &IndoorObject, p: &ShelfProgram) {
        let mut leaf = Assembly::default();
        super::skin::build(&mut leaf, o, p, &self);
        super::pull::build(&mut leaf, &self);
        let pivot = self.pivot();
        let sign = self.hand.sign();
        let q = Quat::from_rotation_y(sign * self.rest_radians);
        for g in leaf.parts.values_mut() {
            if self.rest_radians == 0. {
                for v in &mut g.positions {
                    *v = (Vec3::from_array(*v) + self.center_m).to_array();
                }
            } else {
                let origin = self.center_m - pivot;
                for v in &mut g.positions {
                    *v = (pivot + q * (origin + Vec3::from_array(*v))).to_array();
                }
                for n in &mut g.normals {
                    *n = (q * Vec3::from_array(*n)).to_array();
                }
            }
        }
        let seed = o.seed.wrapping_add(self.bay as u64);
        if a.motion_selected(MotionKind::Cabinet, seed) {
            let end = motion::endpoint(seed, self.rest_radians, 1.15);
            let mut joint = motion::rigid(
                MotionKind::Cabinet,
                seed,
                pivot,
                Vec3::Y,
                sign * (end - self.rest_radians),
                Vec3::ZERO,
            );
            joint.frame = Transform::from_translation(pivot).with_rotation(q);
            a.articulated(leaf, joint);
        } else {
            a.append_assembly(leaf);
        }
    }
}
