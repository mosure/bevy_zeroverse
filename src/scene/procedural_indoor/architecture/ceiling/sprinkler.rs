//! One manufactured head and branch layout per pipe run. Installation offsets
//! use a separate stream and never consume layout or camera randomness.
use super::*;
use crate::scene::procedural_indoor::materials::sprinkler::Finish;

pub(crate) const FITTED_HEADS_VERSION: u32 = 223;
pub(super) const DEFLECTOR_GAUGE_M: f32 = 0.00115;
pub(super) const BOSS_TIP_M: f32 = 0.0014;

// Leave 2 mm of air beyond either the dished rim or the central retaining boss.
fn end_allowance(crown: f32) -> f32 {
    (crown.max(0.) + DEFLECTOR_GAUGE_M * 0.5).max(BOSS_TIP_M) + 0.002
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BranchLayout {
    SingleBank,
    Alternating,
    PairedBanks,
}

#[derive(Clone, Copy)]
pub(crate) struct SprinklerRun {
    pub layout: BranchLayout,
    pub frame: Finish,
    pub bulb: Finish,
    pub slots: u8,
    pub radius: f32,
    pub span: f32,
    pub crown: f32,
    pub slot_root: f32,
    pub slot_width: f32,
    pub bow: f32,
    pub count: u8,
    pitch: f32,
    jitter: f32,
    reach: f32,
    side: f32,
}

pub(crate) struct HeadSite {
    pub x: f32,
    pub z: f32,
    pub clocking: f32,
}

impl CeilingModule {
    pub(crate) fn sprinkler_run(&self, version: u32) -> Option<SprinklerRun> {
        if version < FITTED_HEADS_VERSION {
            return None;
        }
        let CeilingShape::Sprinkler {
            radius_m,
            head_pitch_m,
            branch_reach,
            upright,
        } = self.detail.as_ref()?.shape
        else {
            return None;
        };
        let mut r = stream(self.seed, u64::from_le_bytes(*b"HeadCast"));
        let count = (self.size.x.max(self.size.z) / head_pitch_m)
            .ceil()
            .clamp(1., 10.) as u8;
        let layout = match r.random_range(0..10) {
            0..=2 => BranchLayout::SingleBank,
            3..=7 => BranchLayout::Alternating,
            _ => BranchLayout::PairedBanks,
        };
        let frame = match r.random_range(0..10) {
            0..=4 => Finish::Brass,
            5..=7 => Finish::Nickel,
            _ => Finish::Enamel,
        };
        let crown = r.random_range(-0.0035_f32..0.004);
        // Fit the complete crown and collar on the outlet side, even in a
        // shallow upright installation where the riser may be only millimetres.
        let available = self.size.y * if upright { 0.36 } else { 0.64 };
        let branch_radius = radius_m.min(self.size.y * 0.16) * 0.57;
        let span = r
            .random_range(0.027_f32..0.043)
            .min(available - branch_radius - end_allowance(crown));
        Some(SprinklerRun {
            layout,
            frame,
            bulb: if r.random_bool(0.73) {
                Finish::Ruby
            } else {
                Finish::Amber
            },
            slots: r.random_range(7..=13),
            radius: r.random_range(0.015..0.024),
            span,
            crown,
            slot_root: r.random_range(0.48..0.73),
            slot_width: r.random_range(0.13..0.34),
            bow: r.random_range(0.50..0.78),
            count,
            pitch: self.size.x.max(self.size.z) * 0.80 / f32::from(count),
            jitter: if r.random_bool(0.31) {
                0.
            } else {
                r.random_range(0.045..0.17)
            },
            reach: branch_reach,
            side: if r.random_bool(0.5) { 1. } else { -1. },
        })
    }
}

impl SprinklerRun {
    pub(crate) fn site(self, m: &CeilingModule, i: u8) -> HeadSite {
        let mut r = stream(m.seed, u64::from_le_bytes(*b"HeadSite") ^ u64::from(i));
        let length = m.size.x.max(m.size.z);
        let width = m.size.x.min(m.size.z);
        let center = (f32::from(i) + 0.5) * self.pitch - length * 0.40;
        let mut x = center + self.pitch * self.jitter * r.random_range(-1.0..1.);
        // Keep branch tees clear of the saddle crossbars in their own bay.
        let saddle = x.signum() * length * 0.35;
        let gap = 0.039;
        if (x - saddle).abs() < gap {
            let left = saddle - gap;
            let right = saddle + gap;
            x = if left >= center - self.pitch * 0.42
                && (x < saddle || right > center + self.pitch * 0.42)
            {
                left
            } else {
                right
            };
        }
        let flip = match self.layout {
            BranchLayout::SingleBank => false,
            BranchLayout::Alternating => i % 2 == 1,
            BranchLayout::PairedBanks => (i / 2) % 2 == 1,
        };
        let reach = self.reach * width * 0.43 * (1. + self.jitter * r.random_range(-1.0..1.));
        HeadSite {
            x,
            z: reach.min(width * 0.5 - self.radius - 0.002)
                * if flip { -self.side } else { self.side },
            clocking: r.random_range(0.0..std::f32::consts::TAU),
        }
    }

    /// Height from the bottom of the module to the outlet face.
    pub(crate) fn outlet_y(self, height: f32, upright: bool) -> f32 {
        let end_space = self.span + end_allowance(self.crown);
        if upright {
            height - end_space
        } else {
            end_space
        }
    }
}
