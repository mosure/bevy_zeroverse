//! Stable random namespaces for unrelated placement decisions. A rejected
//! proposal consumes only its own stream, never the next fixture's choices.
//! Physical exclusion and deliberately coordinated assemblies still couple
//! accepted placements; seeded replay is a feature, not a training shortcut.
use super::{stream, ChaCha8Rng, IndoorManifest, ObjectKind};
use rand::{seq::SliceRandom, Rng};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
pub(super) enum Domain {
    SurfaceDressing = 0x5a56_5350_0000_0001,
    PublicDressing = 0x5a56_5350_0000_0002,
    WallOrder = 0x5a56_5350_0000_0003,
    WallArt = 0x5a56_5350_0000_0004,
    Displays = 0x5a56_5350_0000_0005,
    Whiteboards = 0x5a56_5350_0000_0006,
    Clocks = 0x5a56_5350_0000_0007,
    Electrical = 0x5a56_5350_0000_0008,
    WallEquipment = 0x5a56_5350_0000_0009,
    Pendants = 0x5a56_5350_0000_000a,
    WallGroups = 0x5a56_5350_0000_000b,
    ExitSigns = 0x5a56_5350_0000_000c,
    Alarms = 0x5a56_5350_0000_000d,
    Speakers = 0x5a56_5350_0000_000e,
    Detectors = 0x5a56_5350_0000_000f,
    StringLights = 0x5a56_5350_0000_0010,
    PullStations = 0x5a56_5350_0000_0011,
    PowerStandard = 0x5a56_5350_0000_0012,
}

impl Domain {
    pub(super) fn rng(self, seed: u64) -> ChaCha8Rng {
        stream(seed, self as u64)
    }

    /// Fixed two-word keys: early exits, retry budgets and object IDs cannot
    /// shift a subsequent proposal. The child can draw arbitrarily many words.
    pub(super) fn proposal(self, seed: u64, ordinal: usize) -> ChaCha8Rng {
        let mut keys = stream(seed, self as u64 ^ (1 << 63));
        keys.set_word_pos(ordinal as u128 * 2);
        stream(keys.random(), 0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WallPass {
    Groups,
    Art,
    Displays,
    Whiteboards,
    Clocks,
    Electrical,
    ExitSigns,
    Alarms,
    Speakers,
    Detectors,
    Strings,
    PullStations,
}

fn wall_order(seed: u64) -> [WallPass; 12] {
    let mut passes = [
        WallPass::Groups,
        WallPass::Art,
        WallPass::Displays,
        WallPass::Whiteboards,
        WallPass::Clocks,
        WallPass::Electrical,
        WallPass::ExitSigns,
        WallPass::Alarms,
        WallPass::Speakers,
        WallPass::Detectors,
        WallPass::Strings,
        WallPass::PullStations,
    ];
    passes.shuffle(&mut Domain::WallOrder.rng(seed));
    passes
}

impl IndoorManifest {
    /// Functional furniture, projector/screen pairs, primary-room bounds and
    /// ceiling reservations already exist. Unrelated wall installations get a
    /// seedable priority; none is always a leftover of a different category.
    pub(super) fn furnish_wall_installations(&mut self) {
        for pass in wall_order(self.seed) {
            match pass {
                WallPass::Groups => self.compose_wall_groups(),
                WallPass::Art => self.wall_decoration_kind(ObjectKind::WallArt),
                WallPass::Displays => self.wall_decoration_kind(ObjectKind::Display),
                WallPass::Whiteboards => self.wall_decoration_kind(ObjectKind::Whiteboard),
                WallPass::Clocks => self.wall_decoration_kind(ObjectKind::Clock),
                WallPass::Electrical => self.wall_hardware(),
                WallPass::ExitSigns => self.furnish_exit_signs(),
                WallPass::Alarms => self.furnish_wall_service(ObjectKind::FireAlarm),
                WallPass::Speakers => self.furnish_wall_service(ObjectKind::Speaker),
                WallPass::Detectors => self.furnish_detectors(),
                WallPass::Strings => self.furnish_string_lights(),
                WallPass::PullStations => self.furnish_pull_station(),
            }
        }
    }
}

#[cfg(test)]
mod tests;
