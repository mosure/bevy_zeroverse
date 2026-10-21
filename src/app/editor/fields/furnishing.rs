//! Independent per-table population and amount controls.
use super::*;
pub(super) fn append(v: &mut Vec<Item>) {
    let at = v
        .iter()
        .position(|i| matches!(i, Item::Section("AUTOMATION", _)))
        .unwrap_or(v.len());
    v.splice(at..at, [
        section("TABLE BELONGINGS", "Each table independently samples clear or occupied, then a continuous amount. Clear desks can retain workstation screens and input devices. Changes wait for Apply."),
        num("/indoor_clutter/probability", "Tables with belongings", 0., 1., 0.01, "0 leaves tables clear; 1 selects every table. This controls frequency, not item count."),
        num("/indoor_clutter/minimum", "Minimum amount", 0., 1., 0.01, "Lower amount bound on selected tables. Counts scale with usable area and furnishing density; clearance is always enforced."),
        num("/indoor_clutter/maximum", "Maximum amount", 0., 1., 0.01, "Upper amount bound. 0 disables loose table items; 1 allows busy tables with working space preserved."),
        num("/indoor_clutter/bias", "Preference · light / busy", -1., 1., 0.01, "Continuously moves the amount distribution toward light (−1) or busy (+1)."),
        num("/indoor_clutter/spread", "Amount variation", 0., 1., 0.01, "0 fixes the amount at its median; 1 spans the requested range. Frequency remains independent."),
        Item::Distribution(super::super::distribution_preview::DistributionKind::Clutter),
    ]);
}
