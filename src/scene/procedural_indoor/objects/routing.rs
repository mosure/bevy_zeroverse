//! Exhaustive family dispatch; no dynamic registry or allocation on the hot path.
use super::*;

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    match o.kind.family() {
        ObjectFamily::CeilingFans => fans::build(a, o),
        ObjectFamily::Signals => equipment::signals::build(a, o),
        ObjectFamily::Public => public::build_at_version(a, o, version),
        ObjectFamily::Kitchen => kitchen::build(a, o, version),
        ObjectFamily::Equipment => equipment::build(a, o, version),
        ObjectFamily::Utilities => utilities::build(a, o, version),
        ObjectFamily::Tables => tables::build(a, o, version),
        ObjectFamily::Services => services::build(a, o, version),
        ObjectFamily::Accessories => accessories::build(a, o, version),
        ObjectFamily::Hospitality => hospitality::build(a, o),
        ObjectFamily::Chairs => chairs::build(a, o, version),
        ObjectFamily::Seating => seating::build(a, o, version),
        ObjectFamily::Storage => storage::build(a, o, version),
        ObjectFamily::WallFixtures => wall::fixture(a, o),
        ObjectFamily::Desktop => desktop::build(a, o, version),
        ObjectFamily::Tabletop => super::super::clutter::build(a, o),
        ObjectFamily::Plants => super::super::plants::build_at_version(a, o, version),
        ObjectFamily::Bins => containers::bin(a, o),
        ObjectFamily::Displays => computers::display(a, o),
        ObjectFamily::Laptops => computers::laptop(a, o),
        ObjectFamily::WallPanels => wall_panel(a, o),
        ObjectFamily::Literature => literature::build(a, o),
        ObjectFamily::Clocks => clocks::build(a, o),
        ObjectFamily::FloorLamps => standing_lamp::build(a, o),
        ObjectFamily::Floor => floor::build(a, o),
    }
}
