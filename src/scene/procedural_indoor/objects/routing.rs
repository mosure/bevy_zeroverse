//! Exhaustive family dispatch; no dynamic registry or allocation on the hot path.
use super::*;

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    match o.kind.family() {
        ObjectFamily::CeilingFans => fans::build(a, o),
        ObjectFamily::Signals => equipment::signals::build(a, o),
        ObjectFamily::Public => public::build(a, o),
        ObjectFamily::Kitchen => kitchen::build(a, o),
        ObjectFamily::Equipment => equipment::build(a, o),
        ObjectFamily::Utilities => utilities::build(a, o),
        ObjectFamily::Tables => tables::build(a, o),
        ObjectFamily::Services => services::build(a, o),
        ObjectFamily::Accessories => accessories::build(a, o),
        ObjectFamily::Hospitality => hospitality::build(a, o),
        ObjectFamily::Chairs => chairs::build(a, o),
        ObjectFamily::Seating => seating::build(a, o),
        ObjectFamily::Storage => storage::build(a, o),
        ObjectFamily::WallFixtures => wall::fixture(a, o),
        ObjectFamily::Desktop => desktop::build(a, o),
        ObjectFamily::Tabletop => super::super::clutter::build(a, o),
        ObjectFamily::Plants => super::super::plants::build(a, o),
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
