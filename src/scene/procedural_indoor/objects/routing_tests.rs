//! Frozen pre-refactor route; catches wrong-family geometry and material wiring.
use super::*;

fn original_route(a: &mut Assembly, o: &IndoorObject) {
    match o.kind {
        ObjectKind::CeilingFan => fans::build(a, o),
        ObjectKind::ExitSign
        | ObjectKind::FireAlarm
        | ObjectKind::FireDetector
        | ObjectKind::FirePullStation
        | ObjectKind::StringLights => equipment::signals::build(a, o),
        ObjectKind::Elevator
        | ObjectKind::ServiceCounter
        | ObjectKind::DisplayCase
        | ObjectKind::TrayReturn
        | ObjectKind::DrinkDispenser
        | ObjectKind::QueueBarrier
        | ObjectKind::Lectern
        | ObjectKind::Locker
        | ObjectKind::TeachingBoard
        | ObjectKind::ExhibitPlinth
        | ObjectKind::Sculpture
        | ObjectKind::InformationStand
        | ObjectKind::ExhibitionPanel
        | ObjectKind::ExtractionHood
        | ObjectKind::Banquette => public::build(a, o),
        ObjectKind::KitchenCounter
        | ObjectKind::KitchenSink
        | ObjectKind::KitchenCupboard
        | ObjectKind::Pantry
        | ObjectKind::Cooktop
        | ObjectKind::Range
        | ObjectKind::Oven
        | ObjectKind::Microwave
        | ObjectKind::Dishwasher
        | ObjectKind::Refrigerator
        | ObjectKind::DryingRack
        | ObjectKind::SoapDispenser
        | ObjectKind::Sponge
        | ObjectKind::KnifeBlock
        | ObjectKind::UtensilHolder
        | ObjectKind::PaperTowel
        | ObjectKind::KitchenTowel
        | ObjectKind::Dishware
        | ObjectKind::CuttingBoard
        | ObjectKind::FruitBowl
        | ObjectKind::Bread => kitchen::build(a, o),
        ObjectKind::Projector
        | ObjectKind::ProjectionScreen
        | ObjectKind::PowerStrip
        | ObjectKind::Cable
        | ObjectKind::WallShelf
        | ObjectKind::NoticeBoard
        | ObjectKind::WallSconce
        | ObjectKind::CeilingPendant => equipment::build(a, o),
        ObjectKind::Printer | ObjectKind::StorageBox | ObjectKind::CoatRack | ObjectKind::Bag => {
            utilities::build(a, o)
        }
        ObjectKind::Table | ObjectKind::Desk | ObjectKind::CoffeeTable | ObjectKind::SideTable => {
            tables::build(a, o)
        }
        ObjectKind::MobileCart
        | ObjectKind::DrawerUnit
        | ObjectKind::Bench
        | ObjectKind::Ottoman
        | ObjectKind::BaseboardHeater
        | ObjectKind::RoomDivider => services::build(a, o),
        ObjectKind::DeskLamp
        | ObjectKind::Speaker
        | ObjectKind::Headphones
        | ObjectKind::Stapler
        | ObjectKind::TapeDispenser
        | ObjectKind::DocumentTray
        | ObjectKind::DeskOrganizer
        | ObjectKind::TissueBox
        | ObjectKind::DeskFan => accessories::build(a, o),
        ObjectKind::CoffeeMaker | ObjectKind::Kettle | ObjectKind::Vase | ObjectKind::Bowl => {
            hospitality::build(a, o)
        }
        ObjectKind::Chair => chairs::build(a, o),
        ObjectKind::Sofa => seating::build(a, o),
        ObjectKind::Cabinet | ObjectKind::Bookcase => storage::build(a, o),
        ObjectKind::WallOutlet | ObjectKind::LightSwitch => wall::fixture(a, o),
        ObjectKind::Keyboard | ObjectKind::Mouse => desktop::build(a, o),
        ObjectKind::WaterBottle
        | ObjectKind::Mug
        | ObjectKind::CoffeeCup
        | ObjectKind::SodaCan
        | ObjectKind::Notepad
        | ObjectKind::Pencil
        | ObjectKind::Microphone
        | ObjectKind::Phone
        | ObjectKind::PenHolder => super::super::clutter::build(a, o),
        ObjectKind::Plant => super::super::plants::build(a, o),
        ObjectKind::TrashCan => containers::bin(a, o),
        ObjectKind::Display | ObjectKind::Monitor => computers::display(a, o),
        ObjectKind::Laptop => computers::laptop(a, o),
        ObjectKind::Whiteboard | ObjectKind::WallArt => wall_panel(a, o),
        ObjectKind::Notebook | ObjectKind::Books => literature::build(a, o),
        ObjectKind::Clock => clocks::build(a, o),
        ObjectKind::FloorLamp => standing_lamp::build(a, o),
        ObjectKind::Rug | ObjectKind::FloorOutlet => floor::build(a, o),
    }
}

#[test]
fn catalog_routing_preserves_every_geometry_buffer_and_semantic_material_key() {
    for kind in ObjectKind::ALL {
        for seed in [0, 1, 2, 7, 10, 11, 1 << 40, u64::MAX] {
            let object = audit::fixture(kind, seed);
            let mut expected = Assembly::default();
            original_route(&mut expected, &object);
            let mut actual = Assembly::default();
            routing::build(&mut actual, &object);
            assert_eq!(actual.parts.len(), expected.parts.len(), "{kind:?}/{seed}");
            assert!(!actual.parts.is_empty(), "missing builder: {kind:?}");
            for ((key, a), (old_key, b)) in actual.parts.iter().zip(&expected.parts) {
                assert_eq!(key, old_key, "{kind:?}/{seed}");
                assert_eq!(a.positions, b.positions, "positions {kind:?}/{seed}");
                assert_eq!(a.normals, b.normals, "normals {kind:?}/{seed}");
                assert_eq!(a.uvs, b.uvs, "uvs {kind:?}/{seed}");
                assert_eq!(a.indices, b.indices, "indices {kind:?}/{seed}");
            }
        }
    }
}

#[test]
fn catalog_preserves_ordinals_names_semantics_and_placement_capabilities() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/indoor_object_catalog.json"
    ))
    .unwrap();
    let rows = fixture.as_array().unwrap();
    assert_eq!(ObjectKind::ALL.len(), rows.len());
    for (ordinal, (kind, row)) in ObjectKind::ALL.into_iter().zip(rows).enumerate() {
        assert_eq!(kind as usize, ordinal);
        assert_eq!(serde_json::to_value(kind).unwrap(), row["name"]);
        assert_eq!(
            serde_json::from_value::<ObjectKind>(row["name"].clone()).unwrap(),
            kind
        );
        assert_eq!(kind.class_name(), row["class"]);
        assert_eq!(kind.is_surface(), row["surface"]);
        assert_eq!(kind.has_table_outline(), row["table"]);
        assert_eq!(kind.is_kitchen(), row["kitchen"]);
        assert_eq!(kind.is_program_item(), row["program"]);
    }
}
