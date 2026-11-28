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
        ObjectKind::WeightBench
        | ObjectKind::WashingMachine
        | ObjectKind::TumbleDryer
        | ObjectKind::MusicStand
        | ObjectKind::StagePiano
        | ObjectKind::Timpani
        | ObjectKind::BeamSeating
        | ObjectKind::YogaBlock
        | ObjectKind::YogaBolster
        | ObjectKind::LaundryCart
        | ObjectKind::LaundryFoldingTable
        | ObjectKind::BasketStack
        | ObjectKind::DrinkingFountain
        | ObjectKind::WaterCooler
        | ObjectKind::CoffeeGrinder
        | ObjectKind::EspressoMachine
        | ObjectKind::Workbench
        | ObjectKind::TimberOffcut
        | ObjectKind::ArtistEasel
        | ObjectKind::ReceptionDesk
        | ObjectKind::HotFoodWell
        | ObjectKind::PowerRack
        | ObjectKind::DumbbellRack
        | ObjectKind::GroceryDisplay
        | ObjectKind::BleacherBank
        | ObjectKind::VolleyballNet
        | ObjectKind::BasketballGoal
        | ObjectKind::FittingRoom
        | ObjectKind::ApparelDisplay
        | ObjectKind::CheckoutCounter
        | ObjectKind::StationaryBike
        | ObjectKind::Treadmill
        | ObjectKind::ExerciseMat
        | ObjectKind::ExerciseMatRack
        | ObjectKind::Elevator
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
        | ObjectKind::Flatware
        | ObjectKind::CuttingBoard
        | ObjectKind::FruitBowl
        | ObjectKind::Bread => kitchen::build(a, o, bevy_zeroverse_capture::GENERATOR_VERSION),
        ObjectKind::Projector
        | ObjectKind::ProjectionScreen
        | ObjectKind::PowerStrip
        | ObjectKind::Cable
        | ObjectKind::WallShelf
        | ObjectKind::NoticeBoard
        | ObjectKind::WallSconce
        | ObjectKind::CeilingPendant => {
            equipment::build(a, o, bevy_zeroverse_capture::GENERATOR_VERSION)
        }
        ObjectKind::Printer | ObjectKind::StorageBox | ObjectKind::CoatRack | ObjectKind::Bag => {
            utilities::build(a, o, super::super::layout::GENERATOR_VERSION)
        }
        ObjectKind::Table | ObjectKind::Desk | ObjectKind::CoffeeTable | ObjectKind::SideTable => {
            tables::build(a, o, super::super::layout::GENERATOR_VERSION)
        }
        ObjectKind::MobileCart
        | ObjectKind::DrawerUnit
        | ObjectKind::Bench
        | ObjectKind::Ottoman
        | ObjectKind::BaseboardHeater
        | ObjectKind::RoomDivider => services::build(a, o, super::super::layout::GENERATOR_VERSION),
        ObjectKind::DeskLamp
        | ObjectKind::Speaker
        | ObjectKind::Headphones
        | ObjectKind::Stapler
        | ObjectKind::TapeDispenser
        | ObjectKind::DocumentTray
        | ObjectKind::DeskOrganizer
        | ObjectKind::TissueBox
        | ObjectKind::DeskFan => accessories::build(a, o, super::super::layout::GENERATOR_VERSION),
        ObjectKind::CoffeeMaker | ObjectKind::Kettle | ObjectKind::Vase | ObjectKind::Bowl => {
            hospitality::build(a, o, super::super::layout::GENERATOR_VERSION)
        }
        ObjectKind::Chair => chairs::build(a, o, super::super::layout::GENERATOR_VERSION),
        ObjectKind::Sofa => seating::build(a, o, super::super::layout::GENERATOR_VERSION),
        ObjectKind::Cabinet | ObjectKind::Bookcase | ObjectKind::LibraryStack => {
            storage::build(a, o, super::super::layout::GENERATOR_VERSION)
        }
        ObjectKind::Thermostat => wall::thermostat::build(a, o),
        ObjectKind::WallOutlet | ObjectKind::LightSwitch => wall::fixture(a, o),
        ObjectKind::Keyboard | ObjectKind::Mouse | ObjectKind::PcTower => {
            desktop::build(a, o, super::super::layout::GENERATOR_VERSION)
        }
        ObjectKind::TableNapkin
        | ObjectKind::DrinkingGlass
        | ObjectKind::WaterBottle
        | ObjectKind::Mug
        | ObjectKind::CoffeeCup
        | ObjectKind::SodaCan
        | ObjectKind::Notepad
        | ObjectKind::Pencil
        | ObjectKind::Microphone
        | ObjectKind::Phone
        | ObjectKind::PenHolder => {
            super::super::clutter::build(a, o, super::super::layout::GENERATOR_VERSION)
        }
        ObjectKind::Plant | ObjectKind::PlantingBed => super::super::plants::build(a, o),
        ObjectKind::TrashCan => containers::bin(a, o),
        ObjectKind::Display | ObjectKind::Monitor => {
            computers::display(a, o, super::super::layout::GENERATOR_VERSION)
        }
        ObjectKind::Laptop => computers::laptop(a, o, super::super::layout::GENERATOR_VERSION),
        ObjectKind::Whiteboard | ObjectKind::WallArt => wall_panel(a, o),
        ObjectKind::Notebook | ObjectKind::Books => literature::build(
            a,
            o,
            crate::scene::procedural_indoor::layout::GENERATOR_VERSION,
        ),
        ObjectKind::Clock => clocks::build(a, o),
        ObjectKind::FloorLamp => {
            standing_lamp::build(a, o, super::super::layout::GENERATOR_VERSION)
        }
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
            routing::build(
                &mut actual,
                &object,
                super::super::layout::GENERATOR_VERSION,
            );
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

#[test]
fn merging_transformed_parts_reuses_the_first_mesh_and_offsets_subsequent_indices() {
    let mut dest = Assembly::default();
    let mut child = Assembly::default();
    child.box_part(Surface::Concrete, "plant", Vec3::Y, Vec3::ONE, 0.);
    let key = (Surface::Concrete, String::from("plant"));
    let source = child.parts[&key].clone();
    let vertices = child.parts[&key].positions.as_ptr();
    let transform = Transform::from_xyz(0.5, 0.3, 0.9).with_rotation(Quat::from_rotation_y(0.76));
    append_at(&mut dest, child, transform);
    assert_eq!(dest.parts[&key].positions.as_ptr(), vertices);
    let mut second = Assembly::default();
    second.parts.insert(key.clone(), source.clone());
    append_at(&mut dest, second, Transform::IDENTITY);
    let merged = &dest.parts[&key];
    let n = source.positions.len();
    for (i, p) in source.positions.iter().enumerate() {
        assert_eq!(
            merged.positions[i],
            transform.transform_point(Vec3::from_array(*p)).to_array()
        );
        assert_eq!(
            merged.normals[i],
            (transform.rotation * Vec3::from_array(source.normals[i])).to_array()
        );
    }
    assert_eq!(merged.positions[n..], source.positions);
    assert_eq!(merged.uvs, [&source.uvs[..], &source.uvs[..]].concat());
    assert_eq!(merged.indices[..source.indices.len()], source.indices);
    assert!(merged.indices[source.indices.len()..]
        .iter()
        .zip(&source.indices)
        .all(|(a, b)| *a == *b + n as u32));
}
