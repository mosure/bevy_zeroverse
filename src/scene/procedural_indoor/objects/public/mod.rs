//! Bounded, multipart public-space assemblies. All parameters are sampled before
//! admission; construction, collision envelopes and catalog receipts share them.
pub mod banquette;
pub mod barrier;
pub mod basketball;
pub mod baskets;
pub mod beam_seating;
pub mod bleachers;
pub(crate) mod cardio;
pub mod casework;
pub mod cooler;
pub mod dispenser;
#[cfg(test)]
mod diversity_tests;
pub mod dumbbells;
pub mod easel;
pub mod education;
pub mod elevator;
pub mod espresso;
pub(crate) mod fitness;
mod fitting;
pub mod fountain;
pub mod volleyball;
pub mod workbench;
pub use fitting::{fitting_room, CurtainBay, FittingRoomSpec};
pub mod furnishing;
mod hospitality;
pub mod hot_food;
pub mod laundry;
pub mod mats;
pub mod music;
pub mod reception;
#[cfg(test)]
mod repetition_tests;
pub(crate) mod retail;
pub use retail::checkout::{checkout_station, CheckoutForm, CheckoutStation};
pub use retail::{
    apparel_rail, folded_apparel, gondola_stock, produce_stand, refrigerated_case, ApparelRail,
    CaseStock, CrateStyle, DisplayTableStand, FoldedCut, FoldedDisplay, FoldedGarment, FoldedLayer,
    FoldedPile, GondolaRow, GondolaStock, ProduceBase, ProduceBin, ProducePiece, ProduceStand,
    RailForm, RefrigeratedCase,
};
pub mod sculpture;
pub mod service;
pub mod signage;
pub mod strength;
mod supports;
#[cfg(test)]
mod tests;
use super::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PublicItemProgram {
    pub bays: usize,
    pub shelves: usize,
    pub opening: f32,
    pub taper: f32,
    pub curvature: f32,
    pub contents: bool,
    pub glazed: bool,
    pub controls: usize,
    pub finish: Surface,
}
pub fn parameters(o: &IndoorObject) -> PublicItemProgram {
    let mut r = stream(o.seed, 8201);
    let mut finish = stream(o.finish_seed.unwrap_or(o.seed), 8202);
    PublicItemProgram {
        bays: r.random_range(2..=4),
        shelves: r.random_range(2..=4),
        opening: if r.random_bool(0.30) {
            0.
        } else {
            r.random_range(0.25..1.0)
        },
        taper: r.random_range(0.60..0.96),
        curvature: r.random_range(-0.9..0.9),
        contents: r.random_bool(0.92),
        glazed: r.random_bool(0.55),
        controls: r.random_range(1..=3),
        finish: [Surface::Wood, Surface::Metal, Surface::Plastic][finish.random_range(0..3)],
    }
}
pub fn dimensions(kind: ObjectKind, seed: u64) -> Vec3 {
    use ObjectKind::*;
    let mut r = stream(seed, 8203);
    // Metric product ranges, not one XYZ multiplier for unrelated categories.
    // Sample before admission: support areas, navigation and labels see these sizes.
    let (x, y, z) = match kind {
        WashingMachine | TumbleDryer | LaundryFoldingTable => {
            return laundry::dimensions(kind, seed)
        }
        BeamSeating => return beam_seating::dimensions(&mut r),
        YogaBlock | YogaBolster => return mats::props::dimensions(kind, seed),
        LaundryCart => return laundry::cart::dimensions(seed),
        Timpani => return music::timpani::catalog_dimensions(&mut r),
        MusicStand | StagePiano => return music::dimensions(kind, &mut r),
        BasketStack => return baskets::dimensions(seed),
        WaterCooler => return cooler::dimensions(seed),
        DrinkingFountain => return fountain::dimensions(seed),
        EspressoMachine | CoffeeGrinder => return espresso::dimensions(kind, seed),
        BleacherBank => return bleachers::catalog_dimensions(&mut r),
        VolleyballNet => return volleyball::dimensions(seed),
        BasketballGoal => return basketball::dimensions(&mut r),
        FittingRoom => return fitting::dimensions(&mut r),
        Elevator => (
            (elevator::MIN_FOOTPRINT.x, 2.90),
            (2.50, 2.95),
            (elevator::MIN_FOOTPRINT.y, 2.20),
        ),
        ServiceCounter => ((1.45, 3.0), (0.82, 1.10), (0.60, 0.90)),
        DisplayCase => ((0.80, 1.95), (0.88, 1.92), (0.48, 0.84)),
        TrayReturn => ((0.65, 1.25), (0.86, 1.55), (0.46, 0.76)),
        DrinkDispenser => ((0.25, 0.60), (0.37, 0.73), (0.28, 0.46)),
        QueueBarrier => ((0.85, 1.85), (0.76, 1.08), (0.28, 0.44)),
        Lectern => ((0.44, 0.88), (0.92, 1.30), (0.38, 0.66)),
        Locker => ((0.72, 1.80), (1.45, 2.10), (0.38, 0.60)),
        TeachingBoard => ((0.85, 3.35), (1.40, 2.30), (0.48, 0.88)),
        ExhibitPlinth => ((0.42, 1.12), (0.32, 1.35), (0.40, 1.08)),
        Sculpture => ((0.24, 0.78), (0.26, 1.05), (0.22, 0.70)),
        InformationStand => ((0.32, 0.72), (0.84, 1.55), (0.30, 0.58)),
        ExhibitionPanel => ((0.90, 2.35), (1.42, 2.40), (0.38, 0.70)),
        ExtractionHood => ((0.70, 1.85), (0.34, 0.85), (0.48, 0.78)),
        StationaryBike => ((0.55, 0.74), (1.08, 1.39), (1.16, 1.48)),
        Treadmill => ((0.78, 1.02), (1.21, 1.55), (1.78, 2.24)),
        PowerRack => return strength::dimensions(&mut r),
        HotFoodWell => return hot_food::dimensions(&mut r),
        WeightBench => return fitness::bench_dimensions(&mut r),
        DumbbellRack => ((1.05, 2.25), (0.69, 1.04), (0.48, 0.73)),
        GroceryDisplay | ApparelDisplay | CheckoutCounter => {
            return retail::dimensions(kind, &mut r)
        }
        Workbench => ((1.34, 2.46), (0.82, 0.97), (0.70, 0.97)),
        TimberOffcut => ((0.18, 0.75), (0.019, 0.065), (0.075, 0.205)),
        ArtistEasel => ((0.54, 1.09), (1.39, 2.05), (0.51, 0.84)),
        ReceptionDesk => ((1.55, 3.42), (0.99, 1.16), (0.80, 1.08)),
        ExerciseMat => return mats::dimensions(&mut r),
        ExerciseMatRack => ((0.77, 1.56), (0.80, 1.43), (0.64, 0.86)),
        Banquette => ((1.30, 3.0), (0.88, 1.25), (0.60, 0.85)),
        _ => unreachable!("public program item"),
    };
    Vec3::new(
        r.random_range(x.0..x.1),
        r.random_range(y.0..y.1),
        r.random_range(z.0..z.1),
    )
}
pub fn build(a: &mut Assembly, o: &IndoorObject) {
    build_at_version(a, o, super::super::layout::GENERATOR_VERSION);
}

pub(super) fn build_at_version(a: &mut Assembly, o: &IndoorObject, version: u32) {
    match o.kind {
        ObjectKind::WashingMachine | ObjectKind::TumbleDryer | ObjectKind::LaundryFoldingTable => {
            laundry::build(a, o)
        }
        ObjectKind::MusicStand => music::stand(a, o),
        ObjectKind::BeamSeating => beam_seating::build(a, o, version),
        ObjectKind::LaundryCart => laundry::cart::build(a, o),
        ObjectKind::Timpani => music::timpani::build(a, o),
        ObjectKind::StagePiano => music::piano(a, o),
        ObjectKind::BasketStack => baskets::build(a, o),
        ObjectKind::WaterCooler => cooler::build(a, o),
        ObjectKind::DrinkingFountain => fountain::build(a, o),
        ObjectKind::EspressoMachine => espresso::build_machine(a, o),
        ObjectKind::CoffeeGrinder => espresso::build_grinder(a, o),
        ObjectKind::Workbench => workbench::build(a, o),
        ObjectKind::TimberOffcut => workbench::offcut(a, o),
        ObjectKind::ArtistEasel => easel::build(a, o),
        ObjectKind::ReceptionDesk => reception::build(a, o),
        ObjectKind::VolleyballNet => volleyball::build(a, o),
        ObjectKind::BasketballGoal => basketball::build(a, o),
        ObjectKind::BleacherBank => bleachers::build(a, o),
        ObjectKind::PowerRack => strength::build(a, o),
        ObjectKind::HotFoodWell => hot_food::build(a, o),
        ObjectKind::FittingRoom => fitting::build(a, o),
        ObjectKind::Elevator => elevator::build(a, o, version),
        ObjectKind::QueueBarrier => barrier::build(a, o, version),
        ObjectKind::DisplayCase | ObjectKind::TrayReturn | ObjectKind::ExhibitPlinth => {
            casework::build(a, o, version)
        }
        ObjectKind::DrinkDispenser => dispenser::build(a, o),
        ObjectKind::ServiceCounter => service::counter(a, o),
        ObjectKind::ExtractionHood => service::hood::build(a, o, version),
        ObjectKind::StationaryBike | ObjectKind::Treadmill => cardio::build(a, o),
        ObjectKind::DumbbellRack if version >= dumbbells::VERSION => dumbbells::build(a, o),
        ObjectKind::ExerciseMat if version >= mats::VERSION => mats::build(a, o),
        ObjectKind::ExerciseMatRack => mats::build_rack(a, o),
        ObjectKind::YogaBlock | ObjectKind::YogaBolster => mats::props::build(a, o),
        ObjectKind::WeightBench | ObjectKind::DumbbellRack | ObjectKind::ExerciseMat => {
            fitness::build(a, o)
        }
        ObjectKind::GroceryDisplay | ObjectKind::ApparelDisplay | ObjectKind::CheckoutCounter => {
            retail::build(a, o, version)
        }
        ObjectKind::Banquette => banquette::build(a, o, version),
        ObjectKind::Locker => education::build(a, o, version),
        ObjectKind::Lectern | ObjectKind::TeachingBoard => furnishing::build(a, o, version),
        ObjectKind::Sculpture => sculpture::build(a, o, version),
        ObjectKind::InformationStand => signage::stand(a, o, version),
        ObjectKind::ExhibitionPanel => signage::exhibition(a, o),
        _ => unreachable!("public program item"),
    }
}
/// Coordinates are fractions of the pre-admitted object envelope, with Y=0 at contact.
fn panel(
    a: &mut Assembly,
    o: &IndoorObject,
    surface: Surface,
    label: &str,
    p: [f32; 3],
    size: [f32; 3],
) {
    a.box_part(
        surface,
        label,
        Vec3::from_array(p) * o.size,
        Vec3::from_array(size) * o.size,
        0.004,
    );
}
fn rod(
    a: &mut Assembly,
    o: &IndoorObject,
    surface: Surface,
    label: &str,
    p: [f32; 3],
    q: [f32; 3],
    radius: f32,
) {
    a.part(surface, label).rod(
        Vec3::from_array(p) * o.size,
        Vec3::from_array(q) * o.size,
        radius,
    );
}
pub(super) use super::append_at;
