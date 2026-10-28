//! Seeded kitchen construction. Dimensions include every opened door and handle.
//! Programs are pure: layout, geometry, audits and archives share the same seed.
pub mod appliances;
mod cabinet;
pub(crate) mod contents;
pub mod dishware;
mod island;
pub mod packaging;
mod props;
pub mod sink;
pub mod supplies;
#[cfg(test)]
mod supply_program_tests;
#[cfg(test)]
mod tests;
use super::*;
pub use cabinet::fronts::FrontProgram;
pub(crate) use cabinet::{drawer_tray, shell as case_shell};
pub use island::{configure_island, is_island, island_parameters, IslandProgram};
use serde::Serialize;

/// Usable slab bounds, shared by prop admission and countertop geometry.
pub fn counter_top_bounds(o: &IndoorObject) -> (Vec2, Vec2) {
    let half = o.size.xz() * 0.5;
    if is_island(o) {
        (-half, half)
    } else {
        {
            let (d, z) = body(o, &parameters(o));
            (
                Vec2::new(-half.x, z - d * 0.5),
                Vec2::new(half.x, z + d * 0.5 + 0.006),
            )
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct KitchenProgram {
    pub door_style: u8, // solid, glazed, open shelving, swung open
    pub opening_radians: f32,
    pub handles: u8,
    pub hardware: Surface,
    /// 0 hinged, 1 bypass sliding, 2 upper lift-up (lower fronts remain hinged).
    pub cabinet_motion: u8,
    /// Slab, framed inset, raised field, fluted face.
    pub front_profile: u8,
    pub front: FrontProgram,
    pub counter_doors: bool,
    pub upper_bottom_fraction: f32,
    pub upper_depth_fraction: f32,
    pub upper_width_fraction: f32,
    pub upper_offset_fraction: f32,
    pub paired: bool,
    pub shelves: u32,
    pub contents: bool,
    pub gas: bool,
    pub controls: u8, // front, back, right side
    pub viewport: bool,
    pub stacked_ovens: bool,
    pub door_notes: u32,
    pub basins: u32,
    pub faucet: u8,
    pub faucet_handles: u8,
    pub hand_soap: bool,
    pub dish_soap: bool,
    pub sponge: bool,
    pub sink_dishes: u32,
    pub draining_board: bool,
    pub finish: Surface,
    pub top: Surface,
}

pub fn parameters(o: &IndoorObject) -> KitchenProgram {
    let mut r = stream(o.seed, 8010);
    let door_style = r.random_range(0..4);
    let mut front_rng = stream(o.seed, 8013);
    let cabinet_motion = front_rng.random_range(0..3);
    let mut finish_rng = stream(o.finish_seed.unwrap_or(o.seed), 8012);
    KitchenProgram {
        door_style,
        opening_radians: match r.random_range(0..10) {
            0..=4 => 0.,
            5..=6 => r.random_range(0.12..0.55),
            _ => r.random_range(0.65..1.35),
        },
        handles: r.random_range(0..5),
        hardware: [
            Surface::Chrome,
            Surface::Metal,
            Surface::Ink,
            Surface::WoodEdge,
        ][front_rng.random_range(0..4)],
        cabinet_motion,
        front_profile: front_rng.random_range(0..4),
        front: FrontProgram::sample(o.seed),
        counter_doors: front_rng.random_bool(0.55),
        upper_bottom_fraction: front_rng.random_range(0.60..0.72),
        upper_depth_fraction: front_rng.random_range(0.45..0.72),
        upper_width_fraction: front_rng.random_range(0.85..1.0),
        upper_offset_fraction: front_rng.random_range(-0.45..0.45),
        paired: r.random_bool(0.55)
            || (matches!(
                o.kind,
                ObjectKind::KitchenCounter
                    | ObjectKind::KitchenCupboard
                    | ObjectKind::Pantry
                    | ObjectKind::KitchenSink
            ) && o.size.x > 1.25),
        shelves: r.random_range(2..=4),
        contents: r.random_bool(0.90),
        gas: r.random_bool(0.5),
        controls: r.random_range(0..3),
        viewport: r.random_bool(match o.kind {
            ObjectKind::Dishwasher => 0.04,
            ObjectKind::Refrigerator => 0.24,
            ObjectKind::Microwave => 0.94,
            _ => 0.72,
        }),
        stacked_ovens: o.size.y > 1.3 && r.random_bool(0.48),
        door_notes: if r.random_bool(0.68) {
            r.random_range(1..=4)
        } else {
            0
        },
        basins: if o.size.x > 0.95 && r.random_bool(0.48) {
            2
        } else {
            1
        },
        faucet: r.random_range(0..4),
        faucet_handles: r.random_range(0..3),
        hand_soap: r.random_bool(0.65),
        dish_soap: r.random_bool(0.76),
        sponge: r.random_bool(0.62),
        sink_dishes: if r.random_bool(0.58) {
            r.random_range(1..=3)
        } else {
            0
        },
        draining_board: r.random_bool(0.55),
        finish: [
            Surface::Wood,
            Surface::Plastic,
            Surface::Metal,
            Surface::Paint,
        ][finish_rng.random_range(0..4)],
        top: [
            Surface::Wood,
            Surface::Ceramic,
            Surface::Concrete,
            Surface::Metal,
        ][finish_rng.random_range(0..4)],
    }
}

pub fn fixture_size(kind: ObjectKind) -> Vec3 {
    use ObjectKind::*;
    Vec3::from_array(match kind {
        KitchenCounter => [1.05, 0.90, 0.66],
        KitchenSink => [1.25, 1.30, 0.72],
        KitchenCupboard => [1.0, 2.12, 0.67],
        Pantry => [0.85, 2.10, 0.59],
        Cooktop => [0.62, 0.07, 0.52],
        Range => [0.66, 0.90, 0.66],
        Oven => [0.70, 1.95, 0.66],
        Microwave => [0.53, 0.33, 0.40],
        Dishwasher => [0.62, 0.90, 0.64],
        Refrigerator => [0.84, 1.85, 0.68],
        DryingRack => [0.42, 0.22, 0.32],
        SoapDispenser => [0.075, 0.20, 0.095],
        Sponge => [0.11, 0.024, 0.075],
        KnifeBlock => [0.15, 0.34, 0.18],
        UtensilHolder => [0.15, 0.33, 0.15],
        PaperTowel => [0.29, 0.32, 0.16],
        KitchenTowel => [0.24, 0.022, 0.31],
        Dishware => [0.25, 0.15, 0.25],
        CuttingBoard => [0.37, 0.025, 0.27],
        FruitBowl => [0.31, 0.19, 0.29],
        Bread => [0.30, 0.11, 0.17],
        _ => unreachable!("non-kitchen dimensions"),
    })
}

/// Expand the placement envelope before admission, never after mesh generation.
pub fn dimensions(kind: ObjectKind, seed: u64) -> Vec3 {
    let mut o = IndoorObject {
        kind,
        seed,
        size: fixture_size(kind),
        id: 0,
        position: Vec3::ZERO,
        yaw: 0.,
        variant: 0,
        finish_seed: None,
        solid: true,
        support: None,
        neighbor: false,
        interaction_target: None,
        cable_path: None,
        cable_target: None,
    };
    let mut r = stream(seed, 8011);
    o.size *= Vec3::new(
        r.random_range(0.88..1.14),
        r.random_range(0.96..1.04),
        r.random_range(0.94..1.06),
    );
    match kind {
        ObjectKind::Cooktop => {
            o.size = Vec3::new(
                r.random_range(0.32..0.95),
                r.random_range(0.045..0.085),
                r.random_range(0.39..0.60),
            )
        }
        ObjectKind::Microwave => {
            o.size = Vec3::new(
                r.random_range(0.40..0.67),
                r.random_range(0.27..0.43),
                r.random_range(0.33..0.49),
            )
        }
        ObjectKind::Refrigerator => {
            o.size = Vec3::new(
                r.random_range(0.55..1.12),
                r.random_range(1.35..2.10),
                r.random_range(0.54..0.82),
            )
        }
        ObjectKind::Range => {
            o.size.x = r.random_range(0.53..1.02);
            o.size.z = r.random_range(0.57..0.74);
        }
        ObjectKind::Dishwasher => {
            o.size.x = r.random_range(0.44..0.79);
            o.size.z = r.random_range(0.54..0.70);
        }
        _ => {}
    }
    if kind == ObjectKind::Dishware {
        o.size = super::super::items::sizing::dish(props::program(&o).style, &mut r);
    } else if let Some(size) = super::super::items::sizing::loose(kind, &mut r) {
        o.size = size;
    }
    if kind == ObjectKind::Oven && r.random_bool(0.38) {
        o.size.y = 0.90;
    }
    let p = parameters(&o);
    o.size.z += extension(&o, &p);
    o.size
}
fn extension(o: &IndoorObject, p: &KitchenProgram) -> f32 {
    if is_island(o) {
        return 0.;
    }
    if matches!(
        o.kind,
        ObjectKind::KitchenCupboard
            | ObjectKind::Pantry
            | ObjectKind::KitchenCounter
            | ObjectKind::KitchenSink
    ) {
        if p.door_style == 2 || (o.kind == ObjectKind::KitchenCounter && !p.counter_doors) {
            return 0.;
        }
        if p.cabinet_motion == 1 {
            return 0.04;
        }
        let side = o.size.x * if p.paired { 0.5 } else { 1. };
        let travel = if p.cabinet_motion == 2 && o.kind == ObjectKind::KitchenCupboard {
            side.max(o.size.y * (1. - p.upper_bottom_fraction))
        } else {
            side
        };
        return travel * p.opening_radians.sin() + 0.035;
    }
    if p.opening_radians == 0. {
        return 0.;
    }
    let leaf = match o.kind {
        ObjectKind::KitchenCounter | ObjectKind::KitchenSink | ObjectKind::Cooktop => return 0.,
        ObjectKind::Dishwasher => appliances::program::dishwasher_door_height(o),
        ObjectKind::Range | ObjectKind::Oven => appliances::program::oven_fronts(o, p)
            .iter()
            .map(|front| front.height_m)
            .fold(0., f32::max),
        ObjectKind::KitchenCupboard | ObjectKind::Pantry => {
            o.size.x * if p.paired { 0.46 } else { 0.92 }
        }
        ObjectKind::Refrigerator => appliances::program::refrigerator_leaf_width(o, p),
        ObjectKind::Microwave => appliances::program::microwave_door_width(o, p),
        _ => return 0.,
    };
    leaf * p.opening_radians.sin() + 0.035
}
fn body(o: &IndoorObject, p: &KitchenProgram) -> (f32, f32) {
    let d = (o.size.z - extension(o, p) - 0.08).max(0.12);
    (d, -o.size.z * 0.5 + 0.012 + d * 0.5)
}

pub fn metadata(o: &IndoorObject) -> serde_json::Value {
    let p = parameters(o);
    use ObjectKind::*;
    let mut value = match o.kind {
        KitchenSink => {
            serde_json::json!({"door_style":p.door_style,"opening_radians":p.opening_radians,"cabinet_motion":p.cabinet_motion,"front_profile":p.front_profile,"handles":p.handles,"basins":p.basins,"faucet":p.faucet,"faucet_handles":p.faucet_handles,"hand_soap":p.hand_soap,"dish_soap":p.dish_soap,"sponge":p.sponge,"dishes":p.sink_dishes,"draining_board":p.draining_board,"top":p.top})
        }
        KitchenCounter => {
            let island = island_parameters(o);
            let (drawers, doors) = island.as_ref().map_or(
                if p.counter_doors {
                    (
                        0,
                        if p.door_style == 2 {
                            0
                        } else if p.paired {
                            2
                        } else {
                            1
                        },
                    )
                } else {
                    (p.shelves, 0)
                },
                |p| {
                    let banks = (0..p.bays)
                        .filter(|&i| island::is_drawer_bank(o, i))
                        .count() as u32;
                    (banks * 3 + p.bays - banks, p.bays - banks)
                },
            );
            serde_json::json!({"front":p.finish,"top":p.top,"handles":p.handles,"front_profile":p.front_profile,"hardware":p.hardware,"counter_doors":p.counter_doors,"opening_radians":if p.counter_doors {p.opening_radians}else {0.},"drawers":drawers,"doors":doors,
                "installation":if is_island(o) {"island"} else {"fitted"},
                "island":island})
        }
        KitchenCupboard | Pantry => {
            serde_json::json!({"door_style":p.door_style,"opening_radians":p.opening_radians,"upper_bottom_fraction":p.upper_bottom_fraction,"upper_depth_fraction":p.upper_depth_fraction,"upper_width_fraction":p.upper_width_fraction,"upper_offset_fraction":p.upper_offset_fraction,"cabinet_motion":p.cabinet_motion,"front_profile":p.front_profile,"hardware":p.hardware,"paired":p.paired,"handles":p.handles,"shelves":p.shelves,"contents":p.contents,"finish":p.finish})
        }
        Range => {
            serde_json::json!({"gas":p.gas,"controls":p.controls,"viewport":p.viewport,"opening_radians":p.opening_radians})
        }
        Cooktop => serde_json::json!({"gas":p.gas,"controls":p.controls}),
        Oven => {
            serde_json::json!({"installation":if o.size.y>1.3 {"tower"} else {"under_counter"},"stacked":p.stacked_ovens,"viewport":p.viewport,"controls":p.controls,"opening_radians":p.opening_radians})
        }
        Refrigerator => {
            serde_json::json!({"paired":p.paired,"viewport":p.viewport,"opening_radians":p.opening_radians,"door_notes":p.door_notes,"contents":p.contents,"shelves":p.shelves})
        }
        Microwave => {
            serde_json::json!({"viewport":p.viewport,"controls":p.controls,"opening_radians":p.opening_radians,"contents":p.contents})
        }
        Dishwasher => {
            serde_json::json!({"viewport":p.viewport,"opening_radians":p.opening_radians,"contents":p.contents})
        }
        _ => {
            let p = props::program(o);
            match o.kind {
                PaperTowel => serde_json::json!(supplies::paper_towel::parameters(o)),
                SoapDispenser => {
                    serde_json::json!({"pump":p.pump,"construction":supplies::dispenser::parameters(o.seed, p.pump)})
                }
                Bread => serde_json::json!(supplies::bread::parameters(o)),
                KitchenTowel => serde_json::json!(supplies::towel_parameters(o)),
                CuttingBoard => serde_json::json!(supplies::board_parameters(o)),
                Sponge => serde_json::json!(supplies::sponge::parameters(o)),
                Dishware => {
                    serde_json::json!(dishware::parameters(o.size, p.style, p.count, o.seed))
                }
                FruitBowl => serde_json::json!({"basket":p.basket,"fruit_count":p.count}),
                KnifeBlock => serde_json::json!(supplies::knives::parameters(o)),
                UtensilHolder => serde_json::json!(supplies::utensils::parameters(o)),
                DryingRack => {
                    serde_json::json!(supplies::rack::parameters(o.size, p.count, o.seed))
                }
                _ => serde_json::json!({"size_metres":o.size}),
            }
        }
    };
    if o.kind == KitchenSink {
        value["washing_station"] = serde_json::json!(sink::program::parameters(o, &p));
    }
    if matches!(
        o.kind,
        Cooktop | Range | Oven | Microwave | Dishwasher | Refrigerator
    ) {
        value["construction"] = appliances::metadata(o, &p);
        if o.kind == Refrigerator {
            value["shelves"] =
                serde_json::json!(appliances::program::parameters(o).shelf_levels_m.len());
        }
    }
    if matches!(
        o.kind,
        KitchenCounter | KitchenCupboard | Pantry | KitchenSink
    ) {
        value["front_construction"] = serde_json::json!(p.front);
    }
    value
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    match o.kind {
        ObjectKind::KitchenCounter if is_island(o) => island::build(a, o, &p),
        ObjectKind::KitchenCounter | ObjectKind::KitchenCupboard | ObjectKind::Pantry => {
            cabinet::build(a, o, &p)
        }
        ObjectKind::KitchenSink => sink::build(a, o, &p),
        ObjectKind::Cooktop
        | ObjectKind::Range
        | ObjectKind::Oven
        | ObjectKind::Microwave
        | ObjectKind::Dishwasher
        | ObjectKind::Refrigerator => appliances::build(a, o, &p),
        _ => props::build(a, o),
    }
}
fn box_at(a: &mut Assembly, s: Surface, label: &str, center: Vec3, size: Vec3, tf: Transform) {
    a.part(s, label).cuboid(
        size,
        size.min_element().min(0.008) * 0.20,
        Transform::from_translation(tf.transform_point(center)).with_rotation(tf.rotation),
    );
}
fn pipe(a: &mut Assembly, s: Surface, label: &str, points: &[Vec3], radius: f32) {
    // One welded sweep retains the sampled curve and round cross-section,
    // without overlapping spheres and internal caps at every curve sample.
    a.part(s, label).tube(points, radius, 24);
}

/// Adjacent modules share a work plane while retaining independent geometry.
/// Preserve the body depth when a door's reserved travel changes with height.
pub fn align_worktop(o: &mut IndoorObject, height: f32) {
    let old = extension(o, &parameters(o));
    match o.kind {
        ObjectKind::KitchenCounter | ObjectKind::Dishwasher => o.size.y = height,
        ObjectKind::KitchenSink => o.size.y = height / 0.69,
        ObjectKind::Range => o.size.y = height / 0.94,
        ObjectKind::KitchenCupboard => o.size.y = (height - 0.032) / 0.425,
        ObjectKind::Oven if o.size.y < 1.3 => o.size.y = height,
        _ => return,
    }
    o.size.z += extension(o, &parameters(o)) - old;
}

/// Resize a fitted run without consuming its body depth for longer door travel.
pub fn resize_width(o: &mut IndoorObject, width: f32) {
    let old = extension(o, &parameters(o));
    o.size.x = width;
    o.size.z += extension(o, &parameters(o)) - old;
}
