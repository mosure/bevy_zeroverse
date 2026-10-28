//! Built beverage equipment fields; omit controls that a construction does not use.
use super::*;
use crate::scene::procedural_indoor::objects::{hospitality::coffee, public::dispenser};

pub(super) fn record(
    o: &IndoorObject,
    categories: &mut Categories,
    numeric: &mut NumericCollector,
) -> io::Result<()> {
    if o.kind == ObjectKind::CoffeeMaker {
        let p = coffee::parameters(o);
        category(categories, "brewer_construction", format!("{:?}", p.brewer));
        category(categories, "brewer_finish", format!("{:?}", p.finish));
        category(categories, "brewer_trim", format!("{:?}", p.trim));
        numeric.push("brewer_base_height_m", (p.base_fraction * o.size.y) as f64)?;
        if matches!(
            p.brewer,
            coffee::Brewer::Espresso | coffee::Brewer::Automatic
        ) {
            category(categories, "brewer_control_layout", p.control_layout);
            numeric.push("brewer_control_count", p.controls as f64)?;
            numeric.push("brewer_head_height_m", (p.head_height * o.size.y) as f64)?;
            numeric.push("brewer_front_m", (p.front * o.size.z) as f64)?;
        }
        if p.brewer == coffee::Brewer::Filter {
            category(categories, "brewer_thermal_carafe", p.thermal_carafe);
            category(categories, "brewer_handle_side", p.handle_side);
            numeric.push(
                "brewer_carafe_radius_m",
                (o.size.x * p.carafe_radius).min(o.size.z * 0.30) as f64,
            )?;
            numeric.push(
                "brewer_carafe_height_m",
                (o.size.y * p.carafe_height) as f64,
            )?;
        } else {
            category(categories, "brewer_grille_crosswise", p.grille_crosswise);
            numeric.push("brewer_grille_count", p.grille_count as f64)?;
            numeric.push("brewer_tray_width_m", (o.size.x * p.tray_fraction) as f64)?;
            numeric.push("brewer_tray_depth_m", (o.size.z * p.tray_depth) as f64)?;
        }
    } else if o.kind == ObjectKind::DrinkDispenser {
        let p = dispenser::parameters(o);
        category(categories, "dispenser_construction", p.construction);
        category(categories, "dispenser_finish", format!("{:?}", p.finish));
        category(categories, "dispenser_trim", format!("{:?}", p.trim));
        category(categories, "dispenser_grille_crosswise", p.grille_crosswise);
        numeric.push("dispenser_grille_count", p.grille_count as f64)?;
        numeric.push(
            "dispenser_tap_count",
            if p.construction == 3 {
                p.tap_count
            } else {
                p.tanks
            } as f64,
        )?;
        numeric.push("dispenser_tray_width_m", (o.size.x * p.tray_width) as f64)?;
        numeric.push("dispenser_tray_depth_m", (o.size.z * p.tray_depth) as f64)?;
        if p.construction == 3 {
            numeric.push("dispenser_bay_top_m", (o.size.y * p.bay_top) as f64)?;
            numeric.push("dispenser_body_width_m", (o.size.x * p.body_width) as f64)?;
        } else {
            numeric.push("dispenser_tank_count", p.tanks as f64)?;
            if p.construction < 2 {
                numeric.push(
                    "dispenser_fill_fraction",
                    p.fill.min(p.shoulder - 0.03) as f64,
                )?;
            }
        }
    }
    Ok(())
}
