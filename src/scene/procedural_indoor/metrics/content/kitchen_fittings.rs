//! Realized washing/cooking internals; nested construction data also remains in
//! the object audit. Counts refer to admitted parts rather than requested props.
use super::*;
use crate::scene::procedural_indoor::objects::kitchen;
pub(super) fn record(
    o: &IndoorObject,
    categories: &mut Categories,
    numeric: &mut NumericCollector,
) -> io::Result<()> {
    match o.kind {
        ObjectKind::KitchenSink => {
            let p = kitchen::sink::program::parameters(o, &kitchen::parameters(o));
            category(
                categories,
                "sink_basin_finish",
                format!("{:?}", p.basin_finish),
            );
            category(
                categories,
                "sink_faucet_finish",
                format!("{:?}", p.faucet_finish),
            );
            numeric.push("sink_accessories", p.accessories.len() as f64)?;
            numeric.push("sink_dish_stacks", p.dishes.len() as f64)?;
            for b in &p.basins {
                for (name, v) in [
                    ("sink_basin_width_m", b.size.x),
                    ("sink_basin_depth_m", b.size.y),
                    ("sink_bowl_depth_m", p.worktop_m - b.floor_m),
                    ("sink_bowl_roundness", b.roundness),
                    ("sink_bowl_taper", b.taper.x),
                ] {
                    numeric.push(name, v as f64)?;
                }
            }
            for a in &p.accessories {
                numeric.push("sink_accessory_yaw", a.yaw as f64)?;
            }
            numeric.push(
                "sink_tap_reach_m",
                p.faucet_points.last().unwrap().xz().distance(p.mount.xz()) as f64,
            )?;
        }
        ObjectKind::Dishwasher => {
            let p = kitchen::appliances::interior::dishwasher_baskets(o, &kitchen::parameters(o));
            numeric.push("dishwasher_baskets", p.len() as f64)?;
            for b in p {
                numeric.push("dishwasher_basket_height_m", b.position.y as f64)?;
                numeric.push("dishwasher_basket_clearance_m", b.size.y as f64)?;
            }
        }
        ObjectKind::Range | ObjectKind::Oven => {
            let p = kitchen::parameters(o);
            for (i, f) in kitchen::appliances::program::oven_fronts(o, &p)
                .iter()
                .enumerate()
            {
                numeric.push("oven_front_height_m", f.height_m as f64)?;
                numeric.push("oven_front_width_m", f.width_m as f64)?;
                let racks = kitchen::appliances::interior::oven_racks(o, &p, f, i);
                numeric.push("oven_racks_per_chamber", racks.len() as f64)?;
                for r in racks {
                    numeric.push("oven_rack_height_m", r.y_m as f64)?;
                    category(categories, "oven_rack_transverse", r.transverse);
                }
            }
        }
        _ => (),
    }
    Ok(())
}
