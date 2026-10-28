//! Built service-furniture fields, including conditional banks and front sizes.
use super::*;
use crate::scene::procedural_indoor::objects::services;

pub(super) fn record(
    o: &IndoorObject,
    categories: &mut Categories,
    numeric: &mut NumericCollector,
) -> io::Result<()> {
    use ObjectKind::*;
    let finishes = match o.kind {
        DrawerUnit => {
            let p = services::drawers::parameters(o);
            category(categories, "drawer_banks", p.banks.len());
            category(categories, "drawer_base", p.base);
            category(categories, "drawer_hardware", p.hardware);
            numeric.push("drawer_count", p.weights.len() as f64)?;
            for front in p.fronts(o.size.y) {
                numeric.push("drawer_front_width_m", (front.right - front.left) as f64)?;
                numeric.push("drawer_front_height_m", (front.top - front.bottom) as f64)?;
            }
            [p.frame, p.front]
        }
        MobileCart => {
            let p = services::cart::parameters(o);
            [p.frame, p.trays]
        }
        Bench => {
            let p = services::bench::parameters(o);
            [p.frame, p.seat]
        }
        RoomDivider => {
            let p = services::divider::parameters(o);
            [p.frame, p.panel]
        }
        _ => return Ok(()),
    };
    for (role, finish) in ["frame", "surface"].into_iter().zip(finishes) {
        category(
            categories,
            &format!("service_{:?}_{role}_material", o.kind),
            format!("{finish:?}"),
        );
    }
    Ok(())
}
