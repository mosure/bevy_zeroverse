//! Report the formed drinking vessel, visible contents and applied stock.
use super::*;
use crate::scene::procedural_indoor::clutter::beverages::{self, bottle::Closure, cups::CupForm};

pub(super) fn glass(
    o: &IndoorObject,
    categories: &mut Categories,
    numeric: &mut NumericCollector,
) -> io::Result<()> {
    let p = beverages::glassware::Glass::for_object(o);
    category(
        categories,
        "DrinkingGlass_construction",
        format!("{:?}", p.form),
    );
    category(categories, "table_glass_tint", format!("{:?}", p.tint));
    numeric.push("table_glass_wall_m", f64::from(p.wall_m))?;
    numeric.push(
        "table_glass_water_depth_fraction",
        f64::from(p.water_depth_fraction.unwrap_or(0.)),
    )
}

pub(super) fn cup(
    o: &IndoorObject,
    version: u32,
    categories: &mut Categories,
    numeric: &mut NumericCollector,
) -> io::Result<()> {
    let p = beverages::parameters(o);
    let f = CupForm::sample_at(o, &p, version);
    if o.kind == ObjectKind::Mug {
        let profile = if p.belly_fraction > p.taper.max(1.) {
            "barrel"
        } else {
            "tapered"
        };
        let support = if f.saucer.is_some() { "saucer" } else { "bare" };
        category(
            categories,
            "Mug_construction",
            format!("{profile}-{support}"),
        );
    } else {
        let jacket = match &f.sleeve {
            Some(s) if s.printed => "printed",
            Some(s) if s.cut.is_some() => "unprinted",
            Some(_) => "kraft",
            None => "double_wall",
        };
        let opening = if f.lid.is_some() { "lid" } else { "open" };
        category(
            categories,
            "CoffeeCup_construction",
            format!("{jacket}-{opening}"),
        );
        if let Some(cut) = f.sleeve.as_ref().and_then(|s| s.cut.as_ref()) {
            category(
                categories,
                "cup_sleeve_press",
                format!("{:?}", cut.stock.press),
            );
            category(categories, "cup_sleeve_trim", format!("{:?}", cut.trim));
            numeric.push("cup_sleeve_lap_width_m", f64::from(cut.lap_width_m))?;
        }
    }
    if let Some(brew) = f.brew {
        numeric.push("drink_fill_fraction", f64::from(p.fill_fraction))?;
        category(categories, "cup_beverage", format!("{brew:?}"));
    }
    numeric.push("vessel_taper", f64::from(p.taper))?;
    numeric.push("vessel_wall_m", f64::from(f.wall_m))?;
    if let Some(s) = f.saucer {
        numeric.push("saucer_seat_height_m", f64::from(s.seat_y_m))?;
        numeric.push(
            "cup_saucer_offset_m",
            f64::from(f.center_m.distance(s.center_m)),
        )?;
    }
    if let Some(lid) = f.lid {
        numeric.push("cup_lid_rise_m", f64::from(lid.rise_m))?;
        numeric.push(
            "cup_sip_offset_fraction",
            f64::from(lid.port_offset_fraction),
        )?;
    }
    Ok(())
}

pub(super) fn can(
    o: &IndoorObject,
    categories: &mut Categories,
    numeric: &mut NumericCollector,
) -> io::Result<()> {
    let p = beverages::can::parameters(o);
    let state = if p.opened { "open" } else { "closed" };
    let wall = if p.bare { "bare" } else { "printed" };
    category(
        categories,
        "SodaCan_construction",
        format!("{state}-{wall}"),
    );
    category(categories, "can_opened", p.opened);
    category(categories, "can_bare", p.bare);
    if !p.bare {
        numeric.push("can_label_coverage", (p.label_top - p.label_bottom) as f64)?;
    }
    if p.opened {
        numeric.push("can_panel_fold_deg", p.end.fold_radians.to_degrees() as f64)?;
        category(
            categories,
            "can_visible_contents",
            if p.end.fill_fraction.is_some() {
                "drink"
            } else {
                "empty"
            },
        );
        if let Some(fill) = p.end.fill_fraction {
            numeric.push("can_visible_fill_fraction", fill as f64)?;
        }
    }
    let panel_radius = o.size.x.min(o.size.z) * p.radius_fraction * p.neck * p.end.panel_fraction;
    for (key, value) in [
        (
            "can_pour_width_m",
            2. * p.end.port_half_fraction.x * panel_radius,
        ),
        (
            "can_pour_length_m",
            2. * p.end.port_half_fraction.y * panel_radius,
        ),
        (
            "can_score_area_mm2",
            1e6 * std::f32::consts::PI
                * panel_radius.powi(2)
                * p.end.port_half_fraction.x
                * p.end.port_half_fraction.y,
        ),
        ("can_radius_fraction", p.radius_fraction),
        ("can_neck_fraction", p.neck),
        ("can_shoulder_fraction", p.shoulder),
        ("can_lid_yaw_rad", p.lid_yaw),
    ] {
        numeric.push(key, value as f64)?;
    }
    Ok(())
}

pub(super) fn bottle(
    o: &IndoorObject,
    categories: &mut Categories,
    numeric: &mut NumericCollector,
) -> io::Result<()> {
    let p = beverages::parameters(o);
    let cap = Closure::sample(o, &p);
    category(
        categories,
        "WaterBottle_construction",
        [
            "ribbed_pet",
            "long_neck",
            "wide_neck",
            "steel",
            "metal",
            "loop_cap",
        ][p.style as usize],
    );
    numeric.push("bottle_cap_flutes", cap.flutes as f64)?;
    numeric.push(
        "bottle_cap_height_m",
        (o.size.y * cap.cap_height_fraction) as f64,
    )?;
    category(categories, "bottle_tamper_band", cap.tamper_band);
    if let Some(band) = cap.label_band {
        numeric.push("bottle_label_coverage", (band.y - band.x) as f64)?;
    }
    if p.style <= 2 && !p.empty {
        numeric.push("water_fill_fraction", f64::from(p.fill_fraction))?;
        numeric.push(
            "bottle_visible_fill_height_m",
            (o.size.y * p.fill_fraction) as f64,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn beverage_metrics_follow_visible_water_and_actual_can_openings() {
        use crate::scene::procedural_indoor::{layout::IndoorLayout, objects::audit};
        let directory = tempfile::tempdir().unwrap();
        let scene =
            IndoorManifest::generate_with_humans(13, IndoorLayout::Cafe, 0.65, 0, 0.).unwrap();
        let mut numeric = NumericCollector::new(directory.path()).unwrap();
        let mut categories = Categories::new();
        let mut expected_cans = BTreeMap::new();
        let mut water = Vec::new();
        for seed in 0..256 {
            let can = audit::fixture(ObjectKind::SodaCan, seed);
            let p = beverages::can::parameters(&can);
            let state = if p.opened { "open" } else { "closed" };
            let wall = if p.bare { "bare" } else { "printed" };
            *expected_cans.entry(format!("{state}-{wall}")).or_insert(0) += 1;
            record_object(&scene, &can, &mut categories, &mut numeric).unwrap();
            let bottle = audit::fixture(ObjectKind::WaterBottle, seed);
            let p = beverages::parameters(&bottle);
            if p.style <= 2 && !p.empty {
                water.push(f64::from(p.fill_fraction));
            }
            record_object(&scene, &bottle, &mut categories, &mut numeric).unwrap();
        }
        let values = numeric.finish().unwrap();
        assert_eq!(expected_cans.len(), 4);
        assert_eq!(categories["SodaCan_construction"], expected_cans);
        let dist = &values["water_fill_fraction"];
        assert_eq!(dist.count, water.len());
        assert_eq!(dist.max, water.iter().copied().reduce(f64::max).unwrap());
        assert!(
            dist.max > 0.7
                && (dist.mean - water.iter().sum::<f64>() / water.len() as f64).abs() < 1e-9
        );
    }
}
