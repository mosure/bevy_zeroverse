//! Realized washing/cooking internals; nested construction data also remains in
//! the object audit. Counts refer to admitted parts rather than requested props.
use super::*;
use crate::scene::procedural_indoor::objects::kitchen;
pub(super) fn record(
    o: &IndoorObject,
    version: u32,
    categories: &mut Categories,
    numeric: &mut NumericCollector,
) -> io::Result<()> {
    match o.kind {
        ObjectKind::Dishware => {
            let p = kitchen::dishware::object_parameters(o);
            if let Some(form) = &p.plate {
                let slot = kitchen::dishware::PlateForm::rim_slot(
                    p.surface,
                    o.finish_seed.unwrap_or(o.seed),
                );
                category(categories, "plate_contrasting_rim", slot.is_some());
                for (name, value) in [
                    ("plate_outline_exponent", form.outline_exponent),
                    ("plate_foot_height_m", form.foot_height_m),
                    ("plate_foot_diameter_m", 2. * form.foot_radius_m),
                    ("plate_stack_count", p.count as f32),
                    ("plate_installed_height_m", p.stack_height()),
                ] {
                    numeric.push(name, value as f64)?;
                }
                if p.count > 1 {
                    numeric.push("plate_stack_pitch_m", p.interior_floor() as f64)?;
                    numeric.push(
                        "plate_stack_max_offset_m",
                        form.stack_offsets_m
                            .iter()
                            .map(|v| v.length())
                            .fold(0., f32::max) as f64,
                    )?;
                }
            }
        }
        ObjectKind::KitchenSink => {
            let p = kitchen::sink::program::parameters_at(o, &kitchen::parameters(o), version);
            if let Some(tap) = &p.tap {
                category(categories, "sink_spout_form", format!("{:?}", tap.spout));
                numeric.push(
                    "sink_outlet_height_above_worktop_m",
                    (tap.outlet(&p).y - p.worktop_m) as f64,
                )?;
            }
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
                if let Some(waste) = &b.waste {
                    category(categories, "sink_strainer", format!("{:?}", waste.strainer));
                    numeric.push("sink_drain_diameter_m", (b.drain_radius * 2.) as f64)?;
                    numeric.push(
                        "sink_drain_recess_m",
                        (b.drain_radius * waste.depth_ratio) as f64,
                    )?;
                }
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

pub(super) fn record_sink_mats(
    scene: &IndoorManifest,
    group: &crate::scene::procedural_indoor::layout::kitchen::KitchenAssembly,
    numeric: &mut NumericCollector,
) -> io::Result<()> {
    let objects = |id: &usize| scene.objects.get(*id);
    if let Some(sink) = group
        .units
        .iter()
        .filter_map(objects)
        .find(|o| o.kind == ObjectKind::KitchenSink)
    {
        for mat in group
            .props
            .iter()
            .filter_map(objects)
            .filter(|o| o.kind == ObjectKind::Rug)
        {
            let local = sink
                .transform()
                .compute_affine()
                .inverse()
                .transform_point3(mat.position);
            let yaw = mat.yaw - sink.yaw;
            let (sin, cos) = yaw.sin_cos();
            let gap =
                local.z - (sink.size.z + mat.size.x * sin.abs() + mat.size.z * cos.abs()) * 0.5;
            for (name, value) in [
                ("sink_mat_width_m", mat.size.x),
                ("sink_mat_thickness_m", mat.size.y),
                ("sink_mat_lateral_offset_m", local.x),
                ("sink_mat_toe_gap_m", gap),
                ("sink_mat_relative_yaw_rad", sin.atan2(cos)),
            ] {
                numeric.push(name, f64::from(value))?;
            }
        }
    }
    Ok(())
}
