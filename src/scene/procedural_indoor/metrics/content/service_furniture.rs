//! Built service-furniture fields, including conditional banks and front sizes.
use super::*;
use crate::scene::procedural_indoor::objects::services;

pub(super) fn record(
    o: &IndoorObject,
    version: u32,
    categories: &mut Categories,
    numeric: &mut NumericCollector,
) -> io::Result<()> {
    use ObjectKind::*;
    let finishes = match o.kind {
        QueueBarrier
            if version
                >= crate::scene::procedural_indoor::objects::public::barrier::FITTED_VERSION =>
        {
            use crate::scene::procedural_indoor::objects::public::barrier::{self, Infill};
            let p = barrier::Installation::sample(o, &barrier::parameters(o));
            let kind = match p.infill {
                Infill::Webbing { width_m, .. } => {
                    numeric.push("queue_webbing_width_m", width_m.into())?;
                    "retractable_belt"
                }
                Infill::Rope { radius_m, .. } => {
                    numeric.push("queue_rope_diameter_m", (radius_m * 2.).into())?;
                    "socketed_rope"
                }
                Infill::Pickets {
                    pickets, square, ..
                } => {
                    numeric.push("queue_pickets", pickets as f64)?;
                    if square {
                        "square_rails"
                    } else {
                        "round_rails"
                    }
                }
                Infill::Panel {
                    bottom_m,
                    thickness_m,
                    ..
                } => {
                    numeric.push("queue_panel_height_m", (p.connection_y_m - bottom_m).into())?;
                    numeric.push("queue_panel_thickness_m", thickness_m.into())?;
                    "clamped_panel"
                }
            };
            category(categories, "queue_barrier_construction", kind);
            numeric.push("queue_post_spacing_m", (2. * p.post_half_span_m).into())?;
            numeric.push("queue_connection_height_m", p.connection_y_m.into())?;
            numeric.push("queue_barrier_yaw_rad", o.yaw.into())?;
            if let Some(sag) = p.sag_m() {
                numeric.push("queue_span_sag_m", sag.into())?;
            }
            return Ok(());
        }
        Bag => {
            use crate::scene::procedural_indoor::objects::bags;
            let p = bags::parameters(o);
            category(categories, "bag_style", format!("{:?}", p.style));
            let beside = version >= bags::OPEN_TOTE_VERSION && o.variant & bags::BESIDE_SEAT != 0;
            category(
                categories,
                "bag_initial_placement",
                if o.support.is_some() {
                    "on_furniture"
                } else if beside {
                    "beside_seat"
                } else {
                    "floor"
                },
            );
            numeric.push("bag_yaw_rad", f64::from(o.yaw))?;
            if p.style == bags::BagStyle::Tote {
                let cut = (version >= bags::OPEN_TOTE_VERSION)
                    .then(|| bags::ToteCut::sample(o, &p))
                    .flatten();
                category(
                    categories,
                    "tote_mouth",
                    if cut.is_some() { "open" } else { "closed" },
                );
                if let Some(c) = cut {
                    numeric.push(
                        "tote_clear_mouth_depth_m",
                        f64::from(c.clear_mouth_depth_m()),
                    )?;
                    category(categories, "tote_slip_pocket", c.slip_pocket);
                    numeric.push(
                        "tote_draped_handles",
                        c.draped_handles.into_iter().filter(|&v| v).count() as f64,
                    )?;
                    category(categories, "tote_lining", format!("{:?}", c.lining));
                }
            }
            return Ok(());
        }
        CoatRack => {
            use crate::scene::procedural_indoor::objects::utilities::racks;
            let rack = racks::parameters(o);
            category(
                categories,
                "coat_rack_support",
                if rack.rail { "rail" } else { "pegs" },
            );
            if version >= racks::dressing::INTRODUCED {
                let items = racks::dressing::garments(o, &rack);
                category(
                    categories,
                    "coat_rack_placement",
                    if o.variant == racks::ENTRY_RACK {
                        "entry"
                    } else {
                        "room"
                    },
                );
                numeric.push("coat_rack_garments", items.len() as f64)?;
                for item in items {
                    category(categories, "rack_garment_kind", format!("{:?}", item.kind));
                    numeric.push("rack_garment_length_m", f64::from(item.length_m))?;
                }
            }
            return Ok(());
        }
        Locker => {
            use crate::scene::procedural_indoor::objects::public::education;
            let p = education::parameters_at(o, version);
            category(
                categories,
                "locker_front",
                ["vented_metal", "flush", "glazed", "open"][p.front as usize],
            );
            numeric.push("locker_bank_yaw_rad", f64::from(o.yaw))?;
            if let Some(f) = &p.fittings {
                if let Some(install) = education::installation(o) {
                    category(categories, "locker_installation", install);
                }
                if let Some(v) = f.vents {
                    category(
                        categories,
                        "locker_vent",
                        if v.hooded {
                            "pressed_louver"
                        } else {
                            "punched_slot"
                        },
                    );
                    numeric.push("locker_vent_pitch_m", f64::from(v.pitch_m))?;
                }
                if p.front != 3 {
                    for (i, b) in p.bays.iter().enumerate() {
                        for (row, span) in b.levels.windows(2).enumerate() {
                            let lo = span[0] + if row > 0 { p.panel_m * 0.5 } else { 0. };
                            let hi = span[1]
                                - if row + 1 < b.angles.len() {
                                    p.panel_m * 0.5
                                } else {
                                    0.
                                };
                            category(
                                categories,
                                "locker_hinge_side",
                                if f.handing.latch_side(i) > 0. {
                                    "left"
                                } else {
                                    "right"
                                },
                            );
                            numeric.push(
                                "locker_handle_above_bank_base_m",
                                f64::from(f.handle_y(lo, hi)),
                            )?;
                        }
                    }
                }
            }
            return Ok(());
        }
        Printer
            if version
                >= crate::scene::procedural_indoor::objects::utilities::printer::OUTPUT_VERSION =>
        {
            use crate::scene::procedural_indoor::{
                materials::printer_lcd, objects::utilities::printer,
            };
            let p = printer::parameters(o);
            let output = printer::OutputProgram::sample(o, &p);
            category(
                categories,
                "printer_output_stock",
                output.media.unwrap_or("empty"),
            );
            if p.screen {
                category(categories, "printer_panel_state", printer_lcd::status(o));
            }
            numeric.push("printer_delivered_sheets", output.count as f64)?;
            if output.count > 0 {
                category(
                    categories,
                    "printer_print_side",
                    if output.face_up { "up" } else { "down" },
                );
                numeric.push("printer_sheet_width_m", f64::from(output.sheet_m.x))?;
                numeric.push("printer_sheet_length_m", f64::from(output.sheet_m.y))?;
            }
            [p.finish, p.panel]
        }
        BaseboardHeater => {
            use services::heater::{self, Emitter};
            if version < heater::FITTED_HEATING_VERSION {
                let hydronic = heater::parameters(o).hydronic;
                category(
                    categories,
                    "heater_construction",
                    Emitter::Convector { hydronic }.name(),
                );
                return Ok(());
            }
            let stock = heater::Installation::for_object(o);
            category(categories, "heater_construction", stock.emitter.name());
            category(categories, "heater_finish_lot", stock.finish_lot);
            let body = match stock.emitter {
                Emitter::Pressed(p) => {
                    numeric.push("heater_channel_pitch_m", p.pitch_m.into())?;
                    numeric.push("heater_press_depth_m", p.relief_m.into())?;
                    if let Some(spacing) = p.bank_spacing_m {
                        numeric.push("heater_bank_spacing_m", spacing.into())?;
                    }
                    p.body
                }
                Emitter::Tubular(t) => {
                    numeric.push("heater_tube_pitch_m", t.pitch_m.into())?;
                    numeric.push(
                        "heater_tube_open_gap_m",
                        (t.pitch_m - 2. * t.tube_radii_m.x).into(),
                    )?;
                    t.body
                }
                Emitter::Convector { .. } => return Ok(()),
            };
            numeric.push("radiator_floor_clearance_m", body.bottom_m.into())?;
            return Ok(());
        }
        DrawerUnit => {
            let p = services::drawers::parameters(o);
            if p.base == 0 && version >= services::casters::INTRODUCED {
                caster_usage(
                    services::casters::CasterSet::pedestal(o, &p),
                    categories,
                    numeric,
                )?;
            }
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
            let radius = if version >= services::casters::INTRODUCED {
                let wheels = services::casters::CasterSet::cart(o, &p);
                let radius = wheels.radius_m;
                caster_usage(wheels, categories, numeric)?;
                radius
            } else {
                p.wheel_radius
            };
            numeric.push("cart_wheel_radius_m", f64::from(radius))?;
            use services::cart::stock::{CartJob, ShelfLoad, INTRODUCED};
            if version >= INTRODUCED {
                category(
                    categories,
                    "cart_shelf_use",
                    CartJob::read(o, version).name(),
                );
                for level in 0..p.shelf_tops.len() - 1 {
                    let count = match ShelfLoad::sample(o, &p, level, version) {
                        ShelfLoad::Utility(items) => {
                            for item in &items {
                                category(categories, "cart_shelf_stock", item.kind.class_name());
                            }
                            items.len()
                        }
                        ShelfLoad::Service(items) => {
                            for item in &items {
                                category(
                                    categories,
                                    "cart_shelf_stock",
                                    if item.pantry {
                                        "pantry_package"
                                    } else {
                                        "dishware"
                                    },
                                );
                            }
                            items.len()
                        }
                    };
                    numeric.push("cart_lower_bay_items", count as f64)?;
                }
            }
            [p.frame, p.trays]
        }
        Banquette => {
            use crate::scene::procedural_indoor::objects::public::banquette;
            let p = banquette::parameters(o);
            let cover = (version >= banquette::SEWN_BACK_VERSION)
                .then(|| p.sewn_back(o))
                .flatten();
            category(
                categories,
                "banquette_back_construction",
                cover
                    .as_ref()
                    .map_or_else(|| p.stock_name(), |c| c.construction()),
            );
            if let Some(c) = cover {
                numeric.push("banquette_foam_thickness_m", f64::from(c.foam_m))?;
                numeric.push("banquette_seam_depth_m", f64::from(c.tuck_m))?;
                match c.stitch {
                    banquette::Stitch::Channels { pitch_m, .. } => {
                        numeric.push("banquette_channel_pitch_m", f64::from(pitch_m))?
                    }
                    banquette::Stitch::Diamonds { pitch_m, .. } => {
                        numeric.push("banquette_tuft_pitch_x_m", f64::from(pitch_m.x))?;
                        numeric.push("banquette_tuft_pitch_y_m", f64::from(pitch_m.y))?;
                    }
                }
            }
            [p.frame, p.upholstery]
        }
        Bench => {
            let p = services::bench::parameters(o);
            [p.frame, p.seat]
        }
        RoomDivider if services::divider::desktop::is_desktop(o) => {
            let p = services::divider::desktop::parameters(o);
            category(categories, "divider_location", "desk");
            category(categories, "desk_screen_mount", format!("{:?}", p.mount));
            for (key, value) in [
                ("desk_screen_width_m", o.size.x),
                ("desk_screen_height_m", o.size.y),
                ("desk_screen_thickness_m", p.thickness_m),
                ("desk_screen_cable_gap_m", p.cable_gap_m),
            ] {
                numeric.push(key, f64::from(value))?;
            }
            [p.hardware, p.face]
        }
        RoomDivider => {
            category(categories, "divider_location", "floor");
            let p = services::divider::parameters(o);
            category(
                categories,
                "divider_arrangement",
                format!("{:?}", p.plan.arrangement),
            );
            category(categories, "divider_bay_count", p.weights.len());
            numeric.push(
                "divider_insert_thickness_m",
                f64::from(p.insert_thickness_m),
            )?;
            for joint in 1..p.weights.len() {
                numeric.push(
                    "divider_hinge_turn_deg",
                    f64::from(p.plan.joint_angle(joint).to_degrees()),
                )?;
            }
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

fn caster_usage(
    set: services::casters::CasterSet,
    categories: &mut Categories,
    numeric: &mut NumericCollector,
) -> io::Result<()> {
    use services::casters::Wheel;
    let name = match set.wheel {
        Wheel::Solid => "solid",
        Wheel::Spoked(_) => "spoked",
        Wheel::Twin => "twin",
    };
    category(categories, "service_caster_wheel", name);
    category(
        categories,
        "service_caster_fixed_count",
        set.mounts.iter().filter(|m| m.fixed).count(),
    );
    numeric.push("service_caster_diameter_m", f64::from(2. * set.radius_m))?;
    numeric.push("service_caster_tread_width_m", f64::from(set.tread_width_m))?;
    numeric.push(
        "service_caster_swivel_trail_m",
        f64::from(set.swivel_trail_m),
    )
}

/// Positions of admitted independent dining groups alongside the public aisle.
pub(super) fn record_dining_tables(
    scene: &IndoorManifest,
    program: &crate::scene::procedural_indoor::layout::functional::ProgramAssembly,
    numeric: &mut NumericCollector,
) -> io::Result<()> {
    let Some(aisle) = program.dining_aisle() else {
        return Ok(());
    };
    for &id in &program.members {
        let Some(table) = scene.objects.get(id) else {
            continue;
        };
        if table.kind == ObjectKind::Banquette {
            let bench = table;
            if let Some(top) = bench.interaction_target.and_then(|i| scene.objects.get(i)) {
                use crate::scene::procedural_indoor::{layout::dining_places, objects::tables};
                let places = dining_places::booth_places(top, bench, &tables::outline(top)).count();
                let dishes = scene
                    .objects
                    .iter()
                    .filter(|dish| {
                        dish.kind == ObjectKind::Dishware
                            && dish.interaction_target == Some(bench.id)
                    })
                    .count();
                numeric.push("dining_banquette_places", places as f64)?;
                numeric.push("dining_banquette_served_dishes", dishes as f64)?;
            }
            continue;
        }
        if table.kind != ObjectKind::Table
            || table.variant == crate::scene::procedural_indoor::objects::tables::SHARED_ROW_VARIANT
            || table.position.z < aisle.min.y
            || table.position.z > aisle.max.y
        {
            continue;
        }
        let (lo, hi) = table.bounds();
        let (width, center, gap) = if hi.x <= aisle.min.x {
            (
                aisle.min.x - program.min.x,
                aisle.min.x - table.position.x,
                aisle.min.x - hi.x,
            )
        } else if lo.x >= aisle.max.x {
            (
                program.max.x - aisle.max.x,
                table.position.x - aisle.max.x,
                lo.x - aisle.max.x,
            )
        } else {
            continue;
        };
        numeric.push(
            "dining_table_across_bay_fraction",
            f64::from(center / width),
        )?;
        numeric.push("dining_table_aisle_gap_m", f64::from(gap))?;
    }
    Ok(())
}
