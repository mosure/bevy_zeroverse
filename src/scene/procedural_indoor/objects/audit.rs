//! Opt-in geometry diagnostics. Never called by scene generation or rendering.
//! Fixed placement envelopes separate construction diversity from size/pose/color.
use super::*;
use serde::Serialize;

pub const RESOLUTION: usize = 24;
pub const DISTANCE_THRESHOLDS: [f32; 3] = [0.005, 0.015, 0.03];

/// Metric, category-appropriate review fixtures; these are not placement priors.
pub fn fixture(kind: ObjectKind, seed: u64) -> IndoorObject {
    use ObjectKind::*;
    let size = match kind {
        Elevator | ServiceCounter | DisplayCase | TrayReturn | DrinkDispenser | QueueBarrier
        | Lectern | Locker | TeachingBoard | ExhibitPlinth | Sculpture | InformationStand
        | ExhibitionPanel | ExtractionHood | Banquette => public::dimensions(kind, seed).to_array(),
        KitchenCounter | KitchenSink | KitchenCupboard | Pantry | Cooktop | Range | Oven
        | Microwave | Dishwasher | Refrigerator | DryingRack | SoapDispenser | Sponge
        | KnifeBlock | UtensilHolder | PaperTowel | KitchenTowel | Dishware | CuttingBoard
        | FruitBowl | Bread => kitchen::fixture_size(kind).to_array(),

        Table => [1.6, 0.75, 1.2],
        Desk => [1.3, 0.75, 0.75],
        Chair => [0.68, 1.30, 0.68],
        Sofa => [2.1, 0.92, 0.95],
        CoffeeTable => [1.1, 0.43, 0.65],
        SideTable => [0.52, 0.55, 0.48],
        Cabinet => [0.9, 0.85, 0.45],
        Bookcase => [0.95, 1.8, 0.38],
        Plant => [0.8, 1.2, 0.8],
        TrashCan => [0.34, 0.48, 0.34],
        Whiteboard => [1.3, 0.85, 0.09],
        WallArt | NoticeBoard => [0.8, 0.6, 0.045],
        Display => [1.3, 0.8, 0.08],
        Monitor => [0.6, 0.48, 0.24],
        Laptop => [0.37, 0.30, 0.34],
        Notebook => [0.20, 0.024, 0.26],
        Books => [0.24, 0.10, 0.29],
        Mug => [0.13, 0.11, 0.10],
        Rug => [1.8, 0.009, 1.3],
        Clock => [0.35, 0.35, 0.06],
        FloorLamp => standing_lamp::dimensions(seed).to_array(),
        Keyboard => [0.43, 0.03, 0.16],
        Mouse => [0.065, 0.038, 0.11],
        WaterBottle => [0.08, 0.25, 0.08],
        PenHolder => [0.085, 0.16, 0.085],
        Printer => [0.43, 0.30, 0.35],
        StorageBox => [0.38, 0.27, 0.32],
        CoatRack => [0.6, 1.7, 0.6],
        Bag => [0.34, 0.34, 0.20],
        CoffeeCup => [0.095, 0.13, 0.095],
        SodaCan => [0.065, 0.12, 0.065],
        Notepad => [0.16, 0.012, 0.22],
        Pencil => [0.008, 0.008, 0.18],
        Microphone => [0.10, 0.28, 0.12],
        Phone => [0.075, 0.012, 0.15],
        WallOutlet | LightSwitch => [0.15, 0.115, 0.018],
        MobileCart => [0.58, 0.76, 0.43],
        DrawerUnit => [0.42, 0.60, 0.45],
        Bench => [1.4, 0.84, 0.50],
        Ottoman => [0.62, 0.44, 0.58],
        RoomDivider => [1.2, 1.65, 0.4],
        DeskLamp => [0.26, 0.46, 0.26],
        CoffeeMaker => [0.31, 0.37, 0.36],
        Kettle => [0.19, 0.24, 0.27],
        Speaker => [0.14, 0.23, 0.16],
        Headphones => [0.20, 0.21, 0.12],
        Stapler => [0.045, 0.05, 0.14],
        TapeDispenser => [0.06, 0.08, 0.14],
        DocumentTray => [0.26, 0.13, 0.34],
        DeskOrganizer => [0.20, 0.16, 0.13],
        TissueBox => [0.21, 0.12, 0.14],
        Vase => [0.15, 0.24, 0.15],
        Bowl => [0.21, 0.075, 0.21],
        DeskFan => [0.23, 0.35, 0.19],
        Projector => [0.31, 0.12, 0.25],
        ProjectionScreen => [1.4, 1.1, 0.09],
        PowerStrip => [0.29, 0.04, 0.075],
        Cable => [0.3, 0.03, 0.2],
        WallShelf => [0.85, 0.23, 0.27],
        WallSconce => [0.20, 0.30, 0.15],
        CeilingPendant => [0.40, 0.70, 0.40],
        CeilingFan => [1.4, 0.55, 1.4],
        ExitSign => [0.36, 0.20, 0.045],
        FireAlarm => [0.15, 0.20, 0.06],
        FireDetector => [0.14, 0.055, 0.14],
        FirePullStation => [0.11, 0.15, 0.055],
        StringLights => {
            if seed % 2 == 1 {
                [3.6, 0.70, 0.70]
            } else {
                [2.4, 0.38, 0.12]
            }
        }
        FloorOutlet => [0.18, 0.13, 0.16],
    };
    let mut o = IndoorObject {
        id: 0,
        kind,
        size: Vec3::from_array(size),
        position: Vec3::ZERO,
        yaw: 0.,
        variant: (seed
            % match kind {
                Chair => chairs::FAMILIES as u64,
                Plant => super::super::plants::SPECIES as u64,
                _ => 4,
            }) as u32,
        seed,
        finish_seed: Some(0),
        solid: true,
        support: None,
        neighbor: false,
        interaction_target: None,
        cable_path: None,
        cable_target: None,
    };
    if kind.is_kitchen() {
        o.size = kitchen::dimensions(kind, seed);
    }
    if kind == Cable {
        let mut r = stream(seed, 7480);
        let amplitude = r.random_range(0.015..0.08);
        o.cable_path = Some(
            (0..=20)
                .map(|i| {
                    let t = i as f32 / 20.;
                    Vec3::new(-0.14 + 0.28 * t, 0.012, amplitude * (PI * t).sin())
                })
                .collect(),
        );
        o.solid = false;
    }
    o
}

/// Recorded inputs, not a product-of-ranges diversity estimate. Several builders
/// have conditional fields; the depth-map measurements qualify realized geometry.
pub fn program(o: &IndoorObject) -> serde_json::Value {
    use ObjectKind::*;
    match o.kind {
        Microphone => return serde_json::json!(super::super::clutter::microphone::parameters(o)),
        Speaker => return serde_json::json!(accessories::speaker::parameters(o)),
        Headphones => return serde_json::json!(accessories::headphones::parameters(o)),
        Stapler => return serde_json::json!(accessories::stationery::stapler_parameters(o)),
        TapeDispenser => return serde_json::json!(accessories::stationery::tape_parameters(o)),
        Ottoman => return serde_json::json!(services::ottoman::parameters(o)),
        Bench => return serde_json::json!(services::bench::parameters(o)),
        DrawerUnit => return serde_json::json!(services::drawers::parameters(o)),
        MobileCart => return serde_json::json!(services::cart::parameters(o)),
        RoomDivider => return serde_json::json!(services::divider::parameters(o)),
        Kettle => return serde_json::json!(hospitality::kettle::parameters(o)),
        DeskFan => return serde_json::json!(accessories::fan::parameters(o)),
        Printer => return serde_json::json!(utilities::printer::parameters(o)),
        StorageBox => return serde_json::json!(utilities::boxes::parameters(o)),
        CoatRack => return serde_json::json!(utilities::racks::parameters(o)),
        TrashCan => return serde_json::json!(containers::bin_parameters(o)),
        Projector => return serde_json::json!(equipment::projector::parameters(o)),
        Locker => return serde_json::json!(public::education::parameters(o)),
        Banquette => return serde_json::json!(public::banquette::parameters(o)),
        Bookcase => return serde_json::json!(storage::bookcase::parameters(o)),
        WallShelf => return serde_json::json!(equipment::wall_shelf::parameters(o)),
        ProjectionScreen => return serde_json::json!(presentation::projection::parameters(o)),
        ServiceCounter => return serde_json::json!(public::service::counter_parameters(o)),
        ExtractionHood => return serde_json::json!(public::service::hood_parameters(o)),
        Lectern | TeachingBoard => return serde_json::json!(public::furnishing::parameters(o)),
        DisplayCase | TrayReturn | ExhibitPlinth => {
            return serde_json::json!(public::casework::parameters(o))
        }
        DrinkDispenser => return serde_json::json!(public::dispenser::parameters(o)),
        CoffeeMaker => return serde_json::json!(hospitality::coffee::parameters(o)),
        QueueBarrier => return serde_json::json!(public::barrier::parameters(o)),
        Vase | Bowl => return serde_json::json!(vessels::parameters(o)),
        Mug | CoffeeCup | WaterBottle | SodaCan => {
            return serde_json::json!(super::super::clutter::beverages::parameters(o))
        }
        _ => {}
    }
    if o.kind == Elevator {
        return serde_json::json!(public::elevator::parameters(o));
    }
    if o.kind == Sculpture {
        return serde_json::json!(public::sculpture::parameters(o));
    }
    if o.kind == NoticeBoard {
        let p = equipment::parameters(o);
        return serde_json::json!({"frame_fraction":p.board_frame_fraction, "divider_fraction":p.board_divider_fraction, "attachments":equipment::pinups::placements(o)});
    }
    if matches!(o.kind, InformationStand | ExhibitionPanel) {
        return serde_json::json!(public::signage::parameters(o));
    }
    let all = match o.kind {
        Elevator | ServiceCounter | DisplayCase | TrayReturn | DrinkDispenser | QueueBarrier
        | Lectern | Locker | TeachingBoard | ExhibitPlinth | Sculpture | InformationStand
        | ExhibitionPanel | ExtractionHood | Banquette => serde_json::json!(public::parameters(o)),
        KitchenCounter | KitchenSink | KitchenCupboard | Pantry | Cooktop | Range | Oven
        | Microwave | Dishwasher | Refrigerator | DryingRack | SoapDispenser | Sponge
        | KnifeBlock | UtensilHolder | PaperTowel | KitchenTowel | Dishware | CuttingBoard
        | FruitBowl | Bread => kitchen::metadata(o),

        Table | Desk | CoffeeTable | SideTable => serde_json::json!(tables::parameters(o)),
        Chair if o.variant >= 10 => {
            serde_json::json!({"family":if o.variant == 11 {"auditorium"} else {"tablet_arm"}, "folded":chairs::is_folded(o), "seat_height_metres":0.46, "width_metres":o.size.x,"height_metres":o.size.y,"depth_metres":o.size.z})
        }
        Chair if matches!(o.variant, 6 | 7) => serde_json::json!(chairs::stool::parameters(o)),
        Chair if o.variant == 8 => serde_json::json!(seating::parameters(o)),
        Chair => serde_json::json!(chairs::parameters(o)),
        Sofa => serde_json::json!(seating::parameters(o)),
        Cabinet | Bookcase => serde_json::json!(storage::parameters(o)),
        Plant => serde_json::json!({"growth":super::super::plants::Growth::sample(o.seed),
            "pot":super::super::plants::PotProfile::sample(o.seed),
            "pot_radius_height_m":super::super::plants::pot_dimensions(o, &super::super::plants::Growth::sample(o.seed)),
            "species":super::super::plants::species_name(o.variant)}),
        TrashCan => serde_json::json!(containers::bin_parameters(o)),
        TissueBox => serde_json::json!(containers::tissue_parameters(o)),
        Monitor | Display => serde_json::json!(computers::display_parameters(o)),
        Laptop => serde_json::json!(computers::parameters(o)),
        Clock => serde_json::json!(clocks::parameters(o)),
        WallOutlet | LightSwitch => serde_json::json!(wall::parameters(o)),
        Keyboard | Mouse => serde_json::json!(desktop::parameters(o)),
        Books | Notebook => serde_json::json!(literature::parameters(o)),
        FloorLamp => serde_json::json!(standing_lamp::parameters(o)),
        Printer | StorageBox | CoatRack => serde_json::json!(utilities::parameters(o)),
        Bag => serde_json::json!(bags::parameters(o)),
        Whiteboard => serde_json::json!(presentation::parameters(o)),
        CoffeeMaker | Kettle | Vase | Bowl => serde_json::json!(hospitality::parameters(o)),
        Mug | CoffeeCup | WaterBottle | SodaCan => {
            serde_json::json!(super::super::clutter::beverages::parameters(o))
        }
        Phone | Microphone | Pencil | Notepad => {
            serde_json::json!(super::super::clutter::tabletop::parameters(o))
        }
        MobileCart | DrawerUnit | Bench | Ottoman | RoomDivider => {
            serde_json::json!(services::parameters(o))
        }
        CeilingFan => return serde_json::json!(fans::parameters(o)),
        StringLights => return serde_json::json!(equipment::signals::string_program(o)),
        FireAlarm | FireDetector | FirePullStation => {
            return serde_json::json!(equipment::signals::fire::program(o))
        }
        ExitSign => {
            serde_json::json!({"variant":o.variant,"dimensions_m":o.size,"emission":super::super::photometry::decorative_emission(o)})
        }
        DeskLamp => return serde_json::json!(accessories::lamp_parameters(o)),
        DocumentTray | DeskOrganizer => {
            return serde_json::json!(accessories::organizers::parameters(o))
        }
        PowerStrip => return serde_json::json!(equipment::power_strip::parameters(o)),
        Speaker | Headphones | Stapler | TapeDispenser | DeskFan => {
            serde_json::json!(accessories::parameters(o))
        }
        Projector | ProjectionScreen | WallShelf | NoticeBoard | WallSconce | CeilingPendant => {
            serde_json::json!(equipment::parameters(o))
        }
        WallArt => serde_json::json!(super::super::materials::art::frame_parameters(o)),
        Rug | FloorOutlet => serde_json::json!(floor::parameters(o)),
        PenHolder => return serde_json::json!(super::super::clutter::pen_holder::parameters(o)),
        Cable => serde_json::json!({"variant":o.variant}),
    };
    // Exclude known dormant fields of shared equipment/item programs.
    let keys: &[&str] = match o.kind {
        Elevator => &["opening", "bays", "finish"],
        QueueBarrier | DrinkDispenser => &["opening"],
        ServiceCounter => &["bays", "finish"],
        Banquette => &["bays", "curvature", "taper", "finish"],
        DisplayCase => &["shelves", "bays", "contents", "finish"],
        TrayReturn => &["shelves", "contents", "finish"],
        Locker => &["bays", "opening", "contents", "finish"],
        Lectern | ExhibitPlinth => &["finish"],
        Sculpture => &["bays", "curvature", "glazed"],
        ExhibitionPanel => &["taper"],
        Notebook if all["opened"] == true => &[
            "opened",
            "cover_fraction",
            "page_balance",
            "page_crown",
            "envelope_fraction",
            "cover",
        ],
        Notebook => &[
            "opened",
            "cover_fraction",
            "binding",
            "strap",
            "envelope_fraction",
            "cover",
            "corner_fraction",
            "strap_width",
            "strap_position",
            "strap_thickness",
            "ring_spacing",
            "size_variation",
        ],
        Books => &[
            "volumes",
            "cover_fraction",
            "stack_yaw",
            "envelope_fraction",
            "cover",
            "corner_fraction",
            "strap_width",
            "strap_position",
            "strap_thickness",
            "ring_spacing",
            "size_variation",
            "binding",
            "strap",
        ],
        Keyboard => &[
            "keyboard_split",
            "keyboard_rows",
            "keyboard_columns",
            "deck_fraction",
            "key_area_fraction",
            "row_stagger",
            "wrist_rest",
        ],
        Mouse => &[
            "mouse_roundness",
            "mouse_crown",
            "mouse_waist",
            "button_fraction",
            "trackball",
        ],
        Phone => &[
            "case_fraction",
            "bezel_fraction",
            "lens_count",
            "camera_island",
            "phone_roundness",
            "phone_top_border",
            "phone_bottom_border",
            "phone_controls",
            "phone_surface",
            "phone_face_up",
        ],
        Pencil => &["tip_fraction", "barrel_sides", "binding"],
        Notepad => &["binding", "leaf_fraction"],
        Microphone => &["microphone_bend", "microphone_head", "microphone_lateral"],
        Projector => &["corner_m", "lens_fraction", "lens_offset", "vent_count"],
        PowerStrip => &["corner_m", "socket_count"],
        ProjectionScreen => &["screen_drop", "screen_frame"],
        WallShelf => &["shelf_brackets", "shelf_span", "bracket_style"],
        NoticeBoard => &[
            "note_count",
            "board_frame_fraction",
            "board_divider_fraction",
        ],
        WallSconce => &[
            "diffuser_fraction",
            "sconce_width",
            "sconce_height",
            "sconce_taper",
        ],
        CeilingPendant => &["shade_fraction", "diffuser_fraction"],
        Speaker => &[
            "style",
            "body_taper",
            "control_fraction",
            "support_fraction",
            "profile_roundness",
        ],
        Headphones => &["control_fraction", "support_fraction", "profile_roundness"],
        Stapler | TapeDispenser => &["body_taper", "control_fraction", "support_fraction"],
        DocumentTray | DeskOrganizer => &["sections", "support_fraction", "body_taper"],
        DeskFan => &["style", "support_fraction"],
        DeskLamp => &["style", "tilt", "reach"],
        Rug => &["rug_exponent", "rug_taper", "clipped_corners"],
        FloorOutlet => &["lid_width_fraction", "lid_height_fraction"],
        Mug => &["style", "taper", "fill_fraction", "wall_m"],
        CoffeeCup => &["style", "taper", "fill_fraction", "wall_m", "lid"],
        WaterBottle => &[
            "style",
            "fill_fraction",
            "shoulder_fraction",
            "neck_fraction",
        ],
        SodaCan => &["style", "lid", "heel_fraction", "neck_fraction"],
        MobileCart => &["levels", "feet_height"],
        DrawerUnit => &["drawers", "feet_height", "drawer_bias", "handle_fraction"],
        Bench => &["slats", "backed", "roundness"],
        Ottoman => &[
            "base",
            "base_height",
            "roundness",
            "crown",
            "cap_fraction",
            "body_inset",
        ],
        RoomDivider => &["slats", "panel_fraction"],
        StorageBox => &[
            "construction",
            "taper",
            "roundness",
            "lid_fraction",
            "handle_span",
        ],
        Printer => &["construction", "roundness", "split", "lid_fraction"],
        Bag => &[
            "style",
            "body_height",
            "taper",
            "roundness",
            "fullness",
            "handle_span",
            "handle_drop",
            "strap_width",
            "pocket_width",
            "pocket_height",
            "pocket_depth",
            "flap",
            "seam",
            "finish",
        ],
        CoatRack => &["construction", "arms", "branch_height", "handle_span"],
        CoffeeMaker => &["style", "belly", "neck", "taper"],
        Kettle => &["style", "belly", "taper"],
        Bowl => &["style", "taper", "belly", "flutes"],
        Vase => &["style", "taper", "belly", "neck", "flutes"],
        _ => &[],
    };
    if keys.is_empty() {
        all
    } else {
        serde_json::Value::Object(
            keys.iter()
                .filter_map(|k| all.get(*k).map(|v| ((*k).into(), v.clone())))
                .collect(),
        )
    }
}

#[derive(Debug, Serialize)]
pub struct MeshCheck {
    pub triangles: usize,
    pub batches: usize,
    pub max_envelope_overrun_m: f32,
    pub min_y_m: f32,
}

/// Check the actual triangles without constructing a synthetic room around them.
pub fn check(o: &IndoorObject, a: &Assembly) -> Result<MeshCheck, String> {
    let (lo, hi) = a.bounds();
    let excess = (Vec3::new(-o.size.x * 0.5, 0., -o.size.z * 0.5) - lo)
        .max(hi - Vec3::new(o.size.x * 0.5, o.size.y, o.size.z * 0.5))
        .max_element()
        .max(0.);
    if !lo.is_finite() || !hi.is_finite() || excess > 0.003 {
        return Err(format!(
            "{:?}/{} envelope {lo:?}..{hi:?}, excess {excess}",
            o.kind, o.seed
        ));
    }
    let mut triangles = 0;
    for ((_, label), g) in &a.parts {
        if SemanticLabel::from_label(part_label(label)).is_none()
            || g.positions.len() != g.normals.len()
            || g.positions.len() != g.uvs.len()
            || !g.indices.len().is_multiple_of(3)
        {
            return Err("invalid attributes/semantic class".into());
        }
        if g.positions
            .iter()
            .any(|p| !Vec3::from_array(*p).is_finite())
            || g.uvs.iter().any(|p| !Vec2::from_array(*p).is_finite())
            || g.normals.iter().any(|n| {
                !Vec3::from_array(*n).is_finite()
                    || (Vec3::from_array(*n).length() - 1.).abs() > 0.005
            })
        {
            return Err("nonfinite attributes or invalid normals".into());
        }
        for tri in g.indices.as_chunks::<3>().0 {
            if tri.iter().any(|&i| i as usize >= g.positions.len()) {
                return Err("invalid index".into());
            }
            let p = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
            let normal = (p[1] - p[0]).cross(p[2] - p[0]);
            let smooth: Vec3 = tri
                .iter()
                .map(|&i| Vec3::from_array(g.normals[i as usize]))
                .sum();
            if normal.length_squared() < 1e-20 || normal.dot(smooth) < -1e-6 {
                return Err(format!(
                    "{:?}/{} degenerate/inverted triangle {tri:?} in {label}, points {p:?}, orientation {}",
                    o.kind, o.seed, normal.dot(smooth)
                ));
            }
            triangles += 1;
        }
    }
    Ok(MeshCheck {
        triangles,
        batches: a.parts.len(),
        max_envelope_overrun_m: excess,
        min_y_m: lo.y,
    })
}

/// Six orthographic first-surface depth maps, normalized by the fixed envelope.
/// Missing surface is -1. Materials, UVs, vertex counts and triangulation do not
/// contribute to distance. Pixel centers, no culling, nearest/farthest depth.
pub fn descriptor(o: &IndoorObject, a: &Assembly) -> Vec<f32> {
    descriptor_at_resolution(o, a, RESOLUTION)
}

/// Higher resolution for tests of narrow seams and supports. Population studies
/// retain the fixed public resolution so their historical distances are comparable.
pub(crate) fn descriptor_at_resolution(o: &IndoorObject, a: &Assembly, n: usize) -> Vec<f32> {
    assert!((1..=256).contains(&n));
    let mut out = vec![-1.; 6 * n * n];
    for g in a.parts.values() {
        for tri in g.indices.as_chunks::<3>().0 {
            let p = tri.map(|i| {
                Vec3::from_array(g.positions[i as usize]) / o.size + Vec3::new(0.5, 0., 0.5)
            });
            for axis in 0..3 {
                let uv = p.map(|v| Vec2::new(v[(axis + 1) % 3], v[(axis + 2) % 3]));
                let edge = |a: Vec2, b: Vec2, c: Vec2| (b - a).perp_dot(c - a);
                let area = edge(uv[0], uv[1], uv[2]);
                if area.abs() < 1e-10 {
                    continue;
                }
                let lo = uv[0].min(uv[1]).min(uv[2]);
                let hi = uv[0].max(uv[1]).max(uv[2]);
                let start = (lo * n as f32)
                    .floor()
                    .clamp(Vec2::ZERO, Vec2::splat((n - 1) as f32))
                    .as_uvec2();
                let end = (hi * n as f32)
                    .ceil()
                    .clamp(Vec2::ZERO, Vec2::splat((n - 1) as f32))
                    .as_uvec2();
                for y in start.y..=end.y {
                    for x in start.x..=end.x {
                        let q = (Vec2::new(x as f32, y as f32) + Vec2::splat(0.5)) / n as f32;
                        let w = [
                            edge(uv[1], uv[2], q) / area,
                            edge(uv[2], uv[0], q) / area,
                            edge(uv[0], uv[1], q) / area,
                        ];
                        if w.iter().any(|&v| v < -1e-5) {
                            continue;
                        }
                        let z = (w[0] * p[0][axis] + w[1] * p[1][axis] + w[2] * p[2][axis])
                            .clamp(0., 1.);
                        for (side, depth) in [(0, z), (1, 1. - z)] {
                            let i = (axis * 2 + side) * n * n + y as usize * n + x as usize;
                            if out[i] < 0. || depth < out[i] {
                                out[i] = depth;
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

pub fn distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut count = 0;
    let mut error = 0.;
    for (&a, &b) in a.iter().zip(b) {
        if a < 0. && b < 0. {
            continue;
        }
        count += 1;
        error += if a < 0. || b < 0. { 1. } else { (a - b).abs() };
    }
    error / count.max(1) as f32
}

/// Ordered greedy coverage representatives. Not an estimate of independent
/// combinations or perceptual volume; thresholds and sample order are explicit.
pub fn representatives(rows: &[Vec<f32>], threshold: f32) -> Vec<usize> {
    let mut centers: Vec<usize> = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        if centers.iter().all(|&j| distance(row, &rows[j]) > threshold) {
            centers.push(i);
        }
    }
    centers
}

#[cfg(test)]
mod display_qualification;
#[cfg(test)]
mod furniture_qualification;
#[cfg(test)]
mod product_program_tests;
#[cfg(test)]
mod regularity_qualification;
#[cfg(test)]
mod seating_qualification;
#[cfg(test)]
mod tests;
