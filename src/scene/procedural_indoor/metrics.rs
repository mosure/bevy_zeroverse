//! Reproducible, resolution-aware dataset distribution exports, independent of a GPU.
use super::layout::{IndoorLayout, IndoorManifest, ObjectKind, GENERATOR_VERSION, NEIGHBOR_DEPTH};
use bevy::prelude::*;
use serde::Serialize;
use std::{
    collections::BTreeMap,
    fs,
    io::{BufWriter, Write},
    path::Path,
};

const GRID: usize = 24;
const KINDS: [ObjectKind; 28] = [
    ObjectKind::Table,
    ObjectKind::Desk,
    ObjectKind::Chair,
    ObjectKind::Sofa,
    ObjectKind::CoffeeTable,
    ObjectKind::Cabinet,
    ObjectKind::Bookcase,
    ObjectKind::Plant,
    ObjectKind::TrashCan,
    ObjectKind::Whiteboard,
    ObjectKind::Display,
    ObjectKind::Laptop,
    ObjectKind::Monitor,
    ObjectKind::Notebook,
    ObjectKind::Mug,
    ObjectKind::Books,
    ObjectKind::Rug,
    ObjectKind::WallArt,
    ObjectKind::Clock,
    ObjectKind::FloorLamp,
    ObjectKind::Keyboard,
    ObjectKind::Mouse,
    ObjectKind::WaterBottle,
    ObjectKind::PenHolder,
    ObjectKind::Printer,
    ObjectKind::StorageBox,
    ObjectKind::CoatRack,
    ObjectKind::Bag,
];

#[derive(Serialize)]
pub struct NumericDistribution {
    pub count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub standard_deviation: f64,
    pub percentiles_05_25_50_75_95: [f64; 5],
    pub bin_edges: Vec<f64>,
    pub bin_counts: Vec<usize>,
}
impl NumericDistribution {
    pub(super) fn from_values(mut values: Vec<f64>) -> Self {
        values.sort_by(f64::total_cmp);
        let count = values.len();
        assert!(count > 0);
        let min = values[0];
        let max = values[count - 1];
        let mean = values.iter().sum::<f64>() / count as f64;
        let standard_deviation =
            (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count as f64).sqrt();
        let percentiles_05_25_50_75_95 = [0.05, 0.25, 0.5, 0.75, 0.95]
            .map(|q| values[((count - 1) as f64 * q).round() as usize]);
        let span = (max - min).max(1e-6);
        let bin_edges = (0..=20).map(|i| min + span * i as f64 / 20.0).collect();
        let mut bin_counts = vec![0; 20];
        for v in values {
            bin_counts[(((v - min) / span * 20.0) as usize).min(19)] += 1;
        }
        Self {
            count,
            min,
            max,
            mean,
            standard_deviation,
            percentiles_05_25_50_75_95,
            bin_edges,
            bin_counts,
        }
    }
}

#[derive(Serialize)]
pub struct CoverageReport {
    pub schema_version: u32,
    pub generator_version: u32,
    pub scenes: usize,
    pub image_size: [u32; 2],
    pub density: f32,
    pub human_density: f32,
    pub coordinate_convention: &'static str,
    pub count_policy: &'static str,
    pub heatmap_policy: &'static str,
    pub heatmap_grid_size: usize,
    /// Scene-normalized count distributions include zero for absent object kinds.
    pub object_counts_per_scene: BTreeMap<String, BTreeMap<usize, usize>>,
    pub object_counts_by_layout: BTreeMap<String, BTreeMap<usize, usize>>,
    pub categories: BTreeMap<String, BTreeMap<String, usize>>,
    pub numeric: BTreeMap<String, NumericDistribution>,
    pub joint_histograms: BTreeMap<String, Vec<usize>>,
    pub joint_histogram_policy: &'static str,
    pub occupancy_signature_estimate: f64,
    pub occupancy_signature_policy: &'static str,
    pub placement_heatmaps: BTreeMap<String, Vec<usize>>,
    /// First seed observed for each layout x lighting x floor x furniture stratum.
    pub stratified_seeds: BTreeMap<String, u64>,
}

pub fn stratum(scene: &IndoorManifest) -> String {
    format!(
        "{:?}/{:?}/floor{}/furniture{}/{:?}/{}/lights{}/sun{}/electric{}",
        scene.layout,
        scene.lighting,
        scene.floor_style,
        scene.furniture_style,
        scene.architecture_style,
        scene
            .program
            .as_ref()
            .map_or(format!("{:?}", scene.floor_plan), |p| format!(
                "zones{}",
                p.zones.len()
            )),
        scene
            .program
            .as_ref()
            .map_or(
                scene.lighting_design as u32,
                |p| (p.fixture_size.x / p.fixture_size.y).floor() as u8 as u32
            ),
        (super::architecture::sun_illuminance(scene).max(0.1).log10() + 1.0).floor() as u32,
        scene.target_lux.max(1.0).log10().floor() as u32
    )
}

/// Deterministic category-balanced selection; small budgets must not select only
/// the alphabetically first layout or lighting mood.
pub fn select_strata(strata: &BTreeMap<String, u64>, budget: usize) -> Vec<u64> {
    let mut remaining: Vec<_> = strata.iter().collect();
    let mut counts = BTreeMap::<(usize, String), usize>::new();
    let mut selected = Vec::new();
    while !remaining.is_empty() && selected.len() < budget {
        let score = |key: &str| {
            key.split('/')
                .enumerate()
                .map(|(i, value)| {
                    [3.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 3.0, 3.0]
                        .get(i)
                        .copied()
                        .unwrap_or(1.0)
                        / (1 + counts.get(&(i, value.to_owned())).copied().unwrap_or(0)) as f64
                })
                .sum::<f64>()
        };
        let index = (0..remaining.len())
            .max_by(|&a, &b| {
                score(remaining[a].0)
                    .total_cmp(&score(remaining[b].0))
                    .then_with(|| remaining[b].1.cmp(remaining[a].1))
            })
            .unwrap();
        let (key, seed) = remaining.remove(index);
        for (i, value) in key.split('/').enumerate() {
            *counts.entry((i, value.to_owned())).or_default() += 1;
        }
        selected.push(*seed);
    }
    selected
}

#[allow(clippy::too_many_arguments)]
pub fn export_metrics(
    first_seed: u64,
    seeds: usize,
    cameras: usize,
    density: f32,
    layout: IndoorLayout,
    width: u32,
    height: u32,
    directory: &Path,
) -> Result<CoverageReport, String> {
    export_metrics_with_humans(
        first_seed, seeds, cameras, density, layout, width, height, directory, 0.25,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn export_metrics_with_humans(
    first_seed: u64,
    seeds: usize,
    cameras: usize,
    density: f32,
    layout: IndoorLayout,
    width: u32,
    height: u32,
    directory: &Path,
    human_density: f32,
) -> Result<CoverageReport, String> {
    export_metrics_with_camera_settings(
        first_seed,
        seeds,
        cameras,
        density,
        layout,
        width,
        height,
        directory,
        human_density,
        &default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn export_metrics_with_camera_settings(
    first_seed: u64,
    seeds: usize,
    cameras: usize,
    density: f32,
    layout: IndoorLayout,
    width: u32,
    height: u32,
    directory: &Path,
    human_density: f32,
    camera_settings: &super::cameras::CameraSettings,
) -> Result<CoverageReport, String> {
    camera_settings.validate()?;
    if seeds == 0
        || cameras == 0
        || cameras > 256
        || width == 0
        || height == 0
        || !density.is_finite()
        || !(0.0..=1.0).contains(&density)
        || !human_density.is_finite()
        || !(0.0..=1.0).contains(&human_density)
    {
        return Err("metrics require a nonempty audit, 1..=256 cameras, positive image dimensions and density in [0, 1]".into());
    }
    export_inner(
        first_seed,
        seeds,
        cameras,
        density,
        layout,
        width,
        height,
        directory,
        human_density,
        camera_settings,
    )
    .map_err(|e| e.to_string())
}

#[allow(clippy::too_many_arguments)]
fn export_inner(
    first_seed: u64,
    seeds: usize,
    cameras: usize,
    density: f32,
    layout: IndoorLayout,
    width: u32,
    height: u32,
    directory: &Path,
    human_density: f32,
    camera_settings: &super::cameras::CameraSettings,
) -> Result<CoverageReport, Box<dyn std::error::Error>> {
    fs::create_dir_all(directory)?;
    let mut humans = BufWriter::new(fs::File::create(directory.join("humans.csv"))?);
    writeln!(humans,"seed,id,chair,neighbor,pose,outfit,stature_m,build,skin_tone,hairstyle,glasses,x_m,y_m,z_m,yaw_radians")?;
    let mut objects = BufWriter::new(fs::File::create(directory.join("objects.csv"))?);
    let mut camera_rows = BufWriter::new(fs::File::create(directory.join("cameras.csv"))?);
    let mut scenes = BufWriter::new(fs::File::create(directory.join("scenes.csv"))?);
    writeln!(
        objects,
        "seed,layout,id,kind,neighbor,support,x_m,y_m,z_m,width_m,height_m,depth_m,yaw_radians,variant,u,v"
    )?;
    writeln!(
        camera_rows,
        "seed,layout,camera,start_x_m,start_y_m,start_z_m,end_x_m,end_y_m,end_z_m,target_x_m,target_y_m,target_z_m,vertical_fov_deg,horizontal_fov_deg,fx_pixels,fy_pixels,cx_pixels,cy_pixels,near_m,far_m,path_length_m,yaw_deg,pitch_deg"
    )?;
    writeln!(
        scenes,
        "seed,layout,density,lighting,palette,floor,furniture,ceiling,width_m,height_m,depth_m,main_instances,neighbor_instances,main_chairs,rejected_placements"
    )?;
    let mut report = CoverageReport {
        schema_version: 6,
        generator_version: GENERATOR_VERSION,
        scenes: seeds,
        image_size: [width, height],
        density,
        human_density,
        coordinate_convention: "Unaugmented room coordinates, metres; +Y up. Main room normalized u=x/width+0.5, v=z/depth+0.5. Neighbor uses its own 3.2m depth. CSV camera coordinates are local; rendered capture.json contains augmented world extrinsics. Pixel centers use (width/2,height/2), square pixels.",
        count_policy: "Every object kind includes zero-count scenes. main/ and neighbor/ are separate; by-layout distributions have that layout's scene count as denominator.",
        heatmap_policy: "Object centers (not projected visibility or footprint). Row-major v then u; +Z downward. Camera paths contribute 33 equally spaced samples including endpoints per camera. Neighbor and main rooms normalized independently. No failed scenes are silently dropped.",
        heatmap_grid_size: GRID,
        object_counts_per_scene: BTreeMap::new(),
        object_counts_by_layout: BTreeMap::new(),
        categories: BTreeMap::new(),
        numeric: BTreeMap::new(),
        joint_histograms: BTreeMap::new(),
        joint_histogram_policy: "16x16 row-major histograms: zones_vs_main_chairs uses X zone count [1,9), Y chairs [0,64); wood_roughness_vs_repeat uses X roughness [0,1), Y repeat metres [0,2). Clamped endpoints. sun_log10_vs_electric_log10 uses X log10 solar lux [-1,5], Y log10 electric target lux [0,3]; room_area_vs_clutter uses X room area [0,300] m2, Y clutter prior [0,1]. Inspect correlations as well as marginals.",
        occupancy_signature_estimate: 0.0,
        occupancy_signature_policy: "HyperLogLog 4096 registers, about 1.63 percent relative standard error. Hashes 12x12 normalized object-category occupancy and 10cm partition geometry; ignores random seeds, textures and colours. Diagnostic only; not a proof of perceptual uniqueness or training utility.",
        placement_heatmaps: BTreeMap::new(),
        stratified_seeds: BTreeMap::new(),
    };
    let mut signatures = super::program_coverage::OccupancySketch::default();
    let mut numeric = super::metrics_sort::NumericCollector::new(directory)?;
    for offset in 0..seeds {
        let mut scene = IndoorManifest::generate_with_humans(
            first_seed.wrapping_add(offset as u64),
            layout,
            density,
            cameras,
            human_density,
        )?;
        if scene.camera_settings != *camera_settings {
            scene.camera_settings = camera_settings.clone();
            scene.cameras.clear();
            scene.sample_cameras(cameras)?;
        }
        super::validation::validate_layout(&scene)?;
        signatures.insert(&scene);
        if let Some(program) = &scene.program {
            if let Some(d) = &program.domain {
                for (key, value) in [
                    ("exposure_ev100", d.photometry.ev100),
                    (
                        "sky_radiance_mean",
                        d.photometry.sky_radiance.element_sum() / 3.0,
                    ),
                    ("fixture_active_fraction", d.photometry.active_fraction),
                    ("fixture_circuit_contrast", d.photometry.circuit_contrast),
                    ("fixture_gradient_x", d.photometry.fixture_gradient.x),
                    ("fixture_gradient_z", d.photometry.fixture_gradient.y),
                    (
                        "fixture_temperature_gradient_kelvin",
                        d.photometry.temperature_gradient,
                    ),
                    ("facade_pier_fraction", d.facade_pier_fraction),
                    ("blind_coverage", d.blind_coverage),
                    ("blind_tilt_radians", d.blind_tilt),
                    ("ceiling_relief_m", d.ceiling_relief),
                    ("clutter_prior", d.clutter),
                    ("service_density", d.service_density),
                    ("furnishing_disorder", d.disorder),
                ] {
                    numeric.push(key, value as f64)?;
                }
                joint_histogram(
                    &mut report,
                    "sun_log10_vs_electric_log10",
                    (scene.daylight_lux.log10() + 1.0) / 6.0,
                    scene.target_lux.log10() / 3.0,
                );
                joint_histogram(
                    &mut report,
                    "room_area_vs_clutter",
                    scene.room_size.x * scene.room_size.z / 300.0,
                    d.clutter,
                );
            }
            numeric.push("program_zone_count", program.zones.len() as f64)?;
            numeric.push("program_partition_count", program.partitions.len() as f64)?;
            numeric.push("fixture_spacing_x_m", program.light_spacing.x as f64)?;
            numeric.push("fixture_spacing_z_m", program.light_spacing.y as f64)?;
            numeric.push("fixture_drop_m", program.light_drop as f64)?;
            for surface in [
                super::materials::Surface::Glass,
                super::materials::Surface::GlassInterior,
            ] {
                let recipe = super::materials::glass::GlassRecipe::sample(scene.seed, surface);
                for (key, value) in [
                    ("roughness", recipe.roughness),
                    ("ior", recipe.ior),
                    ("attenuation_distance_m", recipe.attenuation_distance_m),
                ] {
                    numeric.push(&format!("glazing_{surface:?}_{key}"), value as f64)?;
                }
            }
            if let Some(f) = &program.finishes {
                numeric.push("ceiling_pitch_x_m", f.ceiling_pitch.x as f64)?;
                numeric.push("ceiling_pitch_z_m", f.ceiling_pitch.y as f64)?;
                numeric.push("wall_panel_pitch_m", f.panel_pitch as f64)?;
                if scene.architecture_style == super::layout::ArchitectureStyle::Classic {
                    if let Some(n) = &f.niche {
                        for (key, value) in [
                            (
                                "niche_width_m",
                                (scene.room_size.x * n.width_fraction).clamp(0.8, 3.6),
                            ),
                            ("niche_bottom_m", scene.room_size.y * n.bottom_fraction),
                            ("niche_height_m", scene.room_size.y * n.height_fraction),
                            ("niche_depth_m", n.depth_m),
                            ("niche_position_fraction", n.position_fraction),
                            ("niche_shelf_pitch_m", n.shelf_pitch_m),
                        ] {
                            numeric.push(key, value as f64)?;
                        }
                    }
                }
            }
            for i in 0..super::architecture::fixture_positions(&scene).len() {
                let (_, lumens) = super::architecture::fixture_photometry(&scene, i);
                let (inner, outer) = super::architecture::fixture_angles(&scene, i);
                numeric.push("fixture_lumens", lumens as f64)?;
                numeric.push("fixture_inner_angle", inner as f64)?;
                numeric.push("fixture_outer_angle", outer as f64)?;
            }
            for zone in &program.zones {
                if let Some(f) = &zone.furnishing {
                    for (key, value) in [
                        ("workstation_stagger", f.stagger),
                        ("workstation_curvature", f.curvature),
                        ("workstation_fan_radians", f.fan_radians),
                        ("workstation_jitter_m", f.jitter),
                        ("workstation_occupancy_gradient_x", f.occupancy_gradient.x),
                        ("workstation_occupancy_gradient_z", f.occupancy_gradient.y),
                        ("workstation_opposing_probability", f.opposing_probability),
                    ] {
                        numeric.push(key, value as f64)?;
                    }
                }
                let size = zone.max - zone.min;
                for (name, value) in [
                    ("zone_area_m2", size.x * size.y),
                    ("zone_aspect", size.x / size.y),
                    ("zone_orientation_radians", zone.orientation),
                    ("zone_aisle_m", zone.aisle),
                    ("zone_desk_width_m", zone.desk_width),
                    ("zone_occupancy", zone.occupancy),
                ] {
                    numeric.push(name, value as f64)?;
                }
            }
            for partition in &program.partitions {
                for (name, value) in [
                    ("partition_door_width_m", partition.door_width),
                    ("partition_glazing_fraction", partition.glazing_fraction),
                    ("partition_thickness_m", partition.thickness),
                    ("partition_mullion_pitch_m", partition.mullion_pitch),
                ] {
                    numeric.push(name, value as f64)?;
                }
                if let Some(f) = partition.transom_fraction {
                    numeric.push("partition_transom_fraction", f as f64)?;
                }
            }
            for material in &program.materials {
                if let Some(l) = &material.layers {
                    for (key, value) in [
                        ("stripe_strength", l.stripe_strength),
                        ("vein_strength", l.vein_strength),
                        ("fleck_strength", l.fleck_strength),
                    ] {
                        numeric.push(
                            &format!("material_{:?}_{key}", material.surface),
                            value as f64,
                        )?;
                    }
                }
                for (parameter, value) in [
                    ("roughness", material.roughness),
                    ("repeat_m", material.period_m),
                    ("relief_m", material.relief_m),
                    ("contrast", material.contrast),
                    ("grain_frequency", material.grain_frequency),
                    ("weathering", material.weathering),
                ] {
                    numeric.push(
                        &format!("material_{:?}_{parameter}", material.surface),
                        value as f64,
                    )?;
                }
            }
            let chairs = scene
                .objects
                .iter()
                .filter(|o| o.kind == ObjectKind::Chair && !o.neighbor)
                .count();
            joint_histogram(
                &mut report,
                "zones_vs_main_chairs",
                (program.zones.len() as f32 - 1.0) / 8.0,
                chairs as f32 / 64.0,
            );
            let wood = &program.materials[super::materials::Surface::Wood as usize];
            joint_histogram(
                &mut report,
                "wood_roughness_vs_repeat",
                wood.roughness,
                wood.period_m / 2.0,
            );
        }
        let layout_name = format!("{:?}", scene.layout);
        for (category, value) in [
            ("layout", layout_name.clone()),
            ("lighting", format!("{:?}", scene.lighting)),
            ("palette", scene.palette.to_string()),
            ("floor", scene.floor_style.to_string()),
            ("furniture", scene.furniture_style.to_string()),
            ("ceiling", scene.ceiling_style.to_string()),
            ("architecture", format!("{:?}", scene.architecture_style)),
            ("floor_plan", format!("{:?}", scene.floor_plan)),
            (
                "furnishing_quarter_turn",
                scene.furnishing_quarter_turn.to_string(),
            ),
            ("lighting_design", scene.lighting_design.to_string()),
            ("blinds", scene.blinds.to_string()),
            ("window_bays", scene.window_bays.to_string()),
        ] {
            if scene.program.is_some()
                && matches!(
                    category,
                    "palette" | "floor_plan" | "furnishing_quarter_turn" | "lighting_design"
                )
            {
                continue;
            }
            *report
                .categories
                .entry(category.into())
                .or_default()
                .entry(value)
                .or_default() += 1;
        }
        report
            .stratified_seeds
            .entry(stratum(&scene))
            .or_insert(scene.seed);
        let count = |kind, neighbor| {
            scene
                .objects
                .iter()
                .filter(|o| o.kind == kind && o.neighbor == neighbor)
                .count()
        };
        for kind in KINDS {
            for neighbor in [false, true] {
                let key = format!("{}/{kind:?}", if neighbor { "neighbor" } else { "main" });
                let n = count(kind, neighbor);
                *report
                    .object_counts_per_scene
                    .entry(key.clone())
                    .or_default()
                    .entry(n)
                    .or_default() += 1;
                *report
                    .object_counts_by_layout
                    .entry(format!("{layout_name}/{key}"))
                    .or_default()
                    .entry(n)
                    .or_default() += 1;
            }
        }
        let main_count = scene.objects.iter().filter(|o| !o.neighbor).count();
        for plant in scene.objects.iter().filter(|o| o.kind == ObjectKind::Plant) {
            let p = super::plants::Growth::sample(plant.seed);
            for (key, value) in [
                ("plant_pot_height_m", plant.size.y * p.pot_fraction),
                (
                    "plant_pot_radius_m",
                    plant.size.x.min(plant.size.z) * 0.44 * p.pot_radius,
                ),
                ("plant_pot_taper", p.pot_taper),
                ("plant_leaf_density", p.density),
                ("plant_phyllotaxis_radians", p.phyllotaxis),
            ] {
                numeric.push(key, value as f64)?;
            }
        }
        writeln!(
            scenes,
            "{},{},{},{:?},{},{},{},{},{},{},{},{},{},{},{}",
            scene.seed,
            layout_name,
            density,
            scene.lighting,
            scene.palette,
            scene.floor_style,
            scene.furniture_style,
            scene.ceiling_style,
            scene.room_size.x,
            scene.room_size.y,
            scene.room_size.z,
            main_count,
            scene.objects.len() - main_count,
            count(ObjectKind::Chair, false),
            scene.rejected_placements
        )?;
        for (name, value) in [
            ("room_area_m2", scene.room_size.x * scene.room_size.z),
            ("room_aspect_ratio", scene.room_size.x / scene.room_size.z),
            ("room_width_m", scene.room_size.x),
            ("room_depth_m", scene.room_size.z),
            ("room_height_m", scene.room_size.y),
            ("sun_elevation_radians", scene.sun_elevation),
            ("sun_azimuth_radians", scene.sun_azimuth),
            ("light_kelvin", scene.light_kelvin),
            ("target_illuminance_lux", scene.target_lux),
            (
                "sun_illuminance_lux",
                super::architecture::sun_illuminance(&scene),
            ),
            ("main_instances", main_count as f32),
            ("main_chairs", count(ObjectKind::Chair, false) as f32),
            ("rejected_placements", scene.rejected_placements as f32),
        ] {
            numeric.push(name, value as f64)?;
        }
        for neighbor in [false, true] {
            let n = scene
                .humans
                .iter()
                .filter(|h| h.neighbor == neighbor)
                .count();
            let key = format!("{}/Person", if neighbor { "neighbor" } else { "main" });
            *report
                .object_counts_per_scene
                .entry(key.clone())
                .or_default()
                .entry(n)
                .or_default() += 1;
            *report
                .object_counts_by_layout
                .entry(format!("{layout_name}/{key}"))
                .or_default()
                .entry(n)
                .or_default() += 1;
        }
        numeric.push("people_per_scene", scene.humans.len() as f64)?;
        numeric.push(
            "ceiling_lights_per_scene",
            super::architecture::fixture_positions(&scene).len() as f64,
        )?;
        numeric.push(
            "camera_primary_room_fraction",
            scene
                .cameras
                .iter()
                .filter(|c| scene.in_primary_room(c.start, 0.0))
                .count() as f64
                / scene.cameras.len().max(1) as f64,
        )?;
        for object in &scene.objects {
            if matches!(
                object.kind,
                ObjectKind::Table | ObjectKind::Desk | ObjectKind::CoffeeTable
            ) {
                let p = super::objects::tables::parameters(object);
                *report
                    .categories
                    .entry("table_support".into())
                    .or_default()
                    .entry(p.support.to_string())
                    .or_default() += 1;
                for (key, value) in [
                    ("table_top_thickness_m", p.top_thickness),
                    ("table_leg_radius_m", p.leg_radius),
                    ("table_leg_rake_m", p.leg_rake),
                    ("table_leg_inset_m", p.leg_inset),
                    ("table_outline_exponent", p.outline_exponent),
                    ("table_taper", p.taper),
                    ("table_pedestal_radius_fraction", p.pedestal_radius),
                    (
                        "table_pedestal_base_radius_fraction",
                        p.pedestal_base_radius,
                    ),
                    ("table_pedestal_base_aspect", p.pedestal_base_aspect),
                ] {
                    numeric.push(key, value as f64)?;
                }
                *report
                    .categories
                    .entry("table_top_surface".into())
                    .or_default()
                    .entry(format!("{:?}", p.top_surface))
                    .or_default() += 1;
            }
            let category = match object.kind {
                ObjectKind::Chair => "chair_family",
                ObjectKind::Laptop => "laptop_family",
                _ => continue,
            };
            *report
                .categories
                .entry(category.into())
                .or_default()
                .entry(object.variant.to_string())
                .or_default() += 1;
            if object.kind == ObjectKind::Laptop {
                let p = super::objects::computers::parameters(object);
                for (name, value) in [
                    ("laptop_aspect", p.aspect_ratio),
                    ("laptop_chassis_m", p.chassis_m),
                    ("laptop_bezel_m", p.bezel_m),
                    ("laptop_keyboard_fraction", p.keyboard_fraction),
                ] {
                    numeric.push(name, value as f64)?;
                }
                numeric.push(
                    "laptop_lid_angle_radians",
                    super::objects::computers::lid_angle(object) as f64,
                )?;
            } else {
                let p = super::objects::chairs::parameters(object);
                *report
                    .categories
                    .entry("chair_headrest".into())
                    .or_default()
                    .entry(p.headrest.to_string())
                    .or_default() += 1;
                for (name, value) in [
                    ("chair_curvature_m", p.curvature),
                    ("chair_taper", p.taper),
                    ("chair_shell_m", p.shell_thickness),
                    ("chair_recline_radians", p.recline),
                    ("chair_arm_height_m", p.arm_height),
                    ("chair_lumbar_m", p.lumbar),
                    ("chair_seat_roundness", p.seat_roundness),
                    ("chair_shoulder_flare", p.shoulder_flare),
                    ("chair_seat_width_fraction", p.seat_width_fraction),
                    ("chair_seat_depth_fraction", p.seat_depth_fraction),
                    ("chair_back_width_fraction", p.back_width_fraction),
                    ("chair_leg_splay", p.leg_splay),
                    ("chair_base_radius_fraction", p.base_radius_fraction),
                ] {
                    numeric.push(name, value as f64)?;
                }
                let yaw = (object.yaw + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
                    - std::f32::consts::PI;
                numeric.push("chair_yaw_radians", yaw as f64)?;
            }
        }
        for human in &scene.humans {
            if let Some(p) = &human.pose_program {
                for (name, value) in [
                    ("pose_lean_x", p.lean.x),
                    ("pose_lean_z", p.lean.y),
                    ("pose_twist", p.torso_twist),
                    ("pose_stance_m", p.stance),
                    ("pose_stride_m", p.stride),
                    ("pose_phase", p.phase),
                ] {
                    numeric.push(name, value as f64)?;
                }
                for i in 0..2 {
                    for (name, value) in [
                        ("pose_arm_reach", p.arm_reach[i]),
                        ("pose_arm_elevation", p.arm_elevation[i]),
                        ("pose_arm_sweep", p.arm_sweep[i]),
                    ] {
                        numeric.push(name, value as f64)?;
                    }
                }
            }
            if let Some(a) = &human.appearance {
                for (name, value) in [
                    ("human_melanin", a.melanin),
                    ("human_skin_roughness", a.skin_roughness),
                    ("human_garment_ease_m", a.garment_ease),
                    ("human_fold_amplitude_m", a.fold_amplitude),
                    ("human_sleeve_coverage", a.sleeve_coverage),
                    ("human_hem_fraction", a.hem_fraction),
                    ("human_weave_scale", a.weave_scale),
                    ("human_weave_rotation_radians", a.weave_rotation),
                    ("human_hair_length_m", a.hair_length),
                    ("human_hair_part", a.hair_part),
                    ("human_hair_curl", a.hair_curl),
                ] {
                    numeric.push(name, value as f64)?;
                }
            }
            for (name, value) in [
                ("human_stature_m", human.stature),
                ("human_build", human.build),
                ("human_head_yaw_radians", human.head_yaw),
            ] {
                numeric.push(name, value as f64)?;
            }
            for (name, value) in [
                ("human_pose", format!("{:?}", human.pose)),
                ("human_outfit", format!("{:?}", human.outfit)),
                ("human_skin_tone", human.skin_tone.to_string()),
                ("human_hairstyle", human.hairstyle.to_string()),
            ] {
                *report
                    .categories
                    .entry(name.into())
                    .or_default()
                    .entry(value)
                    .or_default() += 1;
            }
            heat(
                &mut report,
                if human.neighbor {
                    "neighbor/Person"
                } else {
                    "main/Person"
                },
                human.position.x / scene.room_size.x + 0.5,
                if human.neighbor {
                    (human.position.z - scene.room_size.z * 0.5) / NEIGHBOR_DEPTH
                } else {
                    human.position.z / scene.room_size.z + 0.5
                },
            );
            writeln!(
                humans,
                "{},{},{},{},{:?},{:?},{},{},{},{},{},{},{},{},{}",
                scene.seed,
                human.id,
                human.chair.map(|id| id.to_string()).unwrap_or_default(),
                human.neighbor,
                human.pose,
                human.outfit,
                human.stature,
                human.build,
                human.skin_tone,
                human.hairstyle,
                human.glasses,
                human.position.x,
                human.position.y,
                human.position.z,
                human.yaw
            )?;
        }
        for object in &scene.objects {
            let u = object.position.x / scene.room_size.x + 0.5;
            let v = if object.neighbor {
                (object.position.z - scene.room_size.z * 0.5) / NEIGHBOR_DEPTH
            } else {
                object.position.z / scene.room_size.z + 0.5
            };
            heat(
                &mut report,
                &format!(
                    "{}/{:?}",
                    if object.neighbor { "neighbor" } else { "main" },
                    object.kind
                ),
                u,
                v,
            );
            writeln!(
                objects,
                "{},{},{},{:?},{},{},{},{},{},{},{},{},{},{},{},{}",
                scene.seed,
                layout_name,
                object.id,
                object.kind,
                object.neighbor,
                object.support.map(|x| x.to_string()).unwrap_or_default(),
                object.position.x,
                object.position.y,
                object.position.z,
                object.size.x,
                object.size.y,
                object.size.z,
                object.yaw,
                object.variant,
                u,
                v
            )?;
        }
        for (i, camera) in scene.cameras.iter().enumerate() {
            let fy = height as f32 / (2.0 * (camera.fov_degrees.to_radians() * 0.5).tan());
            let hfov = (width as f32 / (2.0 * fy)).atan().to_degrees() * 2.0;
            let direction = (camera.target - camera.start).normalize();
            let yaw = direction.x.atan2(-direction.z).to_degrees();
            let pitch = direction.y.asin().to_degrees();
            let length = camera.path_length();
            if let Some(m) = &camera.motion {
                numeric.push("camera_roll_start_radians", m.roll[0] as f64)?;
                numeric.push("camera_roll_end_radians", m.roll[1] as f64)?;
                numeric.push(
                    "camera_target_motion_m",
                    m.target_end.distance(camera.target) as f64,
                )?;
                numeric.push(
                    "camera_curve_bend_m",
                    camera
                        .transform_at(0.5)
                        .translation
                        .distance(camera.start.lerp(camera.end, 0.5)) as f64,
                )?;
            }
            writeln!(
                camera_rows,
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},0.1,50,{},{},{}",
                scene.seed,
                layout_name,
                i,
                camera.start.x,
                camera.start.y,
                camera.start.z,
                camera.end.x,
                camera.end.y,
                camera.end.z,
                camera.target.x,
                camera.target.y,
                camera.target.z,
                camera.fov_degrees,
                hfov,
                fy,
                fy,
                width as f32 * 0.5,
                height as f32 * 0.5,
                length,
                yaw,
                pitch
            )?;
            for (name, value) in [
                ("vertical_fov_degrees", camera.fov_degrees),
                ("horizontal_fov_degrees", hfov),
                ("fx_pixels", fy),
                ("fy_pixels", fy),
                ("cx_pixels", width as f32 * 0.5),
                ("cy_pixels", height as f32 * 0.5),
                ("near_m", 0.1),
                ("far_m", 50.0),
                ("camera_height_m", camera.start.y),
                (
                    "camera_start_boundary_clearance_m",
                    (scene.room_size.x * 0.5 - camera.start.x.abs())
                        .min(scene.room_size.z * 0.5 - camera.start.z.abs()),
                ),
                ("trajectory_length_m", length),
                ("camera_yaw_degrees", yaw),
                ("camera_pitch_degrees", pitch),
                ("target_distance_m", camera.start.distance(camera.target)),
            ] {
                numeric.push(name, value as f64)?;
            }
            for (name, position) in [("camera_start", camera.start), ("camera_end", camera.end)] {
                heat(
                    &mut report,
                    name,
                    position.x / scene.room_size.x + 0.5,
                    position.z / scene.room_size.z + 0.5,
                );
            }
            for step in 0..=32 {
                let position = camera.transform_at(step as f32 / 32.0).translation;
                heat(
                    &mut report,
                    "camera_path",
                    position.x / scene.room_size.x + 0.5,
                    position.z / scene.room_size.z + 0.5,
                );
            }
        }
    }
    report.occupancy_signature_estimate = signatures.estimate();
    report.numeric = numeric.finish()?;
    humans.flush()?;
    objects.flush()?;
    camera_rows.flush()?;
    scenes.flush()?;
    fs::write(
        directory.join("metrics.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    write_svg(&report, directory)?;
    Ok(report)
}

fn heat(report: &mut CoverageReport, key: &str, u: f32, v: f32) {
    let x = ((u * GRID as f32).floor() as usize).min(GRID - 1);
    let y = ((v * GRID as f32).floor() as usize).min(GRID - 1);
    report
        .placement_heatmaps
        .entry(key.into())
        .or_insert_with(|| vec![0; GRID * GRID])[y * GRID + x] += 1;
}

fn write_svg(report: &CoverageReport, directory: &Path) -> std::io::Result<()> {
    let keys = [
        "main/Chair",
        "main/Desk",
        "main/Plant",
        "main/Person",
        "neighbor/Person",
        "main/TrashCan",
        "camera_start",
        "camera_end",
        "camera_path",
    ];
    let mut svg = String::from(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1080\" height=\"1140\" viewBox=\"0 0 1080 1140\"><rect width=\"1080\" height=\"1140\" fill=\"white\"/><g font-family=\"sans-serif\" fill=\"#182433\"><text x=\"30\" y=\"30\" font-size=\"21\">Indoor placement and camera coverage</text><text x=\"30\" y=\"54\" font-size=\"13\">Room-normalized centers; +X right, +Z down. Each panel uses its own linear count scale.</text>",
    );
    for (i, key) in keys.iter().enumerate() {
        let x0 = 35 + (i % 3) * 355;
        let y0 = 105 + (i / 3) * 350;
        let values = report
            .placement_heatmaps
            .get(*key)
            .cloned()
            .unwrap_or_else(|| vec![0; GRID * GRID]);
        let max = values.iter().copied().max().unwrap_or(0).max(1);
        svg.push_str(&format!(
            "<text x=\"{x0}\" y=\"{}\" font-size=\"17\">{key}</text>",
            y0 - 15
        ));
        for (j, &value) in values.iter().enumerate() {
            let intensity = (value as f32 / max as f32 * 220.0) as u8;
            svg.push_str(&format!("<rect x=\"{}\" y=\"{}\" width=\"12\" height=\"12\" fill=\"rgb({},{},{})\"><title>{value} samples</title></rect>", x0 + (j % GRID) * 12, y0 + (j / GRID) * 12, 245 - intensity, 249 - intensity / 2, 255 - intensity / 5));
        }
        svg.push_str(&format!("<rect x=\"{x0}\" y=\"{y0}\" width=\"288\" height=\"288\" fill=\"none\" stroke=\"#536777\"/><text x=\"{x0}\" y=\"{}\" font-size=\"12\">0 to {max} samples/bin; total {}</text>", y0 + 308, values.iter().sum::<usize>()));
    }
    svg.push_str("</g></svg>");
    fs::write(directory.join("placement_heatmaps.svg"), svg)
}

fn joint_histogram(report: &mut CoverageReport, name: &str, x: f32, y: f32) {
    let bins = report
        .joint_histograms
        .entry(name.into())
        .or_insert_with(|| vec![0; 256]);
    bins[(y * 16.0).clamp(0.0, 15.0) as usize * 16 + (x * 16.0).clamp(0.0, 15.0) as usize] += 1;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn small_render_budget_covers_layout_and_style_categories() {
        let mut cells = BTreeMap::new();
        let mut seed = 0;
        for layout in 0..4 {
            for lighting in 0..3 {
                for floor in 0..3 {
                    for furniture in 0..3 {
                        for architecture in 0..4 {
                            cells.insert(format!("layout{layout}/light{lighting}/floor{floor}/furniture{furniture}/architecture{architecture}"), seed);
                            seed += 1;
                        }
                    }
                }
            }
        }
        let selected = select_strata(&cells, 4);
        assert_eq!(selected.len(), 4);
        for (dimension, categories) in [(0, 4), (1, 3), (2, 3), (3, 3), (4, 4)] {
            let seen: std::collections::BTreeSet<_> = cells
                .iter()
                .filter(|(_, seed)| selected.contains(seed))
                .map(|(key, _)| key.split('/').nth(dimension).unwrap())
                .collect();
            assert_eq!(seen.len(), categories);
        }
        assert_eq!(select_strata(&cells, usize::MAX).len(), 432);
    }
    #[test]
    fn summaries_preserve_counts_and_constant_intrinsics() {
        let d = NumericDistribution::from_values(vec![50.0; 4]);
        assert_eq!(d.count, 4);
        assert_eq!(d.mean, 50.0);
        assert_eq!(d.standard_deviation, 0.0);
        assert_eq!(d.bin_counts.iter().sum::<usize>(), 4);
        let d = NumericDistribution::from_values(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(d.percentiles_05_25_50_75_95, [0.0, 1.0, 2.0, 3.0, 4.0]);
    }
}
