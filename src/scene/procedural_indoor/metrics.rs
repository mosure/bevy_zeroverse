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
const KINDS: [ObjectKind; 24] = [
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
    pub placement_heatmaps: BTreeMap<String, Vec<usize>>,
    /// First seed observed for each layout x lighting x floor x furniture stratum.
    pub stratified_seeds: BTreeMap<String, u64>,
}

pub fn stratum(scene: &IndoorManifest) -> String {
    format!(
        "{:?}/{:?}/floor{}/furniture{}/{:?}",
        scene.layout,
        scene.lighting,
        scene.floor_style,
        scene.furniture_style,
        scene.architecture_style
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
                    [3.0, 2.0, 1.0, 1.0, 2.0][i]
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
        schema_version: 2,
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
        placement_heatmaps: BTreeMap::new(),
        stratified_seeds: BTreeMap::new(),
    };
    let mut numeric = super::metrics_sort::NumericCollector::new(directory)?;
    for offset in 0..seeds {
        let scene = IndoorManifest::generate_with_humans(
            first_seed.wrapping_add(offset as u64),
            layout,
            density,
            cameras,
            human_density,
        )?;
        super::validation::validate_layout(&scene)?;
        let layout_name = format!("{:?}", scene.layout);
        for (category, value) in [
            ("layout", layout_name.clone()),
            ("lighting", format!("{:?}", scene.lighting)),
            ("palette", scene.palette.to_string()),
            ("floor", scene.floor_style.to_string()),
            ("furniture", scene.furniture_style.to_string()),
            ("ceiling", scene.ceiling_style.to_string()),
            ("architecture", format!("{:?}", scene.architecture_style)),
            ("blinds", scene.blinds.to_string()),
            ("window_bays", scene.window_bays.to_string()),
        ] {
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
            ("room_width_m", scene.room_size.x),
            ("room_depth_m", scene.room_size.z),
            ("room_height_m", scene.room_size.y),
            ("sun_elevation_radians", scene.sun_elevation),
            ("sun_azimuth_radians", scene.sun_azimuth),
            ("light_kelvin", scene.light_kelvin),
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
        for human in &scene.humans {
            for (name, value) in [
                ("human_stature_m", human.stature),
                ("human_build", human.build),
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
            let length = camera.start.distance(camera.end);
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
                let position = camera.start.lerp(camera.end, step as f32 / 32.0);
                heat(
                    &mut report,
                    "camera_path",
                    position.x / scene.room_size.x + 0.5,
                    position.z / scene.room_size.z + 0.5,
                );
            }
        }
    }
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
