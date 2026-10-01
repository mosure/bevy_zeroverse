use bevy_zeroverse_capture::{camera_color, mask_color, GeneratorIdentity, GENERATOR_VERSION};
use bevy_zeroverse_publication::{
    config::{safe_relative, Config, Paper, Protocol},
    dataset::Dataset,
    io, media, paper, pipeline,
    visibility::{decode, Stats},
};
use image::{DynamicImage, GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, io::Cursor, path::Path};

fn png(image: DynamicImage) -> Vec<u8> {
    let mut out = Cursor::new(Vec::new());
    image.write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}
fn stats(count: usize) -> Stats {
    Stats {
        valid_pixels: 1,
        shared_pixels: 1,
        peer_pixels: (0..count).map(|i| u64::from(i == 1)).collect(),
        cardinality_pixels: (0..count).map(|i| u64::from(i == 1)).collect(),
        shared_fraction_valid: 1.0,
        peer_fraction_valid: (0..count).map(|i| f64::from(i == 1)).collect(),
    }
}
#[test]
fn membership_is_numeric_and_background_is_distinct_from_unshared_surface() {
    let mask = png(DynamicImage::ImageLuma16(
        ImageBuffer::from_raw(3, 1, vec![2u16, 0, 0]).unwrap(),
    ));
    let valid = png(DynamicImage::ImageLuma8(
        GrayImage::from_raw(3, 1, vec![1, 1, 0]).unwrap(),
    ));
    let recorded = Stats {
        valid_pixels: 2,
        shared_pixels: 1,
        peer_pixels: vec![0, 1],
        cardinality_pixels: vec![1, 1],
        shared_fraction_valid: 0.5,
        peer_fraction_valid: vec![0.0, 0.5],
    };
    let plane = decode(&mask, &valid, None, [3, 1], 0, 2, &recorded).unwrap();
    assert_eq!(plane.stats.cardinality_pixels, vec![1, 1]);
    assert!(decode(&mask, &valid, None, [3, 1], 1, 2, &recorded).is_err());
    let lossy = png(DynamicImage::ImageLuma8(
        GrayImage::from_raw(3, 1, vec![2, 0, 0]).unwrap(),
    ));
    assert!(decode(&lossy, &valid, None, [3, 1], 0, 2, &recorded).is_err());
}
#[test]
fn high_membership_bits_and_source_exclusion_survive_png() {
    let mask = png(DynamicImage::ImageLuma16(
        ImageBuffer::from_raw(1, 1, vec![32770u16]).unwrap(),
    ));
    let valid = png(DynamicImage::ImageLuma8(
        GrayImage::from_raw(1, 1, vec![1]).unwrap(),
    ));
    let mut report = stats(16);
    report.peer_pixels[15] = 1;
    report.peer_fraction_valid[15] = 1.0;
    report.cardinality_pixels[1] = 0;
    report.cardinality_pixels[2] = 1;
    assert_eq!(
        decode(&mask, &valid, None, [1, 1], 0, 16, &report)
            .unwrap()
            .masks,
        [32770]
    );
    assert!(decode(&mask, &valid, None, [1, 1], 15, 16, &report).is_err());
}
#[test]
fn corrupt_validity_counts_and_previews_fail() {
    let mask = png(DynamicImage::ImageLuma16(
        ImageBuffer::from_raw(1, 1, vec![2u16]).unwrap(),
    ));
    let valid = png(DynamicImage::ImageLuma8(
        GrayImage::from_raw(1, 1, vec![1]).unwrap(),
    ));
    let mut report = stats(2);
    let preview = RgbImage::from_pixel(1, 1, Rgb([0, 0, 0]));
    assert!(decode(&mask, &valid, Some(&preview), [1, 1], 0, 2, &report).is_err());
    report.valid_pixels = 2;
    assert!(decode(&mask, &valid, None, [1, 1], 0, 2, &report).is_err());
    let invalid = png(DynamicImage::ImageLuma8(
        GrayImage::from_raw(1, 1, vec![0]).unwrap(),
    ));
    assert!(decode(&mask, &invalid, None, [1, 1], 0, 2, &stats(2)).is_err());
}
#[test]
fn legend_and_templates_are_closed_contracts() {
    let mut metadata = json!({"schema_version":1,"camera_count":4,"legend":(0..4).map(|i|json!({"bit":i,"camera_index":i,"mask":1u32<<i,"rgb8":camera_color(i,4)})).collect::<Vec<_>>()});
    bevy_zeroverse_publication::visibility::validate_metadata(&metadata, 4).unwrap();
    metadata["legend"][0]["camera_index"] = json!(3);
    assert!(bevy_zeroverse_publication::visibility::validate_metadata(&metadata, 4).is_err());
    assert!(media::fill("@KNOWN@ @TYPO@", &BTreeMap::from([("KNOWN", "ok".into())])).is_err());
    assert!(safe_relative(Path::new("../old-captures/gallery.json")).is_err());
}
#[test]
fn recipe_cannot_replace_sources_or_the_publication_work_directory() {
    let root = tempfile::tempdir().unwrap();
    let original = Config {
        schema_version: 1,
        captures: "out/publication/captures".into(),
        capture: protocol(),
        paper: Paper {
            source: "tex/bevy_zeroverse.tex".into(),
        },
        references: vec![],
    };
    fs::write(
        root.path().join("publication.toml"),
        toml::to_string(&original).unwrap(),
    )
    .unwrap();
    Config::load(root.path()).unwrap();
    for name in ["src", "www/project", "tex", "out", "out/publication"] {
        let mut config = original.clone();
        config.captures = name.into();
        fs::write(
            root.path().join("publication.toml"),
            toml::to_string(&config).unwrap(),
        )
        .unwrap();
        assert!(Config::load(root.path()).is_err(), "accepted cache {name}");
    }
    let mut config = original;
    config.paper.source = "tex/unsupported.tex".into();
    fs::write(
        root.path().join("publication.toml"),
        toml::to_string(&config).unwrap(),
    )
    .unwrap();
    assert!(Config::load(root.path()).is_err());
}
#[test]
fn paper_dependency_closure_is_complete_and_rejects_escape() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::create_dir(root.join("generated")).unwrap();
    fs::write(root.join("arxiv.sty"), "").unwrap();
    fs::write(root.join("references.bib"), "").unwrap();
    fs::write(
        root.join("main.tex"),
        "\\input{generated/table}\\includegraphics{generated/view.png}",
    )
    .unwrap();
    fs::write(root.join("generated/table.tex"), "table").unwrap();
    fs::write(root.join("generated/view.png"), "image").unwrap();
    assert_eq!(
        paper::closure(root, Path::new("main.tex")).unwrap().len(),
        5
    );
    fs::remove_file(root.join("generated/view.png")).unwrap();
    assert!(paper::closure(root, Path::new("main.tex")).is_err());
    fs::write(root.join("main.tex"), "\\input{../secret}").unwrap();
    assert!(paper::closure(root, Path::new("main.tex")).is_err());
}
#[test]
fn install_failure_rolls_back_previously_published_files() {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir_all(root.path().join("out/publication")).unwrap();
    let stage = tempfile::tempdir_in(root.path()).unwrap();
    fs::write(root.path().join("a.txt"), "original").unwrap();
    fs::write(stage.path().join("a.txt"), "new").unwrap();
    let entries = BTreeMap::from([
        ("a.txt".into(), "unused".into()),
        ("missing.txt".into(), "unused".into()),
    ]);
    assert!(pipeline::install(root.path(), stage.path(), &entries).is_err());
    assert_eq!(
        fs::read_to_string(root.path().join("a.txt")).unwrap(),
        "original"
    );
    assert!(!root.path().join("missing.txt").exists());
}

fn protocol() -> Protocol {
    Protocol {
        seed: 0,
        audit_rooms: 4,
        rendered_rooms: 4,
        cameras: 4,
        width: 64,
        height: 64,
        playback_steps: 2,
        density: 0.65,
        human_density: 0.25,
        gi_rays: 1024,
    }
}
fn fixture(root: &Path) -> (GeneratorIdentity, Protocol) {
    let identity = GeneratorIdentity {
        schema_version: 1,
        crate_version: "0.26.0".into(),
        generator_version: GENERATOR_VERSION,
        source_sha256: "f".repeat(64),
    };
    let p = protocol();
    io::write(&root.join("generator_identity.json"), &identity).unwrap();
    io::write(&root.join("metrics.json"),&json!({"schema_version":11,"generator_version":GENERATOR_VERSION,"scenes":4,"image_size":[64,64],"density":0.65,"human_density":0.25,"numeric":{},"object_counts_per_scene":{"main/Chair":{"0":4},"main/Person":{"0":4}},"heatmap_grid_size":2,"placement_heatmaps":{"camera_path":[1,1,1,1]}})).unwrap();
    io::write(
        &root.join("distribution.json"),
        &json!({"invalid_seeds":[],"first_seed":0,"seeds":4,"cameras_per_scene":4}),
    )
    .unwrap();
    io::write(&root.join("render_selection.json"),&json!({"identity":identity,"run_id":"one-run","quality":"Auto","co_visibility":true,"selected_seeds":[0,1,2,3],"playback_steps":2,"gi_rays":1024,"diffuse_gi_enabled":true,"capture_engine":"test"})).unwrap();
    io::write(&root.join("run_complete.json"),&json!({"identity":identity,"run_id":"one-run","selected_seeds":[0,1,2,3],"captured_scenes":4})).unwrap();
    let metadata = json!({"schema_version":1,"camera_count":4,"legend":(0..4).map(|i|json!({"bit":i,"camera_index":i,"mask":1u32<<i,"rgb8":camera_color(i,4)})).collect::<Vec<_>>()});
    let mut rows = String::new();
    for seed in 0..4 {
        let envelope = json!({"footprint":[[-2,-2],[2,-2],[2,2],[-2,2]],"ceiling_drop":[0,0],"floor_patches":[],"pillars":[],"walls":[],"mezzanine":null});
        let row = json!({"seed":seed,"room_size":[4,3,4],"envelope":envelope,"partitions":[]});
        rows.push_str(&format!("{row}\n"));
        let folder = root.join(format!("seed_{seed:06}"));
        fs::create_dir_all(&folder).unwrap();
        io::write(&folder.join("manifest.json"),&json!({"seed":seed,"generator_version":GENERATOR_VERSION,"world_yaw":0.0,"room_size":[4,3,4],"envelope":envelope,"cameras":[0,1,2,3],"objects":[],"humans":[],"layout":"Conference"})).unwrap();
        let mut views = Vec::new();
        for step in 0..2 {
            for camera in 0..4 {
                let index = step * 4 + camera;
                let peer = (camera + 1) % 4;
                let mask = 1u16 << peer;
                let report = Stats {
                    valid_pixels: 4096,
                    shared_pixels: 4096,
                    peer_pixels: (0..4).map(|i| if i == peer { 4096 } else { 0 }).collect(),
                    cardinality_pixels: vec![0, 4096, 0, 0],
                    shared_fraction_valid: 1.0,
                    peer_fraction_valid: (0..4).map(|i| f64::from(i == peer)).collect(),
                };
                views.push(json!({"camera_index":camera,"step_index":step,"time":step,"pose_max_absolute_error":0.0,"world_from_view":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,1,0,1]],"fov_y":std::f32::consts::FRAC_PI_2,"fx_pixels":32,"fy_pixels":32,"near":0.1,"far":100,"semantic_colors":4,"semantic_pixel_counts":{"wall":4096},"co_visibility":report,"annotation_alignment":{"checked_pixels":4096,"depth_position_p99_metres":0,"reprojection_p99_pixels":0,"normal_length_max_error":0,"position_quantization_budget_p99_ratio":0}}));
                for mode in bevy_zeroverse_capture::PUBLICATION_MODES {
                    let color = if mode == "co_visibility" {
                        mask_color(mask, 4)
                    } else {
                        [127, 127, 127]
                    };
                    RgbImage::from_pixel(64, 64, Rgb(color))
                        .save(folder.join(format!("view_{index:02}_{mode}.png")))
                        .unwrap();
                }
                ImageBuffer::<Luma<u16>, Vec<u16>>::from_pixel(64, 64, Luma([mask]))
                    .save(folder.join(format!("view_{index:02}_co_visibility_mask.png")))
                    .unwrap();
                GrayImage::from_pixel(64, 64, Luma([1]))
                    .save(folder.join(format!("view_{index:02}_co_visibility_valid.png")))
                    .unwrap();
            }
        }
        io::write(&folder.join("capture.json"),&json!({"run_id":"one-run","seed":seed,"image_size":[64,64],"elapsed_seconds":1,"annotation_precision":"float32_geometry","capabilities":{"quality":"Auto","shadows":true,"ssao":true},"co_visibility_metadata":metadata,"views":views})).unwrap();
    }
    fs::write(root.join("architecture.jsonl"), rows).unwrap();
    (identity, p)
}
#[test]
fn complete_capture_fixture_passes_and_stale_completion_fails() {
    let temp = tempfile::tempdir().unwrap();
    let (id, p) = fixture(temp.path());
    assert_eq!(Dataset::load(temp.path(), &id, &p).unwrap().scenes.len(), 4);
    io::write(
        &temp.path().join("run_complete.json"),
        &json!({"run_id":"different-run"}),
    )
    .unwrap();
    assert!(Dataset::load(temp.path(), &id, &p).is_err());
}
#[test]
fn missing_annotation_and_duplicate_camera_cannot_publish() {
    let temp = tempfile::tempdir().unwrap();
    let (id, p) = fixture(temp.path());
    let file = temp.path().join("seed_000000/view_00_co_visibility.png");
    let bytes = fs::read(&file).unwrap();
    fs::remove_file(&file).unwrap();
    assert!(Dataset::load(temp.path(), &id, &p).is_err());
    fs::write(file, bytes).unwrap();
    let path = temp.path().join("seed_000000/capture.json");
    let mut capture: Value = io::read(&path).unwrap();
    capture["views"][1]["camera_index"] = json!(0);
    io::write(&path, &capture).unwrap();
    assert!(Dataset::load(temp.path(), &id, &p).is_err());
}
#[test]
fn geometry_mismatch_and_source_upgrade_cannot_relabel_captures() {
    let temp = tempfile::tempdir().unwrap();
    let (id, p) = fixture(temp.path());
    let mut next = id.clone();
    next.source_sha256 = "0".repeat(64);
    assert!(Dataset::load(temp.path(), &next, &p).is_err());
    let path = temp.path().join("seed_000000/manifest.json");
    let mut m: Value = io::read(&path).unwrap();
    m["envelope"]["footprint"][0][0] = json!(-1);
    io::write(&path, &m).unwrap();
    assert!(Dataset::load(temp.path(), &id, &p).is_err());
}
#[test]
fn invalid_alignment_fov_and_histogram_denominators_are_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let (id, p) = fixture(temp.path());
    let path = temp.path().join("seed_000000/capture.json");
    let mut c: Value = io::read(&path).unwrap();
    c["views"][0]["fx_pixels"] = json!(12);
    io::write(&path, &c).unwrap();
    assert!(Dataset::load(temp.path(), &id, &p).is_err());
    c["views"][0]["fx_pixels"] = json!(32);
    c["views"][0]["annotation_alignment"] = Value::Null;
    io::write(&path, &c).unwrap();
    assert!(Dataset::load(temp.path(), &id, &p).is_err());
    let path = temp.path().join("metrics.json");
    let mut metrics: Value = io::read(&path).unwrap();
    metrics["object_counts_per_scene"]["main/Chair"]["0"] = json!(3);
    io::write(&path, &metrics).unwrap();
    assert!(Dataset::load(temp.path(), &id, &p).is_err());
}
#[test]
fn archive_bytes_are_reproducible() {
    let temp = tempfile::tempdir().unwrap();
    let entries = BTreeMap::from([("a.txt".into(), b"test".to_vec())]);
    io::archive(&temp.path().join("one.zip"), &entries).unwrap();
    io::archive(&temp.path().join("two.zip"), &entries).unwrap();
    assert_eq!(
        fs::read(temp.path().join("one.zip")).unwrap(),
        fs::read(temp.path().join("two.zip")).unwrap()
    );
}

#[test]
fn float32_geometry_roundtrip_preserves_integers_exactly() {
    let a = json!(0.32754459977149963);
    let b = json!(0.3275445997714996);
    assert!(bevy_zeroverse_publication::dataset::same_f32(&a, &b));
    assert!(!bevy_zeroverse_publication::dataset::same_f32(
        &json!(16777216u64),
        &json!(16777217u64)
    ));
}
#[test]
fn malformed_histogram_reports_an_error_without_panicking() {
    let temp = tempfile::tempdir().unwrap();
    let (id, p) = fixture(temp.path());
    let path = temp.path().join("metrics.json");
    let mut m: Value = io::read(&path).unwrap();
    m["numeric"] = json!({"example":{"count":2,"bin_edges":[0,1,2],"bin_counts":["bad",1]}});
    io::write(&path, &m).unwrap();
    assert!(Dataset::load(temp.path(), &id, &p).is_err());
    m["numeric"]["example"]["bin_counts"] = json!([u64::MAX, 1]);
    io::write(&path, &m).unwrap();
    assert!(Dataset::load(temp.path(), &id, &p).is_err());
}
#[test]
fn a_rendered_camera_must_match_its_planned_pose() {
    let temp = tempfile::tempdir().unwrap();
    let (id, p) = fixture(temp.path());
    let path = temp.path().join("seed_000000/capture.json");
    let mut c: Value = io::read(&path).unwrap();
    c["views"][0]["pose_max_absolute_error"] = json!(0.5);
    io::write(&path, &c).unwrap();
    assert!(Dataset::load(temp.path(), &id, &p).is_err());
}

#[test]
fn page_counts_and_examples_follow_the_recipe_instead_of_fixed_seeds() {
    let temp = tempfile::tempdir().unwrap();
    let (id, p) = fixture(temp.path());
    let data = Dataset::load(temp.path(), &id, &p).unwrap();
    fs::create_dir_all(temp.path().join("www/project")).unwrap();
    let scenes:Vec<_>=data.scenes.iter().map(|s|json!({"seed":s.capture.seed,"activity":"Conference","features":[],"area_m2":16,"roof_pitch_degrees":0,"frames":[{"views":s.capture.views[..4].iter().map(|v|json!({"camera":v,"images":{"color":"view.webp"}})).collect::<Vec<_>>()}]})).collect();
    let gallery = json!({"scenes":scenes,"featured":(0..4).map(|seed|json!({"seed":seed,"title":"Room program"})).collect::<Vec<_>>(),"audit_rooms":4,"rendered_views":32,"feature_room_counts":data.summary["feature_room_counts"],"co_visibility":{"legend":data.scenes[0].capture.co_visibility_metadata["legend"],"statistics":data.summary["co_visibility"]},"annotation_metrics":data.summary["annotation_metrics"],"semantic_coverage":data.summary["semantic_coverage"],"minimum_semantic_classes":4,"generator_identity":id,"quantized_structural_signatures":1});
    media::page(&gallery, temp.path(), &p).unwrap();
    let html = fs::read_to_string(temp.path().join("www/project/index.html")).unwrap();
    assert!(html.contains("4 of 4 captured rooms") && html.contains("float32 alignment"));
    assert!(
        html.contains("Top: seed 0, cameras 0 and 3. Bottom: seed 1 camera 0, seed 2 camera 1.")
    );
    assert!(html.contains("indoor_seed=0") && !html.contains("data-architecture-seed=\"7\""));
    assert!(html.contains("4-program / 4-rendered-room architectural audit"));
    // The independently frozen reference population remains 512 rooms.
    assert!(html.contains("reference population includes 512 distinct rendered rooms"));
}
