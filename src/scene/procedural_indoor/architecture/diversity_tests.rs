//! Bounded CPU qualification, suitable for running alongside training.
use super::*;
use crate::scene::procedural_indoor::{layout::IndoorLayout, validation};
use std::time::Instant;

#[test]
#[ignore = "64-room architecture distribution and mesh audit; writes a local qualification artifact"]
fn architecture_population_qualification() {
    let directory = std::path::Path::new("out/architecture_diversity");
    std::fs::create_dir_all(directory).unwrap();
    let mut samples = Vec::new();
    let start = Instant::now();
    for seed in (0..64).chain([
        202, 1013005, 65904936, 65991960, 66120987, 66248449, 66242909, 66258278, 66258579,
        66274996, 66275068, 66283662, 66283756,
    ]) {
        let begin = Instant::now();
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 3, 0.25).unwrap();
        let manifest_ms = begin.elapsed().as_secs_f64() * 1000.;
        validation::validate_layout(&scene).unwrap();
        let begin = Instant::now();
        let a = architecture(&scene);
        let construction_ms = begin.elapsed().as_secs_f64() * 1000.;
        let vertices: usize = a.parts.values().map(|g| g.positions.len()).sum();
        let triangles: usize = a.parts.values().map(|g| g.indices.len() / 3).sum();
        let batches = a.parts.len();
        drop(a);
        let stats = validation::validate_geometry(&scene).unwrap();
        assert_eq!(stats.degenerate_triangles, 0, "seed {seed}");
        samples.push(serde_json::json!({"seed":seed,"layout":scene.layout,"room_size":scene.room_size,
            "finishes":scene.program.as_ref().unwrap().finishes,"walls":scene.envelope.as_ref().unwrap().walls,
            "manifest_ms":manifest_ms,"construction_ms":construction_ms,
            "wall_thickness_m":scene.envelope.as_ref().unwrap().wall_thickness(),
            "heaters":scene.objects.iter().filter(|o|o.kind==ObjectKind::BaseboardHeater).collect::<Vec<_>>(),
            "architecture_vertices":vertices,"architecture_triangles":triangles,"architecture_batches":batches,
            "validated_total_triangles":stats.triangles,
            "objects":scene.objects.len(),"humans":scene.humans.len(),"cameras":scene.cameras.len()}));
    }
    let report = serde_json::json!({"generator_revision":super::super::layout::GENERATOR_VERSION,
        "qualification":"CPU manifests, exact meshes and annotation ownership; no GPU captures",
        "elapsed_seconds":start.elapsed().as_secs_f64(),"samples":samples});
    let parameters: Vec<_> = (0..1024)
        .map(|seed| {
            let trim = trim::TrimProgram::sample(seed);
            let interior = interior_glazing::InteriorGlazing::sample(seed);
            trim.validate().unwrap();
            interior.validate().unwrap();
            let thickness = facade::reveal::wall_thickness(seed);
            let (reveal, recess) = facade::reveal::RevealProgram::sample(seed, thickness, 0.08, 0.06);
            reveal.validate(0.08, recess).unwrap();
            serde_json::json!({"seed":seed,"trim":trim,"interior":interior,"reveal":reveal,"recess_m":recess,"frame_depth_m":0.08})
        })
        .collect();
    std::fs::write(
        directory.join("parameters.json"),
        serde_json::to_vec(&parameters).unwrap(),
    )
    .unwrap();
    std::fs::write(
        directory.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!(
        "64-room + 13 regression seed architecture audit passed in {:.2}s",
        start.elapsed().as_secs_f64()
    );
}
