//! CPU oracles for the browser runner. No GPU or network is required to export.
//! The runner checks texture copies and every mip, not just shader compilation.
use super::*;

#[test]
#[ignore = "exports CPU material oracles for scripts/validate_material_webgpu.mjs"]
fn export_webgpu_material_fixtures() {
    let directory = std::env::var_os("ZEROVERSE_MATERIAL_FIXTURES")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "out/material_webgpu/fixtures".into());
    std::fs::create_dir_all(&directory).unwrap();
    let mut cases = Vec::new();
    for seed in [1, 202, u64::MAX - 7] {
        let recipes = program::sample_with_floor(seed, 1).unwrap();
        // Alternate sizes and substrates in one submission, reusing the same
        // scratch. Chrome immediately before cloth detects stale metalness.
        let mut selected: Vec<_> = [
            Surface::Chrome,
            Surface::Fabric,
            Surface::FabricAlt,
            Surface::Leather,
            Surface::Floor,
            Surface::Metal,
            Surface::Plastic,
            Surface::Paint,
            Surface::Accent,
            Surface::Ceiling,
        ]
        .map(|s| recipes[s as usize].clone())
        .into();
        selected.push(super::super::super::human::knit_recipe(seed));
        // Complete-design rugs exercise the same GPU kernel as production,
        // including isolated motif components and quiet/borderless endpoints.
        for variant in 0..6 {
            let mut rug = super::super::super::rugs::recipe(
                &recipes[Surface::FabricAlt as usize],
                variant % 4,
            );
            let d = rug.carpet.as_mut().unwrap().rug.as_mut().unwrap();
            if variant < 4 {
                d.weights = [0.; 4];
                d.weights[variant] = 1.;
            }
            if variant == 4 {
                d.border_width = 0.;
                d.center = 0.;
                d.quiet = 0.98;
            }
            if variant == 5 {
                d.patchwork = 1.;
            }
            selected.push(rug);
        }
        for (variant, mut material) in selected.into_iter().enumerate() {
            if seed == u64::MAX - 7 {
                material.seed = seed;
            }
            let params = recipe::Recipe::new(&material, 1).unwrap();
            let n = params.meta[1];
            let maps = material.maps(1);
            let name = format!("{seed}-{variant}-{:?}", material.surface);
            let files = std::array::from_fn::<_, 3, _>(|map| {
                let filename = format!("{name}-{map}.rgba");
                std::fs::write(directory.join(&filename), maps[map].data.as_ref().unwrap())
                    .unwrap();
                filename
            });
            cases.push(serde_json::json!({
                "name": name,
                "recipe": bytemuck::cast_slice::<recipe::Recipe, u32>(&[params]),
                "size": n,
                "levels": kernel::levels(n).iter().map(|l| serde_json::json!({
                    "size": l.size, "offset": l.offset, "stride": l.stride,
                })).collect::<Vec<_>>(),
                "dispatches": kernel::dispatches(n).iter().map(|d|
                    bytemuck::cast_slice::<kernel::Dispatch, u32>(std::slice::from_ref(d)).to_vec()
                ).collect::<Vec<_>>(),
                "plane_bytes": kernel::plane_bytes(n),
                "expected": files,
                "metalness": if material.surface == Surface::Chrome { 255 } else { 0 },
            }));
        }
    }
    // Preserve the f32 threshold table exactly, including its infinity sentinel.
    std::fs::write(
        directory.join("srgb.bin"),
        bytemuck::cast_slice(kernel::color_table()),
    )
    .unwrap();
    std::fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema": 1, "entries": kernel::ENTRIES, "max_size": kernel::MAX_SIZE,
            "scratch_bytes": {
                "pixels": kernel::plane_bytes(kernel::MAX_SIZE) * 3,
                "heights": kernel::MAX_SIZE * kernel::MAX_SIZE * 4,
                "moments": (kernel::MAX_SIZE * kernel::MAX_SIZE * 4 / 3 + 1) * 16,
            },
            "cases": cases,
        }))
        .unwrap(),
    )
    .unwrap();
    eprintln!(
        "wrote {} browser material fixtures to {}",
        cases.len(),
        directory.display()
    );
}
