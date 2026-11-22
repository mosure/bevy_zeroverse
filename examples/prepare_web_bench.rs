//! Browser CPU preparation diagnostic: full furnished rooms, no renderer or people.
//! Open with ?seed=400&rooms=8 after wasm-bindgen. Output stays in the test page.
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!("Build this diagnostic for wasm32-unknown-unknown with --features web");
}
#[cfg(target_arch = "wasm32")]
mod web {
    use bevy::prelude::*;
    use bevy_zeroverse::scene::procedural_indoor::{
        cameras::CameraSettings,
        gi::IndoorGiSettings,
        layout::IndoorManifest,
        preparation::{PreparedIndoor, StagedAssets},
        validation, IndoorQuality,
    };
    use wasm_bindgen::prelude::*;
    #[wasm_bindgen(
        inline_js = "export function parameters() { const p = new URLSearchParams(location.search); return JSON.stringify({ seed: Number(p.get('seed') ?? 400), rooms: Number(p.get('rooms') ?? 8) }); } export function started() { window.zeroverseBenchStart?.(); } export function finished(report) { window.zeroverseBenchDone?.(JSON.parse(report)); }"
    )]
    extern "C" {
        fn parameters() -> String;
        fn started();
        fn finished(report: &str);
    }
    fn hash(parts: impl IntoIterator<Item = impl AsRef<[u8]>>) -> String {
        let mut h = 0xcbf29ce484222325_u64;
        for part in parts {
            for &b in part.as_ref() {
                h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
            }
        }
        format!("{h:016x}")
    }
    async fn prepare(seed: u64) -> serde_json::Value {
        let start = bevy::platform::time::Instant::now();
        let mut scene =
            IndoorManifest::generate_with_humans(seed, Default::default(), 0.65, 0, 0.).unwrap();
        scene
            .resample_cameras(3, CameraSettings::default(), 1.)
            .unwrap();
        validation::validate_layout(&scene).unwrap();
        let layout_cameras_seconds = start.elapsed().as_secs_f64();
        let mut images = Assets::<Image>::default();
        let mut materials = Assets::<StandardMaterial>::default();
        let mut meshes = Assets::<Mesh>::default();
        let prepared = PreparedIndoor::build(
            scene,
            IndoorQuality::Auto,
            IndoorGiSettings::default(),
            vec![],
            StagedAssets::new(&images),
            StagedAssets::new(&materials),
            StagedAssets::new(&meshes),
        )
        .await;
        let seconds = start.elapsed().as_secs_f64();
        let timings = prepared.timings.clone();
        let manifest = hash([serde_json::to_vec(&prepared.manifest).unwrap()]);
        prepared.images.commit(&mut images);
        prepared.materials.commit(&mut materials);
        prepared.meshes.commit(&mut meshes);
        let mut mesh_hashes: Vec<_> = meshes
            .iter()
            .map(|(_, m)| {
                let mut parts: Vec<_> = m
                    .attributes()
                    .map(|(_, v)| v.get_bytes().to_vec())
                    .collect();
                let indices: Vec<u32> = m.indices().unwrap().iter().map(|i| i as u32).collect();
                parts.push(bytemuck::cast_slice(&indices).to_vec());
                hash(parts)
            })
            .collect();
        let mut image_hashes: Vec<_> = images
            .iter()
            .map(|(_, i)| hash([i.data.as_deref().unwrap_or_default()]))
            .collect();
        mesh_hashes.sort();
        image_hashes.sort();
        serde_json::json!({"seed":seed,"seconds":seconds,"layout_cameras_seconds":layout_cameras_seconds,"timings":timings,
            "mesh_count":meshes.len(),"image_count":images.len(),"material_count":materials.len(),
            "manifest":manifest,"meshes":hash(mesh_hashes),"images":hash(image_hashes)})
    }
    pub fn run() {
        console_error_panic_hook::set_once();
        let params: serde_json::Value = serde_json::from_str(&parameters()).unwrap();
        let seed = params["seed"].as_u64().unwrap();
        let rooms = params["rooms"].as_u64().unwrap().clamp(1, 64);
        bevy::tasks::AsyncComputeTaskPool::get_or_init(|| bevy::tasks::TaskPoolBuilder::new().build())
            .spawn(async move {
                prepare(seed).await;
                started();
                let mut records = Vec::new();
                for i in 0..rooms { records.push(prepare(seed+i).await); }
                finished(&serde_json::json!({"scope":"Browser CPU preparation, full material maps and furniture, human_density=0; excludes GPU upload, render, motion and transport to tensors. Hashing is outside each preparation timing, inside heartbeat observation.","records":records}).to_string());
            }).detach();
    }
}
#[cfg(target_arch = "wasm32")]
fn main() {
    web::run();
}
