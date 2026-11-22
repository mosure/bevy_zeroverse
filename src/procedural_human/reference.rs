//! One lossless AnnyBody reference shared by the viewer, offline geometry and capture workers.
use anyhow::{Context, Result};
use bevy::{
    prelude::{debug, error, Commands, Local, Res},
    tasks::{futures::check_ready, AsyncComputeTaskPool, Task},
};
use bevy_burn_human::BurnHumanAssets;
use burn_human::AnnyBody;
use once_cell::sync::OnceCell;
use std::{io::Read, sync::Arc};

const TENSORS: &[u8] =
    include_bytes!("../../assets/embedded/burn_human/fullbody_default.safetensors.zst");
const METADATA: &[u8] =
    include_bytes!("../../assets/embedded/burn_human/fullbody_default.meta.json");
static REFERENCE: OnceCell<BurnHumanAssets> = OnceCell::new();
type Loaded = Result<&'static BurnHumanAssets>;
#[cfg(test)]
static DECODES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn unpack() -> Result<Vec<u8>> {
    let mut decoder = ruzstd::decoding::StreamingDecoder::new(TENSORS)?;
    let mut tensors = Vec::with_capacity(usize::try_from(decoder.decoder.content_size())?);
    decoder.read_to_end(&mut tensors)?;
    Ok(tensors)
}

fn load() -> Result<AnnyBody> {
    #[cfg(test)]
    DECODES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // Retain the native loose-reference override for existing capture installations.
    #[cfg(not(target_family = "wasm"))]
    if std::env::var_os("BEVY_ASSET_ROOT").is_some() {
        let root = crate::asset::asset_root().join("burn_human");
        let meta = root.join("fullbody_default.meta.json");
        let tensors = root.join("fullbody_default.safetensors");
        if meta.is_file() || tensors.is_file() {
            return AnnyBody::from_reference_paths(tensors, meta)
                .with_context(|| format!("reading AnnyBody override from {}", root.display()));
        }
    }
    AnnyBody::from_reference_bytes(&unpack()?, METADATA).context("decoding bundled AnnyBody")
}

pub(crate) fn get() -> Loaded {
    REFERENCE.get_or_try_init(|| {
        let body = load()?;
        debug!("Decoded session AnnyBody reference");
        Ok(BurnHumanAssets {
            faces: Arc::new(body.faces_quads().clone()),
            uvs: Arc::new(body.metadata().static_data.texture_coordinates.clone()),
            body: Arc::new(body),
        })
    })
}

#[cfg(feature = "human_motion")]
pub(crate) fn installed() -> Option<Arc<AnnyBody>> {
    REFERENCE.get().map(|assets| Arc::clone(&assets.body))
}

pub(super) fn hydrate(
    mut commands: Commands,
    demand: Res<crate::asset::SceneAssetDemand>,
    existing: Option<Res<BurnHumanAssets>>,
    mut task: Local<Option<Task<Loaded>>>,
    mut failed: Local<bool>,
) {
    if existing.is_some() || !demand.humans || *failed {
        return;
    }
    let result = if let Some(assets) = REFERENCE.get() {
        Ok(assets)
    } else {
        let job = task.get_or_insert_with(|| AsyncComputeTaskPool::get().spawn(async { get() }));
        let Some(result) = check_ready(job) else {
            return;
        };
        result
    };
    *task = None;
    match result {
        Ok(assets) => commands.insert_resource(BurnHumanAssets {
            faces: Arc::clone(&assets.faces),
            uvs: Arc::clone(&assets.uvs),
            body: Arc::clone(&assets.body),
        }),
        Err(error) => {
            let message = format!("AnnyBody reference failed to load: {error:#}");
            error!("{message}");
            commands.insert_resource(crate::sample::CaptureFailure(Some(message)));
            *failed = true;
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn bundled_tensors_match_the_original_fullbody_reference() {
        let tensors = super::unpack().unwrap();
        assert_eq!(tensors.len(), 56_774_740);
        assert_eq!(
            bevy_zeroverse_capture::provenance::sha(&tensors),
            "dc2db51d9ccac2a50664f99e6843f54f319f23ca5ad50d26b6648f3764261472"
        );
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn concurrent_callers_people_and_renderer_worlds_decode_once() {
        use super::*;
        use crate::{
            asset::SceneAssetDemand,
            scene::procedural_indoor::{
                humans,
                layout::{IndoorLayout, IndoorManifest},
            },
        };
        use bevy::prelude::{App, Update};
        use std::sync::{atomic::Ordering, Barrier};

        let gate = Barrier::new(8);
        std::thread::scope(|scope| {
            let jobs: Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        gate.wait();
                        let shared = get().unwrap();
                        for _ in 0..128 {
                            assert!(std::ptr::eq(shared, get().unwrap()));
                        }
                        shared
                    })
                })
                .collect();
            for job in jobs {
                assert!(std::ptr::eq(job.join().unwrap(), get().unwrap()));
            }
        });
        let shared = get().unwrap();
        let mut populated = 0;
        for seed in 70..78 {
            let room = IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 1.)
                .unwrap();
            let Some(person) = room.humans.first() else {
                continue;
            };
            assert!(!humans::build_human(person).parts.is_empty());
            let mut app = App::new();
            app.insert_resource(SceneAssetDemand {
                humans: true,
                ..Default::default()
            })
            .add_systems(Update, hydrate);
            app.update();
            let assets = app.world().resource::<BurnHumanAssets>();
            assert!(Arc::ptr_eq(&shared.body, &assets.body));
            assert!(Arc::ptr_eq(&shared.faces, &assets.faces));
            assert!(Arc::ptr_eq(&shared.uvs, &assets.uvs));
            populated += 1;
            if populated == 3 {
                break;
            }
        }
        assert_eq!(populated, 3);
        assert_eq!(DECODES.load(Ordering::Relaxed), 1);
    }
}
