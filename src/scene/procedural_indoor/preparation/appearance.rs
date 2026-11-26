use super::super::motion::{self, AppearancePlayback, LightSource};
use super::*;

/// Resolve secondary pages before upload. A screen with an animation gets a
/// private material; bezel optics and the shared initial texture are preserved.
pub(super) async fn bind(
    scene: &IndoorManifest,
    groups: &mut [Group],
    set: &IndoorMaterials,
    images: &mut StagedAssets<Image>,
    materials: &mut StagedAssets<StandardMaterial>,
) -> Option<AppearancePlayback> {
    let tracks = &scene.indoor_motion.as_ref()?.appearance;
    if tracks.is_empty() {
        return None;
    }
    let mut playback = AppearancePlayback::default();
    for group in groups {
        let Some((id, kind)) = group.object else {
            continue;
        };
        let Ok(index) = tracks.screens.binary_search_by_key(&id, |s| s.object_id) else {
            continue;
        };
        let change = tracks.screens[index];
        let Some((surface, device)) = motion::display(kind) else {
            continue;
        };
        let mut faces = group
            .parts
            .iter_mut()
            .filter(|p| p.surface == Some(surface));
        let Some(first) = faces.next() else { continue };
        let base = materials
            .get(&first.material)
            .expect("prepared screen material");
        let Some(previous) = base.emissive_texture.clone() else {
            continue;
        };
        let material = materials.add(base.clone());
        let next = images.add(materials::screens::next_page(
            change.initial_content_seed,
            device,
            change.next_page_seed,
        ));
        first.material = material.clone();
        for face in faces {
            face.material = material.clone();
        }
        playback.screen(change.at, material, [previous, next]);
        cooperate().await;
    }
    for light in &tracks.lights {
        let handle = match light.source {
            LightSource::Ceiling(id) => set.ceiling_emitter(id),
            LightSource::Object(id) => set.object_emitter(id),
        };
        if let Some(handle) = handle {
            let emission = materials
                .get(handle)
                .expect("prepared emitter material")
                .emissive;
            playback.aperture(light.curve, handle.clone(), emission);
        }
    }
    Some(playback)
}
