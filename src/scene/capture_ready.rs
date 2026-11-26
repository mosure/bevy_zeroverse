//! Opt-in readiness of live indoor RGB textures, including optical resolves.
//! Add the plugin to a Zeroverse app with a renderer. Read the message once or
//! query the resource after `SceneCaptureReadyUpdate` for the current room.
use crate::{
    camera::{lens::LensCamera, CaptureCameraIndex, ZeroverseCamera},
    render::{ssaa::SsaaCamera, RenderMode},
    sample::{CaptureBlocker, CaptureReadiness},
    scene::{procedural_indoor::layout::IndoorManifest, RegenerateSceneEvent, ZeroverseSceneRoot},
};
use bevy::{camera::RenderTarget, prelude::*, render::RenderApp};
use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc,
};

pub(crate) mod render;
#[cfg(test)]
mod tests;

/// Logical camera indices to qualify together. The default follows every camera in the room.
#[derive(Resource, Clone, Default, Debug, PartialEq, Eq)]
pub struct SceneCaptureViews(Option<Vec<usize>>);
impl SceneCaptureViews {
    pub fn all() -> Self {
        Self::default()
    }
    /// Select a nonempty subset. An index outside the current room is reported in the state.
    pub fn only(indices: impl IntoIterator<Item = usize>) -> Result<Self, &'static str> {
        let mut indices: Vec<_> = indices.into_iter().collect();
        if indices.is_empty() {
            return Err("select at least one logical camera");
        }
        indices.sort_unstable();
        if indices.windows(2).any(|v| v[0] == v[1]) {
            return Err("duplicate logical camera index");
        }
        Ok(Self(Some(indices)))
    }
    pub(crate) fn includes(&self, index: usize) -> bool {
        self.0
            .as_ref()
            .is_none_or(|v| v.binary_search(&index).is_ok())
    }
}

#[derive(Clone, Debug)]
pub struct ReadyCameraTexture {
    pub index: usize,
    pub camera: Entity,
    /// The final optical texture, after fisheye projection, SSAA and RGB sensor processing.
    pub image: Handle<Image>,
}

/// Raised once per successfully qualified root and selection. The textures belong to
/// live cameras and may be written by later frames. This is an initialization fence.
#[derive(Message, Clone, Debug)]
pub struct SceneReadyForCapture {
    pub root: Entity,
    pub seed: u64,
    /// App-local ticket. A regeneration of the same seed receives a distinct ticket.
    pub generation: u64,
    /// Sorted by logical camera index.
    pub views: Vec<ReadyCameraTexture>,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneCaptureStage {
    #[default]
    WaitingForRoom,
    Preparing(CaptureBlocker),
    WaitingForColor,
    WaitingForCameras,
    CameraOutOfRange(usize),
    WaitingForDraw,
    Ready,
}

#[derive(Resource, Default)]
pub struct SceneCaptureState {
    root: Option<Entity>,
    retired: Option<Entity>,
    next_ticket: u64,
    intent: Option<Arc<Intent>>,
    ready: Option<SceneReadyForCapture>,
    stage: SceneCaptureStage,
}
impl SceneCaptureState {
    pub(crate) fn wants_editor_draw(&self) -> bool {
        self.stage == SceneCaptureStage::WaitingForCameras || self.needs_residency()
    }
    pub(crate) fn needs_residency(&self) -> bool {
        self.intent
            .as_ref()
            .is_some_and(|i| i.completion.load(Ordering::Acquire) == WAITING)
    }
    pub fn ready(&self) -> Option<&SceneReadyForCapture> {
        self.ready.as_ref()
    }
    pub fn stage(&self) -> SceneCaptureStage {
        self.stage
    }
    fn clear(&mut self, stage: SceneCaptureStage) {
        self.ready = None;
        self.intent = None;
        self.stage = stage;
    }
}

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SceneCaptureReadyUpdate;

pub struct SceneCaptureReadyPlugin;
impl Plugin for SceneCaptureReadyPlugin {
    fn build(&self, app: &mut App) {
        crate::sample::install_preparation_readiness(app);
        app.add_message::<SceneReadyForCapture>()
            .init_resource::<SceneCaptureState>()
            .init_resource::<SceneCaptureViews>()
            .add_systems(
                PostUpdate,
                update
                    .after(crate::sample::CaptureReadinessUpdate)
                    .after(bevy::camera::CameraUpdateSystems)
                    .in_set(SceneCaptureReadyUpdate),
            );
    }
    fn finish(&self, app: &mut App) {
        if app.get_sub_app(RenderApp).is_some() {
            crate::io::image_copy::readiness::install(app);
            render::install(app.sub_app_mut(RenderApp));
        }
    }
}

const WAITING: u8 = 0;
const SUBMITTED: u8 = 1;
const COMPLETE: u8 = 2;

struct Intent {
    receipt: SceneReadyForCapture,
    paths: Vec<Optics>,
    completion: Arc<AtomicU8>,
}
#[derive(PartialEq, Eq)]
struct Optics {
    raster: AssetId<Image>,
    faces: Option<[AssetId<Image>; 6]>,
    lens: Option<AssetId<Image>>,
    resolve: Option<AssetId<Image>>,
}

type Cameras<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static ChildOf,
        &'static CaptureCameraIndex,
        &'static ZeroverseCamera,
        &'static RenderTarget,
        Option<&'static LensCamera>,
        Option<&'static SsaaCamera>,
    ),
>;

fn optics<'a>(
    logical: &ZeroverseCamera,
    target: &'a RenderTarget,
    lens: Option<&'a LensCamera>,
    ssaa: Option<&'a SsaaCamera>,
) -> Option<(Optics, &'a Handle<Image>)> {
    let RenderTarget::Image(target) = target else {
        return None;
    };
    if logical.fisheye.is_some() != lens.is_some() {
        return None;
    }
    let raster = target.handle.id();
    if lens.is_some_and(|l| l.faces[0].id() != raster) {
        return None;
    }
    let intermediate = lens.map_or(&target.handle, |l| &l.output);
    if ssaa.is_some_and(|s| s.source.id() != intermediate.id()) {
        return None;
    }
    let out = ssaa.map_or(intermediate, |s| &s.output);
    Some((
        Optics {
            raster,
            faces: lens.map(|l| std::array::from_fn(|i| l.faces[i].id())),
            lens: lens.map(|l| l.output.id()),
            resolve: ssaa.map(|s| s.output.id()),
        },
        out,
    ))
}

#[allow(clippy::too_many_arguments)]
fn update(
    mut regenerate: MessageReader<RegenerateSceneEvent>,
    scene: Option<Res<IndoorManifest>>,
    roots: Query<Entity, (With<ZeroverseSceneRoot>, With<crate::ovoxel::OvoxelRegion>)>,
    mode: Res<RenderMode>,
    prep: Res<CaptureReadiness>,
    requested: Res<SceneCaptureViews>,
    cameras: Cameras,
    mut state: ResMut<SceneCaptureState>,
    mut events: MessageWriter<SceneReadyForCapture>,
) {
    if !regenerate.is_empty() {
        regenerate.clear();
        state.retired = state.root;
        state.clear(SceneCaptureStage::WaitingForRoom);
    }
    let root = roots.single().ok();
    if state.root != root {
        state.clear(SceneCaptureStage::WaitingForRoom);
    }
    state.root = root;
    // A rejected request can leave the previous root on screen. Surface its
    // failure even when no new root or manifest could be installed.
    if prep.blocker == Some(CaptureBlocker::Failed) {
        state.clear(SceneCaptureStage::Preparing(CaptureBlocker::Failed));
        return;
    }
    let (Some(root), Some(scene)) = (root, scene) else {
        state.clear(SceneCaptureStage::WaitingForRoom);
        return;
    };
    if state.retired == Some(root) {
        state.clear(SceneCaptureStage::WaitingForRoom);
        return;
    }
    state.retired = None;
    if let Some(b) = prep.blocker {
        state.clear(SceneCaptureStage::Preparing(b));
        return;
    }
    if prep.scene_seed != Some(scene.seed) {
        state.clear(SceneCaptureStage::WaitingForRoom);
        return;
    }
    if *mode != RenderMode::Color {
        state.clear(SceneCaptureStage::WaitingForColor);
        return;
    }
    let total = scene.cameras.len();
    if let Some(i) = requested
        .0
        .as_ref()
        .and_then(|v| v.last())
        .filter(|&&i| i >= total)
    {
        state.clear(SceneCaptureStage::CameraOutOfRange(*i));
        return;
    }
    let count = requested.0.as_ref().map_or(total, Vec::len);
    let selected = || {
        cameras
            .iter()
            .filter(|(_, owner, i, ..)| owner.parent() == root && requested.includes(i.0))
    };
    if count == 0 || selected().count() != count {
        state.clear(SceneCaptureStage::WaitingForCameras);
        return;
    }
    let matches = state.intent.as_ref().is_some_and(|intent| {
        intent.receipt.seed == scene.seed
            && intent.receipt.views.len() == count
            && intent
                .receipt
                .views
                .iter()
                .zip(&intent.paths)
                .all(|(v, path)| {
                    cameras
                        .get(v.camera)
                        .is_ok_and(|(_, owner, i, l, t, f, s)| {
                            owner.parent() == root
                                && i.0 == v.index
                                && requested.includes(v.index)
                                && optics(l, t, f, s)
                                    .is_some_and(|(p, out)| p == *path && out == &v.image)
                        })
                })
    });
    if !matches {
        state.clear(SceneCaptureStage::WaitingForCameras);
        let mut views = Vec::with_capacity(count);
        for (camera, _, i, l, t, f, s) in selected() {
            let Some((path, image)) = optics(l, t, f, s) else {
                return;
            };
            if views
                .iter()
                .any(|(v, _): &(ReadyCameraTexture, Optics)| &v.image == image)
            {
                return;
            }
            views.push((
                ReadyCameraTexture {
                    camera,
                    index: i.0,
                    image: image.clone(),
                },
                path,
            ));
        }
        views.sort_unstable_by_key(|(v, _)| v.index);
        if views.windows(2).any(|vv| vv[0].0.index == vv[1].0.index)
            || views.last().is_some_and(|(v, _)| v.index >= total)
        {
            return;
        }
        state.next_ticket = state.next_ticket.wrapping_add(1);
        let (views, paths) = views.into_iter().unzip();
        state.intent = Some(Arc::new(Intent {
            receipt: SceneReadyForCapture {
                root,
                seed: scene.seed,
                generation: state.next_ticket,
                views,
            },
            paths,
            completion: Arc::default(),
        }));
        state.stage = SceneCaptureStage::WaitingForDraw;
    }
    if state.ready.is_none() {
        let Some(receipt) = state
            .intent
            .as_ref()
            .filter(|i| i.completion.load(Ordering::Acquire) == COMPLETE)
            .map(|i| i.receipt.clone())
        else {
            return;
        };
        events.write(receipt.clone());
        state.ready = Some(receipt);
        state.stage = SceneCaptureStage::Ready;
    }
}
