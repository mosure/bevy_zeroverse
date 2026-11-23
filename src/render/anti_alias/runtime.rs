use super::{indoor_fxaa, ConfigurePreviewAa, PreviewAa};
use crate::{
    app::BevyZeroverseConfig,
    camera::{lens::LensCamera, ProcessedEditorCameraMarker, ZeroverseCamera},
    render::RenderMode,
    sample::SamplerState,
    scene::SceneLoadedEvent,
};
use bevy::{
    anti_alias::{
        fxaa::Fxaa,
        smaa::{Smaa, SmaaPreset},
        taa::TemporalAntiAliasing,
    },
    core_pipeline::{
        core_3d::CORE_3D_DEPTH_FORMAT,
        prepass::{
            DepthPrepass, MotionVectorPrepass, MOTION_VECTOR_PREPASS_FORMAT, NORMAL_PREPASS_FORMAT,
        },
    },
    ecs::{query::QueryData, system::EntityCommands},
    prelude::*,
    render::{
        camera::{MipBias, TemporalJitter},
        renderer::{RenderAdapter, RenderDevice},
        view::Msaa,
    },
};

#[cfg(test)]
mod tests;

pub(crate) fn install(app: &mut App) {
    app.add_systems(
        PostUpdate,
        configure
            .in_set(ConfigurePreviewAa)
            .after(crate::sample::camera_activity::CaptureCameraGate)
            .before(crate::camera::lens::LensPhotometry)
            .before(bevy::camera::CameraUpdateSystems),
    );
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Filter {
    Off,
    Fxaa,
    Smaa(SmaaPreset),
    Taa,
    Msaa(Msaa),
}

impl Filter {
    fn samples(self) -> Msaa {
        if let Self::Msaa(msaa) = self {
            msaa
        } else {
            Msaa::Off
        }
    }
}

/// Adapter-specific sample counts are legal only when enabled on the device.
/// In particular WebGPU guarantees 4x on the required formats, but not 2x/8x.
struct SampleCounts(wgpu::TextureFormatFeatureFlags);
impl SampleCounts {
    fn detect(adapter: Option<&RenderAdapter>, device: Option<&RenderDevice>) -> Self {
        let features = device.map_or(wgpu::Features::empty(), RenderDevice::features);
        let flags = [
            wgpu::TextureFormat::Rgba16Float,
            CORE_3D_DEPTH_FORMAT,
            NORMAL_PREPASS_FORMAT,
            MOTION_VECTOR_PREPASS_FORMAT,
        ]
        .into_iter()
        .map(|format| {
            if let Some(adapter) = adapter.filter(|_| {
                features.contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
            }) {
                adapter.get_texture_format_features(format).flags
            } else {
                format.guaranteed_format_features(features).flags
            }
        })
        .fold(wgpu::TextureFormatFeatureFlags::all(), |a, b| a & b);
        Self(flags)
    }
    fn resolve(&self, requested: u32) -> Filter {
        [Msaa::Sample8, Msaa::Sample4, Msaa::Sample2]
            .into_iter()
            .find(|n| n.samples() <= requested && self.0.sample_count_supported(n.samples()))
            .map_or(Filter::Smaa(SmaaPreset::High), Filter::Msaa)
    }
}

fn requested(
    config: &BevyZeroverseConfig,
    mode: &RenderMode,
    editor: bool,
    sampling: bool,
) -> Option<PreviewAa> {
    let visible = if editor {
        !config.camera_grid && (*mode == RenderMode::Color || *mode == RenderMode::CoVisibility)
    } else {
        config.camera_grid && *mode == RenderMode::Color && !config.image_copiers && !sampling
    };
    if config.headless || !config.editor || config.room_schematic || !visible {
        return None;
    }
    match config.preview_aa {
        PreviewAa::Auto if !editor && !config.kiosk => None,
        PreviewAa::Auto => Some(PreviewAa::Taa),
        other => Some(other),
    }
}

fn select(
    request: PreviewAa,
    angular: bool,
    cropped: bool,
    msaa: impl FnOnce(u32) -> Filter,
) -> Filter {
    match request {
        PreviewAa::Taa | PreviewAa::Auto if angular => Filter::Smaa(SmaaPreset::High),
        // Bevy 0.20 TAA samples full-target UVs through a viewport-sized
        // triangle. On an inset viewport this re-applies the inset to RGB.
        PreviewAa::Taa | PreviewAa::Auto if cropped => Filter::Smaa(SmaaPreset::Ultra),
        PreviewAa::Taa | PreviewAa::Auto => Filter::Taa,
        PreviewAa::Smaa => Filter::Smaa(SmaaPreset::High),
        PreviewAa::SmaaUltra => Filter::Smaa(SmaaPreset::Ultra),
        PreviewAa::Off => Filter::Off,
        PreviewAa::Fxaa => Filter::Fxaa,
        PreviewAa::Msaa2 => msaa(2),
        PreviewAa::Msaa4 => msaa(4),
        PreviewAa::Msaa8 => msaa(8),
    }
}

/// Bevy inserts these components when TAA is added, but keeps them when it
/// is removed. Preserve any prepasses already owned by capture or shading.
#[derive(Default)]
struct AddedForTaa {
    depth: bool,
    motion: bool,
    jitter: bool,
    mip_bias: bool,
}
impl AddedForTaa {
    fn release(&self, entity: &mut EntityCommands) {
        if self.depth {
            entity.remove::<DepthPrepass>();
        }
        if self.motion {
            entity.remove::<MotionVectorPrepass>();
        }
        if self.jitter {
            entity.remove::<TemporalJitter>();
        }
        if self.mip_bias {
            entity.remove::<MipBias>();
        }
    }
}

#[derive(Clone, Copy)]
struct HistoryView {
    translation: Vec3,
    rotation: Quat,
    projection: Mat4,
    size: Option<UVec2>,
}

impl HistoryView {
    fn cut_from(&self, previous: &Self) -> bool {
        self.translation.distance_squared(previous.translation) > 1.
            || self.rotation.dot(previous.rotation).abs() < (22.5_f32.to_radians()).cos()
            || self.projection != previous.projection
            || self.size != previous.size
    }
}

#[derive(Component)]
pub(crate) struct ManagedPreviewAa {
    filter: Filter,
    added: AddedForTaa,
    history: Option<HistoryView>,
}

impl ManagedPreviewAa {
    pub(crate) fn label(&self) -> &'static str {
        match self.filter {
            Filter::Off | Filter::Msaa(Msaa::Off) => "Off",
            Filter::Fxaa => "FXAA",
            Filter::Smaa(SmaaPreset::Ultra) => "SMAA Ultra",
            Filter::Smaa(_) => "SMAA High",
            Filter::Taa => "TAA",
            Filter::Msaa(Msaa::Sample2) => "2× MSAA",
            Filter::Msaa(Msaa::Sample4) => "4× MSAA",
            Filter::Msaa(Msaa::Sample8) => "8× MSAA",
        }
    }
}

#[derive(QueryData)]
#[query_data(mutable)]
struct PreviewCamera {
    entity: Entity,
    samples: &'static Msaa,
    camera: &'static Camera,
    transform: &'static Transform,
    projection: &'static Projection,
    editor: Has<ProcessedEditorCameraMarker>,
    angular: Has<LensCamera>,
    depth: Has<DepthPrepass>,
    motion: Has<MotionVectorPrepass>,
    jitter: Has<TemporalJitter>,
    mip_bias: Has<MipBias>,
    taa: Option<&'static mut TemporalAntiAliasing>,
    managed: Option<&'static mut ManagedPreviewAa>,
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn configure(
    mut commands: Commands,
    config: Res<BevyZeroverseConfig>,
    mode: Res<RenderMode>,
    sampler: Option<Res<SamplerState>>,
    copiers: Query<(), With<crate::io::image_copy::ImageCopier>>,
    mut loaded: MessageReader<SceneLoadedEvent>,
    adapter: Option<Res<RenderAdapter>>,
    device: Option<Res<RenderDevice>>,
    mut counts: Local<Option<SampleCounts>>,
    mut cameras: Query<
        PreviewCamera,
        Or<(With<ProcessedEditorCameraMarker>, With<ZeroverseCamera>)>,
    >,
) {
    let scene_cut = loaded.read().count() != 0;
    let sampling = !copiers.is_empty() || sampler.is_some_and(|s| s.enabled);
    for mut view in &mut cameras {
        let filter = requested(&config, &mode, view.editor, sampling).map(|request| {
            let cropped = view.camera.viewport.as_ref().is_some_and(|vp| {
                vp.physical_position != UVec2::ZERO
                    || Some(vp.physical_size) != view.camera.physical_target_size()
            });
            select(request, view.angular, cropped, |n| {
                counts
                    .get_or_insert_with(|| {
                        SampleCounts::detect(adapter.as_deref(), device.as_deref())
                    })
                    .resolve(n)
            })
        });
        let history = (filter == Some(Filter::Taa)).then(|| HistoryView {
            translation: view.transform.translation,
            rotation: view.transform.rotation,
            projection: view.projection.get_clip_from_view(),
            size: view.camera.physical_viewport_size(),
        });
        let previous = view.managed.as_deref().map(|s| s.filter);
        if previous == filter {
            if let Some(filter) = filter {
                if *view.samples != filter.samples() {
                    commands.entity(view.entity).insert(filter.samples());
                }
                if let (Some(next), Some(state), Some(taa)) = (
                    history,
                    view.managed.as_deref_mut(),
                    view.taa.as_deref_mut(),
                ) {
                    let cut = state
                        .history
                        .replace(next)
                        .is_none_or(|p| next.cut_from(&p));
                    if cut || scene_cut {
                        taa.reset = true;
                    }
                }
            }
            continue;
        }
        let mut entity = commands.entity(view.entity);
        if let Some(previous) = &view.managed {
            if previous.filter == Filter::Taa {
                entity.remove::<TemporalAntiAliasing>();
                previous.added.release(&mut entity);
            }
        }
        // Updating Msaa also lets the indoor camera system restore its exact
        // FXAA/SSAO configuration when the camera returns to calibrated capture.
        entity
            .remove::<(Smaa, Fxaa)>()
            .insert(filter.map_or(Msaa::Off, Filter::samples));
        let Some(filter) = filter else {
            entity.remove::<ManagedPreviewAa>();
            continue;
        };
        let mut added = AddedForTaa::default();
        match filter {
            Filter::Taa => {
                added = AddedForTaa {
                    depth: !view.depth,
                    motion: !view.motion,
                    jitter: !view.jitter,
                    mip_bias: !view.mip_bias,
                };
                entity.insert(TemporalAntiAliasing::default());
            }
            Filter::Smaa(preset) => {
                entity.insert(Smaa { preset });
            }
            Filter::Fxaa => {
                entity.insert(indoor_fxaa());
            }
            Filter::Off | Filter::Msaa(_) => {}
        }
        entity.insert(ManagedPreviewAa {
            filter,
            added,
            history,
        });
    }
}
