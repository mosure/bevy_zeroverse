//! Bevy 0.20 selects SDR unconditionally in create_surfaces. Retain its window
//! extraction, camera graph and presentation. Own just the primary swap chain
//! while displaying HDR; the ordinary SDR and offscreen paths need no extra pass.
use super::{transfer::Transfer, BevyZeroverseConfig, DisplayOutput, DisplayStatus};
use bevy::{
    prelude::*,
    render::{
        renderer::{
            RenderAdapter, RenderDevice, RenderGraph, RenderGraphSystems, RenderInstance,
            RenderQueue,
        },
        view::{
            cleanup_view_targets_for_resize, clear_view_attachments,
            window::{create_surfaces, prepare_windows, ExtractedWindow, SurfaceData},
        },
        Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
    },
    window::{Monitor, PrimaryWindow, RawHandleWrapper, WindowPosition},
};
use wgpu::{SurfaceColorSpace as Space, TextureFormat as Format};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Output {
    format: Format,
    space: Space,
    entry: Option<&'static str>,
    status: DisplayOutput,
}

const OUTPUTS: [Output; 3] = [
    Output {
        format: Format::Rgba16Float,
        space: Space::ExtendedSrgbLinear,
        entry: None,
        status: DisplayOutput::ScRgb,
    },
    Output {
        format: Format::Rgba16Float,
        space: Space::ExtendedSrgb,
        entry: Some("extended_srgb"),
        status: DisplayOutput::ExtendedSrgb,
    },
    Output {
        format: Format::Rgb10a2Unorm,
        space: Space::Bt2100Pq,
        entry: Some("bt2100_pq"),
        status: DisplayOutput::Pq,
    },
];

fn choose(caps: &wgpu::SurfaceCapabilities, display: wgpu::DisplayHdrInfo) -> Option<Output> {
    let headroom = display.headroom.and_then(|h| h.potential.or(h.current));
    if headroom
        .map(|h| h > 1.)
        .or_else(|| display.coarse.and_then(|c| c.high_dynamic_range))
        == Some(false)
    {
        return None;
    }
    OUTPUTS.into_iter().find(|o| {
        caps.color_spaces(o.format)
            .contains(o.space.to_color_spaces().unwrap())
    })
}

#[derive(Resource, Default)]
struct Request {
    enabled: bool,
    monitor: Option<Entity>,
    status: DisplayStatus,
}

fn extract(
    mut request: ResMut<Request>,
    args: Extract<Res<BevyZeroverseConfig>>,
    status: Extract<Res<DisplayStatus>>,
    windows: Extract<Query<&Window, With<PrimaryWindow>>>,
    monitors: Extract<Query<(Entity, &Monitor)>>,
) {
    request.enabled =
        args.hdr && args.render_mode == crate::render::RenderMode::Color && !args.room_schematic;
    if status.is_changed() {
        request.status = (**status).clone();
    }
    if !request.enabled {
        return;
    }
    if let Ok(window) = windows.single() {
        let WindowPosition::At(top_left) = window.position else {
            return;
        };
        let center = top_left + window.resolution.physical_size().as_ivec2() / 2;
        request.monitor = monitors
            .iter()
            .find(|(_, m)| {
                let pos = center - m.physical_position;
                pos.min_element() >= 0 && pos.cmplt(m.physical_size().as_ivec2()).all()
            })
            .map(|(entity, _)| entity);
    }
}

// WebGPU objects on an atomics-enabled browser build are thread-affine.
#[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
type SurfaceState<'w> = ResMut<'w, State>;
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
type SurfaceState<'w> = NonSendMut<'w, State>;
#[cfg_attr(
    not(all(target_arch = "wasm32", target_feature = "atomics")),
    derive(Resource)
)]
#[derive(Default)]
struct State {
    attempted: Option<(Entity, Option<Entity>)>,
    pending: bool,
    owner: Option<Owner>,
}

struct Owner {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    transfer: Option<Transfer>,
    output: Option<wgpu::TextureView>,
    // Last to drop: keeps the raw window alive through surface destruction.
    handle: RawHandleWrapper,
}

pub(super) fn install(app: &mut App) {
    let Some(render) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    #[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
    render.world_mut().init_non_send::<State>();
    #[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
    render.init_resource::<State>();
    render
        .init_resource::<Request>()
        .add_systems(ExtractSchedule, extract)
        .add_systems(
            Render,
            handoff
                .in_set(RenderSystems::PrepareViews)
                .before(cleanup_view_targets_for_resize)
                .before(create_surfaces),
        )
        .add_systems(
            Render,
            acquire
                .in_set(RenderSystems::PrepareViews)
                .after(handoff)
                .after(cleanup_view_targets_for_resize)
                .after(clear_view_attachments)
                .before(create_surfaces)
                .before(prepare_windows),
        )
        .add_systems(RenderGraph, encode.in_set(RenderGraphSystems::Finish));
}

fn clear_frame(window: &mut ExtractedWindow) {
    window.swap_chain_texture_view = None;
    window.swap_chain_texture = None;
    window.swap_chain_texture_format = None;
    window.swap_chain_texture_view_format = None;
}

fn handoff(
    mut commands: Commands,
    request: Res<Request>,
    mut state: SurfaceState,
    mut windows: Query<(Entity, &mut ExtractedWindow), With<PrimaryWindow>>,
) {
    let Ok((entity, mut window)) = windows.single_mut() else {
        if state.attempted.is_some() {
            *state = State::default();
            request.status.report(DisplayOutput::Pending);
        }
        return;
    };
    let desired = request.enabled.then_some((entity, request.monitor));
    if desired == state.attempted {
        return;
    }
    let previous = state.attempted;
    state.attempted = desired;
    state.pending = request.enabled;
    request.status.report(DisplayOutput::Pending);
    // A rejected probe is already presenting through Bevy's SDR surface.
    if desired.is_none() && state.owner.is_none() {
        return;
    }
    clear_frame(&mut window);
    window.size_changed = true; // release Bevy's cached window ViewTargets before configure
    if let Some(owner) = state
        .owner
        .take()
        .filter(|_| previous.is_some_and(|p| p.0 == entity))
    {
        commands.entity(entity).insert(owner.handle);
    }
    commands.entity(entity).remove::<SurfaceData>();
}

#[allow(clippy::too_many_arguments)] // Window, wgpu device and optional screenshot are separate Bevy resources.
fn acquire(
    mut commands: Commands,
    shots: Res<super::screenshot::Shots>,
    request: Res<Request>,
    mut state: SurfaceState,
    mut windows: Query<
        (Entity, &mut ExtractedWindow, Option<&RawHandleWrapper>),
        With<PrimaryWindow>,
    >,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    _main_thread: bevy::ecs::system::NonSendMarker,
    instance: Res<RenderInstance>,
    adapter: Res<RenderAdapter>,
    device: Res<RenderDevice>,
) {
    if !request.enabled {
        return;
    }
    let Ok((entity, mut window, handle)) = windows.single_mut() else {
        return;
    };
    if let Some(handle) = handle.filter(|_| state.pending) {
        state.pending = false;
        let target = wgpu::SurfaceTargetUnsafe::RawHandle {
            raw_display_handle: Some(handle.get_display_handle()),
            raw_window_handle: handle.get_window_handle(),
        };
        // SAFETY: the render-world handle is cloned into Owner until after the
        // surface drops. This system runs on the window thread like Bevy's.
        let surface = unsafe { instance.create_surface_unsafe(target) };
        let owner = surface.ok().and_then(|surface| {
            let caps = surface.get_capabilities(&adapter);
            let output = choose(&caps, surface.display_hdr_info(&adapter))?;
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: output.format,
                color_space: output.space,
                width: window.physical_width,
                height: window.physical_height,
                present_mode: present_mode(window.present_mode),
                desired_maximum_frame_latency: window
                    .desired_maximum_frame_latency
                    .map_or(2, |v| v.get()),
                alpha_mode: caps
                    .alpha_modes
                    .first()
                    .copied()
                    .unwrap_or(wgpu::CompositeAlphaMode::Auto),
                view_formats: vec![],
            };
            let transfer = output.entry.map(|entry| {
                Transfer::new(
                    device.wgpu_device(),
                    [config.width, config.height],
                    config.format,
                    entry,
                )
            });
            device.configure_surface(&surface, &config);
            request.status.report(output.status);
            info!("Studio HDR window {:?} / {:?}", output.format, output.space);
            Some(Owner {
                surface,
                config,
                transfer,
                output: None,
                handle: handle.clone(),
            })
        });
        if owner.is_some() {
            // Exactly one swapchain owner. Bevy still extracts and presents this window.
            commands.entity(entity).remove::<RawHandleWrapper>();
        } else {
            request.status.report(DisplayOutput::Sdr);
            info!("Studio HDR requested; this display/adapter offers SDR");
        }
        state.owner = owner;
    }
    let Some(owner) = &mut state.owner else {
        return;
    };
    let size = [window.physical_width, window.physical_height];
    if size != [owner.config.width, owner.config.height] || window.present_mode_changed {
        clear_frame(&mut window);
        owner.output = None;
        [owner.config.width, owner.config.height] = size;
        owner.config.present_mode = present_mode(window.present_mode);
        device.configure_surface(&owner.surface, &owner.config);
        if let Some(t) = &mut owner.transfer {
            t.resize(device.wgpu_device(), size);
        }
    }
    if window.swap_chain_texture.is_some() {
        return;
    }
    owner.output = None;
    let frame = match owner.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(frame)
        | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
        wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
        wgpu::CurrentSurfaceTexture::Outdated => {
            device.configure_surface(&owner.surface, &owner.config);
            return;
        }
        error => {
            warn!("Studio HDR surface unavailable ({error:?}); returning to SDR");
            let owner = state.owner.take().unwrap();
            commands.entity(entity).insert(owner.handle);
            clear_frame(&mut window);
            request.status.report(DisplayOutput::Sdr);
            return;
        }
    };
    let composite = owner.config.color_space != Space::ExtendedSrgbLinear || shots.pending();
    if !composite {
        owner.transfer = None;
    }
    if composite && owner.transfer.is_none() {
        owner.transfer = Some(Transfer::new(
            device.wgpu_device(),
            size,
            owner.config.format,
            "linear_copy",
        ));
    }
    let output = frame.texture.create_view(&Default::default());
    window.swap_chain_texture_view = Some(
        if let Some(t) = owner.transfer.as_ref().filter(|_| composite) {
            owner.output = Some(output);
            t.view.clone()
        } else {
            output.into()
        },
    );
    window.swap_chain_texture_view_format = Some(Format::Rgba16Float);
    window.swap_chain_texture_format = Some(owner.config.format);
    window.swap_chain_texture = Some(frame.into());
}

fn present_mode(mode: bevy::window::PresentMode) -> wgpu::PresentMode {
    // These two wgpu policies include adapter-specific, supported fallbacks.
    match mode {
        bevy::window::PresentMode::Immediate
        | bevy::window::PresentMode::Mailbox
        | bevy::window::PresentMode::AutoNoVsync => wgpu::PresentMode::AutoNoVsync,
        _ => wgpu::PresentMode::AutoVsync,
    }
}

fn encode(
    mut state: SurfaceState,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    shots: Res<super::screenshot::Shots>,
) {
    let Some(owner) = &mut state.owner else {
        return;
    };
    if let (Some(transfer), Some(output)) = (&owner.transfer, owner.output.take()) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Studio display transfer"),
        });
        transfer.encode(&mut encoder, &output);
        queue.submit([encoder.finish()]);
        super::screenshot::read(&shots, &device, &queue, transfer.view.texture());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn output_requires_a_supported_format_color_space_pair_and_display_headroom() {
        let mut caps = wgpu::SurfaceCapabilities::default();
        let mut display = wgpu::DisplayHdrInfo::default();
        assert_eq!(choose(&caps, display.clone()), None);
        for option in OUTPUTS.into_iter().rev() {
            caps.format_capabilities.insert(
                0,
                wgpu::SurfaceFormatCapabilities {
                    format: option.format,
                    color_spaces: option.space.to_color_spaces().unwrap(),
                },
            );
            assert_eq!(choose(&caps, display.clone()), Some(option));
        }
        display.coarse = Some(wgpu::DisplayCoarseRange {
            high_dynamic_range: Some(false),
            ..default()
        });
        assert_eq!(choose(&caps, display.clone()), None);
        display.headroom = Some(wgpu::DisplayHeadroom {
            current: Some(1.),
            potential: Some(2.4),
            reference: None,
        });
        assert_eq!(choose(&caps, display), Some(OUTPUTS[0]));
    }
}
