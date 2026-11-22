//! Bevy's window screenshot extractor only recognizes its own SDR SurfaceData.
//! Keep the public Screenshot/ScreenshotCaptured contract on HDR swap chains.
//! Read the display-linear composite before PQ or extended-sRGB encoding.
use super::{BevyZeroverseConfig, DisplayStatus};
use bevy::{
    asset::RenderAssetUsages,
    camera::RenderTarget,
    prelude::*,
    render::{
        renderer::{RenderDevice, RenderQueue},
        view::screenshot::{Captured, Capturing, Screenshot, ScreenshotCaptured},
        Extract, ExtractSchedule, RenderApp,
    },
    window::{PrimaryWindow, WindowRef},
};
use std::sync::{Arc, Mutex};

#[derive(Default)]
enum Stage {
    #[default]
    Queued,
    Submitted,
    Ready(Box<Image>),
    Failed,
}
#[derive(Component, Clone, Default)]
struct HdrScreenshot(Arc<Mutex<Stage>>);

#[derive(Resource, Default)]
pub(super) struct Shots(Vec<HdrScreenshot>);
impl Shots {
    pub fn pending(&self) -> bool {
        !self.0.is_empty()
    }
}

pub(super) fn install(app: &mut App) {
    app.add_systems(Last, requests);
    if let Some(render) = app.get_sub_app_mut(RenderApp) {
        render
            .init_resource::<Shots>()
            .add_systems(ExtractSchedule, extract);
    }
}

fn requests(
    mut commands: Commands,
    config: Res<BevyZeroverseConfig>,
    display: Res<DisplayStatus>,
    primary: Query<Entity, With<PrimaryWindow>>,
    requested: Query<(Entity, &Screenshot), Without<Capturing>>,
    pending: Query<(Entity, &HdrScreenshot), Without<Captured>>,
) {
    let hdr = config.hdr
        && config.render_mode == crate::render::RenderMode::Color
        && !config.room_schematic
        && !display.sdr_fallback();
    if hdr {
        for (entity, screenshot) in &requested {
            if matches!(&screenshot.0, RenderTarget::Window(WindowRef::Primary))
                || matches!(&screenshot.0, RenderTarget::Window(WindowRef::Entity(w)) if primary.contains(*w))
            {
                commands
                    .entity(entity)
                    .insert((Capturing, HdrScreenshot::default()));
            }
        }
    }
    for (entity, job) in &pending {
        let mut stage = job.0.lock().unwrap();
        match &*stage {
            Stage::Ready(_) => {
                let Stage::Ready(image) = std::mem::take(&mut *stage) else {
                    unreachable!()
                };
                commands
                    .entity(entity)
                    .insert(Captured)
                    .remove::<HdrScreenshot>();
                commands.trigger(ScreenshotCaptured {
                    entity,
                    image: *image,
                });
            }
            Stage::Queued if !hdr => {
                commands
                    .entity(entity)
                    .remove::<(Capturing, HdrScreenshot)>();
            }
            Stage::Failed => {
                commands
                    .entity(entity)
                    .insert(Captured)
                    .remove::<HdrScreenshot>();
            }
            _ => {}
        }
    }
}

fn extract(mut shots: ResMut<Shots>, jobs: Extract<Query<&HdrScreenshot>>) {
    shots.0.clear();
    shots.0.extend(
        jobs.iter()
            .filter(|j| matches!(*j.0.lock().unwrap(), Stage::Queued))
            .cloned(),
    );
}

pub(super) fn read(
    shots: &Shots,
    device: &RenderDevice,
    queue: &RenderQueue,
    source: &wgpu::Texture,
) {
    if !shots.pending() {
        return;
    }
    let size = source.size();
    let row = size.width * 8;
    let stride = row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let bytes = u64::from(stride) * u64::from(size.height);
    let tickets = shots.0.clone();
    if bytes > device.limits().max_buffer_size {
        error!("HDR screenshot is larger than the adapter's readback limit");
        for job in tickets {
            *job.0.lock().unwrap() = Stage::Failed;
        }
        return;
    }
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Studio HDR screenshot"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        source.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: None,
            },
        },
        size,
    );
    for job in &tickets {
        *job.0.lock().unwrap() = Stage::Submitted;
    }
    queue.submit([encoder.finish()]);
    let mapped = buffer.clone();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            if let Err(e) = result {
                error!("HDR screenshot readback failed: {e}");
                for job in tickets {
                    *job.0.lock().unwrap() = Stage::Failed;
                }
                return;
            }
            let image = {
                let data = mapped.slice(..).get_mapped_range().unwrap();
                let pixels = data
                    .chunks_exact(stride as usize)
                    .flat_map(|p| p[..row as usize].iter().copied())
                    .collect();
                Image::new(
                    size,
                    wgpu::TextureDimension::D2,
                    pixels,
                    wgpu::TextureFormat::Rgba16Float,
                    RenderAssetUsages::MAIN_WORLD,
                )
            };
            mapped.unmap();
            // Duplicate explicit requests share one GPU readback.
            if let Some((last, others)) = tickets.split_last() {
                for job in others {
                    *job.0.lock().unwrap() = Stage::Ready(Box::new(image.clone()));
                }
                *last.0.lock().unwrap() = Stage::Ready(Box::new(image));
            }
        });
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(super) fn verify_readback(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) {
    let device = RenderDevice::from(device.clone());
    let jobs = Shots(vec![HdrScreenshot::default(), HdrScreenshot::default()]);
    read(&jobs, &device, &RenderQueue::new(queue.clone()), texture);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    for job in jobs.0 {
        let state = job.0.lock().unwrap();
        let Stage::Ready(image) = &*state else {
            panic!("HDR screenshot did not complete")
        };
        let bytes = image.data.as_ref().unwrap();
        assert_eq!(bytes.len(), 3 * 2 * 8);
        for pixel in bytes.as_chunks::<8>().0 {
            assert_eq!(half::f16::from_le_bytes([pixel[4], pixel[5]]).to_f32(), 4.);
        }
    }
}
