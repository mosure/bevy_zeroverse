//! Once-per-probe IBL convolution. Uses the CPU oracle's exact quadrature and
//! cross-face taps. Only radiance changes between scenes; no host readback.
use super::*;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use wgpu::util::DeviceExt;
mod render;
pub(crate) use render::ReflectionBakePlugin;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

#[derive(Default)]
struct Status {
    encoded: AtomicBool,
    error: Mutex<Option<String>>,
}
#[derive(Clone)]
pub(crate) struct Request {
    pub images: [Handle<Image>; 2],
    source: Arc<Vec<[f32; 4]>>,
    status: Arc<Status>,
}
impl Request {
    pub(crate) fn ready(&self) -> bool {
        self.status.encoded.load(Ordering::Acquire)
    }
    pub(crate) fn failure(&self) -> Option<String> {
        self.status.error.lock().unwrap().clone()
    }
    pub(crate) fn upload_bytes(&self) -> u64 {
        (self.source.len() * std::mem::size_of::<[f32; 4]>()) as u64
    }
}

pub(crate) fn prepare(
    pixels: Vec<Vec3>,
    images: &mut impl super::super::super::preparation::AssetStore<Image>,
    requests: &mut super::super::gpu::Requests,
) -> [Handle<Image>; 2] {
    let source = RadianceMips::new(&pixels, SIZE)
        .0
        .into_iter()
        .flat_map(|(_, pixels)| pixels.into_iter().map(|p| [p.x, p.y, p.z, 0.]))
        .collect();
    let handles = [(16, 1), (SIZE, SIZE.ilog2() + 1)].map(|(size, levels)| {
        let mut image = cube(Vec::new(), size, levels);
        image.data = None;
        image.texture_descriptor.usage |= bevy::render::render_resource::TextureUsages::COPY_SRC;
        images.add(image)
    });
    requests.1.push(Request {
        images: handles.clone(),
        source: Arc::new(source),
        status: Arc::default(),
    });
    handles
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Tap {
    pub addresses: [u32; 4],
    pub blend: [f32; 4],
    pub weight_lod: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Pixel {
    pub taps: u32,
    pub offset: u32,
    pub denominator: f32,
    /// Zero copies one source pixel, one diffuse, two specular.
    pub kind: u32,
}
pub(super) struct Copy {
    pub target: usize,
    pub face: u32,
    pub mip: u32,
    pub size: u32,
    pub offset: u64,
    pub stride: u32,
}
pub(super) struct Program {
    pub taps: Vec<Tap>,
    pub pixels: Vec<Pixel>,
    pub copies: Vec<Copy>,
    pub bytes: u64,
}
pub(super) fn program() -> &'static Program {
    static PROGRAM: std::sync::OnceLock<Program> = std::sync::OnceLock::new();
    PROGRAM.get_or_init(super::stencil::gpu_program)
}
pub(super) fn layout_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    (0..4)
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: binding != 3,
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect()
}
pub(super) struct Workspace {
    taps: wgpu::Buffer,
    pixels: wgpu::Buffer,
    output: wgpu::Buffer,
}
impl Workspace {
    pub fn new(device: &wgpu::Device) -> Self {
        let program = program();
        let input = |label, contents| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        Self {
            taps: input("IBL quadrature", bytemuck::cast_slice(&program.taps)),
            pixels: input("IBL destinations", bytemuck::cast_slice(&program.pixels)),
            output: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("IBL scratch"),
                size: program.bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
        }
    }
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        layout: &wgpu::BindGroupLayout,
        source: &[[f32; 4]],
        images: [&wgpu::Texture; 2],
    ) {
        let source = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("IBL radiance"),
            contents: bytemuck::cast_slice(source),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let buffers = [&source, &self.taps, &self.pixels, &self.output];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, b)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("IBL convolution"),
            layout,
            entries: &entries,
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("IBL convolution"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups((program().pixels.len() as u32).div_ceil(64), 1, 1);
        }
        for c in &program().copies {
            encoder.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.output,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: c.offset,
                        bytes_per_row: Some(c.stride),
                        rows_per_image: Some(c.size),
                    },
                },
                wgpu::TexelCopyTextureInfo {
                    texture: images[c.target],
                    mip_level: c.mip,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: c.face,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: c.size,
                    height: c.size,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}
