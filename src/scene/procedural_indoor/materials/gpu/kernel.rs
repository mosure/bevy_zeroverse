//! Shared GPU dispatch/texture-copy implementation, also exercised by readback qualification.
use super::recipe::Recipe;
use wgpu::util::DeviceExt;

#[cfg(test)]
pub(super) const SHADER: &str = concat!(
    include_str!("synthesis.wgsl"),
    "\n",
    include_str!("paint.wgsl"),
    "\n",
    include_str!("flooring.wgsl")
);
pub(super) const MAX_SIZE: u32 = 512;
pub(super) const ENTRIES: [&str; 8] = [
    "woven_map",
    "leather_map",
    "knit_map",
    "finish_map",
    "carpet_map",
    "paint_map",
    "normals",
    "mips",
];
pub(super) const NORMALS: usize = 6;
pub(super) const MIPS: usize = 7;
pub(super) fn layout_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    (0..6)
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if binding == 0 || binding == 5 {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage {
                        read_only: binding == 4,
                    }
                },
                has_dynamic_offset: false,
                min_binding_size: match binding {
                    0 => wgpu::BufferSize::new(std::mem::size_of::<Recipe>() as u64),
                    5 => wgpu::BufferSize::new(std::mem::size_of::<Dispatch>() as u64),
                    _ => None,
                },
            },
            count: None,
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Level {
    pub size: u32,
    pub offset: u64,
    pub stride: u32,
    pub moment_offset: u32,
}
pub(super) fn levels(size: u32) -> Vec<Level> {
    let mut size = size;
    let mut offset = 0;
    let mut moment_offset = 0;
    let mut result = Vec::new();
    loop {
        let stride = size.max(64) * 4;
        result.push(Level {
            size,
            offset,
            stride,
            moment_offset,
        });
        offset += u64::from(size * stride);
        moment_offset += size * size;
        if size == 1 {
            break;
        }
        size /= 2;
    }
    result
}

/// Shared copy/shader addressing, in u32 elements. Compute it once on the host
/// rather than reconstructing padded mip offsets with per-invocation loops.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Dispatch {
    pub destination: [u32; 4],
    pub source: [u32; 4],
    pub plane: [u32; 4],
}
pub(super) fn dispatches(size: u32) -> Vec<Dispatch> {
    let levels = levels(size);
    let plane = (plane_bytes(size) / 4) as u32;
    let address = |l: &Level| [l.size, (l.offset / 4) as u32, l.stride / 4, l.moment_offset];
    levels
        .iter()
        .enumerate()
        .map(|(i, l)| Dispatch {
            destination: address(l),
            source: address(&levels[i.saturating_sub(1)]),
            plane: [plane, 0, 0, 0],
        })
        .collect()
}
pub(super) fn plane_bytes(size: u32) -> u64 {
    let last = *levels(size).last().unwrap();
    last.offset + u64::from(last.stride)
}

/// Reused sequentially across materials/rooms on the same queue. No per-map scratch allocation.
/// Immutable bind groups hold recipe buffers; each dispatch's copies precede scratch reuse.
pub(super) struct Workspace {
    pub pixels: wgpu::Buffer,
    heights: wgpu::Buffer,
    moments: wgpu::Buffer,
    srgb: wgpu::Buffer,
    levels: [Vec<wgpu::Buffer>; 2],
}
impl Workspace {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let storage = |label, size, extra| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | extra,
                mapped_at_creation: false,
            })
        };
        Self {
            pixels: storage(
                "material encoded mip scratch",
                plane_bytes(MAX_SIZE) * 3,
                wgpu::BufferUsages::COPY_SRC,
            ),
            heights: storage(
                "material relief scratch",
                u64::from(MAX_SIZE * MAX_SIZE * 4),
                wgpu::BufferUsages::empty(),
            ),
            moments: storage(
                "material GGX moment scratch",
                u64::from((MAX_SIZE * MAX_SIZE * 4 / 3 + 1) * 16),
                wgpu::BufferUsages::empty(),
            ),
            srgb: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("material sRGB decode oracle"),
                contents: bytemuck::cast_slice(color_table()),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            levels: [256, 512].map(|size| {
                dispatches(size)
                    .iter()
                    .map(|dispatch| {
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("material mip addresses"),
                            contents: bytemuck::bytes_of(dispatch),
                            usage: wgpu::BufferUsages::UNIFORM,
                        })
                    })
                    .collect()
            }),
        }
    }
    pub(super) fn bindings(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        recipe: &Recipe,
    ) -> Vec<wgpu::BindGroup> {
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("material recipe v1"),
            contents: bytemuck::bytes_of(recipe),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let level_set = match recipe.meta[1] {
            256 => 0,
            512 => 1,
            _ => unreachable!("GPU material recipes only use 256 or 512 texels"),
        };
        self.levels[level_set]
            .iter()
            .map(|level| {
                let buffers = [
                    &params,
                    &self.heights,
                    &self.moments,
                    &self.pixels,
                    &self.srgb,
                    level,
                ];
                let entries: Vec<_> = buffers
                    .iter()
                    .enumerate()
                    .map(|(binding, b)| wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: b.as_entire_binding(),
                    })
                    .collect();
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("material bake"),
                    layout,
                    entries: &entries,
                })
            })
            .collect()
    }
    pub(super) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: [&wgpu::ComputePipeline; 3],
        bindings: &[wgpu::BindGroup],
        size: u32,
        textures: [Option<&wgpu::Texture>; 3],
    ) {
        // Each phase reads storage written by the preceding dispatch. Separate
        // passes make these dependencies explicit on tiled/mobile backends,
        // without queue submissions, host fences, or extra texture allocations.
        for pipeline in &pipelines[..2] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("indoor material base/normal stage"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &bindings[0], &[]);
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(size.div_ceil(8), size.div_ceil(8), 1);
        }
        for (level, binding) in bindings.iter().enumerate().skip(1) {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("indoor material GGX mip stage"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipelines[2]);
            pass.set_bind_group(0, binding, &[]);
            let n = size >> level;
            pass.dispatch_workgroups(n.div_ceil(8), n.div_ceil(8), 1);
        }
        let plane = plane_bytes(size);
        for (map, texture) in textures.into_iter().enumerate() {
            // Normal moments still contribute to GGX mips even when the final
            // material omits a normal map. Skip only its allocation and copies.
            let Some(texture) = texture else {
                continue;
            };
            for (mip, l) in levels(size).into_iter().enumerate() {
                encoder.copy_buffer_to_texture(
                    wgpu::TexelCopyBufferInfo {
                        buffer: &self.pixels,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: map as u64 * plane + l.offset,
                            bytes_per_row: Some(l.stride),
                            rows_per_image: Some(l.size),
                        },
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture,
                        mip_level: mip as u32,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: l.size,
                        height: l.size,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
    }
}

// Match the CPU's *quantized* transfer, including f32 pow rounding at integer
// code boundaries. WGSL pow may differ by a ULP; repeating it at each mip can
// otherwise accumulate a one-code drift even for a perfectly uniform material.
pub(super) fn color_table() -> &'static [f32; 512] {
    static TABLE: std::sync::OnceLock<[f32; 512]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [0.; 512];
        table[..256].copy_from_slice(super::super::srgb8_table());
        for code in 1..256 {
            if ((super::super::linear_to_srgb(1.) * 255.) as usize) < code {
                table[256 + code] = f32::INFINITY;
                continue;
            }
            let mut lo = 0u32;
            let mut hi = 1f32.to_bits();
            while lo + 1 < hi {
                let mid = lo + (hi - lo) / 2;
                if (super::super::linear_to_srgb(f32::from_bits(mid)) * 255.) as usize >= code {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            table[256 + code] = f32::from_bits(hi);
        }
        table
    })
}
#[cfg(test)]
#[test]
fn quantized_transfer_thresholds_match_cpu_at_code_boundaries_and_mixtures() {
    let table = color_table();
    let encode = |v: f32| {
        let mut lo = 0;
        let mut hi = 256;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if v >= table[256 + mid] {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo as u8
    };
    let oracle = |v: f32| (super::super::linear_to_srgb(v) * 255.) as u8;
    for code in 1..256 {
        let t = table[256 + code];
        if t.is_finite() {
            for bits in t.to_bits() - 3..=t.to_bits() + 3 {
                let v = f32::from_bits(bits);
                assert_eq!(encode(v), oracle(v));
            }
        }
    }
    for a in 0..256 {
        for b in 0..256 {
            let v = ((table[a] + table[b]) + table[(a + b) % 256]) + table[(a * 7 + b * 13) % 256];
            let v = v * 0.25;
            assert_eq!(encode(v), oracle(v));
        }
    }
}
