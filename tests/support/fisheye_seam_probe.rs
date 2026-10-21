//! Execute the production composite WGSL on the actual lit face textures at
//! rays approaching each boundary from either side. No CPU approximation of
//! the compositor, mock textures, or camera-rig changes enter this check. The
//! already captured float32 face values are uploaded unchanged for this test
//! only: the native pipelined renderer owns its RenderApp on another thread.
use bevy::{
    prelude::*,
    render::renderer::{RenderDevice, RenderQueue},
};
use bevy_zeroverse::camera::lens::LensCamera;
use wgpu::util::DeviceExt;

fn shader_source() -> String {
    let probe = r#"
@group(1) @binding(0) var<storage,read_write> differences:array<vec4<f32>>;
@compute @workgroup_size(64) fn seam_probe(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=1448u {return;}
    let pairs=array<vec2<u32>,8>(vec2<u32>(0,1),vec2<u32>(0,2),vec2<u32>(0,3),vec2<u32>(0,4),vec2<u32>(1,3),vec2<u32>(1,4),vec2<u32>(2,3),vec2<u32>(2,4));
    let pair=pairs[id.x/181u];let j=f32(id.x%181u)-90.0;
    let ma=params.views[pair.x];let mb=params.views[pair.y];
    let fa=-vec3<f32>(ma[0].z,ma[1].z,ma[2].z);let fb=-vec3<f32>(mb[0].z,mb[1].z,mb[2].z);
    let direction=normalize(fa+fb+normalize(cross(fa,fb))*j/100.0);
    if acos(-direction.z)>params.lens.domain.x {differences[id.x]=vec4<f32>(0.0);return;}
    let before=composite(normalize(direction+0.000001*(fa-fb)));
    let after=composite(normalize(direction-0.000001*(fa-fb)));
    let error=abs(before.rgb-after.rgb);
    let raw=abs(sample_face(pair.x,direction).rgb-sample_face(pair.y,direction).rgb);
    differences[id.x]=vec4<f32>(max(error.x,max(error.y,error.z)),max(raw.x,max(raw.y,raw.z)),1.0,0.0);
}
"#;
    [
        include_str!("../../src/camera/lens/lens.wgsl"),
        include_str!("../../src/camera/lens/resolve.wgsl"),
        probe,
    ]
    .join("\n")
    .lines()
    .filter(|line| !line.starts_with('#'))
    .collect::<Vec<_>>()
    .join("\n")
}

pub fn validate_shader() {
    let source = shader_source();
    let module = wgpu::naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
}

pub fn run(
    render_device: &RenderDevice,
    queue: &RenderQueue,
    lens: &LensCamera,
    faces: &[(Entity, Transform)],
    scale: f32,
    size: UVec2,
    pixels: &[Vec<[f32; 4]>],
) -> serde_json::Value {
    // Shared handles are the public boundary. No App/World/RenderApp is accepted,
    // so this also works when PipelinedRenderingPlugin owns the render world.
    let device = render_device.wgpu_device();
    assert_eq!(pixels.len(), 5);
    let textures: Vec<_> = pixels
        .iter()
        .map(|pixels| {
            assert_eq!(pixels.len(), (size.x * size.y) as usize);
            let extent = wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            };
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("exact captured lighting face for seam probe"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                texture.as_image_copy(),
                bytemuck::cast_slice(pixels),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size.x * 16),
                    rows_per_image: Some(size.y),
                },
                extent,
            );
            texture.create_view(&default())
        })
        .collect();
    let source = shader_source();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("production fisheye seam probe"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    // Explicit layout retains unfilterable float32 texture bindings (the
    // automatic layout would request filtering that the production pass avoids).
    let mut entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }];
    for binding in 1..=6 {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
    }
    let inputs = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("seam inputs"),
        entries: &entries,
    });
    let outputs = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("seam outputs"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("seam probe"),
        bind_group_layouts: &[Some(&inputs), Some(&outputs)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("seam probe"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("seam_probe"),
        compilation_options: default(),
        cache: None,
    });
    // Production Params is 664 float32 words; zero-fill irrelevant preview fields.
    let mut data = vec![0_f32; 664];
    data[..16].copy_from_slice(bytemuck::cast_slice(&lens.calibration.shader_parameters()));
    for i in 0..6 {
        let rotation = faces.get(i).map_or(Quat::IDENTITY, |(_, t)| t.rotation);
        data[16 + i * 16..32 + i * 16]
            .copy_from_slice(&Mat4::from_quat(rotation.inverse()).to_cols_array());
    }
    data[112] = scale;
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("seam parameters"),
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bytes = 1448 * 16;
    let result = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("seam differences"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("seam validation readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut bindings = vec![wgpu::BindGroupEntry {
        binding: 0,
        resource: uniform.as_entire_binding(),
    }];
    for i in 0..6 {
        bindings.push(wgpu::BindGroupEntry {
            binding: i as u32 + 1,
            resource: wgpu::BindingResource::TextureView(&textures[i % 5]),
        });
    }
    let input = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("seam inputs"),
        layout: &inputs,
        entries: &bindings,
    });
    let output = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("seam outputs"),
        layout: &outputs,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: result.as_entire_binding(),
        }],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("seam probe"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &input, &[]);
        pass.set_bind_group(1, &output, &[]);
        pass.dispatch_workgroups(1448_u32.div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&result, 0, &read, 0, bytes);
    queue.submit([encoder.finish()]);
    let (send, receive) = std::sync::mpsc::channel();
    read.slice(..)
        .map_async(wgpu::MapMode::Read, move |value| send.send(value).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receive
        .recv_timeout(std::time::Duration::from_secs(10))
        .unwrap()
        .unwrap();
    let data = read.slice(..).get_mapped_range().unwrap();
    let differences: Vec<[f32; 4]> = data
        .as_chunks::<16>()
        .0
        .iter()
        .map(|v| bytemuck::pod_read_unaligned(v))
        .filter(|v: &[f32; 4]| v[2] > 0.)
        .collect();
    let max = differences.iter().map(|v| v[0]).fold(0_f32, f32::max);
    let mean = differences.iter().map(|v| v[0]).sum::<f32>() / differences.len() as f32;
    let raw_max = differences.iter().map(|v| v[1]).fold(0_f32, f32::max);
    assert!(
        differences.len() > 1000 && max < 0.002 && mean < 0.0001,
        "final compositor discontinuity {max} (mean {mean})"
    );
    serde_json::json!({"world_ray_approaches":differences.len(),"epsilon_radians":0.000001,"linear_rgb_abs_error_mean":mean,"linear_rgb_abs_error_max":max,"raw_face_abs_error_max":raw_max,"production_wgsl":true})
}
