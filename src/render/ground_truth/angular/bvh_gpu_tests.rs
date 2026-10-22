//! Execute the production intersection shader against both acceleration trees.
//! Compare original triangle IDs, distance and barycentrics bit for bit.
use super::*;
use wgpu::util::DeviceExt;

#[test]
#[ignore = "native hardware GPU: exact BVH intersection parity"]
fn surface_area_and_median_trees_produce_identical_gpu_hits() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..default()
    }))
    .unwrap();
    assert_ne!(adapter.get_info().device_type, wgpu::DeviceType::Cpu);
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    let mut input = triangles(4096, false);
    // Architectural surfaces have radically different extents from props.
    for z in [-9., -7., -2.] {
        let mut t = input[0];
        t.p = [[-10., -10., z, 0.], [10., -10., z, 0.], [0., 10., z, 0.]];
        t.info = [17, 0, 0, input.len() as u32];
        input.push(t);
        t.info[3] += 1;
        input.push(t); // Coincident triangles must retain the original-ID tie.
    }
    let mut rays = Vec::<[[f32; 4]; 2]>::new();
    for origin in [Vec3::new(0., 1., 20.), Vec3::new(1., -1., -15.)] {
        for t in &input {
            let center =
                t.p.iter()
                    .map(|p| Vec3::from_array(p[..3].try_into().unwrap()))
                    .sum::<Vec3>()
                    / 3.;
            rays.push([
                origin.extend(0.01).to_array(),
                (center - origin).normalize().extend(100.).to_array(),
            ]);
        }
        for axis in [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::NEG_X,
            Vec3::NEG_Y,
            Vec3::NEG_Z,
        ] {
            rays.push([origin.extend(0.01).to_array(), axis.extend(100.).to_array()]);
        }
    }
    let shader = include_str!("bvh.wgsl")
        .lines()
        .filter(|s| !s.starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
        + r#"
struct Ray { origin:vec4<f32>, direction:vec4<f32> }
@group(0) @binding(0) var<storage,read> rays:array<Ray>;
@group(0) @binding(4) var<storage,read_write> hits:array<vec4<u32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=arrayLength(&rays) {return;}
    let r=rays[id.x];let h=trace(r.origin.xyz,r.direction.xyz,r.origin.w,r.direction.w);
    var ordinal=0xffffffffu;
    if h.index!=0xffffffffu {ordinal=triangles[h.index].info.w;}
    hits[id.x]=vec4<u32>(ordinal,bitcast<u32>(h.distance),bitcast<vec2<u32>>(h.uv));
}"#;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("production angular traversal parity"),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("production angular traversal parity"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: default(),
        cache: None,
    });
    let buffer = |label, bytes: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let input_rays = buffer("rays", bytemuck::cast_slice(&rays));
    let layers = buffer("layers", bytemuck::cast_slice(&[1_u32, 1, 0, 1, 0]));
    let bytes = rays.len() as u64 * 16;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hits"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hit oracle readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let run = |surface_area| {
        let (triangles, nodes) = finish_with_strategy(input.clone(), surface_area);
        let triangles = buffer("triangles", bytemuck::cast_slice(&triangles));
        let nodes = buffer("nodes", bytemuck::cast_slice(&nodes));
        let bindings = [&input_rays, &triangles, &nodes, &layers, &output]
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hit oracle"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &bindings,
        });
        let mut encoder = device.create_command_encoder(&default());
        {
            let mut pass = encoder.begin_compute_pass(&default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((rays.len() as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &read, 0, bytes);
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        read.slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv_timeout(std::time::Duration::from_secs(15))
            .unwrap()
            .unwrap();
        let data = read.slice(..).get_mapped_range().unwrap().to_vec();
        read.unmap();
        data
    };
    let reference = run(false);
    let actual = run(true);
    assert!(
        reference
            .as_chunks::<16>()
            .0
            .iter()
            .filter(|p| u32::from_ne_bytes(p[..4].try_into().unwrap()) != u32::MAX)
            .count()
            > 100
    );
    assert_eq!(
        actual, reference,
        "tree partition must not change any hit bits"
    );
    println!(
        "{} GPU rays: IDs, ranges and barycentrics exactly match median reference",
        rays.len()
    );
}
