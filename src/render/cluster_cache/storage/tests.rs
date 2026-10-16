use super::*;

fn objects() -> GlobalClusterableObjectMeta {
    GlobalClusterableObjectMeta::new(STORAGE)
}

#[test]
fn storage_preserves_offsets_classifications_and_missing_probe_slots() {
    let mut world = World::new();
    let main = world.spawn_empty().id();
    let rendered = world.spawn_empty().id();
    let mut objects = objects();
    objects.entity_to_index.insert(rendered, 17);
    let input = InputKey(vec![
        Record::Counts([1, 0, 0, 1, 1, 0]),
        Record::Object(
            main,
            Some(rendered),
            [true, false, false, true, true, false],
        ),
        Record::Counts([0; 6]),
        Record::Counts([0, 0, 0, 0, 0, 1]),
        Record::Object(
            main,
            Some(rendered),
            [false, false, false, false, false, true],
        ),
    ]);
    assert_eq!(
        pack(&input, &objects, None, None).unwrap(),
        Packed {
            indices: vec![17, u32::MAX, u32::MAX, 17],
            offsets: vec![
                [0, 1, 0, 0, 1, 1, 0, 0],
                [3, 0, 0, 0, 0, 0, 0, 0],
                [3, 0, 0, 0, 0, 0, 1, 0]
            ],
        }
    );
    // Missing render mappings are skipped by Bevy's extractor; missing indices
    // for an existing render entity must fall back to Bevy's diagnostic path.
    objects.entity_to_index.clear();
    assert!(pack(&input, &objects, None, None).is_none());
    let unrendered = InputKey(vec![
        Record::Counts([1, 0, 0, 0, 0, 0]),
        Record::Object(main, None, [true, false, false, false, false, false]),
    ]);
    assert!(pack(&unrendered, &objects, None, None).is_none());
}

#[test]
fn capacity_requires_both_independent_storage_buffers() {
    assert!(fits([64, 256], [4, 32]));
    assert!(fits([64, 256], [64, 256]));
    assert!(!fits([64, 256], [68, 32]));
    assert!(!fits([64, 256], [4, 288]));
}

/// Use Bevy's unmodified extractor/preparer as an independent GPU packing oracle.
/// Reading through a shader also verifies the actual binding sizes and uploads.
#[test]
#[ignore = "requires a native GPU"]
fn gpu_storage_matches_bevy_and_recycles_across_despawn_and_capacity_changes() {
    use bevy::{
        light::cluster::ObjectsInClusterCpu, render::pipelined_rendering::PipelinedRenderingPlugin,
        winit::WinitPlugin,
    };
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(bevy::render::RenderPlugin {
                render_creation: bevy::render::settings::RenderCreation::Automatic(Box::new(
                    bevy::render::settings::WgpuSettings {
                        backends: Some(wgpu::Backends::VULKAN),
                        ..default()
                    },
                )),
                synchronous_pipeline_compilation: true,
                ..default()
            })
            .set(WindowPlugin {
                primary_window: None,
                ..default()
            })
            .disable::<WinitPlugin>()
            .disable::<PipelinedRenderingPlugin>(),
    );
    app.finish();
    app.cleanup();
    app.update();
    let render = app.sub_app(RenderApp).world();
    let device = render.resource::<RenderDevice>().clone();
    let queue = render.resource::<RenderQueue>().clone();
    let pipeline = render.resource::<MeshPipeline>().clone();
    assert!(matches!(
        pipeline.clustered_forward_buffer_binding_type,
        BufferBindingType::Storage { .. }
    ));

    let mut h = super::super::tests::Extraction::new();
    h.render.insert_resource(device.clone());
    h.render.insert_resource(queue.clone());
    h.render.insert_resource(pipeline);
    h.render.init_resource::<Pool>();
    h.render
        .register_component_hooks::<ViewClusterBindings>()
        .on_discard(recycle);
    let mut reference = Schedule::default();
    reference.add_systems(prepare_clusters_for_cpu_clustering);
    let mut extraction = Schedule::new(ExtractSchedule);
    extraction.add_systems(extract_clusters_for_cpu_clustering);
    // Vary membership, empty buffers, growth, shrinkage and entity lifetimes.
    for count in [1, 35, 0, 240, 9, 240, 1] {
        let mut cluster = ObjectsInClusterCpu::default();
        cluster.add_point_light(h.light);
        h.main
            .get_mut::<Clusters>(h.camera)
            .unwrap()
            .clusterable_objects = ClusterableObjects::Cpu(vec![cluster; count]);
        let input = InputKey(h.extract().unwrap());
        let mut main = bevy::render::MainWorld::default();
        std::mem::swap(&mut h.main, &mut *main);
        h.render.insert_resource(main);
        extraction.run(&mut h.render);
        let mut main = h
            .render
            .remove_resource::<bevy::render::MainWorld>()
            .unwrap();
        std::mem::swap(&mut h.main, &mut *main);
        let mut meta = objects();
        let render_entity = **h.main.get::<RenderEntity>(h.light).unwrap();
        meta.entity_to_index.insert(render_entity, 19);
        h.render.insert_resource(meta);
        reference.run(&mut h.render);
        let expected = h
            .render
            .entity_mut(h.view)
            .take::<ViewClusterBindings>()
            .unwrap();
        let meta = h
            .render
            .remove_resource::<GlobalClusterableObjectMeta>()
            .unwrap();
        let actual_entity = h.render.spawn_empty().id();
        h.render.resource_scope(|world, mut pool: Mut<Pool>| {
            let mut commands = world.commands();
            assert!(update(
                actual_entity,
                &input,
                &meta,
                None,
                None,
                None,
                &device,
                &queue,
                &mut pool,
                &mut commands
            ));
        });
        h.render.flush();
        let actual = h.render.get::<ViewClusterBindings>(actual_entity).unwrap();
        for (expected, actual) in [
            (
                expected.clusterable_object_index_lists_binding().unwrap(),
                actual.clusterable_object_index_lists_binding().unwrap(),
            ),
            (
                expected.offsets_and_counts_binding().unwrap(),
                actual.offsets_and_counts_binding().unwrap(),
            ),
        ] {
            let BindingResource::Buffer(ref actual_buffer) = actual else {
                unreachable!()
            };
            let extended_size = actual_buffer.size.unwrap().get();
            let expected = read_storage(&device, &queue, expected, extended_size);
            let actual = read_storage(&device, &queue, actual, extended_size);
            assert_eq!(&actual[..expected.len()], expected, "cluster count {count}");
        }
        h.render.despawn(actual_entity);
        let pool = h.render.resource::<Pool>();
        assert!(!pool.available.is_empty());
        assert!(pool.bytes <= MAX_POOLED_BYTES);
        assert!(pool.available.len() <= MAX_POOLED_VIEWS);
    }
    // Retention is bounded even if a caller repeatedly changes camera counts.
    let mut pool = h.render.resource_mut::<Pool>();
    for _ in 0..(MAX_POOLED_VIEWS * 2) {
        let mut b = ViewClusterBindings::new(STORAGE);
        b.reserve_indices(1);
        b.reserve_clusters(1);
        b.write_buffers(&device, &queue);
        pool.put(b);
    }
    assert_eq!(pool.available.len(), MAX_POOLED_VIEWS);
    assert_eq!(
        pool.bytes,
        pool.available
            .iter()
            .map(|b| sizes(b).unwrap().iter().sum::<u64>())
            .sum::<u64>()
    );
    // A consumer changing the pipeline's binding type must not put uniform
    // buffers into the storage pool, even if the managed marker remains.
    let mut uniform = ViewClusterBindings::new(BufferBindingType::Uniform);
    uniform.write_buffers(&device, &queue);
    assert!(sizes(&uniform).is_none());
    let retained = pool.bytes;
    pool.put(uniform);
    assert_eq!(pool.bytes, retained);
    queue.submit([]);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    drop(h);
    drop(app);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}

fn read_storage(
    device: &RenderDevice,
    queue: &RenderQueue,
    binding: BindingResource<'_>,
    size: u64,
) -> Vec<u8> {
    use bevy::render::render_resource::*;
    let shader = device.wgpu_device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var<storage, read> src: array<u32>;\n@group(0) @binding(1) var<storage, read_write> dst: array<u32>;\n@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) { if id.x < arrayLength(&dst) { dst[id.x] = src[id.x]; } }".into()),
    });
    let pipeline = device
        .wgpu_device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: default(),
            cache: None,
        });
    let output = device.create_buffer(&BufferDescriptor {
        label: None,
        size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&BufferDescriptor {
        label: None,
        size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let group = device
        .wgpu_device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: binding,
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
    let mut encoder = device.create_command_encoder(&default());
    {
        let mut pass = encoder.begin_compute_pass(&default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((size as u32 / 4).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let result = readback.slice(..).get_mapped_range().unwrap().to_vec();
    readback.unmap();
    result
}
