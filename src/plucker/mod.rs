use bevy::{
    prelude::*,
    asset::load_internal_asset,
    core_pipeline::core_3d::graph::{
        Core3d,
        Node3d,
    },
    ecs::system::lifetimeless::Read,
    render::{
        extract_component::{
            ExtractComponent,
            ExtractComponentPlugin,
        },
        render_asset::RenderAssets,
        render_graph::{
            Node,
            RenderGraphApp,
            RenderGraphContext,
            RenderLabel,
        },
        render_resource::{
            binding_types::*,
            AsBindGroup,
            BindGroup,
            BindGroupEntries,
            BindGroupLayout,
            BindGroupLayoutEntries,
            BufferBinding,
            CachedComputePipelineId,
            CachedPipelineState,
            ComputePassDescriptor,
            ComputePipelineDescriptor,
            Extent3d,
            PipelineCache,
            ShaderStages,
            ShaderType,
            TextureDescriptor,
            TextureDimension,
            TextureFormat,
            TextureUsages,
        },
        renderer::{
            RenderContext,
            RenderDevice,
        },
        texture::{
            FallbackImage,
            GpuImage,
        },
        view::{
            ExtractedView,
            ViewUniformOffset,
            ViewUniform,
            ViewUniforms,
        },
        Render,
        RenderApp,
        RenderSet,
    }
};


#[derive(Component, Debug, Reflect)]
pub struct PluckerCamera {
    pub size: UVec2,
}

#[derive(AsBindGroup, Clone, Component, Debug, ExtractComponent, Reflect)]
pub struct PluckerOutput {
    #[storage_texture(0, image_format = Rgba32Float, access = WriteOnly)]
    pub plucker_u: Handle<Image>,

    #[storage_texture(1, image_format = Rgba32Float, access = WriteOnly)]
    pub plucker_v: Handle<Image>,

    #[storage_texture(2, image_format = Rgba32Float, access = WriteOnly)]
    pub visualization: Handle<Image>,
}


const PLUCKER_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(793429261847);

#[derive(
    Debug,
    Hash,
    PartialEq,
    Eq,
    Clone,
    RenderLabel,
)]
struct PluckerLabel;


pub struct PluckerPlugin;
impl Plugin for PluckerPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            PLUCKER_SHADER_HANDLE,
            "plucker.wgsl",
            Shader::from_wgsl
        );

        app.add_plugins(ExtractComponentPlugin::<PluckerOutput>::default());
        app.register_type::<PluckerOutput>();

        app.add_systems(PreUpdate, create_plucker_output);

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_graph_node::<PluckerNode>(
                    Core3d,
                    PluckerLabel,
                ).add_render_graph_edge(
                    Core3d,
                    PluckerLabel,
                    Node3d::EndMainPass,
                );

            render_app
                .add_systems(
                    Render,
                    prepare_plucker_bind_groups.in_set(RenderSet::PrepareBindGroups),
                );
        }
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<PluckerPipeline>();
        }
    }
}


fn create_plucker_output(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    plucker_cameras: Query<
        (
            Entity,
            &PluckerCamera,
        ),
        Without<PluckerOutput>,
    >,
) {
    for (entity, plucker_settings) in plucker_cameras.iter() {
        let size = Extent3d {
            width: plucker_settings.size.x,
            height: plucker_settings.size.y,
            depth_or_array_layers: 1,
        };

        let mut plucker_u = Image {
            texture_descriptor: TextureDescriptor {
                label: Some("plucker_u"),
                size,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Float,
                mip_level_count: 1,
                sample_count: 1,
                usage: TextureUsages::COPY_SRC
                    | TextureUsages::COPY_DST
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            },
            ..Default::default()
        };
        plucker_u.resize(size);
        let plucker_u = images.add(plucker_u);

        let mut plucker_v = Image {
            texture_descriptor: TextureDescriptor {
                label: Some("plucker_v"),
                size,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Float,
                mip_level_count: 1,
                sample_count: 1,
                usage: TextureUsages::COPY_SRC
                    | TextureUsages::COPY_DST
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            },
            ..Default::default()
        };
        plucker_v.resize(size);
        let plucker_v = images.add(plucker_v);

        let mut visualization = Image {
            texture_descriptor: TextureDescriptor {
                label: Some("plucker_visualization"),
                size,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Float,
                mip_level_count: 1,
                sample_count: 1,
                usage: TextureUsages::COPY_SRC
                    | TextureUsages::COPY_DST
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            },
            ..Default::default()
        };
        visualization.resize(size);
        let visualization = images.add(visualization);

        commands
            .entity(entity)
            .insert(PluckerOutput {
                plucker_u,
                plucker_v,
                visualization,
            });
    }
}


#[derive(Resource)]
struct PluckerPipeline {
    pipeline_id: CachedComputePipelineId,
    view_layout: BindGroupLayout,
}

impl FromWorld for PluckerPipeline {
    fn from_world(world: &mut World) -> Self {
        let cache = world.resource::<PipelineCache>();
        let render_device = world.resource::<RenderDevice>();

        let view_layout = render_device.create_bind_group_layout(
            "plucker_view",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer::<ViewUniform>(true),
                ),
            ),
        );

        let pipeline_id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("plucker_kernel".into()),
            layout: vec![
                view_layout.clone(),
                PluckerOutput::bind_group_layout(render_device),
            ],
            push_constant_ranges: vec![],
            shader: PLUCKER_SHADER_HANDLE.clone(),
            shader_defs: vec![],
            entry_point: "plucker_kernel".into(),
        });

        Self {
            pipeline_id,
            view_layout,
        }
    }
}


#[derive(Component)]
struct PluckerBindGroup {
    output: BindGroup,
}


fn prepare_plucker_bind_groups(
    mut commands: Commands,
    images: Res<RenderAssets<GpuImage>>,
    fallback: Res<FallbackImage>,
    plucker_output: Query<(
        Entity,
        &PluckerOutput,
    )>,
    render_device: Res<RenderDevice>,
) {
    for (entity, plucker_output) in plucker_output.iter() {
        let output = plucker_output
            .as_bind_group(
                &PluckerOutput::bind_group_layout(&render_device),
                &render_device,
                &images,
                &fallback,
            )
            .map(|bg| bg.bind_group)
            .unwrap();

        commands
            .entity(entity)
            .insert(PluckerBindGroup {
                output,
            });
    }
}


struct PluckerNode {
    prepared_plucker: QueryState<(
        Read<PluckerBindGroup>,
        Read<ViewUniformOffset>,
        Read<ExtractedView>,
    )>,
}

impl FromWorld for PluckerNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            prepared_plucker: world.query(),
        }
    }
}

impl Node for PluckerNode {
    fn update(&mut self, world: &mut World) {
        self.prepared_plucker.update_archetypes(world);
    }

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<PluckerPipeline>();

        match pipeline_cache.get_compute_pipeline_state(pipeline.pipeline_id) {
            CachedPipelineState::Creating(_) => return Ok(()),
            CachedPipelineState::Err(_) => return Ok(()),
            CachedPipelineState::Queued => return Ok(()),
            CachedPipelineState::Ok(_) => {},
        }

        let view_uniforms_resource = world.resource::<ViewUniforms>();
        let view_uniforms = &view_uniforms_resource.uniforms;
        let view_uniforms_buffer = view_uniforms.buffer().unwrap();

        let view_bind_group = render_context.render_device().create_bind_group(
            None,
            &pipeline.view_layout,
            &BindGroupEntries::sequential((
                BufferBinding {
                    buffer: view_uniforms_buffer,
                    size: Some(ViewUniform::min_size()),
                    offset: 0,
                },
            )),
        );

        if let Some((
            plucker_bindings,
            view_offset,
            view,
        )) = self.prepared_plucker.iter_manual(world).next() {
            let pipeline = pipeline_cache.get_compute_pipeline(pipeline.pipeline_id).unwrap();

            {
                let encoder = render_context.command_encoder();
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());

                pass.set_pipeline(pipeline);

                pass.set_bind_group(
                    0,
                    &view_bind_group,
                    &[view_offset.offset],
                );

                pass.set_bind_group(
                    1,
                    &plucker_bindings.output,
                    &[],
                );

                pass.dispatch_workgroups(
                    view.viewport.z.div_ceil(16),
                    view.viewport.w.div_ceil(16),
                    1,
                );
            }
        }

        Ok(())
    }
}