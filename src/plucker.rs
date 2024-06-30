use bevy::{
    prelude::*,
    asset::load_internal_asset,
    render::{
        extract_component::{
            ExtractComponent,
            ExtractComponentPlugin,
        },
        render_resource::AsBindGroup,
    },
};


const PLUCKER_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(566172342);

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

        // let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
        //     return;
        // };

        // TODO: add plucker pipeline to core_3d graph
    }
}


#[derive(AsBindGroup, Clone, Component, Debug, ExtractComponent, Reflect)]
pub struct PluckerOutput {
    #[storage_texture(0, image_format = Bgra8Unorm, access = WriteOnly)]
    pub plucker_u: Handle<Image>,

    #[storage_texture(1, image_format = Bgra8Unorm, access = WriteOnly)]
    pub plucker_v: Handle<Image>,

    #[storage_texture(2, image_format = Bgra8Unorm, access = WriteOnly)]
    pub visualization: Handle<Image>,
}
