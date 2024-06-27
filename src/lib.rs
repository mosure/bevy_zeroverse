use bevy::{
    prelude::*,
    asset::{
        AssetLoader,
        AsyncReadExt,
        LoadContext,
        io::Reader,
        saver::{AssetSaver, SavedAsset},
    },
    utils::BoxedFuture,
};
use serde::{Deserialize, Serialize};


#[derive(Asset, Reflect)]
struct ZeroverseObject {

}

#[derive(Default, Reflect, Serialize, Deserialize)]
struct ZeroverseSettings {
    scale: Vec3,
    seed: u64,
    sub_object_count: usize,
}

#[derive(Default)]
struct ZeroverseLoader;
impl AssetLoader for ZeroverseLoader {
    type Asset = ZeroverseObject;
    type Error = std::io::Error;
    type Settings = ZeroverseSettings;

    fn load<'a>(
        &'a self,
        reader: &'a mut Reader,
        settings: &'a Self::Settings,
        load_context: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<Self::Asset, Self::Error>> {
        Box::pin(async move {
            let mut bytes = Vec::new();
            reader.read_to_end(&mut bytes).await?;

            Ok(ZeroverseObject { })
        })
    }

    fn extensions(&self) -> &[&str] {
        &["thing"]
    }
}


pub struct BevyZeroversePlugin;

impl Plugin for BevyZeroversePlugin {
    fn build(&self, app: &mut App) {
        info!("initializing BevyZeroversePlugin...");

        app.init_asset::<ZeroverseObject>();
        app.init_asset_loader::<ZeroverseLoader>();
    }
}
