//! Intern complete explicit sampler descriptions, without reading stale CPU assets.
use bevy::{
    image::{ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::{
        render_asset::{ExtractedAssets, RenderAssets},
        render_resource::SamplerId,
        texture::GpuImage,
    },
};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Key {
    Object(SamplerId),
    Description(Description),
}

/// Every sampling parameter participates. Labels have no sampling behavior;
/// float bits preserve even distinctions that a numerical comparison ignores.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Description([u32; 11]);

impl From<&ImageSamplerDescriptor> for Description {
    fn from(descriptor: &ImageSamplerDescriptor) -> Self {
        // Exhaustive destructuring makes a future descriptor field an explicit
        // compatibility decision instead of silently dropping a new setting.
        let ImageSamplerDescriptor {
            label: _,
            address_mode_u,
            address_mode_v,
            address_mode_w,
            mag_filter,
            min_filter,
            mipmap_filter,
            lod_min_clamp,
            lod_max_clamp,
            compare,
            anisotropy_clamp,
            border_color,
        } = descriptor;
        Self([
            *address_mode_u as u32,
            *address_mode_v as u32,
            *address_mode_w as u32,
            *mag_filter as u32,
            *min_filter as u32,
            *mipmap_filter as u32,
            lod_min_clamp.to_bits(),
            lod_max_clamp.to_bits(),
            compare.map_or(0, |v| v as u32 + 1),
            u32::from(*anisotropy_clamp),
            border_color.map_or(0, |v| v as u32 + 1),
        ])
    }
}

#[derive(Resource, Default)]
pub(super) struct Recipes(HashMap<AssetId<Image>, UploadedRecipe>);

struct UploadedRecipe {
    description: Description,
    sampler: Option<SamplerId>,
}

impl Recipes {
    pub(super) fn key(&self, image: AssetId<Image>, sampler: SamplerId) -> Key {
        self.0
            .get(&image)
            .filter(|recipe| recipe.sampler == Some(sampler))
            .map_or(Key::Object(sampler), |recipe| {
                Key::Description(recipe.description)
            })
    }

    pub(super) fn observe_uploads(&mut self, images: &RenderAssets<GpuImage>) {
        for (id, recipe) in &mut self.0 {
            if recipe.sampler.is_none() {
                recipe.sampler = images.get(*id).map(|image| image.sampler.id());
            }
        }
    }

    pub(super) fn capture(&mut self, extracted: &ExtractedAssets<GpuImage>) {
        for id in &extracted.removed {
            self.0.remove(id);
        }
        for (id, image) in &extracted.extracted {
            match &image.sampler {
                ImageSampler::Default => {
                    self.0.remove(id);
                }
                ImageSampler::Descriptor(descriptor) => {
                    self.0.insert(
                        *id,
                        UploadedRecipe {
                            description: descriptor.into(),
                            sampler: None,
                        },
                    );
                }
            }
        }
    }
}

pub(super) fn remember(extracted: Res<ExtractedAssets<GpuImage>>, mut recipes: ResMut<Recipes>) {
    // Run before prepare_assets<GpuImage>. That system removes the old GPU
    // asset before deferring an update and discards superseded retry entries.
    // Thus any GPU image available AFTER preparation matches this description,
    // even with upload throttling, mutation, or RENDER_WORLD-only source data.
    // Bind its sampler identity once: a later direct render-world replacement
    // without a new source extraction falls back to the actual GPU object.
    // Unloaded IDs are evicted here, including assets that never uploaded.
    recipes.capture(&extracted);
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::{asset::uuid::Uuid, image::*};

    #[test]
    fn every_sampler_parameter_changes_the_key_but_labels_do_not() {
        let original = ImageSamplerDescriptor::linear();
        let key = Description::from(&original);
        let mut labeled = original.clone();
        labeled.label = Some("different diagnostic label".into());
        assert_eq!(key, Description::from(&labeled));
        let variants: [fn(&mut ImageSamplerDescriptor); 11] = [
            |d| d.address_mode_u = ImageAddressMode::Repeat,
            |d| d.address_mode_v = ImageAddressMode::Repeat,
            |d| d.address_mode_w = ImageAddressMode::Repeat,
            |d| d.mag_filter = ImageFilterMode::Nearest,
            |d| d.min_filter = ImageFilterMode::Nearest,
            |d| d.mipmap_filter = ImageFilterMode::Nearest,
            |d| d.lod_min_clamp = 1.,
            |d| d.lod_max_clamp = 4.,
            |d| d.compare = Some(ImageCompareFunction::LessEqual),
            |d| d.anisotropy_clamp = 4,
            |d| d.border_color = Some(ImageSamplerBorderColor::OpaqueBlack),
        ];
        for variant in variants {
            let mut changed = original.clone();
            variant(&mut changed);
            assert_ne!(key, Description::from(&changed));
        }
    }

    #[test]
    fn descriptions_follow_extracted_updates_and_unload_pending_assets() {
        let id: AssetId<Image> = Uuid::new_v4().into();
        let image = Image {
            sampler: ImageSampler::linear(),
            ..default()
        };
        let mut extracted = ExtractedAssets::<GpuImage>::default();
        extracted.extracted.push((id, image));
        let mut recipes = Recipes::default();
        recipes.capture(&extracted);
        assert_eq!(recipes.0.len(), 1);
        // Retain the recipe while upload is deferred, with no new extraction.
        extracted.extracted.clear();
        recipes.capture(&extracted);
        assert_eq!(recipes.0.len(), 1);
        // A newer default-sampler update replaces an earlier pending recipe.
        extracted.extracted.push((id, Image::default()));
        recipes.capture(&extracted);
        assert!(recipes.0.is_empty());
        extracted.extracted[0].1.sampler = ImageSampler::nearest();
        recipes.capture(&extracted);
        assert_eq!(recipes.0.len(), 1);
        extracted.extracted.clear();
        extracted.removed.insert(id);
        recipes.capture(&extracted);
        assert!(recipes.0.is_empty());
    }
}
