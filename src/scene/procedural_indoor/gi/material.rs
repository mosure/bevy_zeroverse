//! Scene-linear diffuse reflectance and emitted radiance at a transport hit.
use super::*;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct TransportMap {
    pub size: u32,
    pub address: TextureAddress,
    pub pixels: Arc<[Vec3]>,
}

impl TransportMap {
    const SIDE: usize = 32;

    fn read(
        handle: &Handle<Image>,
        images: &impl super::super::preparation::AssetStore<Image>,
        baked: &super::super::materials::gpu::Requests,
        emission: bool,
    ) -> Option<Self> {
        let image = images.get(handle);
        let address = image.map_or_else(TextureAddress::default, TextureAddress::for_image);
        let linear = srgb8_linear_table();
        let decode = |p: &[u8]| {
            Vec3::new(
                linear[p[0] as usize],
                linear[p[1] as usize],
                linear[p[2] as usize],
            )
        };
        let pixels = if let Some(samples) = baked.albedo(handle.id()) {
            samples.iter().map(|p| decode(p)).collect()
        } else {
            let image = image?;
            let desc = &image.texture_descriptor;
            let data = image.data.as_ref()?;
            if desc.dimension != TextureDimension::D2 || desc.size.depth_or_array_layers != 1 {
                return None;
            }
            let (mut w, mut h) = (image.width() as usize, image.height() as usize);
            let mut pixels = Vec::with_capacity(Self::SIDE * Self::SIDE);
            if desc.format == TextureFormat::Rgba8UnormSrgb {
                // Display content has linear-light area-filtered mips. Reading the
                // closest one above our resolution preserves thin glyphs and avoids
                // scanning or point-decimating the full display. Diffuse finishes
                // retain their existing point reduction, including GPU-baked maps.
                let mut offset = 0;
                if emission {
                    for _ in 1..desc.mip_level_count {
                        if w / 2 < Self::SIDE || h / 2 < Self::SIDE {
                            break;
                        }
                        offset += w * h * 4;
                        w /= 2;
                        h /= 2;
                    }
                }
                let data = data.get(offset..offset + w * h * 4)?;
                for y in 0..Self::SIDE {
                    for x in 0..Self::SIDE {
                        let i = (y * h / Self::SIDE * w + x * w / Self::SIDE) * 4;
                        pixels.push(decode(&data[i..i + 3]));
                    }
                }
            } else {
                for y in 0..Self::SIDE {
                    for x in 0..Self::SIDE {
                        let color = image
                            .get_color_at((x * w / Self::SIDE) as u32, (y * h / Self::SIDE) as u32)
                            .ok()?;
                        pixels.push(super::linear(color));
                    }
                }
            }
            pixels.into()
        };
        Some(Self {
            size: Self::SIDE as u32,
            address,
            pixels,
        })
    }

    fn sample(&self, uv: Vec2) -> Vec3 {
        self.address.sample(uv, self.size, &self.pixels)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum EmissionMap {
    Uniform,
    SharedWithBase,
    Separate(TransportMap),
}

#[derive(Clone)]
pub(super) struct DiffuseMaterial {
    pub albedo: Vec3,
    pub emission: Vec3,
    pub uv_transform: Affine2,
    pub base_map: Option<TransportMap>,
    pub emission_map: EmissionMap,
}

impl DiffuseMaterial {
    pub(super) fn from_standard(
        mat: &StandardMaterial,
        images: &impl super::super::preparation::AssetStore<Image>,
        baked: &super::super::materials::gpu::Requests,
    ) -> Self {
        let emission = Vec3::new(mat.emissive.red, mat.emissive.green, mat.emissive.blue);
        let emissive_texture = mat
            .emissive_texture
            .as_ref()
            .filter(|_| emission != Vec3::ZERO);
        let shared =
            emissive_texture.is_some() && emissive_texture == mat.base_color_texture.as_ref();
        let base_map = mat
            .base_color_texture
            .as_ref()
            .and_then(|h| TransportMap::read(h, images, baked, shared));
        let emission_map = match emissive_texture {
            Some(_) if shared && base_map.is_some() => EmissionMap::SharedWithBase,
            Some(h) if !shared => TransportMap::read(h, images, baked, true)
                .map_or(EmissionMap::Uniform, EmissionMap::Separate),
            _ => EmissionMap::Uniform,
        };
        Self {
            albedo: (linear(mat.base_color) * (1.0 - mat.metallic))
                .clamp(Vec3::ZERO, Vec3::splat(0.95)),
            emission,
            uv_transform: mat.uv_transform,
            base_map,
            emission_map,
        }
    }

    pub(super) fn response(&self, uv: Vec2) -> (Vec3, Vec3) {
        if self.base_map.is_none() && matches!(self.emission_map, EmissionMap::Uniform) {
            return (self.albedo, self.emission);
        }
        let uv = self.uv_transform.transform_point2(uv);
        let base = self.base_map.as_ref().map_or(Vec3::ONE, |m| m.sample(uv));
        let emitted = match &self.emission_map {
            EmissionMap::Uniform => self.emission,
            EmissionMap::SharedWithBase => self.emission * base,
            EmissionMap::Separate(map) => self.emission * map.sample(uv),
        };
        (self.albedo * base, emitted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emitted_energy_uses_its_own_map_and_linear_light_mip_even_on_black_substrates() {
        let mut images = Assets::default();
        let size = Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        };
        // One white texel out of four. Point sampling at every other column and
        // row returns exclusively white. The 32² mip represents 25% radiance.
        let pattern: Vec<u8> = (0..4096)
            .flat_map(|i| {
                let v = if i % 2 == 0 && i / 64 % 2 == 0 {
                    255
                } else {
                    0
                };
                [v, v, v, 255]
            })
            .collect();
        let mut image = Image::new(
            size,
            TextureDimension::D2,
            pattern,
            TextureFormat::Rgba8UnormSrgb,
            default(),
        );
        image
            .data
            .as_mut()
            .unwrap()
            .extend([137, 137, 137, 255].repeat(1024));
        image.texture_descriptor.mip_level_count = 2;
        let content = images.add(image);
        let red = images.add(Image::new_fill(
            size,
            TextureDimension::D2,
            &[128, 0, 0, 255],
            TextureFormat::Rgba8Unorm,
            default(),
        ));
        let mut material = StandardMaterial {
            base_color: Color::linear_rgb(0.02, 0.02, 0.02),
            emissive: LinearRgba::rgb(100., 60., 4.),
            base_color_texture: Some(red),
            emissive_texture: Some(content.clone()),
            ..default()
        };
        let uv = Vec2::splat(0.1);
        let response = |m: &StandardMaterial| {
            DiffuseMaterial::from_standard(m, &images, &default()).response(uv)
        };
        let (base, emitted) = response(&material);
        assert!(base.abs_diff_eq(Vec3::X * (0.02 * 128. / 255.), 1e-6));
        assert!(emitted.abs_diff_eq(Vec3::new(25., 15., 1.), 0.08));
        material.metallic = 1.;
        assert_eq!(response(&material), (Vec3::ZERO, emitted));
        material.base_color_texture = None;
        assert_eq!(response(&material), (Vec3::ZERO, emitted));
        material.base_color_texture = Some(content);
        let shared = DiffuseMaterial::from_standard(&material, &images, &default());
        assert_eq!(shared.emission_map, EmissionMap::SharedWithBase);
        assert_eq!(shared.response(uv), (Vec3::ZERO, emitted));
        material.emissive_texture = None;
        assert_eq!(response(&material).1, Vec3::new(100., 60., 4.));
    }
}
