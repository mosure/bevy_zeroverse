//! Addressing for the reduced albedo maps used by CPU and compute light transport.
//! The two low bit pairs select U/V. The high bit selects white for a border tap.
use bevy::{
    image::{ImageAddressMode, ImageSampler, ImageSamplerBorderColor, ImageSamplerDescriptor},
    prelude::*,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct TextureAddress(u32);

impl TextureAddress {
    pub(super) fn for_image(image: &Image) -> Self {
        match &image.sampler {
            ImageSampler::Descriptor(sampler) => Self::from_descriptor(sampler),
            // Zeroverse uses Bevy's default clamp-to-edge sampler.
            ImageSampler::Default => Self(1 | (1 << 2)),
        }
    }

    pub(super) fn from_descriptor(s: &ImageSamplerDescriptor) -> Self {
        let code = |mode| match mode {
            ImageAddressMode::Repeat => 0,
            ImageAddressMode::ClampToEdge => 1,
            ImageAddressMode::MirrorRepeat => 2,
            ImageAddressMode::ClampToBorder => 3,
        };
        let white = s.border_color == Some(ImageSamplerBorderColor::OpaqueWhite)
            && [s.address_mode_u, s.address_mode_v].contains(&ImageAddressMode::ClampToBorder);
        Self(code(s.address_mode_u) | (code(s.address_mode_v) << 2) | (u32::from(white) << 4))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn gpu_flags(self) -> u32 {
        self.0
    }

    pub(super) fn sample(self, uv: Vec2, n: u32, pixels: &[Vec3]) -> Vec3 {
        if self.0 == 0 {
            let wrapped = uv.rem_euclid(Vec2::ONE);
            let x = (wrapped.x * n as f32) as u32 % n;
            let y = (wrapped.y * n as f32) as u32 % n;
            return pixels[(y * n + x) as usize];
        }
        let index = |value: f32, mode| {
            let coordinate = match mode {
                0 => return Some((value.rem_euclid(1.) * n as f32) as u32 % n),
                2 => 1. - (value.rem_euclid(2.) - 1.).abs(),
                3 if !(0.0..1.0).contains(&value) => return None,
                _ => value.clamp(0., 1.),
            };
            Some(((coordinate * n as f32) as u32).min(n - 1))
        };
        match (index(uv.x, self.0 & 3), index(uv.y, (self.0 >> 2) & 3)) {
            (Some(x), Some(y)) => pixels[(y * n + x) as usize],
            _ if self.0 & 16 != 0 => Vec3::ONE,
            _ => Vec3::ZERO,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_axes_and_exact_endpoints_follow_image_sampling() {
        use ImageAddressMode::*;
        let texels = [Vec3::X, Vec3::Y, Vec3::Z, Vec3::ONE];
        let at = |u, v, uv, border| {
            TextureAddress::from_descriptor(&ImageSamplerDescriptor {
                address_mode_u: u,
                address_mode_v: v,
                border_color: border,
                ..default()
            })
            .sample(uv, 2, &texels)
        };
        let p = Vec2::new(-0.25, 1.25);
        assert_eq!(at(Repeat, Repeat, p, None), Vec3::Y);
        assert_eq!(at(ClampToEdge, ClampToEdge, p, None), Vec3::Z);
        assert_eq!(at(Repeat, ClampToEdge, p, None), Vec3::ONE);
        assert_eq!(at(MirrorRepeat, Repeat, p, None), Vec3::X);
        assert_eq!(
            at(ClampToEdge, MirrorRepeat, Vec2::new(1., 1.), None),
            Vec3::ONE
        );
        assert_eq!(at(Repeat, Repeat, Vec2::ONE, None), Vec3::X);
        assert_eq!(at(Repeat, ClampToBorder, Vec2::ONE, None), Vec3::ZERO);
        assert_eq!(
            at(
                ClampToBorder,
                ClampToEdge,
                p,
                Some(ImageSamplerBorderColor::OpaqueWhite)
            ),
            Vec3::ONE
        );
        assert_eq!(at(ClampToBorder, ClampToEdge, Vec2::ZERO, None), Vec3::X);
    }
}
