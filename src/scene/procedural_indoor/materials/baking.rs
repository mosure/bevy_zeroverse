//! Oven browning and crumb pores share a metric height and roughness field.
use super::*;
use crate::scene::procedural_indoor::{
    layout::stream, objects::public::casework::bakery::VERSION, preparation::AssetStore,
};
use rand::Rng;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Finish {
    Crust,
    Crumb,
}

impl Finish {
    pub(super) const ALL: [Self; 2] = [Self::Crust, Self::Crumb];
    pub(crate) fn surface(self) -> Surface {
        match self {
            Self::Crust => Surface::Terracotta,
            Self::Crumb => Surface::Paper,
        }
    }
    pub(super) fn slot(self) -> usize {
        match self {
            Self::Crust => 1170,
            Self::Crumb => 1171,
        }
    }
    pub(crate) fn label(self, version: u32) -> &'static str {
        match (self, version >= VERSION) {
            (Self::Crust, true) => "food#finish1170#uv_local",
            (Self::Crumb, true) => "food#finish1171#uv_local",
            (Self::Crust, false) => "food#finish103",
            (Self::Crumb, false) => "food#finish104",
        }
    }
    pub(super) fn material(
        self,
        seed: u64,
        images: &mut impl AssetStore<Image>,
    ) -> StandardMaterial {
        let mut r = stream(seed, self.slot() as u64);
        let period = Vec2::splat(r.random_range(0.067..0.098));
        let toast = r.random_range(-0.14..0.21);
        let key = r.random::<u64>();
        let (light, dark, relief, roughness) = match self {
            Self::Crust => ([0.91, 0.73, 0.46], [0.47, 0.24, 0.10], 0.00062, 0.60),
            Self::Crumb => ([0.94, 0.84, 0.65], [0.71, 0.48, 0.26], 0.00086, 0.77),
        };
        sheet::sheet_material::<32>(images, period, 0., 0.55, |uv| {
            let broad = periodic_noise(uv.x, uv.y, 4, 5, key);
            let detail = periodic_noise(uv.x, uv.y, 13, 15, key.rotate_left(31));
            let pore = ((detail - 0.50) * 3.2).clamp(0., 1.);
            let brown = (broad * 1.10 + 0.18 * detail + toast - 0.13).clamp(0., 1.);
            (
                Vec3::from(light).lerp(Vec3::from(dark), brown) * (1. - 0.12 * pore),
                relief * (0.50 * broad - pore * pore),
                roughness + 0.11 * pore + 0.055 * broad,
            )
        })
    }
}

pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    matches!(
        (surface, slot),
        (Surface::Terracotta, 1170) | (Surface::Paper, 1171)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn crust_and_crumb_have_repeatable_dielectric_relief_and_filtered_mips() {
        let mut images = Assets::<Image>::default();
        for finish in Finish::ALL {
            let bake = finish.material(41, &mut images);
            let replay = finish.material(41, &mut images);
            let other = finish.material(61, &mut images);
            assert_eq!(bake.metallic, 0.);
            let maps = |m: &StandardMaterial| {
                [
                    m.base_color_texture.as_ref().unwrap().id(),
                    m.normal_map_texture.as_ref().unwrap().id(),
                    m.metallic_roughness_texture.as_ref().unwrap().id(),
                ]
            };
            for (i, ((a, b), c)) in maps(&bake)
                .into_iter()
                .zip(maps(&replay))
                .zip(maps(&other))
                .enumerate()
            {
                let map = images.get(a).unwrap();
                assert_eq!(map.texture_descriptor.format.is_srgb(), i == 0);
                assert_eq!(map.texture_descriptor.mip_level_count, 6);
                let bytes = map.data.as_ref().unwrap();
                assert_eq!(bytes.len(), 4 * (1024 + 256 + 64 + 16 + 4 + 1));
                assert_eq!(map.data, images.get(b).unwrap().data);
                assert_ne!(map.data, images.get(c).unwrap().data);
                let texels = bytes[..4096].as_chunks::<4>().0;
                match i {
                    0 => assert!(texels.iter().all(|p| p[0] > p[1] && p[1] > p[2])),
                    1 => assert!(texels
                        .iter()
                        .any(|p| p[0].abs_diff(128) > 7 || p[1].abs_diff(128) > 7)),
                    _ => {
                        assert!(bytes.as_chunks::<4>().0.iter().all(|p| p[2] == 0));
                        let (lo, hi) = texels
                            .iter()
                            .map(|p| p[1])
                            .fold((255, 0), |(lo, hi), v| (lo.min(v), hi.max(v)));
                        assert!(lo >= 150 && hi - lo > 16);
                    }
                }
            }
        }
    }
}
