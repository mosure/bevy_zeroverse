//! Non-emitting projection stock and neutral light-absorbing borders.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{stream, ObjectKind},
    objects::presentation::projection::{self, Coating},
    preparation::AssetStore,
};
use rand::Rng;
use std::collections::BTreeMap;

pub(crate) const MASK: &str = "projection_screen#finish600";
pub(crate) fn label(coating: Coating) -> String {
    format!("projection_screen#finish{}", coating.slot())
}
pub(super) fn supports(s: Surface, slot: usize) -> bool {
    (s == Surface::Ink && slot == 600) || (s == Surface::Paper && (600..603).contains(&slot))
}
fn stock(scene_seed: u64, c: Coating, images: &mut impl AssetStore<Image>) -> StandardMaterial {
    let mut r = stream(
        scene_seed,
        u64::from_le_bytes(*b"AVcloth!") + c.slot() as u64,
    );
    let (value, rough, height) = match c {
        Coating::MatteWhite => (
            r.random_range(0.87..0.965),
            r.random_range(0.84..0.95),
            0.000012,
        ),
        Coating::NeutralGray => (
            r.random_range(0.60..0.79),
            r.random_range(0.77..0.92),
            0.000009,
        ),
        Coating::SatinWhite => (
            r.random_range(0.82..0.94),
            r.random_range(0.57..0.74),
            0.000006,
        ),
    };
    let pitch = Vec2::new(
        r.random_range(0.0032..0.0051),
        r.random_range(0.0030..0.0054),
    );
    let salt = r.random();
    sheet::sheet_material::<32>(images, pitch, 0., 1., |uv| {
        let grain = periodic_noise(uv.x, uv.y, 7, 5, salt) - 0.5;
        let weave =
            (uv.x * 4. * std::f32::consts::TAU).cos() * (uv.y * 4. * std::f32::consts::TAU).cos();
        (
            Vec3::splat(value * (1. + 0.009 * grain)),
            height * (weave + grain * 0.48),
            rough + 0.032 * grain,
        )
    })
}

pub(super) fn install(
    scene: &IndoorManifest,
    images: &mut impl AssetStore<Image>,
    mats: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    let mut used = [false; 4];
    if let Some(s) = selection {
        used[0] = s.finishes.contains(&(Surface::Ink, 600));
        for (i, c) in Coating::ALL.into_iter().enumerate() {
            used[i + 1] = s.finishes.contains(&(Surface::Paper, c.slot()));
        }
    } else {
        for o in scene.objects.iter().filter(|o| {
            o.kind == ObjectKind::ProjectionScreen && o.variant & projection::INSTALLED != 0
        }) {
            let p = projection::parameters(o);
            let i = p.installation.unwrap();
            used[0] = true;
            if p.surface_hi.y > p.surface_lo.y
                || scene.indoor_motion.as_ref().is_some_and(|m| {
                    projection::deployment_selected(
                        o,
                        scene.generator_version,
                        m.policy.projection_screen_fraction,
                    )
                })
            {
                used[1 + i.coating as usize] = true;
            }
        }
    }
    if used[0] {
        variants.insert(
            (Surface::Ink, 600),
            mats.add(StandardMaterial {
                base_color: Color::srgb(0.055, 0.055, 0.059),
                perceptual_roughness: 0.95,
                reflectance: 0.28,
                ..default()
            }),
        );
    }
    for (i, c) in Coating::ALL.into_iter().enumerate() {
        if used[i + 1] {
            variants.insert(
                (Surface::Paper, c.slot()),
                mats.add(stock(scene.material_seed(), c, images)),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn optical_stock_is_dielectric_nonemitting_and_has_filtered_metric_pbr_maps() {
        let mut tones = Vec::new();
        for c in Coating::ALL {
            let mut images = Assets::<Image>::default();
            let m = stock(202, c, &mut images);
            assert_eq!(m.metallic, 0.);
            assert_eq!(m.emissive, LinearRgba::BLACK);
            for handle in [
                &m.base_color_texture,
                &m.normal_map_texture,
                &m.metallic_roughness_texture,
            ] {
                let img = images.get(handle.as_ref().unwrap()).unwrap();
                assert_eq!(img.texture_descriptor.mip_level_count, 6);
            }
            let base = images
                .get(m.base_color_texture.as_ref().unwrap())
                .unwrap()
                .data
                .as_ref()
                .unwrap();
            tones.push(
                base[..32 * 32 * 4]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|p| u32::from(p[0]))
                    .sum::<u32>()
                    / 1024,
            );
            assert!(base
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| p[0] == p[1] && p[1] == p[2]));
            assert!(m.uv_transform.matrix2.x_axis.x > 150.);
        }
        assert!(tones[1] + 12 < tones[0] && tones[1] + 10 < tones[2]);
    }
}
