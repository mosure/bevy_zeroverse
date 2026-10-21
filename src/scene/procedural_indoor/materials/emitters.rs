//! Lightweight per-lamp emission bindings; no new image synthesis or lights.
use super::*;
use crate::scene::procedural_indoor::{
    layout::ObjectKind, objects, photometry, preparation::AssetStore,
};
use std::collections::BTreeMap;

pub(crate) fn source_id(label: &str) -> Option<usize> {
    label
        .split('#')
        .find_map(|tag| tag.strip_prefix("source")?.parse().ok())
}

/// Eight reserved channels per instance; unused channels are never allocated.
pub(crate) fn source_key(label: &str) -> Option<(usize, usize)> {
    let id = source_id(label)?;
    let channel = label
        .split('#')
        .find_map(|s| s.strip_prefix("bulb")?.parse::<usize>().ok())
        .unwrap_or(0);
    (channel < 8).then_some((id, channel))
}
pub(crate) fn transport_slot(label: &str) -> Option<usize> {
    let (id, channel) = source_key(label)?;
    usize::MAX.checked_sub(id.checked_mul(8)?.checked_add(channel)?)
}

pub(crate) fn area(object: &super::super::layout::IndoorObject) -> f32 {
    match object.kind {
        ObjectKind::FloorLamp => objects::standing_lamp::emitting_area(object),
        ObjectKind::DeskLamp => objects::accessories::lamps::emitting_area(object),
        ObjectKind::CeilingPendant => {
            let p = objects::equipment::parameters(object);
            std::f32::consts::PI
                * (object.size.x.min(object.size.z) * 0.46 * p.diffuser_fraction).powi(2)
        }
        ObjectKind::WallSconce => {
            let p = objects::equipment::parameters(object);
            object.size.x * p.sconce_width * p.diffuser_fraction * object.size.z * 0.50
        }
        _ => 1.,
    }
}

pub(super) fn build(
    scene: &IndoorManifest,
    template: &Handle<StandardMaterial>,
    materials: &mut impl AssetStore<StandardMaterial>,
) -> BTreeMap<(usize, usize), Handle<StandardMaterial>> {
    let template = materials.get(template).expect("light material").clone();
    scene
        .objects
        .iter()
        .filter(|o| photometry::is_emitter(o.kind))
        .flat_map(|o| (0..photometry::emission_channels(o)).map(move |channel| (o, channel)))
        .map(|(o, channel)| {
            if let Some(emission) = photometry::decorative_emission_channel(o, channel) {
                let mut material = template.clone();
                material.emissive = emission;
                return ((o.id, channel), materials.add(material));
            }
            let power = photometry::lamp(scene, o);
            let c = Color::srgb(power.color.x, power.color.y, power.color.z).to_linear();
            let radiance = power.lumens / (std::f32::consts::PI * area(o).max(0.0001));
            let mut material = template.clone();
            material.emissive =
                LinearRgba::new(c.red * radiance, c.green * radiance, c.blue * radiance, 1.);
            ((o.id, channel), materials.add(material))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn colored_strands_bind_distinct_bounded_channels_without_textures() {
        let mut scene =
            IndoorManifest::generate_with_humans(0, Default::default(), 0.6, 0, 0.).unwrap();
        scene.objects = (0..32)
            .map(|seed| {
                let mut o = objects::audit::fixture(ObjectKind::StringLights, seed);
                o.id = seed as usize;
                o
            })
            .collect();
        let mut materials = Assets::<StandardMaterial>::default();
        let base = materials.add(StandardMaterial::default());
        let bindings = build(&scene, &base, &mut materials);
        let mut slots = std::collections::BTreeSet::new();
        let mut colored = 0;
        for o in &scene.objects {
            let p = objects::equipment::signals::string_program(o);
            colored += usize::from(p.colors_srgb.len() > 1);
            let assembly = objects::build_object(o);
            let mut channels = std::collections::BTreeSet::new();
            for ((surface, label), _) in assembly.parts {
                if surface != Surface::Light {
                    continue;
                }
                let (id, channel) = source_key(&label).unwrap();
                assert_eq!(id, o.id);
                assert!(channels.insert(channel));
                assert!(slots.insert(transport_slot(&label).unwrap()));
                let mat = materials.get(&bindings[&(id, channel)]).unwrap();
                assert!(mat.emissive_texture.is_none() && mat.base_color_texture.is_none());
                assert_eq!(
                    mat.emissive,
                    photometry::decorative_emission_channel(o, channel).unwrap()
                );
                let c = Color::srgb_from_array(p.colors_srgb[channel]).to_linear();
                let rgb = Vec3::new(mat.emissive.red, mat.emissive.green, mat.emissive.blue);
                let expected = Vec3::new(c.red, c.green, c.blue);
                assert!((rgb.normalize() - expected.normalize()).length() < 1e-6);
                assert!(rgb.is_finite() && rgb.max_element() <= 2500.);
            }
            assert_eq!(channels.len(), p.colors_srgb.len());
        }
        assert!(colored > 8);
        assert_eq!(bindings.len(), slots.len());
        assert_eq!(materials.len(), bindings.len() + 1);
        assert!(transport_slot("string_lights#source18446744073709551615#bulb7").is_none());
    }

    #[test]
    fn aperture_materials_match_each_lamp_and_need_no_new_images() {
        let mut scene =
            IndoorManifest::generate_with_humans(6, Default::default(), 0.65, 0, 0.).unwrap();
        scene.objects.clear();
        for kind in [
            ObjectKind::FloorLamp,
            ObjectKind::DeskLamp,
            ObjectKind::CeilingPendant,
            ObjectKind::WallSconce,
        ] {
            for seed in 0..32 {
                let mut o = objects::audit::fixture(kind, seed);
                o.id = scene.objects.len();
                scene.objects.push(o);
            }
        }
        let mut materials = Assets::<StandardMaterial>::default();
        let base = materials.add(StandardMaterial::default());
        let variants = build(&scene, &base, &mut materials);
        assert_eq!(variants.len(), scene.objects.len());
        for o in &scene.objects {
            let power = photometry::lamp(&scene, o);
            let mat = materials.get(&variants[&(o.id, 0)]).unwrap();
            let c = Color::srgb(power.color.x, power.color.y, power.color.z).to_linear();
            let scale = power.lumens / (std::f32::consts::PI * area(o).max(0.0001));
            assert_eq!(
                mat.emissive,
                LinearRgba::new(c.red * scale, c.green * scale, c.blue * scale, 1.)
            );
            assert!(mat.emissive_texture.is_none());
            for ((surface, label), _) in objects::build_object(o).parts {
                if surface == Surface::Light {
                    assert_eq!(source_id(&label), Some(o.id));
                    assert!(
                        variants.contains_key(&source_key(&format!("{label}#motion0")).unwrap())
                    );
                }
            }
        }
    }
}
