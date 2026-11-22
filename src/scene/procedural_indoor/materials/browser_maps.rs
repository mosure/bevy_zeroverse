//! CPU-only texture programs can take longer than a full frame even for one
//! substrate. Evaluate them on the existing Studio worker and transfer the
//! three RGBA planes. The ordinary filter owns mipmaps, PBR formats and samplers.
use super::{mapped_images, program::MaterialRecipe, texture_maps, Surface, TextureMaps};
use crate::scene::procedural_indoor::preparation::pipeline::browser;
use bevy::prelude::Image;
use wasm_bindgen::prelude::*;
use web_sys::js_sys::{Array, Uint8Array};

pub(super) async fn prepare(
    surface: Surface,
    floor: u32,
    seed: u64,
    roughness: f32,
    recipe: Option<&MaterialRecipe>,
) -> [Image; 3] {
    let size = recipe.map_or(256, |r| r.map_size(floor));
    let maps = if browser::worker_enabled() {
        let request = rmp_serde::to_vec_named(&(surface, floor, seed, roughness, recipe))
            .expect("serializable material recipe");
        let result = browser::run("materialMaps", &Uint8Array::from(request.as_slice())).await;
        result.and_then(|value| unpack(value, size))
    } else {
        None
    };
    mapped_images(
        maps.unwrap_or_else(|| texture_maps(surface, floor, seed, roughness, recipe)),
        size,
        recipe,
    )
}

fn unpack(result: JsValue, size: u32) -> Option<TextureMaps> {
    let planes = result.dyn_into::<Array>().ok()?;
    if planes.length() != 3 {
        return None;
    }
    let plane = |index| {
        let plane = planes.get(index).dyn_into::<Uint8Array>().ok()?;
        (plane.length() == size * size * 4).then(|| plane.to_vec())
    };
    Some((plane(0)?, plane(1)?, plane(2)?))
}

#[wasm_bindgen(js_name = zeroverseMaterialMaps)]
pub fn material_maps(request: &[u8]) -> Result<Array, JsError> {
    let (surface, floor, seed, roughness, recipe): (_, _, _, _, Option<MaterialRecipe>) =
        rmp_serde::from_slice(request)?;
    let (color, normal, data) = texture_maps(surface, floor, seed, roughness, recipe.as_ref());
    Ok(Array::of3(
        &Uint8Array::from(color.as_slice()),
        &Uint8Array::from(normal.as_slice()),
        &Uint8Array::from(data.as_slice()),
    ))
}
