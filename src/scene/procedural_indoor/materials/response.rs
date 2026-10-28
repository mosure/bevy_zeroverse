//! Bind the outer surface of a thin conformal coating, after substrate selection.
//! Bevy deliberately leaves clearcoat flat unless its own normal map is bound.
use super::*;

pub(super) fn conformal_coat(material: &mut StandardMaterial) {
    // These callers describe the *finished* surface (pore filling / polish are
    // already in its height field). Share the texture; do not allocate or bake
    // another atlas. Planar cover glass uses a different binding path.
    material.clearcoat_normal_texture = if material.clearcoat > 0. {
        material.normal_map_texture.clone()
    } else {
        None
    };
    material.clearcoat_normal_channel = material.normal_map_channel.clone();
}

#[cfg(test)]
mod tests;
