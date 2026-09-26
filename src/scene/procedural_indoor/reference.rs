//! Lossless geometry/camera bridge for independent offline renderer comparisons.
//! This is an opt-in diagnostic export; it never changes dataset RGB or labels.
use crate::sample::Sample;
use anyhow::{bail, ensure, Context, Result};
use bevy::{
    asset::AssetId,
    mesh::{PrimitiveTopology, VertexAttributeValues},
    prelude::*,
    render::render_resource::{TextureDimension, TextureFormat},
};
use serde_json::{json, Value};
use std::{collections::HashMap, fs, io::Write, path::Path};

/// Export the live, generated assets and the exact sampled camera matrices.
/// Binary arrays are little endian, with offsets measured in bytes. Matrices are
/// column major, metres are the length unit, and textures retain Bevy's UV origin.
pub fn export(world: &mut World, sample: &Sample, size: [u32; 2], directory: &Path) -> Result<()> {
    ensure!(
        sample.indoor.is_some(),
        "reference export requires an indoor scene"
    );
    ensure!(
        !directory.exists(),
        "reference export already exists: {}",
        directory.display()
    );
    fs::create_dir_all(directory.join("textures"))?;
    let instances: Vec<_> = world
        .query::<(
            &Mesh3d,
            &MeshMaterial3d<StandardMaterial>,
            &GlobalTransform,
            Option<&Name>,
        )>()
        .iter(world)
        .map(|(mesh, material, transform, name)| {
            (
                mesh.0.id(),
                material.0.id(),
                transform.to_matrix().to_cols_array_2d(),
                name.map_or_else(String::new, ToString::to_string),
            )
        })
        .collect();
    let mut lights = Vec::new();
    for (light, transform) in world
        .query::<(&DirectionalLight, &GlobalTransform)>()
        .iter(world)
    {
        lights.push(json!({"kind":"sun", "color":light.color.to_linear().to_f32_array(),
            "illuminance":light.illuminance, "world_from_light":transform.to_matrix().to_cols_array_2d()}));
    }
    for (light, transform, name) in world
        .query::<(&PointLight, &GlobalTransform, Option<&Name>)>()
        .iter(world)
    {
        lights.push(
            json!({"kind":"point", "color":light.color.to_linear().to_f32_array(),
            "intensity":light.intensity,"radius":light.radius,
            "name":name.map(ToString::to_string),
            "world_from_light":transform.to_matrix().to_cols_array_2d()}),
        );
    }
    for (light, transform, name) in world
        .query::<(&SpotLight, &GlobalTransform, Option<&Name>)>()
        .iter(world)
    {
        lights.push(
            json!({"kind":"spot", "color":light.color.to_linear().to_f32_array(),
            "intensity":light.intensity,"radius":light.radius,
            "inner_angle":light.inner_angle,"outer_angle":light.outer_angle,
            "name":name.map(ToString::to_string),
            "world_from_light":transform.to_matrix().to_cols_array_2d()}),
        );
    }
    let meshes = world.resource::<Assets<Mesh>>();
    let materials = world.resource::<Assets<StandardMaterial>>();
    let images = world.resource::<Assets<Image>>();
    let mut mesh_ids = HashMap::new();
    let mut material_ids = HashMap::new();
    let mut image_ids = HashMap::new();
    let mut exported_meshes = Vec::new();
    let mut exported_materials = Vec::new();
    let mut exported_instances = Vec::new();
    let mut binary = std::io::BufWriter::new(fs::File::create(directory.join("geometry.bin"))?);
    let mut offset = 0u64;
    for (mesh_id, material_id, matrix, name) in instances {
        let next_mesh = mesh_ids.len();
        let mesh_index = *mesh_ids.entry(mesh_id).or_insert(next_mesh);
        if mesh_index == exported_meshes.len() {
            let mesh = meshes
                .get(mesh_id)
                .context("reference mesh missing from main world")?;
            ensure!(
                mesh.primitive_topology() == PrimitiveTopology::TriangleList,
                "unsupported reference topology"
            );
            let Some(VertexAttributeValues::Float32x3(positions)) =
                mesh.attribute(Mesh::ATTRIBUTE_POSITION)
            else {
                bail!("missing float3 positions")
            };
            let Some(VertexAttributeValues::Float32x3(normals)) =
                mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
            else {
                bail!("missing float3 normals")
            };
            let Some(VertexAttributeValues::Float32x2(uvs)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0)
            else {
                bail!("missing float2 UVs")
            };
            ensure!(
                positions.len() == normals.len() && positions.len() == uvs.len(),
                "reference attribute counts differ"
            );
            let indices: Vec<u32> = mesh.indices().map_or_else(
                || (0..positions.len() as u32).collect(),
                |indices| indices.iter().map(|i| i as u32).collect(),
            );
            ensure!(
                indices.len().is_multiple_of(3)
                    && indices.iter().all(|&i| (i as usize) < positions.len()),
                "invalid reference indices"
            );
            let start = offset;
            for values in [
                positions.as_flattened(),
                normals.as_flattened(),
                uvs.as_flattened(),
            ] {
                for &v in values {
                    ensure!(v.is_finite(), "nonfinite reference vertex");
                    binary.write_all(&v.to_le_bytes())?;
                    offset += 4;
                }
            }
            for &v in &indices {
                binary.write_all(&v.to_le_bytes())?;
                offset += 4;
            }
            exported_meshes
                .push(json!({"offset":start, "vertices":positions.len(), "indices":indices.len()}));
        }
        let next_material = material_ids.len();
        let material_index = *material_ids.entry(material_id).or_insert(next_material);
        if material_index == exported_materials.len() {
            let m = materials
                .get(material_id)
                .context("reference material missing")?;
            let mut texture = |handle: &Option<Handle<Image>>| -> Result<Value> {
                export_texture(handle, images, &mut image_ids, directory)
            };
            let uv = m.uv_transform;
            exported_materials.push(json!({
                "base_color":m.base_color.to_linear().to_f32_array(),
                "emissive":m.emissive.to_f32_array(), "emissive_exposure_weight":m.emissive_exposure_weight,
                "roughness":m.perceptual_roughness, "metallic":m.metallic, "reflectance":m.reflectance,
                "specular_transmission":m.specular_transmission,"diffuse_transmission":m.diffuse_transmission,
                "thickness":m.thickness,"ior":m.ior,
                "anisotropy_strength":m.anisotropy_strength,"anisotropy_rotation":m.anisotropy_rotation,
                "attenuation_color":m.attenuation_color.to_linear().to_f32_array(),
                "attenuation_distance":if m.attenuation_distance.is_finite() {Some(m.attenuation_distance)} else {None},
                "double_sided":m.double_sided,"alpha_mode":format!("{:?}",m.alpha_mode),
                "unlit":m.unlit,"uv_matrix":[uv.matrix2.x_axis.to_array(),uv.matrix2.y_axis.to_array(),uv.translation.to_array()],
                "base_color_texture":texture(&m.base_color_texture)?,
                "normal_map_texture":texture(&m.normal_map_texture)?,
                "metallic_roughness_texture":texture(&m.metallic_roughness_texture)?,
                "emissive_texture":texture(&m.emissive_texture)?,
            }));
        }
        exported_instances.push(json!({"mesh":mesh_index,"material":material_index,"world_from_mesh":matrix,"name":name}));
    }
    binary.flush()?;
    let environment = world.resource::<super::IndoorEnvironment>();
    let cameras: Vec<_> = sample.views.iter().enumerate().map(|(index,v)| json!({
        "index":index,"world_from_view":v.world_from_view,"fov_y":v.fovy,"near":v.near,"far":v.far,"time":v.time
    })).collect();
    // The reference uses actual emission and visibility, not the raster lighting proxies.
    let document = json!({
        "format":"zeroverse-reference-v1","seed":sample.indoor.as_ref().unwrap().seed,
        "capture_engine":crate::CAPTURE_ENGINE_IDENTITY,
        "image_size":size,"geometry_bytes":offset,"annotation_aabb":sample.aabb,
        "coordinates":"right handed Y up; camera -Z forward +Y up; column-major matrices; metres",
        "textures":"PNG; Bevy top-left UV origin; color textures sRGB, data textures linear",
        "radiometry":"RGB photometric proxy units: directional irradiance in lux, point intensity in lumens, emission in cd/m2; no spectral conversion",
        "ev100":environment.ev100,"environment_intensity":environment.map.intensity,
        "world_yaw":sample.indoor.as_ref().unwrap().world_yaw,
        "meshes":exported_meshes,"materials":exported_materials,"instances":exported_instances,
        "lights":lights,"cameras":cameras,
    });
    fs::write(
        directory.join("scene.json"),
        serde_json::to_vec_pretty(&document)?,
    )?;
    Ok(())
}

fn export_texture(
    handle: &Option<Handle<Image>>,
    images: &Assets<Image>,
    ids: &mut HashMap<AssetId<Image>, Value>,
    directory: &Path,
) -> Result<Value> {
    let Some(handle) = handle else {
        return Ok(Value::Null);
    };
    if let Some(value) = ids.get(&handle.id()) {
        return Ok(value.clone());
    }
    let img = images
        .get(handle)
        .context("reference image missing from main world")?;
    let desc = &img.texture_descriptor;
    ensure!(
        desc.dimension == TextureDimension::D2 && desc.size.depth_or_array_layers == 1,
        "reference material texture must be 2D"
    );
    ensure!(
        matches!(
            desc.format,
            TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb
        ),
        "unsupported reference texture format"
    );
    let count = desc.size.width as usize * desc.size.height as usize * 4;
    let pixels = img
        .data
        .as_ref()
        .context("reference texture has no CPU pixels")?
        .get(..count)
        .context("truncated reference texture")?;
    let name = format!("textures/{:03}.png", ids.len());
    image::save_buffer(
        directory.join(&name),
        pixels,
        desc.size.width,
        desc.size.height,
        image::ColorType::Rgba8,
    )?;
    let result = json!({"path":name,"srgb":desc.format == TextureFormat::Rgba8UnormSrgb});
    ids.insert(handle.id(), result.clone());
    Ok(result)
}
