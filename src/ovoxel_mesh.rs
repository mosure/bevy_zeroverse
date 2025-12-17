use std::{collections::HashMap, fs, io::Write, path::Path};

use anyhow::{anyhow, Context, Result};
use bevy::{asset::RenderAssetUsages, prelude::*, render::render_resource::PrimitiveTopology};
use bevy_mesh::{Indices, MeshVertexAttribute, VertexAttributeValues, VertexFormat};
use bytemuck::cast_slice;
use serde_json::{json, Map, Value};

use crate::{ovoxel::OvoxelVolume, render::semantic::SemanticLabel};

/// Custom vertex attribute for semantic ids exported from O-Voxel volumes.
pub const ATTRIBUTE_SEMANTIC_ID: MeshVertexAttribute =
    MeshVertexAttribute::new("SemanticId", 0x5E11_1C01, VertexFormat::Uint32);

/// Builds a CPU mesh from an `OvoxelVolume` by emitting dual-grid faces.
pub fn ovoxel_to_mesh(volume: &OvoxelVolume) -> Mesh {
    ovoxel_to_mesh_internal(volume, false)
}

/// Builds a CPU mesh from an `OvoxelVolume` with colors derived from semantic labels.
pub fn ovoxel_to_semantic_mesh(volume: &OvoxelVolume) -> Mesh {
    ovoxel_to_mesh_internal(volume, true)
}

fn ovoxel_to_mesh_internal(volume: &OvoxelVolume, semantic_colors: bool) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );

    if volume.coords.is_empty() {
        return mesh;
    }

    let min = Vec3::from(volume.aabb[0]);
    let max = Vec3::from(volume.aabb[1]);
    let extent = (max - min).max(Vec3::splat(f32::EPSILON));
    let res_f = volume.resolution.max(1) as f32;
    let voxel_size = extent / res_f;

    let mut index_map = HashMap::new();
    for (i, c) in volume.coords.iter().enumerate() {
        index_map.insert((c[0], c[1], c[2]), i);
    }

    let dual_pos: Vec<Vec3> = volume
        .coords
        .iter()
        .zip(volume.dual_vertices.iter())
        .map(|(c, dual)| {
            let voxel_min = min + Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32) * voxel_size;
            let dual_offset = Vec3::new(
                dual[0] as f32 / 255.0,
                dual[1] as f32 / 255.0,
                dual[2] as f32 / 255.0,
            );
            voxel_min + voxel_size * dual_offset
        })
        .collect();

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut colors: Vec<[f32; 4]> = Vec::new();
    let mut semantics: Vec<u32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let color_scale = 1.0 / 255.0;

    // helper to push a quad (two triangles) with consistent winding
    let mut push_quad = |quad: [Vec3; 4], color: [f32; 4], semantic_id: u32| {
        let base = positions.len() as u32;
        let n = (quad[1] - quad[0])
            .cross(quad[2] - quad[0])
            .normalize_or_zero();
        for v in quad {
            positions.push([v.x, v.y, v.z]);
            normals.push([n.x, n.y, n.z]);
            colors.push(color);
            semantics.push(semantic_id);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };

    // For each voxel, emit quads only for the three canonical edges (+X, +Y, +Z from the voxel min corner).
    // Each quad connects the four dual vertices of the cells touching that primal edge. A quad is emitted if
    // any of those four cells reports an intersection flag for the corresponding axis.
    for coord in volume.coords.iter() {
        let base = (coord[0], coord[1], coord[2]);
        let idx_at = |dx: i32, dy: i32, dz: i32| -> Option<usize> {
            let key = (
                (base.0 as i32 + dx) as u32,
                (base.1 as i32 + dy) as u32,
                (base.2 as i32 + dz) as u32,
            );
            index_map.get(&key).copied()
        };

        let color_for = |i: usize| -> [f32; 4] {
            let semantic_id = *volume.semantics.get(i).unwrap_or(&0) as u32;
            let semantic_color = semantic_colors
                .then(|| palette_color(semantic_id as u16, &volume.semantic_labels))
                .flatten();
            let base_color = volume
                .base_color
                .get(i)
                .copied()
                .unwrap_or([255, 255, 255, 255]);
            if let Some(sem) = semantic_color {
                sem
            } else {
                [
                    base_color[0] as f32 * color_scale,
                    base_color[1] as f32 * color_scale,
                    base_color[2] as f32 * color_scale,
                    base_color[3] as f32 * color_scale,
                ]
            }
        };
        let sem_for = |i: usize| *volume.semantics.get(i).unwrap_or(&0) as u32;

        // Edge along +X: cells (0,0,0), (0,1,0), (0,0,1), (0,1,1)
        if let (Some(a), Some(b), Some(c), Some(d)) = (
            idx_at(0, 0, 0),
            idx_at(0, 1, 0),
            idx_at(0, 0, 1),
            idx_at(0, 1, 1),
        ) {
            push_quad(
                [dual_pos[a], dual_pos[b], dual_pos[d], dual_pos[c]],
                color_for(a),
                sem_for(a),
            );
        }

        // Edge along +Y: cells (0,0,0), (1,0,0), (0,0,1), (1,0,1)
        if let (Some(a), Some(b), Some(c), Some(d)) = (
            idx_at(0, 0, 0),
            idx_at(1, 0, 0),
            idx_at(0, 0, 1),
            idx_at(1, 0, 1),
        ) {
            push_quad(
                [dual_pos[a], dual_pos[b], dual_pos[d], dual_pos[c]],
                color_for(a),
                sem_for(a),
            );
        }

        // Edge along +Z: cells (0,0,0), (1,0,0), (0,1,0), (1,1,0)
        if let (Some(a), Some(b), Some(c), Some(d)) = (
            idx_at(0, 0, 0),
            idx_at(1, 0, 0),
            idx_at(0, 1, 0),
            idx_at(1, 1, 0),
        ) {
            push_quad(
                [dual_pos[a], dual_pos[b], dual_pos[d], dual_pos[c]],
                color_for(a),
                sem_for(a),
            );
        }
    }

    if positions.is_empty() {
        return mesh;
    }

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_attribute(
        ATTRIBUTE_SEMANTIC_ID,
        VertexAttributeValues::Uint32(semantics),
    );
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn palette_color(semantic_id: u16, labels: &[String]) -> Option<[f32; 4]> {
    let label = labels.get(semantic_id as usize)?;
    let known = SemanticLabel::from_label(label)?;
    let c = known.color().to_linear();
    Some([c.red, c.green, c.blue, c.alpha])
}

fn align_to(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

#[allow(clippy::too_many_arguments)]
fn push_view(
    buffer: &mut Vec<u8>,
    component_bytes: &[u8],
    target: Option<u32>,
    buffer_views: &mut Vec<Value>,
    accessors: &mut Vec<Value>,
    accessor_type: &str,
    component_type: u32,
    count: usize,
    min_max: Option<(Vec<f32>, Vec<f32>)>,
) -> usize {
    let start = align_to(buffer.len(), 4);
    buffer.resize(start, 0);
    buffer.extend_from_slice(component_bytes);

    let view_index = buffer_views.len();
    buffer_views.push(json!({
        "buffer": 0,
        "byteOffset": start,
        "byteLength": component_bytes.len(),
        "target": target,
    }));

    let mut accessor = json!({
        "bufferView": view_index,
        "componentType": component_type,
        "count": count,
        "type": accessor_type,
    });

    if let Some((min, max)) = min_max {
        if let Value::Object(map) = &mut accessor {
            map.insert("min".to_string(), json!(min));
            map.insert("max".to_string(), json!(max));
        }
    }

    accessors.push(accessor);
    accessors.len() - 1
}

/// Writes a minimal GLB with positions/normals/colors and the semantic id attribute.
pub fn write_mesh_as_glb(mesh: &Mesh, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let positions: Vec<[f32; 3]> = mesh
        .attribute(Mesh::ATTRIBUTE_POSITION)
        .and_then(|a| a.as_float3())
        .map(|v| v.to_vec())
        .ok_or_else(|| anyhow!("mesh missing positions attribute"))?;

    let normals: Option<Vec<[f32; 3]>> =
        mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
            .and_then(|a| match a {
                VertexAttributeValues::Float32x3(v) => Some(v.clone()),
                _ => None,
            });

    let colors: Vec<[f32; 4]> = mesh
        .attribute(Mesh::ATTRIBUTE_COLOR)
        .and_then(|a| match a {
            VertexAttributeValues::Float32x4(v) => Some(v.clone()),
            VertexAttributeValues::Float32x3(v) => Some(
                v.iter()
                    .map(|c| [c[0], c[1], c[2], 1.0])
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .unwrap_or_else(|| vec![[1.0, 1.0, 1.0, 1.0]; positions.len()]);

    let semantics: Option<Vec<u32>> =
        mesh.attribute(ATTRIBUTE_SEMANTIC_ID.id)
            .and_then(|a| match a {
                VertexAttributeValues::Uint32(v) => Some(v.clone()),
                _ => None,
            });

    let indices: Vec<u32> = match mesh.indices() {
        Some(Indices::U32(idx)) => idx.clone(),
        Some(Indices::U16(idx)) => idx.iter().map(|i| *i as u32).collect(),
        None => (0..positions.len() as u32).collect(),
    };

    let mut buffer = Vec::new();
    let mut buffer_views = Vec::new();
    let mut accessors = Vec::new();

    let pos_floats: Vec<f32> = positions.iter().flat_map(|p| p.iter().copied()).collect();
    let pos_min = [
        positions.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min),
        positions.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min),
        positions.iter().map(|p| p[2]).fold(f32::INFINITY, f32::min),
    ];
    let pos_max = [
        positions
            .iter()
            .map(|p| p[0])
            .fold(f32::NEG_INFINITY, f32::max),
        positions
            .iter()
            .map(|p| p[1])
            .fold(f32::NEG_INFINITY, f32::max),
        positions
            .iter()
            .map(|p| p[2])
            .fold(f32::NEG_INFINITY, f32::max),
    ];
    let pos_accessor = push_view(
        &mut buffer,
        cast_slice(&pos_floats),
        Some(34962),
        &mut buffer_views,
        &mut accessors,
        "VEC3",
        5126,
        positions.len(),
        Some((pos_min.to_vec(), pos_max.to_vec())),
    );

    let normal_accessor = normals.as_ref().map(|normals| {
        let normals_f: Vec<f32> = normals.iter().flat_map(|n| n.iter().copied()).collect();
        push_view(
            &mut buffer,
            cast_slice(&normals_f),
            Some(34962),
            &mut buffer_views,
            &mut accessors,
            "VEC3",
            5126,
            normals.len(),
            None,
        )
    });

    let color_f: Vec<f32> = colors.iter().flat_map(|c| c.iter().copied()).collect();
    let color_accessor = push_view(
        &mut buffer,
        cast_slice(&color_f),
        Some(34962),
        &mut buffer_views,
        &mut accessors,
        "VEC4",
        5126,
        colors.len(),
        None,
    );

    let semantic_accessor = semantics.as_ref().map(|sem| {
        push_view(
            &mut buffer,
            cast_slice(sem),
            Some(34962),
            &mut buffer_views,
            &mut accessors,
            "SCALAR",
            5125,
            sem.len(),
            None,
        )
    });

    let idx_accessor = push_view(
        &mut buffer,
        cast_slice(&indices),
        Some(34963),
        &mut buffer_views,
        &mut accessors,
        "SCALAR",
        5125,
        indices.len(),
        Some((
            vec![0.0],
            vec![indices.iter().copied().max().unwrap_or(0) as f32],
        )),
    );

    let mut attributes = Map::new();
    attributes.insert("POSITION".to_string(), json!(pos_accessor));
    if let Some(norm_idx) = normal_accessor {
        attributes.insert("NORMAL".to_string(), json!(norm_idx));
    }
    attributes.insert("COLOR_0".to_string(), json!(color_accessor));
    if let Some(sem_idx) = semantic_accessor {
        attributes.insert("_SEMANTIC_ID".to_string(), json!(sem_idx));
    }

    let gltf = json!({
        "asset": { "version": "2.0" },
        "extensionsUsed": ["KHR_materials_unlit"],
        "buffers": [{ "byteLength": buffer.len() }],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "materials": [{
            "doubleSided": true,
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0
            },
            "extensions": {
                "KHR_materials_unlit": {}
            }
        }],
        "meshes": [{
            "primitives": [{
                "attributes": attributes,
                "indices": idx_accessor,
                "mode": 4,
                "material": 0
            }]
        }],
        "nodes": [{ "mesh": 0 }],
        "scenes": [{ "nodes": [0] }],
        "scene": 0
    });

    let mut json_bytes = serde_json::to_vec(&gltf)?;
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' ');
    }
    while buffer.len() % 4 != 0 {
        buffer.push(0);
    }

    let total_length = 12 + 8 + json_bytes.len() as u32 + 8 + buffer.len() as u32;

    let mut glb = Vec::with_capacity(total_length as usize);
    glb.extend_from_slice(b"glTF");
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&total_length.to_le_bytes());

    glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"JSON");
    glb.extend_from_slice(&json_bytes);

    glb.extend_from_slice(&(buffer.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"BIN\0");
    glb.extend_from_slice(&buffer);

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(path).with_context(|| format!("create {:?}", path))?;
    file.write_all(&glb)
        .with_context(|| format!("write glb {:?}", path))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn builds_mesh_from_single_voxel() {
        let volume = OvoxelVolume {
            coords: vec![[0, 0, 0]],
            dual_vertices: vec![[0, 0, 0]],
            intersected: vec![1],
            base_color: vec![[255, 0, 0, 255]],
            semantics: vec![7],
            semantic_labels: vec!["unlabeled".into(), "chair".into()],
            resolution: 2,
            aabb: [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        };

        let mesh = ovoxel_to_mesh(&volume);
        let positions = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(|a| a.as_float3())
            .unwrap();
        let semantics: Vec<u32> = match mesh.attribute(ATTRIBUTE_SEMANTIC_ID.id) {
            Some(VertexAttributeValues::Uint32(v)) => v.clone(),
            _ => Vec::new(),
        };
        assert!(!positions.is_empty());
        assert!(semantics.iter().all(|s| *s == 7));
    }

    #[test]
    fn writes_glb() {
        let volume = OvoxelVolume {
            coords: vec![[0, 0, 0]],
            dual_vertices: vec![[0, 0, 0]],
            intersected: vec![1],
            base_color: vec![[0, 0, 0, 255]],
            semantics: vec![0],
            semantic_labels: vec!["unlabeled".into()],
            resolution: 1,
            aabb: [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        };
        let mesh = ovoxel_to_mesh(&volume);
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.glb");
        write_mesh_as_glb(&mesh, &path).expect("glb written");
        let data = fs::read(&path).expect("glb exists");
        assert!(data.starts_with(b"glTF"));
    }

    #[test]
    fn semantic_mesh_uses_palette_colors() {
        let volume = OvoxelVolume {
            coords: vec![[0, 0, 0]],
            dual_vertices: vec![[0, 0, 0]],
            intersected: vec![1],
            base_color: vec![[0, 0, 0, 255]],
            semantics: vec![1],
            semantic_labels: vec!["unlabeled".into(), "chair".into()],
            resolution: 1,
            aabb: [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        };

        let mesh = ovoxel_to_semantic_mesh(&volume);
        let colors = match mesh.attribute(Mesh::ATTRIBUTE_COLOR) {
            Some(VertexAttributeValues::Float32x4(v)) => v.clone(),
            _ => panic!("mesh missing color attribute"),
        };
        let chair = SemanticLabel::Chair.color().to_linear();
        let first = colors.first().unwrap();
        assert!((first[0] - chair.red).abs() < 1e-5);
        assert!((first[1] - chair.green).abs() < 1e-5);
        assert!((first[2] - chair.blue).abs() < 1e-5);
    }
}
