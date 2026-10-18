//! Preserve static annotation geometry when only a skin pose changes. Updates
//! are computed with the existing skin oracle and committed atomically; any
//! invalid pose falls back to full extraction and its ordinary diagnostics.
use super::*;

pub(super) struct VertexUpdate {
    pub base_generation: u64,
    pub ranges: Vec<Range<usize>>,
}

pub(super) struct DeformedMesh {
    pub range: Range<usize>,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
}

/// No topology, instance ordinals, validation rules or floating-point arithmetic
/// change. Adjacent changed spans share a write, without copying static gaps.
pub(super) fn apply(
    geometry: &mut ExtractedGeometry,
    meshes: Vec<DeformedMesh>,
) -> Option<VertexUpdate> {
    if meshes.iter().any(|mesh| {
        mesh.range.end > geometry.vertices.len()
            || mesh.range.start > mesh.range.end
            || mesh.positions.len() != mesh.range.len()
            || mesh.normals.len() != mesh.range.len()
    }) {
        return None;
    }
    let mut ranges: Vec<Range<usize>> = Vec::new();
    for mesh in meshes {
        let vertices = &mut geometry.vertices[mesh.range.clone()];
        let changed = vertices.iter().zip(&mesh.positions).zip(&mesh.normals).any(
            |((vertex, position), normal)| {
                bytemuck::bytes_of(&vertex.position) != bytemuck::bytes_of(position)
                    || bytemuck::bytes_of(&vertex.normal) != bytemuck::bytes_of(normal)
            },
        );
        if !changed {
            continue;
        }
        for ((vertex, position), normal) in
            vertices.iter_mut().zip(mesh.positions).zip(mesh.normals)
        {
            vertex.position = position;
            vertex.normal = normal;
        }
        if let Some(last) = ranges
            .last_mut()
            .filter(|last| last.end == mesh.range.start)
        {
            last.end = mesh.range.end;
        } else {
            ranges.push(mesh.range);
        }
    }
    Some(VertexUpdate {
        base_generation: geometry.generation,
        ranges,
    })
}
