//! Match Bevy's four-weight skinning and inverse-transpose normal transform.
use super::*;
use bevy::mesh::skinning::SkinnedMeshInverseBindposes;

#[allow(clippy::type_complexity)]
pub(super) fn bake(
    mesh: &Mesh,
    skin: &SkinnedMesh,
    poses: &Assets<SkinnedMeshInverseBindposes>,
    joints: &Query<Ref<GlobalTransform>>,
) -> Result<(Vec<[f32; 3]>, Vec<[f32; 3]>), String> {
    let poses = poses
        .get(&skin.inverse_bindposes)
        .ok_or("inverse bind poses unavailable")?;
    if poses.len() != skin.joints.len() {
        return Err("joint/bind-pose lengths differ".into());
    }
    let matrices = skin
        .joints
        .iter()
        .zip(poses.iter())
        .map(|(joint, bind)| {
            joints
                .get(*joint)
                .map(|t| t.to_matrix() * *bind)
                .map_err(|_| "joint transform unavailable".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (
        Some(VertexAttributeValues::Float32x3(positions)),
        Some(VertexAttributeValues::Float32x3(normals)),
        Some(VertexAttributeValues::Uint16x4(indices)),
        Some(VertexAttributeValues::Float32x4(weights)),
    ) = (
        mesh.attribute(Mesh::ATTRIBUTE_POSITION),
        mesh.attribute(Mesh::ATTRIBUTE_NORMAL),
        mesh.attribute(Mesh::ATTRIBUTE_JOINT_INDEX),
        mesh.attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT),
    )
    else {
        return Err("expected four joint indices/weights and position/normal attributes".into());
    };
    if [normals.len(), indices.len(), weights.len()]
        .iter()
        .any(|n| *n != positions.len())
    {
        return Err("inconsistent skin attribute lengths".into());
    }
    let mut out_positions = Vec::with_capacity(positions.len());
    let mut out_normals = Vec::with_capacity(normals.len());
    for (((position, normal), indices), weights) in
        positions.iter().zip(normals).zip(indices).zip(weights)
    {
        let mut matrix = Mat4::ZERO;
        for (&index, &weight) in indices.iter().zip(weights) {
            if weight != 0.0 {
                matrix += *matrices
                    .get(index as usize)
                    .ok_or("joint index out of bounds")?
                    * weight;
            }
        }
        let position = matrix.transform_point3(Vec3::from_array(*position));
        let normal = matrix
            .inverse()
            .transpose()
            .transform_vector3(Vec3::from_array(*normal))
            .normalize();
        if !position.is_finite() || !normal.is_finite() {
            return Err("nonfinite or singular skin deformation".into());
        }
        out_positions.push(position.to_array());
        out_normals.push(normal.to_array());
    }
    Ok((out_positions, out_normals))
}
