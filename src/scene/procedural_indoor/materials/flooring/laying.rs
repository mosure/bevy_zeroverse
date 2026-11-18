//! A single metric laying direction and origin across the entire walkable floor.
use super::*;
use crate::scene::procedural_indoor::{
    geometry::Geometry, layout::IndoorManifest, materials::Surface,
};
use bevy::math::{Affine2, Vec2};
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FloorLaying {
    /// Angle from the longest envelope edge towards +Z in room X/Z.
    pub turn_radians: f32,
    /// Starting offset in fractions of one material atlas along U and V.
    pub phase: Vec2,
}

impl FloorLaying {
    pub(super) fn sample(seed: u64) -> Self {
        let mut r = stream(seed, 0x464c_4f4f_524c_4159);
        let quarter = if r.random_bool(0.5) { FRAC_PI_2 } else { 0. };
        let choice = r.random::<f32>();
        let bias = if choice < 0.70 {
            0.
        } else if choice < 0.9 {
            FRAC_PI_4
        } else {
            r.random_range(0.12..1.42)
        };
        Self {
            turn_radians: quarter + bias,
            phase: Vec2::new(r.random(), r.random()),
        }
    }

    pub(super) fn validate(self) -> Result<(), String> {
        if self.turn_radians.is_finite()
            && (0.0..PI).contains(&self.turn_radians)
            && self.phase.cmpge(Vec2::ZERO).all()
            && self.phase.cmplt(Vec2::ONE).all()
        {
            Ok(())
        } else {
            Err("invalid floor laying angle or phase".into())
        }
    }
}

/// Existing horizontal UVs are (x, -z) metres. Rotate before applying
/// material repeat so rectangular stock retains its physical dimensions.
pub(crate) fn mapping(scene: &IndoorManifest) -> Option<Affine2> {
    let recipe = scene
        .program
        .as_ref()?
        .materials
        .get(Surface::Floor as usize)?;
    let laying = recipe.flooring.as_ref()?.laying?;
    let footprint = &scene.envelope.as_ref()?.footprint;
    let (a, b) = crate::scene::procedural_indoor::envelope::polygon::edges(footprint)
        .max_by(|(a, b), (c, d)| a.distance_squared(*b).total_cmp(&c.distance_squared(*d)))?;
    let heading = ((b - a).to_angle() + laying.turn_radians).rem_euclid(PI);
    Some(Affine2::from_angle_translation(
        heading,
        laying.phase * recipe.period_uv(),
    ))
}

pub(crate) fn apply(g: &mut Geometry, frame: Affine2) {
    for ((uv, normal), point) in g.uvs.iter_mut().zip(&g.normals).zip(&g.positions) {
        if normal[1] > 0.9999 {
            *uv = frame
                .transform_point2(Vec2::new(point[0], -point[2]))
                .to_array();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{architecture::architecture, layout::IndoorLayout};
    use bevy::{
        mesh::{Mesh, VertexAttributeValues},
        prelude::Vec3,
    };

    #[test]
    fn actual_floor_shares_metric_uvs_and_normal_frame_across_patches() {
        let mut scene =
            IndoorManifest::generate_with_humans(47, IndoorLayout::LectureHall, 0.65, 0, 0.)
                .unwrap();
        let key = (Surface::Floor, "floor".to_string());
        let installed = architecture(&scene).parts.remove(&key).unwrap();
        let frame = mapping(&scene).unwrap();
        let recipe = &mut scene.program.as_mut().unwrap().materials[Surface::Floor as usize];
        let floor = recipe.flooring.as_mut().unwrap();
        let mut archive = serde_json::to_value(&*floor).unwrap();
        archive.as_object_mut().unwrap().remove("laying");
        *floor = serde_json::from_value(archive).unwrap();
        assert!(mapping(&scene).is_none());
        let legacy = architecture(&scene).parts.remove(&key).unwrap();
        assert_eq!(installed.positions, legacy.positions);
        assert_eq!(installed.normals, legacy.normals);
        assert_eq!(installed.indices, legacy.indices);
        let u = frame.matrix2.x_axis;
        let tangent = Vec3::new(u.x, 0., u.y);
        for ((point, normal), uv) in installed
            .positions
            .iter()
            .zip(&installed.normals)
            .zip(&installed.uvs)
        {
            if normal[1] > 0.9999 {
                let p = Vec3::from_array(*point);
                assert!((uv[0] - frame.translation.x - p.dot(tangent)).abs() < 2e-5);
                assert!((uv[1] - frame.translation.y - p.dot(Vec3::Y.cross(tangent))).abs() < 2e-5);
            }
        }
        let mesh = installed.into_mesh();
        let Some(VertexAttributeValues::Float32x4(tangents)) =
            mesh.attribute(Mesh::ATTRIBUTE_TANGENT)
        else {
            panic!("missing floor tangents");
        };
        let Some(VertexAttributeValues::Float32x3(normals)) =
            mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
        else {
            panic!("missing floor normals");
        };
        for (normal, t) in normals.iter().zip(tangents) {
            if normal[1] > 0.9999 {
                assert!(Vec3::new(t[0], t[1], t[2]).dot(tangent) > 0.999);
            }
        }
    }
}
