//! Shared camera presentation for SVG/PNG exports and the native/WebGPU 2D view.
//! Calibration determines lens type; legacy FOV fields never override it.
use super::*;
#[cfg(test)]
#[path = "camera_tests.rs"]
mod tests;

pub(crate) struct CameraGizmo {
    pub footprint: Vec<[f32; 3]>,
    pub origin: [f32; 3],
    pub axis_end: [f32; 3],
    pub fisheye: bool,
    pub label: String,
}
impl CameraGizmo {
    pub fn marker_radius(&self) -> f32 {
        if self.fisheye {
            7.
        } else {
            5.
        }
    }
    pub fn lens_name(&self) -> &'static str {
        if self.fisheye {
            "fisheye"
        } else {
            "pinhole"
        }
    }
}

pub(crate) fn camera_gizmo(c: &Camera, length: f32) -> Result<CameraGizmo, String> {
    let footprint = camera_footprint(c, length)?;
    let m = Mat4::from_cols_array_2d(&c.world_from_view);
    let lens = c.calibration.as_ref().and_then(|c| c.fisheye);
    let kind = lens.map_or_else(
        || "pinhole".to_string(),
        |l| format!("fisheye {:.0}°", 2. * l.max_angle.to_degrees()),
    );
    let axis = m.transform_vector3(Vec3::NEG_Z);
    // A vertical optical axis vanishes in an X/Z projection. Name its direction
    // instead of drawing a fictitious horizontal heading for ceiling mounts.
    let vertical = if axis.xz().length() < 0.02 {
        if axis.y < 0. {
            " · down"
        } else {
            " · up"
        }
    } else {
        ""
    };
    Ok(CameraGizmo {
        footprint,
        origin: m.transform_point3(Vec3::ZERO).to_array(),
        axis_end: m.transform_point3(Vec3::NEG_Z * length).to_array(),
        fisheye: lens.is_some(),
        label: format!("{} · {kind}{vertical}", c.label),
    })
}

pub(crate) fn camera_footprint(c: &Camera, length: f32) -> Result<Vec<[f32; 3]>, String> {
    let m = Mat4::from_cols_array_2d(&c.world_from_view);
    if !m.is_finite()
        || !length.is_finite()
        || length <= 0.
        || !c.fov_y.is_finite()
        || !c.aspect.is_finite()
        || c.principal_point_offset.is_some_and(|p| !p.is_finite())
    {
        return Err("invalid schematic camera matrix or lens".into());
    }
    let rigid = Mat3::from_mat4(m);
    let gram = rigid.transpose() * rigid;
    if (gram - Mat3::IDENTITY)
        .to_cols_array()
        .iter()
        .any(|v| v.abs() > 0.01)
        || (rigid.determinant() - 1.).abs() > 0.01
        || m.row(3).distance(Vec4::W) > 0.001
    {
        return Err("schematic cameras require rigid world_from_view matrices".into());
    }
    let d = length;
    let offset = c.principal_point_offset.unwrap_or_default();
    let corners: Vec<Vec3> = if let Some(calibration) = &c.calibration {
        calibration.validate()?;
        if calibration.fisheye.is_some() {
            let size = calibration.image_size.map(|x| x as f32);
            (0..=32)
                .flat_map(|y| {
                    (0..=32).map(move |x| [x as f32 / 32. * size[0], y as f32 / 32. * size[1]])
                })
                .filter_map(|p| calibration.ray(p))
                .map(|r| Vec3::from_array(r) * d)
                .collect()
        } else {
            let k = calibration.k;
            let [width, height] = calibration.image_size.map(|x| x as f32);
            [[0., 0.], [width, 0.], [width, height], [0., height]]
                .map(|[u, v]| {
                    let y = (v - k[1][2]) / k[1][1];
                    let x = (u - k[0][2] - k[0][1] * y) / k[0][0];
                    Vec3::new(x * d, -y * d, -d)
                })
                .to_vec()
        }
    } else {
        if !(0.01..3.13).contains(&c.fov_y)
            || !c.aspect.is_finite()
            || c.aspect <= 0.
            || !offset.is_finite()
            || offset.abs().cmpgt(Vec2::splat(0.5)).any()
        {
            return Err("invalid schematic pinhole lens".into());
        }
        let h = d * (c.fov_y * 0.5).tan();
        let w = h * c.aspect;
        [
            Vec3::new(-w, -h, -d),
            Vec3::new(w, -h, -d),
            Vec3::new(w, h, -d),
            Vec3::new(-w, h, -d),
        ]
        .map(|p| p + Vec3::new(-2. * offset.x * w, 2. * offset.y * h, 0.))
        .to_vec()
    };
    // Display hull of the optical center and calibrated rays: pinhole image
    // corners or sampled fisheye domain. This is not an occluded floor footprint.
    let mut hull: Vec<_> = std::iter::once(Vec3::ZERO)
        .chain(corners)
        .map(|v| m.transform_point3(v).xz())
        .collect();
    hull.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    hull.dedup();
    fn chain(vertices: impl Iterator<Item = Vec2>) -> Vec<Vec2> {
        let mut result: Vec<Vec2> = vec![];
        for v in vertices {
            while result.len() >= 2
                && (result[result.len() - 1] - result[result.len() - 2])
                    .perp_dot(v - result[result.len() - 1])
                    <= 0.
            {
                result.pop();
            }
            result.push(v);
        }
        result.pop();
        result
    }
    let mut poly = chain(hull.iter().copied());
    poly.extend(chain(hull.iter().rev().copied()));
    Ok(poly.into_iter().map(|p| [p.x, 0., p.y]).collect())
}
