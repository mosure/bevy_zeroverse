//! Trace the actual luminaire apertures into static reflection maps. Emissive
//! faces remain excluded from diffuse transport to avoid counting lights twice.
use super::*;

pub(crate) struct ReflectionEmitters {
    triangles: Vec<Triangle>,
    nodes: Vec<Node>,
    emission: Vec<Vec3>,
}

impl ReflectionEmitters {
    pub(crate) fn new(scene: &IndoorManifest) -> Self {
        let mut triangles = Vec::new();
        let mut emission = Vec::new();
        let area = architecture::fixtures::emitting_area(scene);
        for (i, position) in architecture::fixture_positions(scene)
            .into_iter()
            .enumerate()
        {
            let (color, lumens) = architecture::fixture_photometry(scene, i);
            let luminance = lumens / (PI * area);
            let color = linear(Color::srgb(color.x, color.y, color.z));
            emission.push(color * luminance);
            for ((surface, _), geometry) in
                architecture::fixtures::geometry(scene, i, position).parts
            {
                if surface != Surface::Light {
                    continue;
                }
                for indices in geometry.indices.as_chunks::<3>().0 {
                    let a = Vec3::from_array(geometry.positions[indices[0] as usize]);
                    let b = Vec3::from_array(geometry.positions[indices[1] as usize]);
                    let c = Vec3::from_array(geometry.positions[indices[2] as usize]);
                    triangles.push(Triangle {
                        a,
                        ab: b - a,
                        ac: c - a,
                        // Match the visible emitter's back-face culling without a sqrt.
                        normal: (b - a).cross(c - a),
                        uv: [Vec2::ZERO; 3],
                        material: i,
                    });
                }
            }
        }
        let nodes = bvh::build_probe_tree(&mut triangles);
        Self {
            triangles,
            nodes,
            emission,
        }
    }

    /// The opaque scene's nearest hit is the limit, so housings, louvers and
    /// room geometry still occlude emission. Holes and gaps remain unlit.
    pub(crate) fn radiance(&self, origin: Vec3, direction: Vec3, limit: f32) -> Option<Vec3> {
        let (index, _, _) = trace_triangles::<true>(
            &self.triangles,
            &self.nodes,
            origin,
            direction,
            limit,
            false,
        )?;
        Some(self.emission[self.triangles[index].material])
    }
}

#[cfg(test)]
mod tests;
