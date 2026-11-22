//! Trace the actual luminaire apertures into static reflection maps. Emissive
//! faces remain excluded from diffuse transport to avoid counting lights twice.
use super::*;
use crate::scene::procedural_indoor::photometry::source_color;

pub(crate) struct ReflectionEmitters {
    triangles: Vec<Triangle>,
    nodes: Vec<Node>,
    emission: Vec<Vec3>,
}

impl ReflectionEmitters {
    pub(crate) fn new(scene: &IndoorManifest) -> Self {
        let mut triangles = Vec::new();
        let mut emission = Vec::new();
        let positions = architecture::fixture_positions(scene);
        for (i, position) in positions.iter().copied().enumerate() {
            let area = architecture::fixtures::emitting_area_at(scene, position);
            let (color, lumens) = architecture::fixture_photometry_at(scene, i, &positions);
            let luminance = lumens / (PI * area);
            let color = linear(source_color(scene, color));
            emission.push(color * luminance);
            for ((surface, _), geometry) in
                architecture::fixtures::geometry(scene, i, position).parts
            {
                if surface != Surface::Light {
                    continue;
                }
                append_face_triangles(&mut triangles, &geometry, i, None);
            }
        }
        for o in scene
            .objects
            .iter()
            .filter(|o| o.kind == super::super::layout::ObjectKind::DeskLamp)
        {
            if !super::super::photometry::stationary_source(scene, o) {
                continue;
            }
            let power = super::super::photometry::lamp(scene, o);
            if power.lumens == 0. {
                continue;
            }
            let (aperture, area) = objects::accessories::lamps::reflection_aperture(o);
            let c = super::super::photometry::lamp_emission(scene, power, area);
            append_face_triangles(
                &mut triangles,
                &aperture,
                emission.len(),
                Some(o.transform()),
            );
            emission.push(Vec3::new(c.red, c.green, c.blue));
        }
        objects::equipment::signals::strings::reflection_apertures(scene, |geometry, tf, c| {
            append_face_triangles(&mut triangles, geometry, emission.len(), Some(tf));
            emission.push(Vec3::new(c.red, c.green, c.blue));
        });
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

fn append_face_triangles(
    out: &mut Vec<Triangle>,
    g: &super::super::geometry::Geometry,
    material: usize,
    tf: Option<Transform>,
) {
    let point = |i: u32| {
        let p = Vec3::from_array(g.positions[i as usize]);
        tf.map_or(p, |t| t.transform_point(p))
    };
    for &[a, b, c] in g.indices.as_chunks::<3>().0 {
        let a = point(a);
        let ab = point(b) - a;
        let ac = point(c) - a;
        out.push(Triangle {
            a,
            ab,
            ac,
            normal: ab.cross(ac),
            uv: [Vec2::ZERO; 3],
            material,
        });
    }
}

#[cfg(test)]
mod tests;
