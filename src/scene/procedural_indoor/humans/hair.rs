//! Opaque fitted grooms: directional scalp relief, a tapered hairline and curved
//! bundles. Roots follow the Anny head and supported long ends follow the torso.
mod coils;
mod curtain;
mod cut;
mod groom;
mod program;
mod scalp;
mod strands;
use super::{super::geometry::Geometry, IndoorHuman};
use bevy::prelude::*;
pub use cut::ScalpCut;
pub(crate) use program::torso_weight;
pub use program::{HairProgram, HairStyle};

pub(super) fn finish_material(material: &mut StandardMaterial, h: &IndoorHuman) {
    material.reflectance = 0.28;
    if let Some(style @ (HairStyle::TightCurls | HairStyle::Afro)) = HairStyle::from_id(h.hairstyle)
    {
        let radius = h
            .appearance
            .as_ref()
            .map_or(0.012, |a| a.hair_program.curl_radius_m);
        coils::finish(material, style, radius);
    } else if let Some(a) = &h.appearance {
        material.perceptual_roughness = 0.64 + 0.10 * (a.hair_curl / 0.06).clamp(0., 1.);
        if h.hairstyle == HairStyle::Buzz as u8 {
            if let Some(cut) = a.hair_program.scalp_cut {
                let coverage = cut.buzz_coverage();
                material.perceptual_roughness = 0.78 + 0.10 * (1. - coverage);
                material.anisotropy_strength = 0.08 + 0.18 * coverage;
                // V follows the fibres in the shared, mipmapped groom atlas.
                let along_fibre = (0.0075 / cut.clip_length_m).clamp(1., 7.5);
                material.uv_transform *=
                    bevy::math::Affine2::from_scale(Vec2::new(1., along_fibre));
            }
        }
    }
}

pub(super) struct HairFrame {
    pub head: Mat4,
    pub torso: Mat4,
    pub head_origin: Vec3,
}
impl HairFrame {
    fn point(&self, h: &IndoorHuman, p: Vec3) -> Vec3 {
        let scale = self.head.x_axis.truncate().length();
        let t = torso_weight(h.hairstyle, (self.head_origin.y - p.y) * scale, h.stature);
        self.head
            .transform_point3(p)
            .lerp(self.torso.transform_point3(p), t)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn append(
    h: &IndoorHuman,
    rest: &[Vec3],
    posed: &[Vec3],
    normals: &[Vec3],
    faces: &[i64],
    dominant: &[usize],
    labels: &[String],
    rotation: Quat,
    frame: HairFrame,
    g: &mut Geometry,
) {
    if h.hairstyle == 5 {
        return;
    }
    let style = HairStyle::from_id(h.hairstyle).expect("validated hair style");
    let head = |i: usize| labels[dominant[i]] == "head";
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for (_, p) in rest.iter().enumerate().filter(|(i, _)| head(*i)) {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    let groom = groom::Groom::new(h, lo, hi);
    let inv_rotation = rotation.inverse();
    scalp::append(
        &groom, style, rest, posed, normals, faces, head, rotation, g,
    );
    let start = g.positions.len();
    let first_index = g.indices.len();
    if style.loose() {
        curtain::append(h, &groom, rest, dominant, labels, g);
    }
    if matches!(
        style,
        HairStyle::Bun
            | HairStyle::Ponytail
            | HairStyle::HighPonytail
            | HairStyle::Braid
            | HairStyle::TwinBraids
    ) {
        let scalp: Vec<_> = rest
            .iter()
            .enumerate()
            .filter(|(i, _)| head(*i))
            .map(|(i, &p)| (p, (inv_rotation * normals[i]).normalize_or(Vec3::Y)))
            .collect();
        let roots = strands::roots(h, &groom, &scalp, frame.head.x_axis.truncate().length());
        strands::tied(h, &groom, &roots, g);
    }
    if (groom.program.bangs > 0. || matches!(style, HairStyle::Pixie | HairStyle::Swept))
        && !matches!(
            style,
            HairStyle::Buzz
                | HairStyle::Bald
                | HairStyle::TightCurls
                | HairStyle::Afro
                | HairStyle::Locs
        )
    {
        let triangles: Vec<_> = faces
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|q| q.iter().all(|&i| head(i as usize)))
            .flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
            .map(|tri| tri.map(|i| rest[i as usize]))
            .collect();
        strands::fringe(h, &groom, &triangles, g);
    }
    // New fall meshes are constructed in the fitted rest frame. Upper roots
    // follow the head and supported ends follow the torso in static poses too.
    for p in &mut g.positions[start..] {
        *p = frame.point(h, Vec3::from_array(*p)).to_array();
    }
    strands::recompute_normals(g, start, first_index);
}

#[cfg(test)]
mod tests {
    use super::super::{build_human, sample_person, HumanPoseKind, HumanSurface};
    use bevy::prelude::*;

    #[test]
    fn groom_extremes_remain_bounded_on_the_anny_head() {
        for style in 0..super::HairStyle::ALL.len() as u8 {
            for (length, curl, part) in [(0.006, 0.0, -0.6), (0.11, 0.06, 0.6)] {
                let mut h = sample_person(
                    31,
                    0,
                    Vec3::ZERO,
                    0.0,
                    HumanPoseKind::StandingRelaxed,
                    None,
                    false,
                );
                h.hairstyle = style;
                let appearance = h.appearance.as_mut().unwrap();
                appearance.hair_length = length;
                appearance.hair_curl = curl;
                appearance.hair_part = part;
                appearance.body_gender = Some(if curl == 0. { 0.05 } else { 0.95 });
                appearance.hair_program.drop_m = if curl == 0. { 0.08 } else { 0.65 };
                appearance.hair_program.layers = if curl == 0. { 0. } else { 1. };
                appearance.hair_program.spread = if curl == 0. { 0.7 } else { 1.4 };
                let mut assembly = build_human(&h);
                let hair = assembly.parts.remove(&HumanSurface::Hair);
                if style == 5 {
                    assert!(hair.is_none_or(|g| g.indices.is_empty()));
                    continue;
                }
                let hair = hair.unwrap();
                assert!(
                    !hair.indices.is_empty() && hair.positions.len() < 100_000,
                    "style {style}: {} vertices",
                    hair.positions.len()
                );
                for p in &hair.positions {
                    let p = Vec3::from_array(*p);
                    assert!(
                        p.is_finite() && p.distance(h.joints[4]) < 1.15,
                        "style {style}: {p}"
                    );
                }
                for normal in &hair.normals {
                    assert!((Vec3::from_array(*normal).length() - 1.0).abs() < 0.005);
                }
                for uv in &hair.uvs {
                    assert!(Vec2::from_array(*uv).is_finite());
                }
                assert!(hair
                    .indices
                    .iter()
                    .all(|&i| (i as usize) < hair.positions.len()));
                for triangle in hair.indices.as_chunks::<3>().0 {
                    let [a, b, c] = triangle.map(|i| Vec3::from_array(hair.positions[i as usize]));
                    assert!(
                        (b - a).cross(c - a).length_squared() > 1e-18,
                        "collapsed hair triangle: style {style}"
                    );
                }
            }
        }
    }
}
