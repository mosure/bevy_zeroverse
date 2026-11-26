//! The placing side and the validating side use identical body envelopes. Only
//! candidates near a person need to build that person's articulated capsules.
use super::*;
use crate::scene::procedural_indoor::layout::IndoorObject;

pub(super) fn at_workstation(person: &IndoorHuman, object: &IndoorObject) -> bool {
    person
        .workstation
        .is_some_and(|id| object.id == id || object.support == Some(id))
}

pub(super) fn worktop_capsules(person: &IndoorHuman, tf: Transform) -> Vec<(Vec3, Vec3, f32)> {
    // The chest is broad across the shoulders, not equally deep front-to-back.
    // Five overlapping longitudinal capsules bound that elliptical volume and
    // let an abdomen approach a desk edge without shrinking the lateral envelope.
    let mut table_capsules: Vec<_> = collision_capsules_inner(person, true)
        .into_iter()
        .map(|(a, b, r)| (tf.transform_point(a), tf.transform_point(b), r))
        .collect();
    let (waist, chest, width) = table_capsules.remove(1);
    // Anny's weight anchor does not shrink the chest/bust linearly with `build`.
    // Draped cloth also bridges the chest-to-waist profile instead of following
    // the abdomen inward. Retain an adult torso-depth floor scaled by stature.
    let depth = (0.145 * person.build.max(person.stature / 1.75)
        + person.appearance.as_ref().map_or(0.028, |a| {
            // The fitted shell and buttons project beyond the rest body's
            // ease. Reserve that construction before admitting a seated pose.
            a.garment_ease + a.garment.torso_ease_m * (person.stature / 1.75) + 0.005
        }))
    .min(width);
    let right = tf.rotation * (person.joints[9] - person.joints[5]).normalize_or(Vec3::X);
    // Discrete capsule centres leave scalloped gaps between neighbouring ribs.
    // Bound the continuous elliptical torso with the half-spacing hypotenuse,
    // rather than admitting cuffs/abdomen into one of those small gaps at a
    // curved or oblique tabletop. No body mesh/model is needed during placement.
    let half_spacing = (width - depth) * 0.25;
    let radius = depth.hypot(half_spacing);
    for fraction in [-1., -0.5, 0., 0.5, 1.] {
        let offset = (width - depth) * fraction;
        table_capsules.push((waist + right * offset, chest + right * offset, radius));
    }
    // The wider circulation skeleton is not a superset of the anatomical arm
    // positions: it moves wrists sideways. Include the actual limbs at close
    // worktops, with room for the skin's sole-grounding offset and finger tips.
    let s = person.stature / 1.75;
    let ground = Vec3::Y * (0.025 * s);
    for base in [5, 9] {
        if person
            .worktop_contact
            .as_ref()
            .is_some_and(|c| c.palms[(base - 5) / 4])
        {
            continue;
        }
        let p = &person.joints;
        let tip = p[base + 3] + (p[base + 3] - p[base + 2]).normalize_or(Vec3::NEG_Y) * 0.095 * s;
        for (a, b, r) in [
            (p[base], p[base + 1], 0.095 * person.build + 0.025 * s),
            (p[base + 1], p[base + 2], 0.065 * person.build + 0.025 * s),
            (p[base + 2], tip, 0.080 * s),
        ] {
            table_capsules.push((
                tf.transform_point(a - ground),
                tf.transform_point(b - ground),
                r,
            ));
        }
    }
    table_capsules
}

/// Test a supported rectangular prop against people already placed in its room.
/// A worker's actual reach differs from the broader walking skeleton.
pub(crate) fn supported_prop_clear(scene: &IndoorManifest, prop: &IndoorObject) -> bool {
    if scene.humans.is_empty() {
        return true;
    }
    let (lo, hi) = prop.bounds();
    let pad = Vec3::splat(0.01);
    scene
        .humans
        .iter()
        .filter(|h| h.neighbor == prop.neighbor)
        .all(|h| {
            let (a, b) = h.bounds();
            if !a.cmplt(hi + pad).all() || !b.cmpgt(lo - pad).all() {
                return true;
            }
            let tf = h.transform();
            let capsules = if at_workstation(h, prop) {
                worktop_capsules(h, tf)
            } else {
                collision_capsules(h)
                    .into_iter()
                    .map(|(a, b, r)| (tf.transform_point(a), tf.transform_point(b), r))
                    .collect()
            };
            let inverse = prop.transform().compute_affine().inverse();
            let size = prop.size;
            let min = Vec3::new(-size.x * 0.5, 0., -size.z * 0.5);
            let max = Vec3::new(size.x * 0.5, size.y, size.z * 0.5);
            !capsules.into_iter().any(|(a, b, r)| {
                box_hit(
                    inverse.transform_point3(a),
                    inverse.transform_point3(b),
                    r,
                    min,
                    max,
                )
            })
        })
}
