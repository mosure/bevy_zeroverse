//! Continuous outer shell, inset face and physically distinct driver surfaces.
use super::*;
use crate::scene::procedural_indoor::geometry::{ProfileJoints, RevolvedOutline};
use std::sync::OnceLock;

fn corners(half: Vec2, z: f32) -> [Vec3; 4] {
    [
        Vec3::new(-half.x, -half.y, z),
        Vec3::new(half.x, -half.y, z),
        Vec3::new(half.x, half.y, z),
        Vec3::new(-half.x, half.y, z),
    ]
}

pub(super) fn shell(a: &mut Assembly, p: &SpeakerProgram, tf: Transform, label: &str) {
    let g = a.part(p.finish, label);
    let s = 0.5 * p.body;
    let t = p.wall;
    let front = s.z;
    let rings = [
        corners(Vec2::new(s.x * p.rear_width_ratio, s.y) - t * 0.46, -s.z),
        corners(Vec2::new(s.x * p.rear_width_ratio, s.y), -s.z + t * 0.70),
        corners(s.xy(), front - t * 0.70),
        corners(s.xy() - t * 0.46, front),
        corners(s.xy() - t, front - p.face_recess()),
    ]
    .map(|ring| ring.map(|v| tf.transform_point(v + Vec3::Y * s.y)));
    g.metric_quad(rings[0], tf.rotation * -Vec3::Z);
    for pair in rings.windows(2) {
        for i in 0..4 {
            let j = (i + 1) % 4;
            let face = [pair[0][i], pair[0][j], pair[1][j], pair[1][i]];
            let normal = (face[1] - face[0]).cross(face[2] - face[0]);
            g.metric_quad(face, normal);
        }
    }
    if p.covered {
        let half = s.xy() - t;
        let uv = [
            [-half.x, -half.y],
            [half.x, -half.y],
            [half.x, half.y],
            [-half.x, half.y],
        ];
        a.part(p.grille, p.acoustic_label())
            .vertex_quad(rings[4], [tf.rotation * Vec3::Z; 4], uv);
    }
}

pub(super) fn driver(a: &mut Assembly, p: &SpeakerProgram, d: &Driver, face: Transform) {
    static RIM: OnceLock<RevolvedOutline> = OnceLock::new();
    let ring = RIM.get_or_init(|| RevolvedOutline::round(32, 0., 0));
    // Axis points out of the baffle. All three profiles meet at the same rings.
    let tf = face
        * Transform::from_translation(d.center.extend(0.))
            .with_rotation(Quat::from_rotation_x(FRAC_PI_2));
    let r = d.radius;
    let z = d.depth;
    // The surround fits inside the advertised envelope, even on large exposed drivers.
    let bead = (0.085 * z).min(p.face_recess());
    ring.sweep(
        a.part(Surface::Rubber, acoustic::DAMPER_LABEL),
        &[
            (r, 0.),
            (0.94 * r, 0.),
            (0.88 * r, bead),
            (0.82 * r, 0.),
            (0.79 * r, -0.11 * z),
        ],
        ProfileJoints::Smooth,
        tf,
    );
    let cone = a.part(Surface::Plastic, p.acoustic_label());
    ring.sweep(
        cone,
        &[
            (0.79 * r, -0.11 * z),
            (0.54 * r, -0.59 * z),
            (d.dome * r, -z),
        ],
        ProfileJoints::Smooth,
        tf,
    );
    // A shallow spherical cap closes the voice-coil opening, with a crease at its foot.
    let dome = r * d.dome;
    ring.sweep(
        cone,
        &[
            (dome, -z),
            (0.77 * dome, -0.73 * z),
            (0.43 * dome, -0.61 * z),
            (0., -0.57 * z),
        ],
        ProfileJoints::Smooth,
        tf,
    );
}
