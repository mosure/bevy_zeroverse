//! Connected monitor bases/risers and wall brackets. Dimensions are resolved
//! inside the same support envelope before any vertices are emitted.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct MountProgram {
    pub base_height: f32,
    pub rail_fraction: f32,
    pub roundness: f32,
    pub riser_style: u8,
    pub riser_radius: f32,
    pub elbow_fraction: f32,
    pub finish: Surface,
    pub wall_span: Vec2,
}
pub fn parameters(o: &IndoorObject) -> MountProgram {
    let mut r = stream(o.seed, 0x444953504d4f554e);
    MountProgram {
        base_height: r.random_range(0.006_f32..0.021).min(o.size.y * 0.065),
        rail_fraction: r.random_range(0.055..0.12),
        roundness: r.random_range(2.0..3.8),
        riser_style: r.random_range(0..3),
        riser_radius: r.random_range(0.0045..0.012),
        elbow_fraction: r.random_range(0.26..0.66),
        finish: [Surface::Metal, Surface::Chrome, Surface::Plastic][r.random_range(0..3)],
        wall_span: Vec2::new(r.random_range(0.14..0.38), r.random_range(0.12..0.42)),
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &DisplayProgram, d: &DisplayGeometry) {
    let m = &p.mount;
    let label = o.kind.class_name();
    let span = Vec2::new(d.width_m * p.stand_width, o.size.z * p.stand_depth);
    let h = m.base_height;
    let rail = span.min_element() * m.rail_fraction;
    let mut foot = Vec3::Y * h * 0.5;
    match p.stand {
        0 => a.part(m.finish, label).housing(
            &[(0., span * 0.5), (h, span * 0.48)],
            m.roundness,
            Transform::IDENTITY,
        ),
        1 => {
            foot.z = -span.y * 0.37;
            for side in [-1., 1.] {
                a.part(m.finish, label).rod(
                    foot,
                    Vec3::new(side * (span.x * 0.5 - h), h * 0.5, span.y * 0.40),
                    h * 0.5,
                );
            }
        }
        2 => a.box_part(
            m.finish,
            label,
            foot,
            Vec3::new(span.x, h, span.y),
            h * 0.25,
        ),
        3 | 5 => {
            for side in [-1., 1.] {
                a.box_part(
                    m.finish,
                    label,
                    Vec3::new(side * (span.x - rail) * 0.5, h * 0.5, 0.),
                    Vec3::new(rail, h, span.y),
                    h * 0.1,
                );
            }
            let zs: &[f32] = if p.stand == 3 { &[-1., 1., 0.] } else { &[-1.] };
            for &z in zs {
                a.box_part(
                    m.finish,
                    label,
                    Vec3::new(0., h * 0.5, z * (span.y - rail) * 0.5),
                    Vec3::new(span.x - rail * 2., h, rail),
                    h * 0.1,
                );
            }
            if p.stand == 5 {
                foot.z = -(span.y - rail) * 0.5;
            }
        }
        _ => {
            for side in [-1., 1.] {
                let x = side * span.x * 0.40;
                a.part(m.finish, label).rod(
                    Vec3::new(x, h * 0.5, -span.y * 0.5 + h),
                    Vec3::new(x, h * 0.5, span.y * 0.5 - h),
                    h * 0.5,
                );
            }
            a.part(m.finish, label).rod(
                Vec3::new(-span.x * 0.40, h * 0.5, 0.),
                Vec3::new(span.x * 0.40, h * 0.5, 0.),
                h * 0.5,
            );
        }
    }
    let x = d.width_m * p.mount_offset;
    let sag = if d.curvature_radius_m > 0. {
        d.curvature_radius_m - (d.curvature_radius_m.powi(2) - x * x).sqrt()
    } else {
        0.
    };
    let attachment = d.panel_transform.transform_point(Vec3::new(
        x,
        -d.height_m * 0.22,
        sag - d.panel_m * 0.5 + 0.0006,
    ));
    let radius = m.riser_radius.min(h * 0.46);
    let elbow = Vec3::new(
        foot.x,
        attachment.y * m.elbow_fraction,
        d.panel_transform
            .translation
            .z
            .clamp(-o.size.z * 0.5 + radius * 2., o.size.z * 0.5 - radius * 2.),
    );
    let mut strut = |from: Vec3, to: Vec3, scale: f32| {
        if m.riser_style == 0 {
            a.part(m.finish, label).rod(from, to, radius * scale);
        } else {
            let delta = to - from;
            a.part(m.finish, label).cuboid(
                Vec3::new(radius * 2. * scale, delta.length(), radius * 1.5 * scale),
                radius * 0.15,
                Transform::from_translation((from + to) * 0.5)
                    .with_rotation(Quat::from_rotation_arc(Vec3::Y, delta.normalize())),
            );
        }
    };
    strut(foot, elbow, 1.);
    strut(
        elbow,
        attachment,
        if m.riser_style == 2 { 0.68 } else { 1. },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::objects::{audit, build_object};
    #[test]
    fn display_mounts_replay_diversify_and_keep_their_contact_planes() {
        let mut stands = std::collections::BTreeSet::new();
        let mut cases = std::collections::BTreeSet::new();
        for seed in 0..128 {
            for kind in [ObjectKind::Monitor, ObjectKind::Display] {
                let o = audit::fixture(kind, seed);
                let p = display_parameters(&o);
                assert_eq!(
                    serde_json::to_value(&p).unwrap(),
                    serde_json::to_value(display_parameters(&o)).unwrap()
                );
                let a = build_object(&o);
                audit::check(&o, &a).unwrap_or_else(|e| panic!("{kind:?}/{seed}: {e}"));
                let (lo, _) = a.bounds();
                if kind == ObjectKind::Monitor {
                    assert!(lo.y.abs() < 0.00005);
                    stands.insert((p.stand, p.mount.riser_style));
                } else {
                    assert!((lo.z + o.size.z * 0.5).abs() < 0.00005);
                }
                cases.insert((
                    (p.back_fraction.x * 100.) as u32,
                    (p.back_fraction.y * 100.) as u32,
                ));
            }
        }
        assert_eq!(stands.len(), 18);
        assert!(cases.len() > 95);
    }
}
