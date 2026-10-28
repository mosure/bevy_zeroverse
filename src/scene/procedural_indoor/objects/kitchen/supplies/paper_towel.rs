//! Fitted, supported roll holders. Fullness changes paper radius independently
//! of the cardboard core, spindle and holder; no loose sheet floats off the roll.
use super::*;

#[derive(Debug, Clone, Serialize)]
pub struct RollProgram {
    pub horizontal: bool,
    pub base: Vec2,
    pub base_height: f32,
    pub base_style: u8,
    pub radius: f32,
    pub full_radius: f32,
    pub core_radius: f32,
    pub length: f32,
    pub center_height: f32,
    pub arm_x: f32,
    pub arm_radius: f32,
    pub arm_style: u8,
    pub spindle_radius: f32,
    pub tail_drop: f32,
    pub tail_width: f32,
    pub finish: Surface,
}

pub fn parameters(o: &IndoorObject) -> RollProgram {
    let mut r = stream(o.seed, 0x524f4c4c484f4c44);
    let horizontal = props::program(o).horizontal;
    let s = o.size;
    let base_height = r.random_range(0.006_f32..0.018).min(s.y * 0.07);
    let full_radius = if horizontal {
        (s.z * 0.43).min(s.y * 0.32)
    } else {
        s.x.min(s.z) * 0.43
    };
    let core_radius = full_radius * r.random_range(0.19..0.32);
    let radius = core_radius + (full_radius - core_radius) * r.random_range(0.28..1.);
    let length = if horizontal {
        s.x * r.random_range(0.62..0.81)
    } else {
        s.y * r.random_range(0.64..0.87)
    };
    let base = if horizontal {
        s.xz() * Vec2::new(r.random_range(0.88..0.98), r.random_range(0.72..0.98))
    } else {
        Vec2::splat(full_radius * r.random_range(2.02..2.27))
    };
    let center_height = if horizontal {
        base_height + full_radius + r.random_range(0.005_f32..0.025).min(s.y * 0.08)
    } else {
        base_height + length * 0.5
    };
    let tail_drop = if horizontal && r.random_bool(0.58) {
        (center_height - base_height) * r.random_range(0.25..0.92)
    } else {
        0.
    };
    RollProgram {
        horizontal,
        base,
        base_height,
        base_style: r.random_range(0..if horizontal { 3 } else { 2 }),
        radius,
        full_radius,
        core_radius,
        length,
        center_height,
        arm_x: if horizontal {
            (length + (s.x - length) * 0.40) * 0.5
        } else {
            0.
        },
        arm_radius: (full_radius * r.random_range(0.045_f32..0.08)).min(base_height * 0.44),
        arm_style: if horizontal { r.random_range(0..2) } else { 0 },
        spindle_radius: core_radius * r.random_range(0.42..0.72),
        tail_drop,
        tail_width: length * r.random_range(0.84..0.98),
        finish: [
            Surface::Metal,
            Surface::Chrome,
            Surface::Wood,
            Surface::Plastic,
        ][r.random_range(0..4)],
    }
}

pub(crate) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let l = o.kind.class_name();
    if p.base_style == 0 {
        a.part(p.finish, l).housing(
            &[(0., p.base * 0.5), (p.base_height, p.base * 0.47)],
            2.,
            Transform::IDENTITY,
        );
    } else if p.base_style == 1 || !p.horizontal {
        a.box_part(
            p.finish,
            l,
            Vec3::Y * p.base_height * 0.5,
            Vec3::new(p.base.x, p.base_height, p.base.y),
            p.base_height * 0.22,
        );
    } else {
        // Two feet support a shared rear cross-member and the roll axle.
        let w = (p.base.x - p.arm_x * 2.).max(p.arm_radius * 3.);
        for sign in [-1., 1.] {
            a.box_part(
                p.finish,
                l,
                Vec3::new(sign * p.arm_x, p.base_height * 0.5, 0.),
                Vec3::new(w, p.base_height, p.base.y),
                p.base_height * 0.18,
            );
        }
        a.box_part(
            p.finish,
            l,
            Vec3::new(0., p.base_height * 0.5, -p.base.y * 0.35),
            Vec3::new(p.arm_x * 2. - w, p.base_height, p.arm_radius * 2.),
            p.arm_radius * 0.2,
        );
    }
    let tf = if p.horizontal {
        Transform::from_xyz(-p.length * 0.5, p.center_height, 0.)
            .with_rotation(Quat::from_rotation_z(-FRAC_PI_2))
    } else {
        Transform::from_xyz(0., p.base_height, 0.)
    };
    let roll = |g: &mut Geometry, outer: f32, inner: f32| {
        g.lathe(
            &[
                (outer, 0.),
                (outer, p.length),
                (inner, p.length),
                (inner, 0.),
                (outer, 0.),
            ],
            32,
            tf,
        );
    };
    roll(a.part(Surface::Paper, l), p.radius, p.core_radius);
    roll(
        a.part(Surface::WoodEdge, l),
        p.core_radius * 0.996,
        p.core_radius * 0.84,
    );
    if p.horizontal {
        a.part(p.finish, l).rod(
            Vec3::new(-p.arm_x, p.center_height, 0.),
            Vec3::new(p.arm_x, p.center_height, 0.),
            p.spindle_radius,
        );
        for sign in [-1., 1.] {
            let foot = Vec3::new(
                sign * if p.base_style == 2 {
                    p.arm_x
                } else {
                    p.length * 0.42
                },
                p.base_height * 0.5,
                -p.base.y * 0.25,
            );
            let top = Vec3::new(sign * p.arm_x, p.center_height, 0.);
            if p.arm_style == 0 {
                a.part(p.finish, l).rod(foot, top, p.arm_radius);
            } else {
                let rear = Vec3::new(foot.x, top.y, foot.z);
                a.part(p.finish, l).rod(foot, rear, p.arm_radius);
                a.part(p.finish, l).rod(rear, top, p.arm_radius);
            }
        }
        if p.tail_drop > 0. {
            // Tangency at the front of the roll; a suspended sheet, with its
            // top edge physically joined to the paper and its bottom above base.
            a.box_part(
                Surface::Paper,
                l,
                Vec3::new(0., p.center_height - p.tail_drop * 0.5, p.radius + 0.00015),
                Vec3::new(p.tail_width, p.tail_drop, 0.0003),
                0.,
            );
        }
    } else {
        let remaining = o.size.y - p.base_height - p.length;
        let h = p.base_height + p.length + remaining * 0.40;
        a.part(p.finish, l)
            .cylinder(p.spindle_radius, h, Transform::from_xyz(0., h * 0.5, 0.));
        a.part(p.finish, l).ellipsoid(
            Vec3::splat((p.spindle_radius * 1.2).min(remaining * 0.45)),
            Transform::from_xyz(0., h, 0.),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{
        items::sizing,
        objects::{audit, build_object},
    };
    #[test]
    fn holders_keep_rolls_supported_hollow_and_inside_metric_envelopes() {
        let mut styles = std::collections::BTreeSet::new();
        let mut radii = std::collections::BTreeSet::new();
        for seed in 0..128 {
            let mut o = audit::fixture(ObjectKind::PaperTowel, seed);
            o.size = sizing::loose(ObjectKind::PaperTowel, &mut stream(seed, 748)).unwrap();
            let p = parameters(&o);
            assert_eq!(
                serde_json::to_value(&p).unwrap(),
                serde_json::to_value(parameters(&o)).unwrap()
            );
            let a = build_object(&o);
            audit::check(&o, &a).unwrap_or_else(|e| panic!("roll {seed}: {e}"));
            let (lo, hi) = a.bounds();
            assert!(lo.y.abs() < 0.00001);
            assert!(lo
                .cmpge(Vec3::new(-o.size.x * 0.5, 0., -o.size.z * 0.5) - Vec3::splat(0.00005))
                .all());
            assert!(hi
                .cmple(Vec3::new(o.size.x * 0.5, o.size.y, o.size.z * 0.5) + Vec3::splat(0.00005))
                .all());
            assert!(p.spindle_radius < p.core_radius * 0.84 && p.core_radius < p.radius);
            assert!(p.tail_drop == 0. || p.center_height - p.tail_drop > p.base_height);
            styles.insert((p.horizontal, p.base_style, p.arm_style));
            radii.insert((p.radius * 1000.) as u32);
        }
        assert_eq!(styles.len(), 8);
        assert!(radii.len() > 30);
    }
}
