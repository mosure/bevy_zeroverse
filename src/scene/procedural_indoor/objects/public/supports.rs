//! Small freestanding supports. Top attachment and every foot share one envelope.
use super::*;

#[derive(Clone, Debug, serde::Serialize)]
pub struct SupportProgram {
    pub foundation: u8,
    pub span: Vec2,
    pub base_fraction: f32,
    pub stem_fraction: f32,
    pub foot_phase: f32,
    pub feet: usize,
}
impl SupportProgram {
    pub fn sample(seed: u64) -> Self {
        let mut r = stream(seed, 0x50554242415345);
        Self {
            foundation: r.random_range(0..4),
            span: Vec2::new(r.random_range(0.60..0.94), r.random_range(0.62..0.94)),
            base_fraction: r.random_range(0.018..0.042),
            stem_fraction: r.random_range(0.025..0.065),
            foot_phase: r.random_range(0.0..TAU),
            feet: r.random_range(3..=5),
        }
    }
}

pub fn build(a: &mut Assembly, o: &IndoorObject, p: &SupportProgram, finish: Surface, top: Vec3) {
    let label = o.kind.class_name();
    let s = o.size;
    let h = (s.y * p.base_fraction).min(0.045);
    let radius = s.x.min(s.z) * p.stem_fraction;
    let span = s.xz() * p.span;
    let base = Vec3::Y * h;
    match p.foundation {
        0 | 1 => {
            if p.foundation == 0 {
                a.box_part(
                    finish,
                    label,
                    base * 0.5,
                    Vec3::new(span.x, h, span.y),
                    h * 0.2,
                );
            } else {
                a.part(finish, label).housing(
                    &[(0., span * 0.48), (h * 0.45, span * 0.5), (h, span * 0.43)],
                    2.,
                    Transform::IDENTITY,
                );
            }
            a.part(finish, label).rod(base, top, radius);
        }
        2 => {
            let root = base.with_y((top.y * 0.18).max(h));
            for i in 0..p.feet {
                let angle = p.foot_phase + TAU * i as f32 / p.feet as f32;
                let foot = Vec3::new(
                    angle.cos() * span.x * 0.43,
                    h * 0.5,
                    angle.sin() * span.y * 0.43,
                );
                a.part(finish, label).rod(foot, root, h * 0.48);
                a.part(Surface::Rubber, label).cylinder(
                    h * 0.75,
                    h * 0.25,
                    Transform::from_translation(foot.with_y(h * 0.125)),
                );
            }
            a.part(finish, label).rod(root, top, radius);
        }
        _ => {
            for sign in [-1., 1.] {
                let x = sign * span.x * 0.44;
                a.box_part(
                    finish,
                    label,
                    Vec3::new(x, h * 0.5, 0.),
                    Vec3::new(radius * 2., h, span.y),
                    h * 0.2,
                );
                a.part(finish, label).rod(
                    Vec3::new(x, h, 0.),
                    top + Vec3::X * x * 0.18,
                    radius * 0.65,
                );
            }
            a.part(finish, label).rod(
                top - Vec3::X * span.x * 0.08,
                top + Vec3::X * span.x * 0.08,
                radius,
            );
        }
    }
}
