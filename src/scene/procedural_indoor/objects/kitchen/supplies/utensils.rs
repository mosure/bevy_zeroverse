//! Individually sized tools packed through a hollow holder. The bounded proposal
//! loop admits physical shafts and heads, never a fixed ring or parity sequence.
use super::*;
use crate::scene::procedural_indoor::placement::Footprint;

#[derive(Clone, Debug, Serialize)]
pub struct ToolProgram {
    pub style: u8,
    pub position: Vec3,
    pub rotation: Quat,
    pub height: f32,
    pub width: f32,
    pub shaft: f32,
    pub thickness: f32,
    pub head: Surface,
    pub handle: Surface,
    pub tines: u32,
    #[serde(skip)]
    pub(crate) footprint: Footprint,
}
#[derive(Clone, Debug, Serialize)]
pub struct HolderProgram {
    pub radius: f32,
    pub height: f32,
    pub foot: f32,
    pub wall: f32,
    pub segments: u32,
    pub finish: Surface,
    pub tools: Vec<ToolProgram>,
}
pub fn parameters(o: &IndoorObject) -> HolderProgram {
    let mut r = stream(o.seed, 0x5554454e53494c53);
    let radius = o.size.x.min(o.size.z) * r.random_range(0.32..0.43);
    let height = o.size.y * r.random_range(0.36..0.52);
    let wall = (radius * r.random_range(0.055..0.085)).min(0.004);
    let foot = r.random_range(0.74..0.96);
    let mut p = HolderProgram {
        radius,
        height,
        wall,
        foot,
        segments: if r.random_bool(0.25) {
            r.random_range(10..=16)
        } else {
            32
        },
        finish: [
            Surface::Ceramic,
            Surface::Metal,
            Surface::Plastic,
            Surface::Wood,
        ][r.random_range(0..4)],
        tools: Vec::new(),
    };
    let target = r.random_range(1..=7);
    for _ in 0..target {
        for _ in 0..32 {
            let h = o.size.y * r.random_range(0.73..0.94);
            let width = (h * r.random_range(0.075..0.125)).min(radius * 0.65);
            let shaft = (width * r.random_range(0.12..0.19)).min(0.004);
            let lean =
                Vec2::new(r.random_range(-1.0..1.0), r.random_range(-1.0..1.0)) * radius * 0.70;
            let axis = Vec3::new(lean.x, h, lean.y).normalize();
            let rotation = Quat::from_rotation_arc(Vec3::Y, axis)
                * Quat::from_rotation_y(r.random_range(-PI..PI));
            let root = Vec2::new(r.random_range(-1.0..1.0), r.random_range(-1.0..1.0))
                * radius
                * foot
                * 0.77;
            let position = Vec3::new(
                root.x,
                wall + shaft * (1. - axis.y * axis.y).max(0.).sqrt(),
                root.y,
            );
            // A faceted inner wall lies inside its circumcircle. Use its
            // inradius, including the thicker shaft crossing at an angle.
            let inset = (PI / p.segments as f32).cos();
            let at_rim = position.xz() + axis.xz() * (height - position.y) / axis.y;
            if root.length() + shaft > (radius * foot - wall) * inset
                || at_rim.length() + shaft / axis.y > (radius - wall) * inset
            {
                continue;
            }
            let thickness = shaft * 0.8;
            let (lo, hi) = tool_bounds(position, rotation, h, width, shaft.max(thickness));
            let head_low =
                position.y + axis.y * h * 0.72 - width * 0.5 * (1. - axis.y * axis.y).sqrt();
            if head_low <= height + 0.004
                || lo.x < -o.size.x * 0.5
                || hi.x > o.size.x * 0.5
                || lo.z < -o.size.z * 0.5
                || hi.z > o.size.z * 0.5
                || hi.y > o.size.y
            {
                continue;
            }
            let footprint = Footprint {
                center: (lo.xz() + hi.xz()) * 0.5,
                half: (hi.xz() - lo.xz()) * 0.5,
                yaw: 0.,
            };
            if p.tools
                .iter()
                .any(|t| footprint.overlaps(t.footprint, 0.0015))
            {
                continue;
            }
            let style = r.random_range(0..4);
            p.tools.push(ToolProgram {
                style,
                position,
                rotation,
                height: h,
                width,
                shaft,
                thickness,
                head: [Surface::Wood, Surface::Metal, Surface::Plastic][r.random_range(0..3)],
                handle: [Surface::Wood, Surface::Metal, Surface::Plastic][r.random_range(0..3)],
                tines: r.random_range(3..=5),
                footprint,
            });
            break;
        }
    }
    p
}
fn tool_bounds(
    position: Vec3,
    rotation: Quat,
    height: f32,
    width: f32,
    thickness: f32,
) -> (Vec3, Vec3) {
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for x in [-width * 0.5, width * 0.5] {
        for y in [0., height] {
            for z in [-thickness, thickness] {
                let v = position + rotation * Vec3::new(x, y, z);
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
    }
    (lo, hi)
}
pub(crate) fn tool(a: &mut Assembly, p: &ToolProgram, seed: u64) {
    let label = |s| {
        format!(
            "kitchen_utensil#finish{}",
            super::super::super::super::materials::variants::object_slot(s, seed, None)
        )
    };
    let tf = Transform::from_translation(p.position).with_rotation(p.rotation);
    a.part(p.handle, &label(p.handle)).tube(
        &[p.position, tf.transform_point(Vec3::Y * p.height * 0.76)],
        p.shaft,
        10,
    );
    let w = p.width;
    let h = p.height;
    match p.style {
        0 => a.part(p.head, &label(p.head)).ellipsoid(
            Vec3::new(w * 0.5, h * 0.14, p.thickness),
            tf.mul_transform(Transform::from_xyz(0., h * 0.86, 0.)),
        ),
        1 => a.part(p.head, &label(p.head)).housing(
            &[
                (h * 0.72, Vec2::new(w * 0.19, p.thickness)),
                (h * 0.85, Vec2::new(w * 0.5, p.thickness)),
                (h, Vec2::new(w * 0.45, p.thickness)),
            ],
            5.,
            tf,
        ),
        _ => {
            let slotted = p.style == 2;
            let n = if slotted { 4 } else { p.tines };
            // Real openings between separate strips; no overlaid black marks.
            let bar = w / n as f32 * if slotted { 0.50 } else { 0.43 };
            for i in 0..n {
                let x = (i as f32 / (n - 1) as f32 - 0.5) * (w - bar);
                let bottom = h * 0.77;
                let top = if slotted {
                    h * 0.955
                } else {
                    h * (0.98 + 0.02 * (1. - (x / w).abs()))
                };
                a.part(p.head, &label(p.head)).cuboid(
                    Vec3::new(bar, top - bottom, p.thickness * 2.),
                    bar.min(p.thickness) * 0.2,
                    tf.mul_transform(Transform::from_xyz(x, (bottom + top) * 0.5, 0.)),
                );
            }
            for (bottom, top) in [(h * 0.72, h * 0.77), (h * 0.955, h)]
                .into_iter()
                .take(if slotted { 2 } else { 1 })
            {
                a.part(p.head, &label(p.head)).cuboid(
                    Vec3::new(w, top - bottom, p.thickness * 2.),
                    p.thickness * 0.2,
                    tf.mul_transform(Transform::from_xyz(0., (bottom + top) * 0.5, 0.)),
                );
            }
        }
    }
}
pub(crate) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let r = p.radius;
    let h = p.height;
    let w = p.wall;
    let label = o.kind.class_name();
    a.part(p.finish, label).lathe(
        &[
            (0., 0.),
            (r * p.foot, 0.),
            (r, h),
            (r - w, h),
            (r * p.foot - w, w),
            (0., w),
        ],
        p.segments,
        Transform::IDENTITY,
    );
    for (i, t) in p.tools.iter().enumerate() {
        tool(a, t, o.seed ^ (0x544f4f4c46494e49 + i as u64));
    }
}
