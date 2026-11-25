//! Original rectangular shelf for archived manifests.
use super::*;

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &ShelfProgram) {
    let s = o.size;
    let l = o.kind.class_name();
    let under = s.y - p.thickness_m;
    a.box_part(
        p.board,
        l,
        Vec3::new(0., s.y - p.thickness_m * 0.5, 0.),
        Vec3::new(s.x, p.thickness_m, s.z),
        0.001,
    );
    let t = p.strap_thickness_m;
    let rear = -s.z * 0.5;
    let end = rear + s.z * p.reach_fraction;
    for &x in &p.bracket_x_m {
        a.box_part(
            p.hardware,
            l,
            Vec3::new(x, under * 0.5, rear + t * 0.5),
            Vec3::new(p.strap_width_m, under, t),
            0.001,
        );
        // The load strap meets the shelf underside and rear plate; braces
        // terminate inside these members, rather than floating below the shelf.
        a.box_part(
            p.hardware,
            l,
            Vec3::new(x, under - t * 0.5, (rear + t + end) * 0.5),
            Vec3::new(p.strap_width_m, t, end - rear - t),
            0.001,
        );
        if p.style == 1 {
            continue;
        }
        let from = Vec3::new(x, p.brace_radius_m * 1.2, rear + t * 0.5);
        let to = Vec3::new(x, under - t * 0.5, end - p.brace_radius_m * 1.5);
        if p.style == 0 {
            a.part(p.hardware, l).rod(from, to, p.brace_radius_m);
        } else {
            let points: Vec<_> = (0..=8)
                .map(|i| {
                    let u = i as f32 / 8.;
                    let q = from.lerp(to, u);
                    q + Vec3::Y * (under * p.brace_bend * u * (1. - u))
                })
                .collect();
            a.part(p.hardware, l).tube(&points, p.brace_radius_m, 6);
        }
    }
}
