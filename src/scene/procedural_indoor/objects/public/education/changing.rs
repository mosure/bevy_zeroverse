//! Furniture inside an open or glazed changing locker. All fittings belong to
//! the cabinet instance; loosely placed personal belongings have separate owners.
use super::*;

pub(super) fn fittings(
    a: &mut Assembly,
    o: &IndoorObject,
    p: &LockerProgram,
    bay: usize,
    row: usize,
    (bottom, top): (f32, f32),
    label: &str,
) {
    let height = top - bottom;
    if height < 0.44 {
        return;
    }
    let mut r = stream(
        o.seed,
        u64::from_le_bytes(*b"CoatRail") ^ ((bay as u64) << 8) ^ row as u64,
    );
    let compartment = &p.bays[bay];
    let center = compartment.left + compartment.width * 0.5;
    let rear = -o.size.z * 0.5 + p.panel_m;
    let depth = p.body_depth_m - p.panel_m;
    let mut below = top;
    if height > 0.86 && r.random_bool(0.78) {
        let thickness = r.random_range(0.011..0.020);
        let length = depth * r.random_range(0.76..0.92);
        below -= r.random_range(0.18..0.30);
        a.box_part(
            p.finish,
            label,
            Vec3::new(center, below + thickness * 0.5, rear + length * 0.5),
            Vec3::new(compartment.width, thickness, length),
            0.0018,
        );
    }
    if height > 0.78 && depth >= 0.39 && r.random_bool(0.62) {
        let z = rear + depth * r.random_range(0.40..0.55);
        let y = below - r.random_range(0.055..0.10);
        let radius = r.random_range(0.008..0.014);
        let ends = [
            compartment.left + 0.005,
            compartment.left + compartment.width - 0.005,
        ];
        for x in ends {
            a.box_part(
                p.accent,
                label,
                Vec3::new(x, y, z),
                Vec3::new(0.010, radius * 3.3, radius * 3.3),
                0.002,
            );
        }
        a.part(p.accent, label)
            .rod(Vec3::new(ends[0], y, z), Vec3::new(ends[1], y, z), radius);
    } else if r.random_bool(0.72) {
        let count = if compartment.width > 0.34 && r.random_bool(0.51) {
            2
        } else {
            1
        };
        let y = below - height.min(0.68) * r.random_range(0.19..0.37);
        for i in 0..count {
            let x = center + compartment.width * ((i as f32 + 0.5) / count as f32 - 0.5);
            let base = Vec3::new(x, y, rear + 0.007);
            let elbow = base + Vec3::new(0., -0.008, r.random_range(0.027..0.042));
            a.box_part(p.accent, label, base, Vec3::new(0.024, 0.051, 0.014), 0.002);
            let g = a.part(p.accent, label);
            g.rod(base, elbow, 0.0055);
            g.rod(elbow, elbow + Vec3::new(0., 0.024, 0.008), 0.0055);
        }
    }
}
