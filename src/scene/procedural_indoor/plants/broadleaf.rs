//! Large paddle leaves, palmate fans and paired waxy leaflets. Petioles and
//! leaflets attach to the exact emitted stem; no alpha cards or external assets.
use super::*;

pub(super) fn build(
    a: &mut Assembly,
    o: &IndoorObject,
    g: &Growth,
    base: Vec3,
    spread: f32,
    soil_radius: f32,
) {
    let mut r = stream(o.seed, 6442);
    let height = o.size.y - base.y;
    match o.variant % SPECIES {
        14 => {
            for j in 0..g.count(9) {
                let dir = radial(g.angle(j));
                let start = base + dir * soil_radius * r.random_range(0.10..0.40);
                let tip =
                    base + dir * spread * 0.18 + Vec3::Y * height * r.random_range(0.30..0.62);
                stem(
                    a,
                    &[start, start.lerp(tip, 0.55) - dir * spread * 0.04, tip],
                    o.size.y * 0.004,
                );
                blade(
                    a,
                    tip,
                    dir * spread * r.random_range(0.48..0.66)
                        + Vec3::Y * height * r.random_range(0.18..0.30),
                    spread * r.random_range(0.12..0.18),
                    0,
                    r.random(),
                    g,
                );
            }
        }
        15 => {
            let crown = base + Vec3::Y * height * r.random_range(0.19..0.36);
            stem(a, &[base, crown], o.size.y * 0.014);
            for j in 0..g.count(8) {
                let dir = radial(g.angle(j));
                let tip = crown
                    + dir * spread * r.random_range(0.25..0.37)
                    + Vec3::Y * height * r.random_range(0.18..0.32);
                stem(a, &[crown, crown.lerp(tip, 0.55), tip], o.size.y * 0.0035);
                let side = Vec3::Y.cross(dir);
                let fingers = r.random_range(7..=11);
                let fan = r.random_range(0.76..1.16);
                for k in 0..fingers {
                    let angle = (k as f32 / (fingers - 1) as f32 * 2. - 1.) * fan;
                    let d = (dir * 0.70 + Vec3::Y * 0.55) * angle.cos() + side * angle.sin();
                    blade(
                        a,
                        tip,
                        d * spread * r.random_range(0.52..0.60),
                        spread * 0.055,
                        1,
                        r.random(),
                        g,
                    );
                }
            }
        }
        _ => {
            for j in 0..g.count(8) {
                let dir = radial(g.angle(j));
                let start = base + dir * soil_radius * 0.45;
                let tip = base
                    + dir * spread * r.random_range(0.30..0.45)
                    + Vec3::Y * height * r.random_range(0.62..0.91);
                let leader = [start, start.lerp(tip, 0.42) - dir * spread * 0.06, tip];
                stem(a, &leader, o.size.y * 0.005);
                let pairs = r.random_range(5..=8);
                for k in 0..pairs {
                    let t = 0.26 + g.node(k, pairs) * 0.69;
                    let p = along(&leader, t);
                    for side in [-1., 1.] {
                        let d = (Vec3::Y.cross(dir) * side + dir * 0.35).normalize();
                        let root = p + d * spread * 0.035;
                        stem(a, &[p, root], o.size.y * 0.0013);
                        blade(
                            a,
                            root,
                            d * spread * (0.36 - t * 0.13) + Vec3::Y * height * 0.025,
                            spread * 0.075,
                            0,
                            r.random(),
                            g,
                        );
                    }
                }
            }
        }
    }
}
