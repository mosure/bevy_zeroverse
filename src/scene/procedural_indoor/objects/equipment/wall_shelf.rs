//! Full rectangular support with sampled, connected mounting hardware.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct ShelfProgram {
    /// Triangulated rods, angle straps, curved braces, or hidden supports.
    pub style: u8,
    pub thickness_m: f32,
    pub bracket_x_m: Vec<f32>,
    pub strap_width_m: f32,
    pub strap_thickness_m: f32,
    pub reach_fraction: f32,
    pub brace_radius_m: f32,
    pub brace_bend: f32,
    pub board: Surface,
    pub hardware: Surface,
}

pub fn parameters(o: &IndoorObject) -> ShelfProgram {
    let mut r = stream(o.seed, 0x57414c4c5348454c);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x5348454c46494e49);
    let style = r.random_range(0..4);
    let thickness = r
        .random_range(if style == 3 {
            0.030_f32..0.065
        } else {
            0.014..0.038
        })
        .min(o.size.y * 0.30);
    let count = if style == 3 {
        0
    } else {
        (o.size.x / r.random_range(0.44..0.75)).ceil().clamp(2., 4.) as usize
    };
    let span = o.size.x * r.random_range(0.46..0.82);
    let offset = (o.size.x - span) * r.random_range(-0.12..0.12);
    let regular = r.random_bool(0.35);
    let mut bracket_x = Vec::with_capacity(count);
    for i in 0..count {
        let t = i as f32 / (count - 1) as f32;
        let jitter = if regular || i == 0 || i + 1 == count {
            0.
        } else {
            r.random_range(-0.14..0.14) / count as f32
        };
        bracket_x.push(offset + span * (t + jitter - 0.5));
    }
    let strap_thickness = r.random_range(0.006_f32..0.014).min(o.size.y * 0.09);
    ShelfProgram {
        style,
        thickness_m: thickness,
        bracket_x_m: bracket_x,
        strap_width_m: r.random_range(0.014_f32..0.036).min(o.size.x * 0.045),
        strap_thickness_m: strap_thickness,
        reach_fraction: r.random_range(0.62..0.94),
        brace_radius_m: r
            .random_range(0.0035_f32..0.0070)
            .min(strap_thickness * 0.48),
        brace_bend: r.random_range(0.18..0.72),
        board: [Surface::Wood, Surface::Plastic, Surface::Metal][f.random_range(0..3)],
        hardware: [Surface::Metal, Surface::Chrome, Surface::WoodEdge][f.random_range(0..3)],
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
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
