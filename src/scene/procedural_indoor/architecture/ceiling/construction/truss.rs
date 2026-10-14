use super::*;

pub(super) fn build(
    a: &mut Assembly,
    f: &Frame,
    web: TrussWeb,
    radius: f32,
    pitch: f32,
    taper: f32,
) -> Vec<Vec3> {
    let radius = radius.min(f.height * 0.095).min(f.width * 0.12);
    let n = (f.length / pitch).ceil().clamp(2., 24.) as usize;
    let ends = f.length * 0.47 - radius;
    let high = f.height - radius - 0.009;
    let low = radius + 0.016;
    let upper = |x: f32| high - (high - low) * (1. - taper) * (x / ends).abs() * 0.45;
    let sides: &[f32] = if web == TrussWeb::Space {
        &[-0.30, 0.30]
    } else {
        &[0.]
    };
    let mut anchors = Vec::new();
    for &side in sides {
        let z = f.width * side;
        f.rod(
            a,
            Surface::Metal,
            Vec3::new(-ends, low, z),
            Vec3::new(ends, low, z),
            radius,
        );
        let top = [
            Vec3::new(-ends, upper(-ends), z),
            Vec3::new(0., upper(0.), z),
            Vec3::new(ends, upper(ends), z),
        ];
        f.tube(a, Surface::Metal, &top, radius, 10);
        for i in 0..n {
            let x = (i as f32 / n as f32 * 2. - 1.) * ends;
            let next = ((i + 1) as f32 / n as f32 * 2. - 1.) * ends;
            let (p, q) = if web == TrussWeb::Pratt {
                (Vec3::new(x, upper(x), z), Vec3::new(next, low, z))
            } else {
                (
                    Vec3::new(x, if i % 2 == 0 { low } else { upper(x) }, z),
                    Vec3::new(next, if i % 2 == 0 { upper(next) } else { low }, z),
                )
            };
            f.rod(a, Surface::Metal, p, q, radius * 0.48);
            if web == TrussWeb::Pratt || i == 0 {
                f.rod(
                    a,
                    Surface::Metal,
                    Vec3::new(x, low, z),
                    Vec3::new(x, upper(x), z),
                    radius * 0.48,
                );
            }
        }
        f.rod(
            a,
            Surface::Metal,
            Vec3::new(ends, low, z),
            Vec3::new(ends, upper(ends), z),
            radius * 0.48,
        );
        for x in [-ends, ends] {
            f.block(
                a,
                Surface::Metal,
                Vec3::new(x, (low + upper(x)) * 0.5, z),
                Vec3::new(0.012, upper(x) - low + radius * 1.4, radius * 2.7),
                0.002,
            );
            anchors.push(Vec3::new(x, upper(x), z));
        }
    }
    if web == TrussWeb::Space {
        for i in 0..=n {
            let x = (i as f32 / n as f32 * 2. - 1.) * ends;
            f.rod(
                a,
                Surface::Metal,
                Vec3::new(x, upper(x), -f.width * 0.30),
                Vec3::new(x, upper(x), f.width * 0.30),
                radius * 0.55,
            );
            if i < n {
                let next = ((i + 1) as f32 / n as f32 * 2. - 1.) * ends;
                f.rod(
                    a,
                    Surface::Metal,
                    Vec3::new(x, low, -f.width * 0.30),
                    Vec3::new(next, low, f.width * 0.30),
                    radius * 0.45,
                );
            }
        }
    }
    anchors
}
