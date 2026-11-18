//! Stock colors are sampled once per floor. Metric installation, relief and
//! roughness use other streams. Bakes on both backends consume these same colors.
use bevy::color::Color;
use rand::Rng;

pub(super) fn hsl(h: f32, s: f32, l: f32) -> [f32; 3] {
    let c = Color::hsl(h.rem_euclid(360.), s, l).to_srgba();
    [c.red, c.green, c.blue]
}

fn heather(r: &mut impl Rng, lightness: f32) -> [f32; 3] {
    // Undyed wool, warm greige and cool slate mixtures.
    let h = if r.random_bool(0.64) {
        r.random_range(22.0..54.0)
    } else {
        r.random_range(192.0..237.0)
    };
    hsl(h, r.random_range(0.025..0.17), lightness)
}

fn dye(r: &mut impl Rng, lightness: f32) -> [f32; 3] {
    let hue = match r.random_range(0..10) {
        0..=3 => r.random_range(180.0..239.0), // blue, petrol, indigo
        4..=5 => r.random_range(4.0..48.0),    // brick, ochre
        6 => r.random_range(80.0..173.0),      // moss, emerald
        _ => r.random_range(0.0..360.0),
    };
    hsl(hue, r.random_range(0.14..0.49), lightness)
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    std::array::from_fn(|i| a[i] + t * (b[i] - a[i]))
}

pub(super) fn yarns(r: &mut impl Rng) -> [[f32; 3]; 3] {
    let l = r.random_range(0.14..0.61);
    let base = if r.random_bool(0.56) {
        heather(r, l)
    } else {
        dye(r, l)
    };
    let high = (l + r.random_range(0.045..0.27)).min(0.82);
    let low = l * r.random_range(0.38..0.86);
    let binder = heather(r, high);
    let fleck = if r.random_bool(0.57) {
        dye(r, low)
    } else {
        heather(r, low)
    };
    [
        base,
        mix(base, binder, r.random_range(0.42..0.96)),
        mix(base, fleck, r.random_range(0.52..0.98)),
    ]
}

pub(super) fn mineral(r: &mut impl Rng, veining: f32) -> [f32; 3] {
    let l = r.random_range(0.18..0.79);
    match r.random_range(0..100) {
        0..=45 => heather(r, l),
        46..=77 => hsl(r.random_range(17.0..55.0), r.random_range(0.09..0.32), l),
        78..=93 => {
            // Veined green stone and blue slate carry much less chroma than
            // solution-dyed yarn. Keep the wide-area stone field subdued.
            let hue = if veining > 0.5 {
                r.random_range(133.0..177.0)
            } else {
                r.random_range(194.0..235.0)
            };
            hsl(hue, r.random_range(0.035..0.19), l)
        }
        _ => {
            // A glazed or integrally pigmented product can be strongly colored.
            // Saturated chartreuse over an entire floor is a poor prior.
            let hue = r.random_range(0.0..360.0);
            let saturation = if (60.0..135.0).contains(&hue) {
                r.random_range(0.035..0.14)
            } else {
                r.random_range(0.22..0.46)
            };
            hsl(hue, saturation, 0.13 + l * 0.68)
        }
    }
}

pub(super) fn joint(r: &mut impl Rng, base: [f32; 3], carpet: bool) -> [f32; 3] {
    let l = (base[0] + base[1] + base[2]) / 3.;
    if carpet {
        // A cut between carpet modules shows pile and backing, not light grout.
        let shade = r.random_range(0.48..0.94);
        base.map(|v| v * shade)
    } else {
        let lightness = if r.random_bool(0.76) {
            (l + r.random_range(-0.14..0.13)).clamp(0.10, 0.82)
        } else {
            r.random_range(0.12..0.82)
        };
        let mortar = heather(r, lightness);
        mix(mortar, base, r.random_range(0.03..0.35))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::layout::stream;
    use bevy::math::Vec3;

    #[test]
    fn independent_yarns_and_stock_specific_joints_retain_quiet_and_chromatic_floors() {
        let mut mixtures = 0;
        let mut quiet = 0;
        let mut colorful = 0;
        let mut matched_mortar = 0;
        for seed in 0..1024 {
            let mut r = stream(seed, 123);
            let v = yarns(&mut r);
            assert_eq!(v, yarns(&mut stream(seed, 123)));
            let [a, b, c] = v.map(Vec3::from);
            let chroma = |v: Vec3| v - Vec3::splat(v.element_sum() / 3.);
            // Relative channel proportions differ beyond a mere move along the
            // grey axis (which would leave all yarns on one chromaticity line).
            mixtures += usize::from(
                chroma(a)
                    .cross(chroma(b))
                    .length()
                    .max(chroma(a).cross(chroma(c)).length())
                    > 0.001,
            );
            let backing = joint(&mut r, v[0], true);
            assert!((0..3).all(|i| backing[i] <= v[0][i] && backing[i] > 0.));
            let stone = Vec3::from(mineral(&mut r, 0.6));
            let range = stone.max_element() - stone.min_element();
            if stone.y > stone.x && stone.z < stone.x {
                assert!(range < 0.145, "saturated yellow-green stone {stone:?}");
            }
            quiet += usize::from(range < 0.09);
            colorful += usize::from(range > 0.17);
            let grout = Vec3::from(joint(&mut r, stone.to_array(), false));
            matched_mortar +=
                usize::from((grout.element_sum() - stone.element_sum()).abs() / 3. < 0.14);
        }
        assert!(mixtures > 180, "{mixtures}");
        assert!(quiet > 350 && colorful > 60, "{quiet} {colorful}");
        assert!(
            matched_mortar > 750 && matched_mortar < 995,
            "{matched_mortar}"
        );
    }
}
