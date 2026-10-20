//! Metric loose-item dimensions, sampled before placement/admission. Correlated
//! dimensions preserve round vessels and recognizable paper/device proportions.
use super::*;

pub(crate) fn loose(kind: ObjectKind, r: &mut impl Rng) -> Option<Vec3> {
    use ObjectKind::*;
    Some(match kind {
        Mug => {
            let d = r.random_range(0.065..0.115);
            Vec3::new(
                d * r.random_range(1.22..1.40),
                d * r.random_range(0.85..1.30),
                d,
            )
        }
        CoffeeCup => {
            let d = r.random_range(0.060..0.100);
            Vec3::new(d, d * r.random_range(1.12..1.90), d)
        }
        WaterBottle => {
            let d = r.random_range(0.050..0.100);
            Vec3::new(d, d * r.random_range(2.8..3.6), d)
        }
        SodaCan => {
            let d = r.random_range(0.053..0.073);
            Vec3::new(d, d * r.random_range(1.65..2.55), d)
        }
        PenHolder => {
            let d = r.random_range(0.065..0.115);
            Vec3::new(d, r.random_range(0.12..0.21), d)
        }
        Pencil => {
            let d = r.random_range(0.0065..0.010);
            Vec3::new(d, d * 1.12, r.random_range(0.095..0.205))
        }
        Notepad | Notebook => {
            let w = r.random_range(0.085..0.225);
            Vec3::new(
                w,
                r.random_range(0.006..0.033),
                w * r.random_range(1.12..1.55),
            )
        }
        Phone => {
            let w = r.random_range(0.058..0.086);
            Vec3::new(
                w,
                r.random_range(0.007..0.016),
                w * r.random_range(1.85..2.18),
            )
        }
        Keyboard => Vec3::new(
            r.random_range(0.28..0.46),
            r.random_range(0.014..0.033),
            r.random_range(0.105..0.185),
        ),
        Mouse => {
            let length = r.random_range(0.085..0.125);
            Vec3::new(
                length * r.random_range(0.50..0.68),
                length * r.random_range(0.25..0.37),
                length,
            )
        }
        Books => {
            let w = r.random_range(0.12..0.29);
            Vec3::new(
                w,
                r.random_range(0.028..0.16),
                w * r.random_range(1.05..1.45),
            )
        }
        Bowl => {
            let d = r.random_range(0.10..0.30);
            Vec3::new(d, d * r.random_range(0.22..0.42), d)
        }
        SoapDispenser => {
            let w = r.random_range(0.055..0.105);
            Vec3::new(w, r.random_range(0.13..0.24), w * r.random_range(1.1..1.45))
        }
        Sponge => {
            let w = r.random_range(0.075..0.145);
            Vec3::new(
                w,
                r.random_range(0.014..0.035),
                w * r.random_range(0.52..0.75),
            )
        }
        DryingRack => {
            let w = r.random_range(0.29..0.54);
            Vec3::new(
                w,
                r.random_range(0.16..0.28),
                w * r.random_range(0.60..0.84),
            )
        }
        KnifeBlock => {
            let w = r.random_range(0.11..0.20);
            Vec3::new(
                w,
                r.random_range(0.27..0.39),
                w * r.random_range(1.05..1.35),
            )
        }
        UtensilHolder => {
            let d = r.random_range(0.10..0.19);
            Vec3::new(d, r.random_range(0.26..0.40), d)
        }
        PaperTowel => {
            // Accommodates either holder orientation and standard roll widths.
            let width = r.random_range(0.23..0.33);
            Vec3::new(
                width,
                width * r.random_range(1.0..1.15),
                r.random_range(0.12..0.19),
            )
        }
        KitchenTowel => {
            let w = r.random_range(0.16..0.34);
            Vec3::new(
                w,
                r.random_range(0.012..0.035),
                w * r.random_range(1.05..1.48),
            )
        }
        CuttingBoard => {
            let w = r.random_range(0.22..0.48);
            Vec3::new(
                w,
                r.random_range(0.014..0.038),
                w * r.random_range(0.58..0.86),
            )
        }
        FruitBowl => {
            let w = r.random_range(0.21..0.38);
            Vec3::new(
                w,
                w * r.random_range(0.47..0.66),
                w * r.random_range(0.82..1.0),
            )
        }
        Bread => {
            let w = r.random_range(0.16..0.38);
            Vec3::new(
                w,
                w * r.random_range(0.28..0.40),
                w * r.random_range(0.48..0.70),
            )
        }
        _ => return None,
    })
}

/// Envelopes of the actual dish builder, not interchangeable width/height
/// jitter: small cups, tall glasses, shallow bowls and dinner/side plates.
pub(crate) fn dish(style: u8, r: &mut impl Rng) -> Vec3 {
    let (diameter, height, diameter_fraction) = match style {
        0 => (
            r.random_range(0.105..0.31),
            r.random_range(0.035..0.10),
            0.86,
        ),
        1 => {
            let d = r.random_range(0.10..0.26);
            (d, d * r.random_range(0.22..0.44) / 0.72, 0.86)
        }
        2 | 3 => (
            r.random_range(0.052..0.108),
            r.random_range(0.055..0.14) / 0.87,
            0.5332,
        ),
        _ => (
            r.random_range(0.050..0.092),
            r.random_range(0.075..0.20) / 0.87,
            0.5332,
        ),
    };
    let envelope = diameter / diameter_fraction;
    Vec3::new(envelope, height, envelope)
}

#[cfg(test)]
mod tests;
