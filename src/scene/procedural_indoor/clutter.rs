pub mod beverages;
pub(crate) mod tabletop;
#[cfg(test)]
mod tests;
use super::{
    layout::{stream, IndoorObject, ObjectKind},
    materials::Surface,
    objects::Assembly,
};
use bevy::prelude::*;
use rand::Rng;

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    match o.kind {
        ObjectKind::Mug | ObjectKind::CoffeeCup | ObjectKind::WaterBottle | ObjectKind::SodaCan => {
            beverages::build(a, o)
        }
        ObjectKind::Notepad | ObjectKind::Pencil | ObjectKind::Microphone | ObjectKind::Phone => {
            tabletop::build(a, o)
        }
        ObjectKind::PenHolder => {
            let r = s.x.min(s.z) * 0.47;
            let height = s.y * 0.60;
            a.part(Surface::Metal, "other_prop").lathe(
                &[
                    (0.0, 0.0),
                    (r, 0.0),
                    (r, height),
                    (r * 0.86, height),
                    (r * 0.86, height * 0.1),
                    (0.0, height * 0.1),
                ],
                24,
                Transform::IDENTITY,
            );
            let mut rng = stream(o.seed, 8);
            for _ in 0..7 {
                let p = Vec3::new(
                    rng.random_range(-r * 0.45..r * 0.45),
                    height * 0.1,
                    rng.random_range(-r * 0.45..r * 0.45),
                );
                let top = Vec3::new(p.x * 1.4, s.y * rng.random_range(0.78..0.98), p.z * 1.4);
                a.part(
                    if rng.random_bool(0.5) {
                        Surface::Wood
                    } else {
                        Surface::Ink
                    },
                    "other_prop",
                )
                .rod(p, top, 0.0025);
            }
        }
        _ => unreachable!("non-clutter object"),
    }
}
