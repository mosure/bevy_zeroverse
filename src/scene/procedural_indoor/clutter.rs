pub mod beverages;
pub mod microphone;
pub mod pen_holder;
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
    match o.kind {
        ObjectKind::Mug | ObjectKind::CoffeeCup | ObjectKind::WaterBottle | ObjectKind::SodaCan => {
            beverages::build(a, o)
        }
        ObjectKind::Notepad | ObjectKind::Pencil | ObjectKind::Microphone | ObjectKind::Phone => {
            tabletop::build(a, o)
        }
        ObjectKind::PenHolder => pen_holder::build(a, o),
        _ => unreachable!("non-clutter object"),
    }
}
