use super::*;
pub mod pad;
pub mod phone;
#[cfg(test)]
mod tests;
pub mod writing;
use std::f32::consts::{FRAC_PI_2, TAU};

#[derive(Debug, Clone, serde::Serialize)]
pub struct DesktopItemProgram {
    pub binding: u8,
    pub leaf_fraction: f32,
    pub tip_fraction: f32,
    pub barrel_sides: u32,
    pub case_fraction: f32,
    pub bezel_fraction: f32,
    pub lens_count: usize,
    pub camera_island: bool,
    pub microphone_bend: f32,
    pub microphone_head: f32,
    pub microphone_lateral: f32,
    pub phone_roundness: f32,
    pub phone_top_border: f32,
    pub phone_bottom_border: f32,
    pub phone_controls: u32,
    pub phone_surface: Surface,
    pub phone_face_up: bool,
}
pub fn parameters(o: &IndoorObject) -> DesktopItemProgram {
    let mut r = stream(o.seed, 7450);
    let mut phone = stream(o.seed, 7451);
    DesktopItemProgram {
        binding: r.random_range(0..4),
        leaf_fraction: r.random_range(0.58..0.87),
        tip_fraction: r.random_range(0.08..0.23),
        barrel_sides: [6, 8, 20][r.random_range(0..3)],
        case_fraction: r.random_range(0.70..0.91),
        bezel_fraction: r.random_range(0.035..0.15),
        lens_count: r.random_range(1..=3),
        camera_island: r.random_bool(0.65),
        microphone_bend: r.random_range(0.18..0.57),
        microphone_head: r.random_range(0.045..0.09),
        microphone_lateral: r.random_range(-0.27..0.27),
        phone_roundness: phone.random_range(4.0..18.),
        phone_top_border: phone.random_range(0.035..0.10),
        phone_bottom_border: phone.random_range(0.035..0.16),
        phone_controls: phone.random_range(0..3),
        phone_surface: [Surface::Metal, Surface::Plastic, Surface::Chrome]
            [phone.random_range(0..3)],
        phone_face_up: phone.random_bool(0.76),
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    match o.kind {
        ObjectKind::Notepad => pad::build(a, o),
        ObjectKind::Pencil => writing::build(a, o),
        ObjectKind::Microphone => super::microphone::build(a, o),
        ObjectKind::Phone => phone::build(a, o),
        _ => unreachable!(),
    }
}
