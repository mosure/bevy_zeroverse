use super::*;
use std::f32::consts::{PI, TAU};
const N: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pattern {
    Solid,
    Sprigs,
    Ribbons,
    Pebbles,
}
pub(super) fn pattern(seed: u64, slot: usize) -> Pattern {
    match stream(seed, u64::from_le_bytes(*b"TissInk ") ^ slot as u64).random_range(0..10) {
        0..=2 => Pattern::Solid,
        3..=5 => Pattern::Sprigs,
        6..=7 => Pattern::Ribbons,
        _ => Pattern::Pebbles,
    }
}

fn pigment(h: f32, s: f32, l: f32) -> [u8; 3] {
    let c = Color::hsl(h.rem_euclid(360.), s, l).to_srgba();
    [c.red, c.green, c.blue].map(|v| (v * 255.).round() as u8)
}
/// A pointed or rounded ink stamp. Wrap the stamped area itself, so the side
/// panel chart can continue through a corner or cross the texture seam.
fn mark(tile: &mut [u8], center: Vec2, axis: Vec2, radii: Vec2, ink: [u8; 3], pointed: bool) {
    let b = axis.abs() * radii.x + axis.perp().abs() * radii.y + 1.;
    let lo = (center - b).floor().as_ivec2();
    let hi = (center + b).ceil().as_ivec2();
    for y in lo.y..=hi.y {
        for x in lo.x..=hi.x {
            let v = Vec2::new(x as f32 + 0.5, y as f32 + 0.5) - center;
            let u = v.dot(axis) / radii.x;
            let w = v.dot(axis.perp()) / radii.y;
            let edge = 1. - u * u - if pointed { w.abs() } else { w * w };
            let coverage = (0.5 + edge * radii.min_element() * 0.72).clamp(0., 1.);
            if coverage <= 0. {
                continue;
            }
            let i = (y.rem_euclid(N as i32) as usize * N + x.rem_euclid(N as i32) as usize) * 4;
            for (old, new) in tile[i..i + 3].iter_mut().zip(ink) {
                *old =
                    (f32::from(*old) + (f32::from(new) - f32::from(*old)) * coverage).round() as u8;
            }
        }
    }
}
pub(super) fn material(
    seed: u64,
    slot: usize,
    images: &mut impl AssetStore<Image>,
) -> StandardMaterial {
    let mut r = stream(seed, u64::from_le_bytes(*b"TissDyes") ^ slot as u64);
    let family = pattern(seed, slot);
    let hue = r.random_range(0.0..360.);
    let pale = r.random_bool(0.79);
    let base = pigment(
        hue,
        r.random_range(0.10..0.40),
        if pale {
            r.random_range(0.78..0.94)
        } else {
            r.random_range(0.22..0.48)
        },
    );
    let a = pigment(
        hue + r.random_range(-40.0..38.),
        r.random_range(0.22..0.56),
        if pale {
            r.random_range(0.40..0.67)
        } else {
            r.random_range(0.66..0.84)
        },
    );
    let b = pigment(
        hue + r.random_range(22.0..78.),
        r.random_range(0.13..0.54),
        if pale {
            r.random_range(0.57..0.78)
        } else {
            r.random_range(0.46..0.67)
        },
    );
    let mut mat = StandardMaterial {
        perceptual_roughness: r.random_range(0.57..0.87),
        ..default()
    };
    if family == Pattern::Solid {
        mat.base_color = Color::srgb_u8(base[0], base[1], base[2]);
        return mat;
    }
    let mut tile = [base[0], base[1], base[2], 255].repeat(N * N);
    match family {
        Pattern::Sprigs => {
            for _ in 0..r.random_range(4..=9) {
                let start = Vec2::new(r.random_range(0.0..N as f32), r.random_range(0.0..N as f32));
                let stem = Vec2::from_angle(r.random_range(-PI..PI));
                let len = r.random_range(20.0..42.);
                mark(
                    &mut tile,
                    start + stem * len * 0.5,
                    stem,
                    Vec2::new(len * 0.51, 0.9),
                    a,
                    true,
                );
                for j in 0..r.random_range(4..=7) {
                    let side = if j % 2 == 0 { 1. } else { -1. };
                    let axis = (stem * r.random_range(0.34..0.86) + stem.perp() * side).normalize();
                    let radius = r.random_range(5.0..11.);
                    let c = start + stem * (len * (j as f32 + 1.) / 8.) + axis * radius * 0.80;
                    mark(
                        &mut tile,
                        c,
                        axis,
                        Vec2::new(radius, radius * r.random_range(0.30..0.52)),
                        if j % 3 == 0 { b } else { a },
                        true,
                    );
                }
            }
        }
        Pattern::Ribbons => {
            let phase = r.random_range(-PI..PI);
            let amplitude = r.random_range(0.065..0.21);
            let diagonal = r.random_range(-1_i32..=1) as f32;
            let width = r.random_range(0.09..0.32);
            for x in 0..N {
                let u = x as f32 / N as f32;
                let offset = diagonal * u
                    + amplitude * (u * TAU + phase).sin()
                    + 0.038 * (u * TAU * 2. - phase).sin();
                for y in 0..N {
                    let v = (y as f32 / N as f32 + offset).rem_euclid(1.);
                    let ink = if v < width {
                        a
                    } else if (width + 0.055..width + 0.105).contains(&v) {
                        b
                    } else {
                        base
                    };
                    tile[(y * N + x) * 4..(y * N + x) * 4 + 3].copy_from_slice(&ink);
                }
            }
        }
        Pattern::Pebbles => {
            let angle = r.random_range(-PI..PI);
            for _ in 0..r.random_range(18..=46) {
                let c = Vec2::new(r.random_range(0.0..N as f32), r.random_range(0.0..N as f32));
                let axis = Vec2::from_angle(angle + r.random_range(-0.9..0.9));
                let radius = r.random_range(2.5..10.);
                mark(
                    &mut tile,
                    c,
                    axis,
                    Vec2::new(radius, radius * r.random_range(0.32..0.78)),
                    if r.random_bool(0.59) { a } else { b },
                    false,
                );
            }
        }
        Pattern::Solid => unreachable!(),
    }
    let repeat = Vec2::new(r.random_range(0.10..0.24), r.random_range(0.09..0.21));
    mat.uv_transform = bevy::math::Affine2::from_scale_angle_translation(
        repeat.recip(),
        r.random_range(-0.36..0.36),
        Vec2::new(r.random(), r.random()),
    );
    mat.base_color_texture = Some(images.add(mip_image(tile, N as u32, MapType::Color)));
    mat
}
