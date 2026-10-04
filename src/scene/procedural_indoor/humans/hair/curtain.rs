//! Layered closed tresses. Front and rear lengths route around the torso rather
//! than expanding radially into a cape around the shoulders.
use super::{groom::Groom, strands, Geometry, HairStyle, IndoorHuman};
use bevy::prelude::*;
use std::f32::consts::{PI, TAU};

fn smooth(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0., 1.);
    t * t * (3. - 2. * t)
}

pub(super) fn append(
    h: &IndoorHuman,
    groom: &Groom,
    rest: &[Vec3],
    dominant: &[usize],
    labels: &[String],
    g: &mut Geometry,
) {
    let style = HairStyle::from_id(h.hairstyle).unwrap();
    let bob = matches!(style, HairStyle::Bob | HairStyle::AsymmetricBob);
    let drop = if bob {
        0.008 + groom.program.drop_m * 0.13
    } else {
        groom.program.drop_m
    };
    let lo = groom.centre.y - groom.size.y * 0.5;
    let top = groom.centre.y + groom.size.y * 0.475;
    let bottom = lo - drop;
    const BANDS: usize = 18;
    let mut support = [Vec2::new(-0.025, 0.035); BANDS];
    for (i, &p) in rest.iter().enumerate() {
        let name = &labels[dominant[i]];
        if !["spine", "neck", "root", "breast", "clavicle", "pelvis"]
            .iter()
            .any(|prefix| name.starts_with(prefix))
            || !(bottom - 0.07..lo + 0.05).contains(&p.y)
        {
            continue;
        }
        for (j, band) in support.iter_mut().enumerate() {
            let y = bottom + (lo - bottom) * j as f32 / (BANDS - 1) as f32;
            if (p.y - y).abs() < 0.065 {
                band.x = band.x.min(p.z);
                band.y = band.y.max(p.z);
            }
        }
    }
    let profile = |y: f32| {
        let f = ((y - bottom) / (lo - bottom) * (BANDS - 1) as f32).clamp(0., (BANDS - 1) as f32);
        let i = (f.floor() as usize).min(BANDS - 2);
        support[i].lerp(support[i + 1], smooth(0., 1., f - i as f32))
    };
    let count = ((groom.size.x * 1.8 / groom.program.clump_width_m) as usize).clamp(18, 32);
    // The ear opening separates forward locks from the rear fall. No surface
    // bridges through a shoulder or wraps around its arm.
    for (a_lo, a_hi, locks, front) in [
        (-1.42_f32, 1.42_f32, count, false),
        (-PI * 0.73, -1.70, 4, true),
        (1.70, PI * 0.73, 4, true),
    ] {
        let mut backing = Vec::with_capacity(locks);
        for i in 0..locks {
            let angle = a_lo + (a_hi - a_lo) * (i as f32 + 0.5) / locks as f32;
            let cluster = groom.noise(Vec3::new(angle * 3., 0.2, 0.5), 2.);
            let angle = angle + cluster * (a_hi - a_lo) / locks as f32 * 0.13;
            let cut_layer = if style == HairStyle::LayeredLong {
                0.16
            } else {
                0.075
            };
            let tip = bottom
                + (0.5 + cluster * 0.5) * groom.program.layers * cut_layer
                + if style == HairStyle::AsymmetricBob {
                    angle / PI * 0.10
                } else {
                    0.
                };
            // Neighbouring locks share a wave envelope. Independent phases
            // produced rope lattices with large holes instead of dense hair.
            let phase = angle * 0.15 + groom.part * 2. + cluster * 0.10;
            let curl = match style {
                HairStyle::WavyLong => groom.program.curl_radius_m * 0.8,
                HairStyle::CurlyLong => groom.program.curl_radius_m * 1.5,
                _ => 0.001,
            };
            let centre = |t: f32| {
                let y = top + (tip - top) * t;
                let head_y = ((y - groom.centre.y) / (groom.size.y * 0.5)).clamp(0., 0.99);
                let head_radius = (1. - head_y * head_y).sqrt();
                let neck = smooth(lo + 0.015, lo - 0.15, y);
                let body = profile(y);
                let wave = (top - y) / groom.program.wave_length_m * TAU + phase;
                let fade = smooth(0.08, 0.32, t);
                let x = groom.centre.x
                    + angle.sin()
                        * (groom.size.x * 0.5 * head_radius + 0.005)
                        * (1. + neck * (groom.program.spread - 1.) * 0.3)
                    + neck * groom.program.sweep * 0.008
                    + curl * fade * wave.sin();
                let z_head =
                    groom.centre.z + angle.cos() * (groom.size.z * 0.5 * head_radius + 0.005);
                let z_body = if front {
                    body.x - 0.046
                } else {
                    body.y + 0.046
                };
                let z = z_head * (1. - neck)
                    + z_body * neck
                    + curl * fade * (1. + wave.cos()) * if front { -1. } else { 1. };
                Vec3::new(x, y, z)
            };
            let pitch = if front {
                groom.size.x * 0.045
            } else {
                groom.size.x * 1.04 / locks as f32
            };
            let section = if style == HairStyle::Locs {
                Vec2::splat(groom.program.clump_width_m * 0.48)
            } else {
                Vec2::new(pitch * 0.80, 0.0045 + groom.volume * 0.0015)
            };
            let rings = if style == HairStyle::CurlyLong {
                44
            } else {
                32
            };
            if !front && style != HairStyle::Locs {
                backing.push(
                    (0..=rings)
                        .map(|row| centre(row as f32 / rings as f32))
                        .collect(),
                );
            }
            strands::tress(
                g,
                rings,
                10,
                section,
                centre,
                |t| {
                    // Retain a dense fall. Tapering over the last fifth of the
                    // entire length left long, isolated needles at the hem.
                    let end_zone = (0.022 + groom.program.layers * 0.014) / (top - tip);
                    (0.4 + 0.6 * smooth(0., 0.16, t)) * (1. - 0.65 * smooth(1. - end_zone, 1., t))
                },
                phase * 0.01,
            );
            if style == HairStyle::CurlyLong && i % 6 == 0 {
                strands::tube(
                    g,
                    44,
                    8,
                    0.00065 + groom.program.flyaways * 0.0007,
                    |t| {
                        let p = centre(t);
                        let r = (groom.program.curl_radius_m * groom.program.flyaways).min(0.003)
                            * smooth(0.15, 0.60, t)
                            * (PI * t).sin();
                        let theta = phase + t * (top - tip) / groom.program.wave_length_m * TAU;
                        p + Vec3::new(r * theta.sin(), 0., r * theta.cos())
                    },
                    |t| 1. - 0.75 * t.powi(5),
                    phase * 0.013,
                );
            }
        }
        // A narrow closed inner layer fills gaps in loose hair, without widening
        // around shoulders. Locs deliberately retain individually separated locks.
        strands::backing(g, &backing);
    }
}
