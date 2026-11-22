//! Joined street frontages on a common pedestrian apron. Recessed bays on the
//! street and exposed ends supply parallax without another transmissive layer.
use super::{
    bay::{Aperture, Face},
    *,
};
use crate::scene::procedural_indoor::architecture::outdoors::midground::Terrace;
use rand::Rng;

struct Stock {
    masonry: Surface,
    wall: String,
    jamb: String,
    door_jamb: String,
    window: String,
    door: String,
}

impl Stock {
    fn sample(seed: u64, masonry: Surface) -> Self {
        Self {
            masonry,
            wall: finish(masonry, "other_structure", seed),
            jamb: finish(Surface::Metal, "window", seed),
            door_jamb: finish(Surface::Metal, "door", seed),
            window: finish(Surface::Glass, "window", seed),
            door: finish(Surface::Paint, "door", seed),
        }
    }
}

fn bays(
    a: &mut Assembly,
    face: &Face,
    span: Vec2,
    levels: usize,
    entrance: bool,
    stock: &Stock,
    r: &mut impl Rng,
) {
    let max_cols = (span.x / 1.52).floor().clamp(1., 3.);
    let cols = (span.x / r.random_range(1.4..2.25))
        .round()
        .clamp(1., max_cols) as usize;
    let pitch = span.x / cols as f32;
    let door_col = r.random_range(0..cols);
    let width = (pitch * r.random_range(0.47..0.77)).min(1.61);
    let sill = r.random_range(0.73..1.16);
    let lintel = span.y - r.random_range(0.38..0.69);
    let recess = r.random_range(0.074..0.24);
    let frame = r.random_range(0.032..0.061);
    for level in 0..levels {
        for col in 0..cols {
            let at = Vec2::new(
                (col as f32 + 0.5) * pitch - span.x * 0.5,
                level as f32 * span.y,
            );
            let door = entrance && level == 0 && col == door_col;
            let w = if door {
                (pitch - 0.22).clamp(0.74, 1.08)
            } else {
                width
            }
            .min(pitch - 2. * (0.26 + frame));
            let aperture = Aperture {
                lo: at + Vec2::new(-w * 0.5, if door { 0.071 } else { sill }),
                hi: at + Vec2::new(w * 0.5, if door { 2.17 } else { lintel }),
            };
            let outer = aperture.expanded(frame);
            let limits = Aperture {
                lo: at - Vec2::X * (pitch * 0.5),
                hi: at + Vec2::new(pitch * 0.5, span.y),
            };
            outer.wall(
                face,
                a.part(stock.masonry, &stock.wall),
                &limits,
                recess + 0.017,
                true,
            );
            aperture.sash(
                face,
                a.part(
                    Surface::Metal,
                    if door { &stock.door_jamb } else { &stock.jamb },
                ),
                &outer,
                recess - 0.014,
                0.014,
            );
            let (surface, pigment) = if door {
                (Surface::Paint, &stock.door)
            } else {
                (Surface::Glass, &stock.window)
            };
            aperture.pane(face, a.part(surface, pigment), recess);
        }
    }
}

pub(in super::super) fn build(a: &mut Assembly, row: &Terrace) {
    let mut r = stream(row.seed, 0x5445_5252_4143_4553);
    let axis = row.axis().extend(0.).xzy();
    let normal = row.front.extend(0.).xzy();
    // Exterior grade is -0.070 m. Bury the foundation bottom so that neither
    // the render mesh nor the annotation intersection sees coincident planes.
    let outer_front = row.center.extend(-0.075).xzy() + normal * row.size.y * 0.5;
    let count = (row.size.x / r.random_range(2.65..4.30))
        .round()
        .clamp(1., 4.) as usize;
    let weights: Vec<f32> = (0..count).map(|_| r.random_range(0.82..1.18)).collect();
    let total: f32 = weights.iter().sum();
    let masonry = [Surface::Paint, Surface::Concrete, Surface::Terracotta][r.random_range(0..3)];
    let pitch = r.random_range(2.76_f32..3.35).min(row.eaves_limit - 0.22);
    let max_levels = ((row.eaves_limit - 0.16) / pitch).floor().clamp(1., 3.) as usize;
    let typical = r.random_range(1..=max_levels);
    let mut left = -row.size.x * 0.5;
    for (idx, w) in weights.into_iter().enumerate() {
        let plot = row.size.x * w / total;
        let width = plot - 0.024;
        let setback = r.random_range(0.06..0.29);
        let depth = row.size.y - setback;
        let x = left + plot * 0.5;
        left += plot;
        let levels = if r.random_bool(0.38) {
            r.random_range(1..=max_levels)
        } else {
            typical
        };
        let height = levels as f32 * pitch;
        let fascia = r.random_range(0.07_f32..0.15);
        let front_h = height + fascia;
        let back_h = front_h + r.random_range(0.0_f32..0.72).min(row.eaves_limit - front_h);
        let stock = Stock::sample(
            if idx == 0 || r.random_bool(0.65) {
                row.finish_seed
            } else {
                row.seed
            },
            masonry,
        );
        let face = Face {
            origin: outer_front + axis * x - normal * setback,
            axis,
            normal,
        };
        bays(
            a,
            &face,
            Vec2::new(width, pitch),
            levels,
            true,
            &stock,
            &mut r,
        );
        let half = width * 0.5;
        face.panel(
            a.part(masonry, &stock.wall),
            Vec2::new(-half, height),
            Vec2::new(half, front_h),
            0.,
        );
        let rear = Face {
            origin: face.origin - normal * depth,
            axis: -axis,
            normal: -normal,
        };
        rear.panel(
            a.part(masonry, &stock.wall),
            Vec2::new(-half, 0.),
            Vec2::new(half, back_h),
            0.,
        );
        let point = |x, y, z| face.point(Vec2::new(x, y), z);
        for (sign, exposed) in [(-1., idx == 0), (1., idx + 1 == count)] {
            let x = sign * half;
            let side = Face {
                origin: point(x, 0., depth * 0.5),
                axis: -normal,
                normal: axis * sign,
            };
            if exposed {
                bays(
                    a,
                    &side,
                    Vec2::new(depth, pitch),
                    levels,
                    false,
                    &stock,
                    &mut r,
                );
            }
            let foot = if exposed { height } else { 0. };
            a.part(masonry, &stock.wall).metric_quad(
                [
                    point(x, foot, 0.),
                    point(x, foot, depth),
                    point(x, back_h, depth),
                    point(x, front_h, 0.),
                ],
                axis * sign,
            );
        }
        a.part(Surface::Metal, "other_structure").metric_quad(
            [
                point(-half, front_h, 0.),
                point(half, front_h, 0.),
                point(half, back_h, depth),
                point(-half, back_h, depth),
            ],
            Vec3::Y,
        );
        a.part(masonry, &stock.wall).metric_quad(
            [
                point(-half, 0., 0.),
                point(half, 0., 0.),
                point(half, 0., depth),
                point(-half, 0., depth),
            ],
            Vec3::NEG_Y,
        );
        // Each sidewalk panel reaches its own threshold. The slab bottom is
        // buried in grade and the expansion joint separates adjacent panels.
        face.block(
            a.part(
                Surface::Concrete,
                &finish(Surface::Concrete, "floor", row.finish_seed),
            ),
            Vec3::new(0., 0.032, -(row.apron + setback) * 0.5),
            Vec3::new(plot - 0.009, 0.070, row.apron + setback),
        );
    }
}
