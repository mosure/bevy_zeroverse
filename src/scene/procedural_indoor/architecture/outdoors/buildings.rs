//! Connected wings, supported setbacks and coherent per-building facade rhythms.
use super::*;
mod bay;
#[cfg(test)]
mod bay_tests;
mod interior;
mod roofs;
pub(super) mod terrace;
use crate::scene::procedural_indoor::materials::exterior::label as finish;

/// Cheap threshold query for dataset metrics. The entrance and mesh use the
/// same bay grid, rebate clearance and full-depth wing of the ground storey.
pub(super) fn main_entrance(p: &Parcel, b: &Building) -> Option<streets::Entrance> {
    let width = p.size.x * b.split;
    let grid = bay::Bays::new(
        width,
        b.window_installation.recess(Vec2::new(width, p.size.y)),
        b,
    );
    let column = grid.columns / 2;
    let u = grid.offset(column);
    let door_width = grid.door_width(column, grid.available(u))?;
    let outward = Vec2::new(p.yaw.sin(), p.yaw.cos());
    let axis = Vec2::new(outward.y, -outward.x);
    Some(streets::Entrance {
        threshold: p.center + axis * (u - (p.size.x - width) * 0.5) + outward * (p.size.y * 0.5),
        outward,
        width: door_width,
    })
}

pub(super) fn build(
    a: &mut Assembly,
    p: &Parcel,
    b: &Building,
    viewed_side: Option<usize>,
) -> Option<streets::Entrance> {
    let mut entrance = None;
    let mut wings = vec![(Vec2::ZERO, p.size, b.levels)];
    if b.split < 1. {
        let left = p.size.x * b.split;
        let right = p.size.x - left;
        wings = vec![
            (
                Vec2::new(-right * 0.5, 0.),
                Vec2::new(left, p.size.y),
                b.levels,
            ),
            (
                Vec2::new(left * 0.5, (p.size.y * (b.wing_depth - 1.)) * 0.5),
                Vec2::new(right, p.size.y * b.wing_depth),
                b.wing_levels,
            ),
        ];
    }
    for (origin, footprint, levels) in wings {
        let mut top = (origin, footprint);
        for level in 0..levels {
            let inset = if level >= b.terrace_level && footprint.min_element() > 2.8 {
                b.setback
            } else {
                0.
            };
            let size = footprint * (1. - inset);
            let offset = origin + footprint * inset * 0.5 * b.shift;
            let bottom = level as f32 * b.pitch - 0.075;
            let center = Vec3::new(offset.x, bottom + b.pitch * 0.5, offset.y);
            // Only the caps are solid planes. Each wall builds the complement
            // of its openings and closes the bottom of every recessed bay.
            let rotation = Quat::from_rotation_y(p.yaw);
            let translate = Vec3::new(p.center.x, 0., p.center.y);
            let g = a.part(b.wall, &finish(b.wall, "other_structure", p.seed));
            for (y, normal) in [(bottom, Vec3::NEG_Y), (bottom + b.pitch - 0.021, Vec3::Y)] {
                let corners = [
                    Vec2::new(-0.5, -0.5),
                    Vec2::new(0.5, -0.5),
                    Vec2::new(0.5, 0.5),
                    Vec2::new(-0.5, 0.5),
                ];
                g.metric_quad(
                    corners.map(|uv| {
                        let q = offset + uv * size;
                        translate + rotation * Vec3::new(q.x, y, q.y)
                    }),
                    normal,
                );
            }
            let viewed = viewed_side.filter(|&side| {
                level < 3
                    && (b.split >= 1.
                        || !(side == 1 && origin.x < 0. || side == 3 && origin.x > 0.))
            });
            let doorway = facade(
                a,
                p,
                b,
                Storey {
                    offset,
                    size,
                    bottom,
                    level,
                },
                viewed,
            );
            // On a setback wing a diagonal approach could hit the adjoining
            // wing. Use a door on the foremost plane of the assembled footprint.
            if level == 0 && origin.y + footprint.y * 0.5 >= p.size.y * 0.5 - 0.001 {
                entrance = doorway;
            }
            // A shallow slab overlaps its wall at an internal face, leaving no
            // coplanar exterior cap. Upper floors remain supported inside it.
            box_at(
                a,
                p,
                Surface::Concrete,
                "other_structure",
                center.with_y(bottom + b.pitch - 0.015),
                Vec3::new(size.x - 0.018, 0.10, size.y + 0.08),
                0.,
            );
            top = (offset, size);
        }
        roofs::build(a, p, b, top.0, top.1, levels as f32 * b.pitch - 0.045);
    }
    entrance
}

struct Storey {
    offset: Vec2,
    size: Vec2,
    bottom: f32,
    level: usize,
}

fn facade(
    a: &mut Assembly,
    p: &Parcel,
    b: &Building,
    storey: Storey,
    viewed_side: Option<usize>,
) -> Option<streets::Entrance> {
    let Storey {
        offset,
        size,
        bottom,
        level,
    } = storey;
    let mut r = stream(p.seed, 6382 + level as u64);
    let rotation = Quat::from_rotation_y(p.yaw);
    let world = Vec3::new(p.center.x, 0., p.center.y);
    let recess = b.window_installation.recess(size);
    let rebate = b.window_installation.rebate_m;
    let wall = finish(b.wall, "other_structure", p.seed);
    let frames = ["window", "door"].map(|name| finish(b.frame, name, p.seed));
    let panes = ["window", "door"].map(|name| finish(Surface::Glass, name, p.seed));
    let mut entrance = None;
    for side in 0..4 {
        let (axis, normal, width, depth) = if side % 2 == 0 {
            (Vec3::X, Vec3::Z, size.x, size.y)
        } else {
            (Vec3::Z, Vec3::X, size.y, size.x)
        };
        let normal = normal * if side < 2 { 1. } else { -1. };
        let origin = Vec3::new(offset.x, bottom, offset.y) + normal * depth * 0.5;
        let face = bay::Face {
            origin: world + rotation * origin,
            axis: rotation * axis,
            normal: rotation * normal,
        };
        let reveal = recess + rebate + 0.018;
        let room = (viewed_side == Some(side)).then(|| {
            let seed =
                p.seed ^ u64::from(offset.x.to_bits()).rotate_left(17) ^ (4 * level + side) as u64;
            interior::Room::sample(seed, width, depth, b.pitch, reveal)
        });
        let mut has_opening = false;
        let mut shade_stock = stream(p.seed, 0x5348414445 + side as u64);
        let installed = shade_stock.random_bool(b.window_installation.roller_probability);
        let typical_drop: f32 = shade_stock.random_range(0.0..0.88);
        let mut shade_rng = stream(p.seed, 0x53484144524f57 + (level * 4 + side) as u64);
        let grid = bay::Bays::new(width, recess, b);
        let columns = grid.columns;
        let cell = grid.cell;
        let window_w = (cell * b.window_fraction).max(0.16);
        let window_h = b.pitch * b.window_height;
        let row_shift = if level % 2 == 1 {
            b.stagger * cell * (1. - b.window_fraction)
        } else {
            0.
        };
        for column in 0..columns {
            let u = grid.offset(column) + row_shift;
            // The aperture budget includes both frame jambs and a real joint
            // between bays. High glazing ratios must not overlap thick frames.
            // Leave the material behind the adjacent corner's own cavity.
            let available = grid.available(u);
            let door_width = if side == 0 && level == 0 {
                grid.door_width(column, available)
            } else {
                None
            };
            let door = door_width.is_some();
            let window_w = door_width.unwrap_or(window_w.min(available));
            let window_h = if door { 2.12 } else { window_h };
            let label = if door { "door" } else { "window" };
            let sill = if door { 0.07 } else { b.pitch * b.sill };
            let center = origin + axis * u + Vec3::Y * (sill + window_h * 0.5);
            let f = b.frame_width;
            let clear = bay::Aperture {
                lo: Vec2::new(u - window_w * 0.5, sill),
                hi: Vec2::new(u + window_w * 0.5, sill + window_h),
            };
            let outer = clear.expanded(f);
            let bounds = bay::Aperture {
                lo: Vec2::new(
                    if column == 0 {
                        -width * 0.5
                    } else {
                        u - cell * 0.5
                    },
                    0.,
                ),
                hi: Vec2::new(
                    if column + 1 == columns {
                        width * 0.5
                    } else {
                        u + cell * 0.5
                    },
                    b.pitch - 0.021,
                ),
            };
            if window_w < 0.16 {
                // A deep jamb at a narrow corner can consume the bay. Preserve
                // the structural pier rather than emitting an inverted cutout.
                face.panel(a.part(b.wall, &wall), bounds.lo, bounds.hi, 0.);
                continue;
            }
            outer.wall(
                &face,
                a.part(b.wall, &wall),
                &bounds,
                reveal,
                room.is_none(),
            );
            clear.sash(
                &face,
                a.part(b.frame, &frames[usize::from(door)]),
                &outer,
                recess,
                rebate,
            );
            if room.is_some() {
                clear.clear_pane(&face, a.part(Surface::Glass, label), recess + rebate);
                has_opening = true;
            } else {
                clear.pane(
                    &face,
                    a.part(Surface::Glass, &panes[usize::from(door)]),
                    recess + rebate,
                );
            }
            let sash_center = center - normal * recess;
            if door {
                entrance = Some(streets::Entrance {
                    threshold: face.point(Vec2::new(u, sill), 0.).xz(),
                    outward: face.normal.xz(),
                    width: window_w,
                });
                box_at(
                    a,
                    p,
                    b.frame,
                    "door",
                    sash_center + axis * window_w * 0.32 + normal * 0.025,
                    axis.abs() * 0.025 + Vec3::Y * 0.38 + normal.abs() * 0.08,
                    0.,
                );
            }
            // Embed mullion/transom end caps into the surrounding frame. Ending
            // exactly at the pane edge makes their cap faces coplanar with glass.
            if !door && window_w > 0.85 && r.random_bool(0.7) {
                box_at(
                    a,
                    p,
                    b.frame,
                    label,
                    sash_center + normal * 0.004,
                    axis.abs() * (f * 0.65) + Vec3::Y * (window_h + 0.006) + normal.abs() * 0.043,
                    0.,
                );
            }
            if !door && r.random_bool(0.4) {
                box_at(
                    a,
                    p,
                    b.frame,
                    label,
                    sash_center + normal * 0.009 + Vec3::Y * window_h * 0.22,
                    axis.abs() * (window_w + 0.006) + Vec3::Y * (f * 0.6) + normal.abs() * 0.045,
                    0.,
                );
            }
            if installed && !door && shade_rng.random_bool(0.82) {
                // As on a real elevation, adjacent offices can share one stock
                // while leaving their screens raised, part-drawn, or nearly shut.
                let drop = (typical_drop + shade_rng.random_range(-0.42..0.42)).clamp(0., 0.97);
                let head = sash_center + Vec3::Y * (window_h * 0.5 + f * 0.37);
                let width = window_w + f * 1.3;
                box_at(
                    a,
                    p,
                    b.frame,
                    "blinds",
                    head + normal * 0.031,
                    axis.abs() * width + Vec3::Y * 0.064 + normal.abs() * 0.074,
                    0.,
                );
                let length = (window_h * drop).max(0.);
                if length > 0.055 {
                    let top = head - Vec3::Y * 0.029 + normal * 0.049;
                    box_at(
                        a,
                        p,
                        Surface::FabricAlt,
                        "blinds",
                        top - Vec3::Y * length * 0.5,
                        axis.abs() * (width - 0.018) + Vec3::Y * length + normal.abs() * 0.005,
                        0.,
                    );
                    box_at(
                        a,
                        p,
                        b.frame,
                        "blinds",
                        top - Vec3::Y * length,
                        axis.abs() * (width - 0.013) + Vec3::Y * 0.015 + normal.abs() * 0.012,
                        0.,
                    );
                }
            }
            if b.hood > 0. {
                box_at(
                    a,
                    p,
                    b.frame,
                    "other_structure",
                    center + Vec3::Y * (window_h * 0.5 + f + 0.024) + normal * b.hood * 0.5,
                    axis.abs() * (window_w + f * 2.) + Vec3::Y * 0.035 + normal.abs() * b.hood,
                    0.,
                );
            }
            if b.fins && column < columns - 1 {
                box_at(
                    a,
                    p,
                    b.frame,
                    "other_structure",
                    origin + axis * (u + cell * 0.5) + Vec3::Y * b.pitch * 0.5 + normal * 0.07,
                    axis.abs() * 0.04 + Vec3::Y * (b.pitch - 0.18) + normal.abs() * 0.15,
                    0.,
                );
            }
        }
        if let Some(room) = room.filter(|_| has_opening) {
            room.build(a, &face);
        }
        // Plinth/sill courses and rainwater goods have metric rather than
        // fixed relative sizes, while respecting each facade's bay envelope.
        if level == 0 && side != 0 {
            box_at(
                a,
                p,
                Surface::Concrete,
                "other_structure",
                origin + Vec3::Y * 0.12 + normal * 0.022,
                axis.abs() * width + Vec3::Y * 0.22 + normal.abs() * 0.06,
                0.,
            );
        }
    }
    entrance
}
