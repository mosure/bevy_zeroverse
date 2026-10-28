//! Freestanding cabinetry. The placement envelope is the entire countertop,
//! including its overhangs, rather than a wall-run carcass stretched in depth.
use super::*;

// Reserved explicitly; ordinary counter variants use 0..3. Keeping the same
// object kind preserves counter semantics, instance IDs and support APIs.
const ISLAND_VARIANT: u32 = 0x100;

#[derive(Clone, Debug, Serialize)]
pub struct IslandProgram {
    pub slab_thickness: f32,
    pub end_overhang: f32,
    pub front_overhang: f32,
    pub rear_overhang: f32,
    pub plinth_height: f32,
    pub plinth_recess: f32,
    pub bays: u32,
    pub bay_weights: Vec<f32>,
    pub raised_base: bool,
    pub waterfall_ends: bool,
}

pub fn is_island(o: &IndoorObject) -> bool {
    o.kind == ObjectKind::KitchenCounter && o.variant == ISLAND_VARIANT
}

pub(super) fn is_drawer_bank(o: &IndoorObject, bay: u32) -> bool {
    stream(o.seed, 8083 + bay as u64).random_bool(0.48)
}

/// Establish the island's complete envelope before placement or prop sampling.
pub fn configure_island(o: &mut IndoorObject) {
    assert_eq!(o.kind, ObjectKind::KitchenCounter);
    o.variant = ISLAND_VARIANT;
    let mut r = stream(o.seed, 8081);
    o.size.x = r.random_range(1.30..2.70);
    o.size.z = r.random_range(0.82..1.28);
}

pub fn island_parameters(o: &IndoorObject) -> Option<IslandProgram> {
    if !is_island(o) {
        return None;
    }
    let mut r = stream(o.seed, 8082);
    let bays = (o.size.x / r.random_range(0.48..0.76))
        .round()
        .clamp(2., 4.) as u32;
    let raised_base = r.random_bool(0.32);
    Some(IslandProgram {
        slab_thickness: r.random_range(0.022..0.064),
        end_overhang: r.random_range(0.025..0.045),
        // Includes the front's 22 mm panels and projecting handles.
        front_overhang: r.random_range(0.060..0.075),
        rear_overhang: if r.random_bool(0.45) {
            r.random_range(0.18..0.28)
        } else {
            r.random_range(0.035..0.055)
        },
        plinth_height: if raised_base {
            r.random_range(0.15..0.24)
        } else {
            r.random_range(0.065..0.14)
        },
        plinth_recess: r.random_range(0.045..0.065),
        bays,
        bay_weights: (0..bays).map(|_| r.random_range(0.82..1.30)).collect(),
        raised_base,
        waterfall_ends: r.random_bool(0.28),
    })
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, finish: &KitchenProgram) {
    let p = island_parameters(o).expect("island construction requires island installation");
    let w = o.size.x - 2. * p.end_overhang;
    let d = o.size.z - p.front_overhang - p.rear_overhang;
    let z = (p.rear_overhang - p.front_overhang) * 0.5;
    let top = o.size.y - p.slab_thickness;
    let h = top - p.plinth_height;
    if p.raised_base {
        for x in [-1., 1.] {
            for side in [-1., 1.] {
                a.box_part(
                    finish.hardware,
                    "cabinet",
                    Vec3::new(
                        x * (w * 0.5 - 0.065),
                        p.plinth_height * 0.5,
                        z + side * (d * 0.5 - 0.065),
                    ),
                    Vec3::new(0.042, p.plinth_height, 0.042),
                    0.004,
                );
            }
        }
    } else {
        a.box_part(
            finish.finish,
            "cabinet",
            Vec3::new(0., p.plinth_height * 0.5, z),
            Vec3::new(
                w - 2. * p.plinth_recess,
                p.plinth_height,
                d - 2. * p.plinth_recess,
            ),
            0.002,
        );
    }
    cabinet::shell(
        a,
        Vec3::new(w, h, d),
        p.plinth_height,
        z,
        finish.finish,
        "cabinet",
    );
    a.box_part(
        finish.top,
        "counter",
        Vec3::new(0., o.size.y - p.slab_thickness * 0.5, 0.),
        Vec3::new(o.size.x, p.slab_thickness, o.size.z),
        0.004,
    );
    if p.waterfall_ends {
        // Vertical worktop returns terminate against the underside of the
        // slab. Their inside faces remain outside the cabinet side panels.
        let thickness = (p.end_overhang * 0.78).min(p.slab_thickness);
        for side in [-1., 1.] {
            a.box_part(
                finish.top,
                "counter",
                Vec3::new(side * (o.size.x - thickness) * 0.5, top * 0.5, 0.),
                Vec3::new(thickness, top, o.size.z),
                0.002,
            );
        }
    }
    let total = p.bay_weights.iter().sum::<f32>();
    let mut cursor = -w * 0.5;
    for i in 0..p.bays {
        let bay = w * p.bay_weights[i as usize] / total;
        let x = cursor + bay * 0.5;
        cursor += bay;
        // Drawer banks and door bays use real panel gaps, never a giant single
        // drawer stretched across a two-metre island.
        let drawers = is_drawer_bank(o, i);
        let rows = if drawers { 3 } else { 2 };
        let upper = if drawers {
            h / 3.
        } else {
            h * stream(o.seed, 8090 + i as u64).random_range(0.18..0.32)
        };
        let mut bottom = p.plinth_height;
        if i > 0 {
            a.box_part(
                finish.finish,
                "cabinet",
                Vec3::new(x - bay * 0.5, p.plinth_height + h * 0.5, z),
                Vec3::new(0.016, h - 0.036, d - 0.016),
                0.001,
            );
        }
        for row in 0..rows {
            let rh = if drawers {
                upper
            } else if row == 0 {
                h - upper
            } else {
                upper
            };
            let tf = Transform::from_xyz(x, bottom + rh * 0.5, z + d * 0.5 + 0.014);
            use super::super::super::motion::{construction as motion, MotionKind};
            let seed = o.seed ^ ((i as u64) << 32) ^ row as u64 ^ 0x49534c45;
            let drawer = drawers || row > 0;
            motion::hinged(
                a,
                MotionKind::Cabinet,
                seed,
                || {
                    let side = if seed.is_multiple_of(2) { -1. } else { 1. };
                    let mut joint = motion::rigid(
                        MotionKind::Cabinet,
                        seed,
                        tf.translation + Vec3::X * side * (bay - 0.006) * 0.5,
                        Vec3::Y,
                        if drawer {
                            0.
                        } else {
                            side * motion::endpoint(seed, 0., 1.25)
                        },
                        if drawer {
                            Vec3::Z * d * 0.65
                        } else {
                            Vec3::ZERO
                        },
                    );
                    joint.frame = tf;
                    joint
                },
                |a| {
                    // Drawer fronts are opaque; paneled doors can be glazed.
                    let mut panel_finish = finish.clone();
                    if drawer {
                        panel_finish.door_style = 0;
                    }
                    // Fluted lower fields pair with clean utility drawers;
                    // avoid repeating dense reeds on every shallow front.
                    if row > 0 && panel_finish.front_profile == 3 {
                        panel_finish.front_profile = 0;
                    }
                    cabinet::front_panel(
                        a,
                        Vec2::new(bay - 0.006, rh - 0.006),
                        tf,
                        &panel_finish,
                        "cabinet",
                    );
                    if drawer {
                        cabinet::drawer_tray(a, bay - 0.006, rh - 0.006, d, tf, "cabinet");
                    }
                    let pull = cabinet::fronts::top_pull(
                        Vec2::new(bay - 0.006, rh - 0.006),
                        &panel_finish,
                    );
                    cabinet::handle(a, tf, pull.x, pull.y, finish, true, "cabinet");
                },
            );
            bottom += rh;
        }
        // A finished rear face is essential for an object viewed from all sides.
        let rear = Transform::from_xyz(x, p.plinth_height + h * 0.5, z - d * 0.5 - 0.014)
            .with_rotation(Quat::from_rotation_y(std::f32::consts::PI));
        cabinet::panel(
            a,
            Vec2::new(bay - 0.006, h - 0.006),
            rear,
            finish.finish,
            "cabinet",
            false,
        );
    }
}
