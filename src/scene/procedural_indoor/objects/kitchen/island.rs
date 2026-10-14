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
}

pub fn is_island(o: &IndoorObject) -> bool {
    o.kind == ObjectKind::KitchenCounter && o.variant == ISLAND_VARIANT
}

pub(super) fn is_drawer_bank(o: &IndoorObject, bay: u32) -> bool {
    (o.seed.wrapping_add(bay as u64)).is_multiple_of(2)
}

/// Establish the island's complete envelope before placement or prop sampling.
pub fn configure_island(o: &mut IndoorObject) {
    assert_eq!(o.kind, ObjectKind::KitchenCounter);
    o.variant = ISLAND_VARIANT;
    let mut r = stream(o.seed, 8081);
    o.size.x = r.random_range(1.50..2.35);
    o.size.z = r.random_range(0.92..1.18);
}

pub fn island_parameters(o: &IndoorObject) -> Option<IslandProgram> {
    if !is_island(o) {
        return None;
    }
    let mut r = stream(o.seed, 8082);
    Some(IslandProgram {
        slab_thickness: r.random_range(0.028..0.047),
        end_overhang: r.random_range(0.025..0.045),
        // Includes the front's 22 mm panels and projecting handles.
        front_overhang: r.random_range(0.060..0.075),
        rear_overhang: if r.random_bool(0.45) {
            r.random_range(0.18..0.28)
        } else {
            r.random_range(0.035..0.055)
        },
        plinth_height: r.random_range(0.085..0.12),
        plinth_recess: r.random_range(0.045..0.065),
        bays: (o.size.x / r.random_range(0.52..0.68))
            .round()
            .clamp(2., 4.) as u32,
    })
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, finish: &KitchenProgram) {
    let p = island_parameters(o).expect("island construction requires island installation");
    let w = o.size.x - 2. * p.end_overhang;
    let d = o.size.z - p.front_overhang - p.rear_overhang;
    let z = (p.rear_overhang - p.front_overhang) * 0.5;
    let top = o.size.y - p.slab_thickness;
    let h = top - p.plinth_height;
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
    let bay = w / p.bays as f32;
    for i in 0..p.bays {
        let x = -w * 0.5 + bay * (i as f32 + 0.5);
        // Drawer banks and door bays use real panel gaps, never a giant single
        // drawer stretched across a two-metre island.
        let drawers = is_drawer_bank(o, i);
        let rows = if drawers { 3 } else { 2 };
        let upper = if drawers { h / 3. } else { h * 0.25 };
        let mut bottom = p.plinth_height;
        for row in 0..rows {
            let rh = if drawers {
                upper
            } else if row == 0 {
                h - upper
            } else {
                upper
            };
            let tf = Transform::from_xyz(x, bottom + rh * 0.5, z + d * 0.5 + 0.014);
            cabinet::panel(
                a,
                Vec2::new(bay - 0.006, rh - 0.006),
                tf,
                finish.finish,
                "cabinet",
                false,
            );
            cabinet::handle(
                a,
                tf,
                0.,
                if !drawers && row == 0 { rh * 0.30 } else { 0. },
                finish,
                true,
                "cabinet",
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
