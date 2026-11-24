//! Shared washing-station construction and fitted accessory poses.
use super::*;
mod basin;
pub mod tap;
pub mod waste;
pub(crate) const FITTED_PLUMBING_VERSION: u32 = 218;
pub mod program;
#[cfg(test)]
mod tests;

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &KitchenProgram, version: u32) {
    let plan = program::parameters_at(o, p, version);
    let top = plan.worktop_m;
    let underside = top - worktop::RUN_THICKNESS;
    let s = o.size;
    let (d, z) = body(o, p);
    let toe = 0.11;
    cabinet::open_shell(
        a,
        Vec3::new(s.x - 0.02, underside - toe, d),
        toe,
        z,
        p.finish,
        "cabinet",
    );
    a.box_part(
        Surface::Rubber,
        "cabinet",
        Vec3::new(0., toe * 0.5, z - 0.015),
        Vec3::new(s.x - 0.09, toe, d - 0.05),
        0.004,
    );
    let door_bottom = 0.12;
    cabinet::doors(
        a,
        s.x - 0.035,
        door_bottom,
        underside - door_bottom,
        Vec2::new(0., z + d * 0.5 - 0.012),
        p,
        "cabinet",
    );
    worktop::pierced(
        a,
        o,
        p.top,
        (
            Vec2::new(-s.x * 0.5, z - d * 0.5),
            Vec2::new(s.x * 0.5, z + d * 0.5),
        ),
        (plan.aperture_min, plan.aperture_max),
        top,
    );
    for (i, bowl) in plan.basins.iter().enumerate() {
        let shared_edges = [i > 0, i + 1 < plan.basins.len()];
        basin::build(a, bowl, top, plan.basin_finish, shared_edges);
    }
    for dish in &plan.dishes {
        dishware::build(
            a,
            Transform::from_translation(dish.position)
                .with_rotation(Quat::from_rotation_y(dish.yaw)),
            &dish.dish,
            dish.seed,
        );
    }
    if let Some(tap) = &plan.tap {
        tap.build(a, &plan, p.faucet_handles);
    } else {
        legacy_faucet(a, &plan, p.faucet_handles);
    }
    for prop in &plan.accessories {
        let mut child = Assembly::default();
        match prop.kind {
            ObjectKind::SoapDispenser => {
                supplies::dispenser::build(&mut child, Vec3::ZERO, prop.size, prop.pump, prop.seed)
            }
            ObjectKind::Sponge => {
                let mut obj = o.clone();
                obj.kind = prop.kind;
                obj.size = prop.size;
                obj.seed = prop.seed;
                supplies::sponge::build(&mut child, &obj);
            }
            ObjectKind::DryingRack => {
                props::rack(&mut child, Vec3::ZERO, prop.size, p.sink_dishes, prop.seed)
            }
            _ => unreachable!(),
        }
        let rotation = Quat::from_rotation_y(prop.yaw);
        for ((surface, label), mut g) in child.parts {
            for v in &mut g.positions {
                *v = (rotation * Vec3::from_array(*v) + prop.position).to_array();
            }
            for n in &mut g.normals {
                *n = (rotation * Vec3::from_array(*n)).to_array();
            }
            a.part(surface, &label).append(g);
        }
    }
}

fn legacy_faucet(a: &mut Assembly, plan: &program::SinkProgram, handles: u8) {
    pipe(
        a,
        plan.faucet_finish,
        "faucet",
        &plan.faucet_points,
        plan.faucet_radius_m,
    );
    a.part(plan.faucet_finish, "faucet").cylinder(
        plan.faucet_radius_m * 2.0,
        0.018,
        Transform::from_translation(plan.mount),
    );
    for &h in &plan.handles {
        a.part(plan.faucet_finish, "faucet").cylinder(
            0.016,
            0.035,
            Transform::from_translation(h + Vec3::Y * 0.014),
        );
        let points = match handles {
            1 => [
                h + Vec3::new(-0.025, 0.034, 0.),
                h + Vec3::new(0.025, 0.034, 0.),
            ],
            2 => [h + Vec3::Y * 0.034, h + Vec3::new(0.035, 0.034, 0.)],
            _ => [h + Vec3::Y * 0.03, h + Vec3::new(0., 0.075, 0.015)],
        };
        pipe(a, plan.faucet_finish, "faucet", &points, 0.004);
    }
}
