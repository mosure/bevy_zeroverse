//! Shared washing-station construction and fitted accessory poses.
use super::*;
mod basin;
pub mod program;
#[cfg(test)]
mod tests;

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &KitchenProgram) {
    let plan = program::parameters(o, p);
    let top = plan.worktop_m;
    let s = o.size;
    let (d, z) = body(o, p);
    cabinet::open_shell(
        a,
        Vec3::new(s.x - 0.02, top - 0.11, d),
        0.11,
        z,
        p.finish,
        "cabinet",
    );
    a.box_part(
        Surface::Rubber,
        "cabinet",
        Vec3::new(0., 0.055, z - 0.015),
        Vec3::new(s.x - 0.09, 0.11, d - 0.05),
        0.004,
    );
    cabinet::doors(
        a,
        s.x - 0.035,
        0.12,
        top - 0.15,
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
    for bowl in &plan.basins {
        basin::build(a, bowl, top, plan.basin_finish);
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
        let points = match p.faucet_handles {
            1 => [
                h + Vec3::new(-0.025, 0.034, 0.),
                h + Vec3::new(0.025, 0.034, 0.),
            ],
            2 => [h + Vec3::Y * 0.034, h + Vec3::new(0.035, 0.034, 0.)],
            _ => [h + Vec3::Y * 0.03, h + Vec3::new(0., 0.075, 0.015)],
        };
        pipe(a, plan.faucet_finish, "faucet", &points, 0.004);
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
