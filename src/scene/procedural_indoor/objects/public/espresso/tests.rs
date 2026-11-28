use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object, part_label};

#[test]
fn espresso_stock_has_metric_group_pitch_open_cup_bays_and_hopper_contents() {
    let mut groups = [0; 3];
    let mut fill = [0; 3];
    for seed in 0..96 {
        let o = audit::fixture(ObjectKind::EspressoMachine, seed);
        let machine = Machine::for_object(&o);
        groups[machine.groups - 1] += 1;
        let a = build_object(&o);
        let check = audit::check(&o, &a).unwrap();
        assert!(check.triangles < 12000 && check.min_y_m.abs() < 0.001);
        assert!(a.parts.keys().all(|(_, l)| part_label(l) == CLASS));
        // No triangle crosses a cup-sized volume between the front spouts and grate.
        let first = -(machine.groups as f32 - 1.) * 0.5 * machine.group_pitch_m;
        for i in 0..machine.groups {
            let center = Vec3::new(
                first + i as f32 * machine.group_pitch_m,
                machine.tray_height_m + machine.cup_clearance_m * 0.47,
                machine.spout_z_m,
            );
            let half = Vec3::new(0.028, machine.cup_clearance_m * 0.42, 0.028);
            for mesh in a.parts.values() {
                for tri in mesh.indices.as_chunks::<3>().0 {
                    let [u, v, w] = tri.map(|j| Vec3::from_array(mesh.positions[j as usize]));
                    let lo = u.min(v).min(w);
                    let hi = u.max(v).max(w);
                    assert!(
                        !((center + half).cmpgt(lo).all() && (center - half).cmplt(hi).all()),
                        "cup bay obstructed on seed {seed}"
                    );
                }
            }
        }
        let g = audit::fixture(ObjectKind::CoffeeGrinder, seed);
        let stock = Grinder::for_object(&g);
        let a = build_object(&g);
        let check = audit::check(&g, &a).unwrap();
        assert!(check.triangles < 6000 && check.min_y_m.abs() < 0.001);
        assert_eq!(
            a.parts.keys().any(|(_, s)| s.contains("#finish1060")),
            stock.fill_fraction > 0.
        );
        assert!(a.parts.keys().any(|(s, _)| *s == Surface::ContainerGlass));
        assert!(a.parts.keys().all(|(_, s)| part_label(s) == CLASS));
        fill[if stock.fill_fraction == 0. {
            0
        } else if stock.fill_fraction < 0.5 {
            1
        } else {
            2
        }] += 1;
    }
    assert!(
        groups.iter().all(|n| *n >= 11) && fill.iter().all(|n| *n >= 4),
        "{groups:?} {fill:?}"
    );
}
