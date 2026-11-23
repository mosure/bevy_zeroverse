//! Physical device sockets. The mesh builder, routing and replay validation all
//! resolve connector position and departure normal through these same frames.
use super::*;
pub(crate) type Port = (Vec3, Vec3);

#[derive(Clone, Copy)]
pub(crate) enum TowerJack {
    Mains,
    Video(bool),
    Keyboard,
    Mouse,
}
impl TowerJack {
    pub(super) const ALL: [Self; 5] = [
        Self::Mains,
        Self::Video(false),
        Self::Video(true),
        Self::Keyboard,
        Self::Mouse,
    ];
}
pub(crate) fn tower_local(o: &IndoorObject, jack: TowerJack) -> Vec3 {
    let (x, y) = match jack {
        TowerJack::Mains => (-0.26, 0.30),
        TowerJack::Video(false) => (0.16, 0.38),
        TowerJack::Video(true) => (0.16, 0.60),
        TowerJack::Keyboard => (-0.26, 0.76),
        TowerJack::Mouse => (0.16, 0.81),
    };
    Vec3::new(o.size.x * x, o.size.y * y, -o.size.z * 0.5 - 0.001)
}

fn input_local(o: &IndoorObject, p: &InputProgram) -> Vec3 {
    let x = if o.kind == ObjectKind::Mouse {
        0.
    } else if p.keyboard_split {
        -o.size.x * 0.26
    } else {
        o.size.x * 0.19
    };
    Vec3::new(
        x,
        (o.size.y * 0.39).clamp(0.006, 0.015),
        -o.size.z * 0.5 - 0.001,
    )
}
pub(super) fn input_socket(a: &mut Assembly, o: &IndoorObject, p: &InputProgram) {
    let pt = input_local(o, p);
    let depth = if o.kind == ObjectKind::Keyboard && p.keyboard_split {
        o.size.z * 0.17
    } else {
        0.018
    };
    a.box_part(
        Surface::Rubber,
        o.kind.class_name(),
        pt + Vec3::Z * (depth * 0.5),
        Vec3::new(0.010, 0.0055, depth),
        0.001,
    );
}

pub(crate) fn monitor_local(o: &IndoorObject, source: ObjectKind) -> Option<Vec3> {
    let (x, y) = match source {
        ObjectKind::Monitor => (-0.024, 0.031),
        ObjectKind::Keyboard => (-0.024, 0.012),
        ObjectKind::Mouse => (0.024, 0.012),
        _ => return None,
    };
    Some(Vec3::new(x, y, -o.size.z * 0.5 - 0.001))
}
fn rear(o: &IndoorObject, local: Vec3) -> Port {
    let tf = o.transform();
    (tf.transform_point(local), tf.rotation * Vec3::NEG_Z)
}
pub(crate) fn data_ports(source: &IndoorObject, target: &IndoorObject) -> Option<[Port; 2]> {
    use ObjectKind::*;
    if (source.kind, target.kind) == (Laptop, Monitor) {
        // A dock-capable display can provide power and video on one USB-C lead.
        return Some([
            (
                equipment::power_port(source, true)?,
                equipment::power_port_normal(source, true)?,
            ),
            rear(target, monitor_local(target, Monitor)?),
        ]);
    }
    let (s, t) = match (source.kind, target.kind) {
        (Keyboard | Mouse, Monitor) => (
            input_local(source, &parameters(source)),
            monitor_local(target, source.kind)?,
        ),
        (Keyboard | Mouse, PcTower) => (
            input_local(source, &parameters(source)),
            tower_local(
                target,
                if source.kind == Keyboard {
                    TowerJack::Keyboard
                } else {
                    TowerJack::Mouse
                },
            ),
        ),
        (Monitor, PcTower) => (
            monitor_local(source, Monitor)?,
            tower_local(
                target,
                TowerJack::Video(target.interaction_target != Some(source.id)),
            ),
        ),
        _ => return None,
    };
    Some([rear(source, s), rear(target, t)])
}
pub(crate) fn connection(source: &IndoorObject, target: &IndoorObject) -> Option<[Port; 2]> {
    if let Some(ports) = data_ports(source, target) {
        return Some(ports);
    }
    use ObjectKind::*;
    if !(matches!(target.kind, PowerStrip | FloorOutlet | WallOutlet)
        && matches!(
            source.kind,
            Laptop | Projector | Monitor | PcTower | PowerStrip
        ))
        || source.kind == PowerStrip && target.kind == PowerStrip
        || source.kind == Laptop && target.kind != PowerStrip
    {
        return None;
    }
    let low = source.kind == Laptop;
    Some([
        (
            equipment::source_power_port(source)?,
            equipment::source_power_normal(source)?,
        ),
        (
            equipment::power_port(target, low)?,
            equipment::power_port_normal(target, low)?,
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::semantic::SemanticLabel;

    // Intersect the port center with the actual outward-facing socket triangles
    // in the local XY plane, independent of the analytic routing formula.
    fn hits_socket(mesh: &Assembly, center: Vec3) -> bool {
        mesh.parts
            .iter()
            .filter(|((surface, _), _)| *surface == Surface::Rubber)
            .any(|(_, g)| {
                g.indices.chunks_exact(3).any(|tri| {
                    let [a, b, c] =
                        [tri[0], tri[1], tri[2]].map(|i| Vec3::from_array(g.positions[i as usize]));
                    if [a, b, c].iter().any(|p| (p.z - center.z).abs() > 0.00055)
                        || (b - a).cross(c - a).z >= -1e-8
                    {
                        return false;
                    }
                    [(a, b), (b, c), (c, a)]
                        .into_iter()
                        .all(|(u, v)| (v - u).xy().perp_dot((center - u).xy()) <= 1e-8)
                })
            })
    }

    #[test]
    fn casework_ports_and_peripheral_plugs_meet_the_rendered_housings() {
        let mut measured_styles = [false; 3];
        for seed in 0..32 {
            for kind in [
                ObjectKind::PcTower,
                ObjectKind::Monitor,
                ObjectKind::Mouse,
                ObjectKind::Keyboard,
            ] {
                let mut o = audit::fixture(kind, seed);
                if kind == ObjectKind::PcTower {
                    o.size = tower::dimensions(&mut stream(seed, 27), seed % 2 == 0);
                    measured_styles[if o.size.y < 0.12 {
                        0
                    } else if o.size.x < 0.14 {
                        1
                    } else {
                        2
                    }] = true;
                }
                let mesh = build_object(&o);
                audit::check(&o, &mesh).unwrap_or_else(|e| panic!("{kind:?} {seed}: {e}"));
                let sockets: Vec<_> = match kind {
                    ObjectKind::PcTower => {
                        let (lo, hi) = mesh.bounds();
                        assert!(lo.y.abs() < 0.001 && (hi.y - o.size.y).abs() < 0.001);
                        assert!(mesh
                            .parts
                            .keys()
                            .all(|(_, label)| SemanticLabel::from_label(part_label(label))
                                == Some(SemanticLabel::PcTower)));
                        TowerJack::ALL.map(|j| tower_local(&o, j)).to_vec()
                    }
                    ObjectKind::Monitor => {
                        [ObjectKind::Monitor, ObjectKind::Mouse, ObjectKind::Keyboard]
                            .map(|k| monitor_local(&o, k).unwrap())
                            .to_vec()
                    }
                    _ => vec![input_local(&o, &parameters(&o))],
                };
                for p in sockets {
                    assert!(hits_socket(&mesh, p), "{kind:?} {seed} socket {p:?}");
                }
            }
        }
        assert_eq!(measured_styles, [true; 3]);
    }
}
