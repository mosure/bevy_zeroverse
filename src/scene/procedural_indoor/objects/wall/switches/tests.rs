use super::*;
use crate::scene::procedural_indoor::{
    envelope::ray_triangle,
    objects::{audit, build_object},
};

fn surface_at(a: &Assembly, q: Vec2) -> (f32, Surface) {
    let from = q.extend(0.10);
    a.parts
        .iter()
        .flat_map(|((surface, _), mesh)| {
            mesh.indices
                .as_chunks::<3>()
                .0
                .iter()
                .filter_map(move |tri| {
                    let p = tri.map(|i| Vec3::from_array(mesh.positions[i as usize]));
                    ray_triangle(from, Vec3::NEG_Z, p).map(|d| (from.z - d, *surface))
                })
        })
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .expect("ray hits installed plate")
}
#[track_caller]
fn same_height(x: f32, y: f32) {
    assert!((x - y).abs() < 1.5e-6, "{x} != {y}");
}

#[test]
fn switch_apertures_and_moving_faces_agree_with_the_installed_settings() {
    let mut observed = std::collections::BTreeMap::new();
    for seed in 0..192 {
        let mut o = audit::fixture(ObjectKind::LightSwitch, seed);
        o.size = [
            Vec3::new(0.062, 0.094, 0.016),
            Vec3::new(0.16, 0.115, 0.022),
            Vec3::new(0.27, 0.14, 0.026),
        ][seed as usize % 3];
        let b = parameters(&o).switch_bank.unwrap();
        let a = build_object(&o);
        audit::check(&o, &a).unwrap();
        let (lo, hi) = a.bounds();
        same_height(lo.z, -0.35 * o.size.z);
        assert!(
            hi.z <= o.size.z * 0.5 + 1e-5
                && lo.x >= -o.size.x * 0.5 - 1e-5
                && hi.y <= o.size.y + 1e-5
        );
        assert!(a
            .parts
            .keys()
            .all(|(_, l)| super::super::super::part_label(l) == o.kind.class_name()));
        for (i, &m) in b.modules.iter().enumerate() {
            *observed.entry(m.name()).or_insert(0) += 1;
            let c = b.center(i);
            let s = b.module_size - b.gap_m * 2.;
            let base = b.face_z_m - b.gap_m * 0.42;
            let rim = Vec2::X * (b.module_size.x * 0.5 + b.gap_m * 0.3);
            same_height(surface_at(&a, c + rim).0, b.face_z_m);
            let pocket = surface_at(&a, c + Vec2::X * (b.module_size.x * 0.5 - b.gap_m * 0.33));
            assert_eq!(pocket.1, Surface::Ink);
            same_height(pocket.0, b.pocket_z_m);
            let check_paddle = |center: Vec2, angle: f32| {
                let bottom = surface_at(&a, center - Vec2::Y * s.y * 0.28);
                let top = surface_at(&a, center + Vec2::Y * s.y * 0.28);
                assert_eq!((bottom.1, top.1), (Surface::Plastic, Surface::Plastic));
                same_height(top.0 - bottom.0, angle.tan() * s.y * 0.56);
                assert!(angle.abs() > 0.065 && angle.abs() <= 0.216);
            };
            match m {
                Actuator::Paddle { angle_rad } => check_paddle(c, angle_rad),
                Actuator::Split { angles_rad } => {
                    let width = (s.x - b.gap_m) * 0.5;
                    for (j, angle) in angles_rad.into_iter().enumerate() {
                        check_paddle(c + Vec2::X * ((j as f32 - 0.5) * (width + b.gap_m)), angle);
                    }
                    same_height(surface_at(&a, c).0, b.pocket_z_m);
                }
                Actuator::Rotary { angle_rad } => {
                    let v = Vec2::from_angle(angle_rad).rotate(Vec2::Y)
                        * (s.min_element() * 0.417 * 0.57);
                    let mark = surface_at(&a, c + v);
                    let plain = surface_at(&a, c + v.perp());
                    assert_eq!((mark.1, plain.1), (Surface::Ink, Surface::Plastic));
                    assert!(mark.0 > plain.0 && plain.0 > base);
                }
                Actuator::Slider { level } => {
                    let y = s.y * (level - 0.5) * (0.79 - 0.159);
                    assert!(surface_at(&a, c + Vec2::Y * y).0 > base);
                    same_height(
                        surface_at(&a, c - Vec2::Y * (if y >= 0. { 0.33 } else { -0.33 }) * s.y).0,
                        b.pocket_z_m,
                    );
                }
                Actuator::Buttons { latched } => {
                    for (j, off) in [-0.227, 0.227].into_iter().enumerate() {
                        let hit = surface_at(&a, c + Vec2::Y * s.y * off);
                        assert_eq!(hit.1, Surface::Plastic);
                        assert_eq!(hit.0 < base, latched == Some(j));
                    }
                }
                Actuator::Toggle { angle_rad } => {
                    // Locate the thrown tip independently along the circuit's
                    // centreline and compare it with the opposing side of the slot.
                    let (h, off) = (0..=32)
                        .map(|j| {
                            let off = Vec2::Y * (j as f32 * 0.0005 * angle_rad.signum());
                            (surface_at(&a, c + off).0, off)
                        })
                        .max_by(|a, b| a.0.total_cmp(&b.0))
                        .unwrap();
                    assert!(off.length() >= 0.001 && h > base + 0.002);
                    assert!(h - surface_at(&a, c - off).0 > 0.002);
                }
            }
        }
    }
    assert_eq!(observed.len(), 6);
    assert!(observed.values().all(|&n| n >= 6));
}

#[test]
fn room_hardware_can_match_without_locking_circuit_states_or_other_products() {
    let mut shared = 0;
    let mut mixed = 0;
    let mut toggle_states = [0, 0];
    let mut levels = Vec::new();
    for room in 0..128_u64 {
        let mut families = Vec::new();
        for local in 0..8 {
            let seed = room * 8 + local;
            let mut o = audit::fixture(ObjectKind::LightSwitch, seed);
            o.finish_seed = None;
            o.size = Vec3::new(0.26, 0.12, 0.023);
            install(&mut o, room, (room % 3) as u32);
            let p = parameters(&o);
            assert_eq!(o.variant % 3, (room % 3) as u32);
            let bank = p.switch_bank.unwrap();
            families.push(p.style);
            mixed += usize::from(
                bank.modules
                    .iter()
                    .any(|m| m.name() != bank.modules[0].name()),
            );
            for m in bank.modules {
                match m {
                    Actuator::Toggle { angle_rad } => {
                        toggle_states[usize::from(angle_rad > 0.)] += 1
                    }
                    Actuator::Slider { level } => levels.push(level),
                    _ => {}
                }
            }
            o.variant = (room % 3) as u32;
            assert!(parameters(&o).switch_bank.is_none());
        }
        shared += families
            .iter()
            .skip(1)
            .filter(|&&f| f == families[0])
            .count();
    }
    assert!(
        (500..850).contains(&shared) && mixed > 90,
        "shared {shared}, mixed {mixed}"
    );
    assert!(toggle_states.into_iter().all(|v| v > 75));
    assert!(levels.iter().any(|&v| v < 0.06) && levels.iter().any(|&v| v > 0.94));
}
