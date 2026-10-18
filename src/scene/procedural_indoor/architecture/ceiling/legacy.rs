use super::*;

pub(super) fn build(a: &mut Assembly, scene: &IndoorManifest, m: &CeilingModule) {
    let (lo, hi) = m.body_bounds();
    let axis = if m.size.x > m.size.z {
        Vec3::X
    } else {
        Vec3::Z
    };
    let cross = Vec3::new(axis.z, 0., axis.x);
    let length = m.size.dot(axis);
    let width = m.size.dot(cross);
    let start = m.center - axis * (length * 0.5);
    let end = m.center + axis * (length * 0.5);
    let material = match m.system {
        CeilingSystem::AcousticCloud | CeilingSystem::Baffles => Surface::Fabric,
        CeilingSystem::TimberSlats => Surface::Wood,
        CeilingSystem::SprinklerMain => Surface::Accent,
        _ => Surface::Metal,
    };
    let panel = |a: &mut Assembly, surface, center, size, bevel| {
        a.box_part(surface, "ceiling", center, size, bevel);
    };
    match m.system {
        // These systems require recorded detail; manifest validation rejects
        // a missing shape instead of silently inventing a legacy substitute.
        CeilingSystem::Coffers | CeilingSystem::Ventilation => return,
        CeilingSystem::AcousticCloud => {
            let thickness = m.thickness_m.min(m.size.y * 0.35);
            panel(
                a,
                material,
                m.center.with_y(lo.y + thickness * 0.5),
                Vec3::new(m.size.x, thickness, m.size.z),
                0.007,
            );
        }
        CeilingSystem::Baffles | CeilingSystem::TimberSlats => {
            let count = (width / m.pitch_m).floor().max(2.) as usize;
            let w = m.thickness_m.min(width / count as f32 * 0.35);
            for i in 0..count {
                let p = m.center + cross * ((i as f32 + 0.5) * width / count as f32 - width * 0.5);
                panel(
                    a,
                    material,
                    p.with_y(lo.y + m.size.y * 0.34),
                    axis * (length - 0.006) + cross * w + Vec3::Y * (m.size.y * 0.68),
                    0.003,
                );
            }
            // Upper battens carry the separate slats, with vertical face gaps.
            for side in [-0.34, 0.34] {
                panel(
                    a,
                    Surface::Metal,
                    (m.center + axis * length * side).with_y(lo.y + m.size.y * 0.74),
                    axis * 0.026 + cross * width + Vec3::Y * (m.size.y * 0.09),
                    0.002,
                );
            }
        }
        CeilingSystem::RoundDuct | CeilingSystem::RectangularDuct => {
            let radius = (width * 0.46).min(m.size.y * 0.36);
            let y = lo.y + radius + 0.002;
            let p = start.with_y(y) + axis * 0.004;
            let q = end.with_y(y) - axis * 0.004;
            if m.system == CeilingSystem::RoundDuct {
                a.part(material, "ceiling").rod(p, q, radius);
                let count = (length / 1.05).ceil().max(1.) as usize;
                for i in 1..count {
                    let c = p.lerp(q, i as f32 / count as f32);
                    a.part(Surface::Chrome, "ceiling").rod(
                        c - axis * 0.009,
                        c + axis * 0.009,
                        radius + 0.002,
                    );
                }
            } else {
                panel(
                    a,
                    material,
                    (p + q) * 0.5,
                    axis * (length - 0.008) + cross * (radius * 2.) + Vec3::Y * (radius * 1.8),
                    0.003,
                );
                // Noncoplanar transverse flanges and a downward supply grille.
                for t in [0.24, 0.73] {
                    let c = p.lerp(q, t);
                    panel(
                        a,
                        Surface::Chrome,
                        c,
                        axis * 0.018
                            + cross * (radius * 2. + 0.008)
                            + Vec3::Y * (radius * 1.8 + 0.008),
                        0.002,
                    );
                }
            }
            // An actual grille drops clear of the shell, avoiding dark decals
            // on coincident faces and retaining geometric annotation ownership.
            let vent = m.center.with_y(lo.y + 0.010);
            panel(
                a,
                Surface::Ink,
                vent,
                axis * 0.30_f32.min(length * 0.4) + cross * (radius * 1.3) + Vec3::Y * 0.008,
                0.,
            );
            for i in 0..7 {
                panel(
                    a,
                    Surface::Metal,
                    vent + axis * (i as f32 - 3.) * 0.032 + Vec3::NEG_Y * 0.005,
                    axis * 0.012 + cross * (radius * 1.25) + Vec3::Y * 0.002,
                    0.,
                );
            }
        }
        CeilingSystem::SprinklerMain => {
            let radius = 0.016 + (m.seed % 17) as f32 * 0.0006;
            let y = hi.y - 0.040;
            a.part(material, "ceiling").rod(
                start.with_y(y) + axis * 0.006,
                end.with_y(y) - axis * 0.006,
                radius,
            );
            let count = (length / 1.8).ceil().max(1.) as usize;
            for i in 0..count {
                let center = start.lerp(end, (i as f32 + 0.5) / count as f32);
                let head = center.with_y(lo.y + 0.030);
                a.part(Surface::Chrome, "ceiling")
                    .rod(head, center.with_y(y), 0.008);
                // Deflector disc, narrow frame and red thermal bulb.
                a.part(Surface::Chrome, "ceiling").cylinder(
                    0.022,
                    0.004,
                    Transform::from_translation(head - Vec3::Y * 0.020),
                );
                for side in [-1., 1.] {
                    a.part(Surface::Chrome, "ceiling").rod(
                        head + cross * side * 0.013,
                        head + cross * side * 0.013 - Vec3::Y * 0.018,
                        0.0025,
                    );
                }
                a.part(Surface::Accent, "ceiling").rod(
                    head - Vec3::Y * 0.002,
                    head - Vec3::Y * 0.016,
                    0.003,
                );
            }
        }
        CeilingSystem::Truss => {
            let radius = m.thickness_m.min(0.027);
            let lower = lo.y + radius + 0.002;
            let upper = hi.y - radius - 0.025;
            let p = start + axis * (radius + 0.008);
            let q = end - axis * (radius + 0.008);
            for y in [lower, upper] {
                a.part(material, "ceiling")
                    .rod(p.with_y(y), q.with_y(y), radius);
            }
            let cells = (length / 0.55).ceil().max(2.) as usize;
            for i in 0..cells {
                let c = p.lerp(q, i as f32 / cells as f32);
                let d = p.lerp(q, (i + 1) as f32 / cells as f32);
                a.part(material, "ceiling").rod(
                    c.with_y(if i % 2 == 0 { lower } else { upper }),
                    d.with_y(if i % 2 == 0 { upper } else { lower }),
                    radius * 0.55,
                );
            }
        }
        _ => return,
    }
    // Suspension rises to the actual local roof, not a nominal room height.
    // Its mount terminates within one millimetre of the actual ceiling.
    for t in [0.16, 0.84] {
        let p = start.lerp(end, t);
        let roof = scene.ceiling_height(p.xz());
        let bottom = match m.system {
            CeilingSystem::AcousticCloud => lo.y + m.thickness_m.min(m.size.y * 0.35),
            CeilingSystem::Baffles | CeilingSystem::TimberSlats => lo.y + m.size.y * 0.78,
            CeilingSystem::RoundDuct | CeilingSystem::RectangularDuct => {
                let r = (width * 0.46).min(m.size.y * 0.36);
                lo.y + r
                    + 0.002
                    + r * if m.system == CeilingSystem::RoundDuct {
                        1.
                    } else {
                        0.9
                    }
                    - 0.001
            }
            CeilingSystem::SprinklerMain => {
                hi.y - 0.040 + 0.016 + (m.seed % 17) as f32 * 0.0006 - 0.001
            }
            _ => hi.y - 0.028,
        };
        a.part(Surface::Metal, "ceiling")
            .rod(p.with_y(bottom), p.with_y(roof - 0.001), 0.002);
        panel(
            a,
            Surface::Metal,
            p.with_y(roof - 0.003),
            Vec3::new(0.020, 0.004, 0.020),
            0.001,
        );
    }
}
