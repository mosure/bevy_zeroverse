//! Coordinated luminaire construction, placement envelope and emitting area.
//! Room circuits share a construction language, with individual metric sizes.
use super::*;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum FixtureShape {
    Panel,
    Louver,
    Downlight,
    Drum,
    Ring,
    Tube,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FixtureProgram {
    pub shape: FixtureShape,
    pub trim: Surface,
    pub finish: u8,
    pub depth_m: f32,
    pub border_fraction: f32,
    pub opening_fraction: f32,
    pub louver_pitch_m: f32,
}

pub fn parameters(scene: &IndoorManifest) -> FixtureProgram {
    let mut r = stream(scene.seed, 7592);
    let shape = [
        FixtureShape::Panel,
        FixtureShape::Louver,
        FixtureShape::Downlight,
        FixtureShape::Drum,
        FixtureShape::Ring,
        FixtureShape::Tube,
    ][r.random_range(0..6)];
    FixtureProgram {
        shape,
        trim: [
            Surface::Metal,
            Surface::Chrome,
            Surface::Paint,
            Surface::WoodEdge,
        ][r.random_range(0..4)],
        finish: r.random_range(0..12),
        depth_m: r.random_range(0.045..0.12),
        border_fraction: r.random_range(0.05..0.13),
        opening_fraction: r.random_range(0.60..0.82),
        louver_pitch_m: r.random_range(0.07..0.18),
    }
}

/// Circuit construction with deterministic, bounded fixture-level differences.
/// Geometry and photometry must consume the identical realized aperture.
pub fn parameters_at(scene: &IndoorManifest, position: Vec3) -> FixtureProgram {
    let mut p = parameters(scene);
    let key = position.x.to_bits() as u64 | (position.z.to_bits() as u64) << 32;
    let mut r = stream(scene.seed ^ key, 0x4649585441504552);
    p.opening_fraction = (p.opening_fraction + r.random_range(-0.14..0.12)).clamp(0.42, 0.93);
    p.border_fraction *= r.random_range(0.65..1.35);
    p.louver_pitch_m *= r.random_range(0.65..1.40);
    p
}

fn cells(size: Vec3) -> (usize, f32, f32) {
    let length = size.x.max(size.z);
    let width = size.x.min(size.z);
    let count = (length / (width * 1.3)).floor().clamp(1., 6.) as usize;
    (
        count,
        length / count as f32,
        width.min(length / count as f32) * 0.46,
    )
}

/// Area of the downward emitting face, including the faceted circular meshes.
pub fn emitting_area(scene: &IndoorManifest) -> f32 {
    area(parameters(scene), super::fixture_size(scene))
}

pub fn emitting_area_at(scene: &IndoorManifest, position: Vec3) -> f32 {
    area(parameters_at(scene, position), size(scene, position))
}

pub fn size(scene: &IndoorManifest, position: Vec3) -> Vec3 {
    super::fixture_layout::size(scene, position)
}

fn area(p: FixtureProgram, s: Vec3) -> f32 {
    let inner = 1. - p.border_fraction * 2.;
    match p.shape {
        FixtureShape::Panel | FixtureShape::Louver => s.x * s.z * inner * inner,
        FixtureShape::Tube => s.x.max(s.z) * 0.92 * 4. * (s.x.min(s.z) * 0.09).min(s.y * 0.23),
        _ => {
            let (n, _, r) = cells(s);
            let area = 16. * (std::f32::consts::TAU / 32.).sin() * r * r;
            n as f32
                * area
                * match p.shape {
                    FixtureShape::Ring => 1. - p.opening_fraction.powi(2),
                    _ => p.opening_fraction.powi(2),
                }
        }
    }
}

/// Outside the emitting aperture, shared by direct lighting and GI.
pub fn emitter(scene: &IndoorManifest, position: Vec3) -> Vec3 {
    position + direction(scene, position) * (size(scene, position).y * 0.5 + 0.006)
}

/// Seeded circuit orientation shared by mesh construction and placement audits.
pub fn rotation(scene: &IndoorManifest, position: Vec3) -> Quat {
    super::fixture_layout::rotation(scene, position)
}

pub fn direction(scene: &IndoorManifest, position: Vec3) -> Vec3 {
    rotation(scene, position) * Vec3::NEG_Y
}

pub fn geometry(scene: &IndoorManifest, index: usize, position: Vec3) -> Assembly {
    let mut a = Assembly::default();
    let p = parameters_at(scene, position);
    let s = size(scene, position);
    let label = format!("lamp#{index}");
    let trim = if super::super::materials::variants::supports(p.trim) {
        format!("{label}#finish{}", p.finish)
    } else {
        label.clone()
    };
    let y = -s.y * 0.5;
    let tf = |v| Transform::from_translation(position + v);
    let inner = 1. - p.border_fraction * 2.;
    match p.shape {
        FixtureShape::Panel | FixtureShape::Louver => {
            // Four rails leave a real opening. The backing is above the diffuser.
            for axis in [0, 2] {
                for sign in [-1., 1.] {
                    let mut size = s;
                    size[axis] = s[axis] * p.border_fraction;
                    if axis == 2 {
                        size.x *= inner;
                    }
                    let mut c = Vec3::ZERO;
                    c[axis] = sign * (s[axis] - size[axis]) * 0.5;
                    a.part(p.trim, &trim).cuboid(size, 0.002, tf(c));
                }
            }
            let light_y = y + if p.shape == FixtureShape::Louver {
                s.y * 0.72
            } else {
                0.004
            };
            a.part(Surface::Light, &label).cuboid(
                Vec3::new(s.x * inner, 0.002, s.z * inner),
                0.,
                tf(Vec3::Y * light_y),
            );
            a.part(p.trim, &trim).cuboid(
                Vec3::new(s.x * inner, 0.003, s.z * inner),
                0.,
                tf(Vec3::Y * (s.y * 0.5 - 0.0015)),
            );
            if p.shape == FixtureShape::Louver {
                let n = (s.x / p.louver_pitch_m).floor().clamp(2., 20.) as usize;
                for j in 1..n {
                    a.part(Surface::Chrome, &label).cuboid(
                        Vec3::new(0.003, s.y * 0.60, s.z * inner),
                        0.0005,
                        tf(Vec3::new(
                            s.x * inner * (j as f32 / n as f32 - 0.5),
                            y + s.y * 0.34,
                            0.,
                        )),
                    );
                }
            }
        }
        FixtureShape::Tube => {
            let along = if s.x >= s.z { Vec3::X } else { Vec3::Z };
            let across = Vec3::new(along.z, 0., along.x);
            let length = s.x.max(s.z) * 0.92;
            let radius = (s.x.min(s.z) * 0.09).min(s.y * 0.23);
            a.part(p.trim, &trim)
                .cuboid(s.with_y(s.y * 0.18), 0.002, tf(Vec3::Y * s.y * 0.36));
            for sign in [-1., 1.] {
                let c = position + across * sign * s.x.min(s.z) * 0.22 + Vec3::Y * (y + radius);
                a.part(Surface::Light, &label).rod(
                    c - along * length * 0.5,
                    c + along * length * 0.5,
                    radius,
                );
                for end in [-1., 1.] {
                    let q = c + along * end * (length * 0.5 + 0.003);
                    a.part(Surface::Ceramic, &label).rod(
                        q - along * 0.004,
                        q + along * 0.004,
                        radius * 1.05,
                    );
                    a.part(p.trim, &trim)
                        .rod(q, q.with_y(position.y + s.y * 0.36), radius * 0.50);
                }
            }
        }
        _ => {
            let (n, pitch, r) = cells(s);
            let along = if s.x >= s.z { Vec3::X } else { Vec3::Z };
            for j in 0..n {
                let c = along * pitch * (j as f32 + 0.5 - n as f32 * 0.5);
                let hole = r * p.opening_fraction;
                if p.shape == FixtureShape::Ring {
                    a.part(p.trim, &trim).lathe(
                        &[
                            (r, y + 0.004),
                            (r, s.y * 0.45),
                            (hole, s.y * 0.45),
                            (hole, y + 0.004),
                        ],
                        32,
                        tf(c),
                    );
                    a.part(Surface::Light, &label).lathe(
                        &[(hole, y + 0.003), (r, y + 0.003)],
                        32,
                        tf(c),
                    );
                } else {
                    let inset = if p.shape == FixtureShape::Downlight {
                        s.y * 0.70
                    } else {
                        0.005
                    };
                    a.part(p.trim, &trim).lathe(
                        &[(hole, y), (r, y), (r, s.y * 0.48), (0., s.y * 0.48)],
                        32,
                        tf(c),
                    );
                    a.part(Surface::Chrome, &label).lathe(
                        &[(hole, y + inset), (hole, y + 0.001)],
                        32,
                        tf(c),
                    );
                    a.part(Surface::Light, &label).lathe(
                        &[(0., y + inset), (hole, y + inset)],
                        32,
                        tf(c),
                    );
                }
                // A carrier connects round arrays instead of floating heads.
                if n > 1 {
                    a.part(p.trim, &trim).rod(
                        position + c + Vec3::Y * s.y * 0.43,
                        position
                            + c
                            + along * if j + 1 == n { -pitch * 0.2 } else { pitch }
                            + Vec3::Y * s.y * 0.43,
                        0.004,
                    );
                }
            }
        }
    }
    let rotation = rotation(scene, position);
    for g in a.parts.values_mut() {
        for v in &mut g.positions {
            *v = (position + rotation * (Vec3::from_array(*v) - position)).to_array();
        }
        for n in &mut g.normals {
            *n = (rotation * Vec3::from_array(*n)).to_array();
        }
    }
    let anchors = if matches!(
        p.shape,
        FixtureShape::Downlight | FixtureShape::Drum | FixtureShape::Ring
    ) {
        let (n, pitch, r) = cells(s);
        let axis = if s.x >= s.z { Vec3::X } else { Vec3::Z };
        [-1., 1.].map(|sign| axis * sign * ((n - 1) as f32 * pitch * 0.5 + r * 0.86))
    } else {
        [-1., 1.].map(|sign| Vec3::X * sign * s.x * 0.32)
    };
    for offset in anchors {
        let offset = rotation * (offset + Vec3::Y * s.y * 0.44);
        let roof = scene
            .envelope
            .as_ref()
            .and_then(|e| e.mezzanine.as_ref())
            .filter(|m| m.deck.contains(position.xz()) && position.y < m.deck.height)
            .map_or_else(
                || scene.ceiling_height((position + offset).xz()),
                |m| m.deck.height - m.thickness,
            );
        a.part(Surface::Chrome, &label).rod(
            position + offset,
            (position + offset).with_y(roof - 0.001),
            0.002,
        );
    }
    a
}
