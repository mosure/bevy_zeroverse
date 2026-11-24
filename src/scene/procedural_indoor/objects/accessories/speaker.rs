//! Dimensioned speaker cabinets, driver cells and grille-covered enclosures.
mod cabinet;
mod mount;
use super::*;
use crate::scene::procedural_indoor::materials::speaker as acoustic;
pub use mount::{Mount, MountKind};

#[derive(Clone, Debug, serde::Serialize)]
pub struct Driver {
    pub low: f32,
    pub high: f32,
    pub center: Vec2,
    pub radius: f32,
    pub depth: f32,
    pub dome: f32,
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct SpeakerProgram {
    pub body: Vec3,
    pub wall: f32,
    pub horizontal: bool,
    pub covered: bool,
    pub mount: Mount,
    pub rear_width_ratio: f32,
    pub light_facing: bool,
    pub finish: Surface,
    pub grille: Surface,
    pub drivers: Vec<Driver>,
    pub control_x: f32,
    pub control_radius: f32,
}
fn body_size(size: Vec3, r: &mut impl Rng) -> Vec3 {
    size * Vec3::new(
        r.random_range(0.83..0.99),
        r.random_range(0.79..0.98),
        r.random_range(0.78..0.98),
    )
}
/// Mounting queries share the body draws without allocating driver geometry.
pub(crate) fn case_depth(o: &IndoorObject) -> f32 {
    body_size(o.size, &mut stream(o.seed, 0x535045414b455250)).z
}
pub fn parameters(o: &IndoorObject) -> SpeakerProgram {
    let mut r = stream(o.seed, 0x535045414b455250);
    let (mount, body) = Mount::fit(o, body_size(o.size, &mut r));
    let wall = body.min_element() * r.random_range(0.025..0.045);
    let horizontal = body.x > body.y * 1.6;
    let length = if horizontal { body.x } else { body.y };
    let width = if horizontal { body.y } else { body.x };
    let count = ((length / width * r.random_range(0.8..1.6)).round() as usize).clamp(1, 6);
    let weights: Vec<_> = (0..count).map(|_| r.random_range(0.65..1.65)).collect();
    let total = weights.iter().sum::<f32>();
    let mut cursor = -length * 0.5 + wall;
    let mut drivers = Vec::new();
    for weight in weights {
        let high = cursor + (length - 2. * wall) * weight / total;
        let radius = (width * 0.5 - wall).min((high - cursor) * 0.5) * r.random_range(0.58..0.87);
        let center = Vec2::new(
            (width * 0.5 - wall - radius) * r.random_range(-0.60..0.60),
            (cursor + high) * 0.5,
        );
        drivers.push(Driver {
            low: cursor,
            high,
            center,
            radius,
            depth: (radius * r.random_range(0.16..0.32)).min(body.z * 0.24),
            dome: r.random_range(0.19..0.38),
        });
        cursor = high;
    }
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x535045414b46494e);
    SpeakerProgram {
        body,
        wall,
        horizontal,
        drivers,
        covered: r.random_bool(0.43),
        finish: [Surface::Wood, Surface::Metal, Surface::Plastic][f.random_range(0..3)],
        grille: if f.random_bool(0.51) {
            Surface::Fabric
        } else {
            Surface::Metal
        },
        mount,
        light_facing: f.random_bool(0.29),
        rear_width_ratio: r.random_range(0.83..1.0),
        control_x: r.random_range(-0.30..0.30),
        control_radius: body.x.min(body.z) * r.random_range(0.035..0.065),
    }
}
impl SpeakerProgram {
    pub(crate) fn acoustic_label(&self) -> &'static str {
        if self.light_facing {
            acoustic::PALE_LABEL
        } else {
            acoustic::DARK_LABEL
        }
    }
    fn face_recess(&self) -> f32 {
        if self.covered {
            self.wall * 0.42
        } else {
            self.drivers
                .iter()
                .fold(self.wall * 0.25, |t, d| t.min(0.12 * d.depth))
        }
    }
    pub(crate) fn face_name(&self) -> &'static str {
        if !self.covered {
            "ExposedDrivers"
        } else if self.grille == Surface::Fabric {
            "WovenCloth"
        } else {
            "PerforatedSteel"
        }
    }
}

/// A rectangular baffle cell with a real circular driver aperture. Corner
/// samples are explicit; adjoining cells meet at their edges without overlap.
fn baffle(g: &mut Geometry, p: &SpeakerProgram, d: &Driver, tf: Transform) {
    let half = if p.horizontal { p.body.y } else { p.body.x } * 0.5 - p.wall;
    // Each rectangle corner lies outside both cardinal tangents of its quadrant.
    // Fan the corresponding arc into that corner, then fill the four gaps along
    // the straight sides. The fixed topology also works for an off-centre driver.
    const SIDES: usize = 32;
    static CIRCLE: std::sync::OnceLock<[Vec2; SIDES]> = std::sync::OnceLock::new();
    let circle = CIRCLE
        .get_or_init(|| std::array::from_fn(|i| Vec2::from_angle(TAU * i as f32 / SIDES as f32)));
    let start = g.positions.len() as u32;
    let normal = (tf.rotation * Vec3::Z).to_array();
    let corners = [
        Vec2::new(half, d.high),
        Vec2::new(-half, d.high),
        Vec2::new(-half, d.low),
        Vec2::new(half, d.low),
    ];
    for q in circle
        .iter()
        .map(|v| d.center + d.radius * *v)
        .chain(corners)
    {
        g.positions
            .push(tf.transform_point(q.extend(0.)).to_array());
        g.normals.push(normal);
        g.uvs.push(q.to_array());
    }
    for quadrant in 0..4 {
        let corner = start + (SIDES + quadrant) as u32;
        let previous = start + (SIDES + (quadrant + 3) % 4) as u32;
        let first = quadrant * (SIDES / 4);
        g.indices.extend([previous, corner, start + first as u32]);
        for i in first..first + SIDES / 4 {
            g.indices
                .extend([corner, start + ((i + 1) % SIDES) as u32, start + i as u32]);
        }
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let label = o.kind.class_name();
    let s = p.body;
    let tf = p.mount.cabinet(s);
    cabinet::shell(a, &p, tf, label);
    if !p.covered {
        let rotation = if p.horizontal {
            Quat::from_rotation_z(-FRAC_PI_2)
        } else {
            Quat::IDENTITY
        };
        let face = tf
            * Transform::from_xyz(0., s.y * 0.5, s.z * 0.5 - p.face_recess())
                .with_rotation(rotation);
        for d in &p.drivers {
            baffle(a.part(p.finish, label), &p, d, face);
            cabinet::driver(a, &p, d, face);
        }
    }
    p.mount.build_yoke(a, s);
    if p.mount.kind == MountKind::Desktop {
        let exposed = (p.wall * 0.55).min(o.size.y - s.y);
        a.part(Surface::Plastic, label).cylinder(
            p.control_radius,
            p.wall * 0.72,
            Transform::from_xyz(
                s.x * p.control_x,
                s.y + exposed - p.wall * 0.36,
                -s.z * 0.14,
            ),
        );
    }
}

#[cfg(test)]
mod tests;
