//! One realized optical surface for direct lighting, RGB and reflection probes.
use super::*;

pub(super) struct Fitting {
    pub(super) program: FixtureProgram,
    pub(super) size: Vec3,
    pub(super) insert: OpticalInsert,
    pub(super) linear: Option<LinearFixture>,
}

impl Fitting {
    pub(super) fn new(scene: &IndoorManifest, position: Option<Vec3>) -> Self {
        let (program, size) = if let Some(p) = position {
            (parameters_at(scene, p), super::size(scene, p))
        } else {
            (parameters(scene), super::super::fixture_size(scene))
        };
        let insert = optical_insert(scene, &program);
        let linear = LinearFixture::fit(scene, program.shape, position, size);
        Self {
            program,
            size,
            insert,
            linear,
        }
    }

    pub(super) fn area(&self) -> f32 {
        self.linear.map_or_else(
            || super::area(&self.program, self.size, self.insert),
            |l| l.projected_area_m2(),
        )
    }

    pub(super) fn reflector(&self, hole: f32) -> (f32, f32, f32, Surface) {
        let h = self.size.y;
        let y = -h * 0.5;
        match self.insert {
            OpticalInsert::Cup {
                throat,
                recess,
                reflector,
            } => (
                optics::cup_depth(h, hole, recess),
                hole * throat,
                y,
                reflector,
            ),
            _ if self.program.shape == FixtureShape::Downlight => {
                (h * 0.70, hole, y + 0.001, Surface::Chrome)
            }
            _ => (0.005, hole, y + 0.001, Surface::Chrome),
        }
    }

    pub(super) fn emit(&self, g: &mut Geometry, position: Vec3) {
        let p = &self.program;
        let s = self.size;
        let y = -s.y * 0.5;
        let tf = |v| Transform::from_translation(position + v);
        let along = if s.x >= s.z { Vec3::X } else { Vec3::Z };
        match p.shape {
            FixtureShape::Panel | FixtureShape::Louver => {
                let inner = 1. - p.border_fraction * 2.;
                if let OpticalInsert::Basket { width, recess, .. } = self.insert {
                    optics::Basket::new(position, s * Vec3::new(inner, 1., inner), width, recess)
                        .strip(g, 1);
                } else {
                    let light_y = y + if p.shape == FixtureShape::Louver {
                        s.y * 0.72
                    } else {
                        0.004
                    };
                    g.cuboid(
                        Vec3::new(s.x * inner, 0.002, s.z * inner),
                        0.,
                        tf(Vec3::Y * light_y),
                    );
                }
            }
            FixtureShape::Tube => {
                if let Some(linear) = self.linear {
                    return linear.aperture(g, position);
                }
                let across = Vec3::new(along.z, 0., along.x);
                let length = s.x.max(s.z) * 0.92;
                let radius = (s.x.min(s.z) * 0.09).min(s.y * 0.23);
                for sign in [-1., 1.] {
                    let c = position + across * sign * s.x.min(s.z) * 0.22 + Vec3::Y * (y + radius);
                    g.rod(c - along * length * 0.5, c + along * length * 0.5, radius);
                }
            }
            _ => {
                let (n, pitch, r) = cells(s);
                let hole = r * p.opening_fraction;
                for j in 0..n {
                    let c = along * pitch * (j as f32 + 0.5 - n as f32 * 0.5);
                    if p.shape == FixtureShape::Ring {
                        g.lathe(&[(hole, y + 0.003), (r, y + 0.003)], 32, tf(c));
                    } else {
                        let (inset, throat, _, _) = self.reflector(hole);
                        g.lathe(&[(0., y + inset), (throat, y + inset)], 32, tf(c));
                    }
                }
            }
        }
    }
}

pub(super) fn rotate(g: &mut Geometry, position: Vec3, rotation: Quat) {
    for v in &mut g.positions {
        *v = (position + rotation * (Vec3::from_array(*v) - position)).to_array();
    }
    for n in &mut g.normals {
        *n = (rotation * Vec3::from_array(*n)).to_array();
    }
}
