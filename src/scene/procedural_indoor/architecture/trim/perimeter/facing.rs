//! Surface frame of an extruded board. Grain follows its length and wraps
//! continuously over the installed profile, including across cutouts.
use super::*;

pub(super) struct Facing<'a> {
    pub(super) along: Vec3,
    across: Vec3,
    curved: Option<(&'a TrimSection, bool)>,
    distance: f32,
    timber_origin: Option<Vec2>,
}

pub(super) struct Band {
    along: Vec3,
    distances: [f32; 2],
    normals: [Vec3; 2],
    timber_origin: Option<Vec2>,
}

impl<'a> Facing<'a> {
    pub(super) fn new(run: &Run, section: Option<(&'a TrimSection, bool)>) -> Self {
        let pitched = |d: Vec2| Vec3::new(d.x, run.gradient.dot(d), d.y);
        Self {
            along: pitched(run.along).normalize(),
            across: pitched(run.inward),
            curved: section
                .filter(|(s, _)| matches!(s.profile, TrimProfile::Round | TrimProfile::Cove)),
            distance: 0.,
            timber_origin: run.timber_origin,
        }
    }

    pub(super) fn band(&mut self, lo: f32, hi: f32, depth_delta: f32) -> Band {
        let transverse = Vec3::Y * (hi - lo) + self.across * depth_delta;
        let start = self.distance;
        self.distance += transverse.reject_from_normalized(self.along).length();
        let flat = self.along.cross(transverse).normalize();
        let mut normals = [flat; 2];
        if let Some((s, crown)) = self.curved {
            let height = |h: f32| {
                if crown {
                    1. - h / s.height_m
                } else {
                    h / s.height_m
                }
            };
            if height((lo + hi) * 0.5) > s.shoulder {
                // Tangents of the quarter ellipse in metric section space.
                // Finite at both tips, even where dz/dh is infinite.
                normals = [lo, hi].map(|h| {
                    let u = ((height(h) - s.shoulder) / (1. - s.shoulder)).clamp(0., 1.);
                    let (rise, run) = if s.profile == TrimProfile::Round {
                        ((1. - u * u).sqrt(), -u)
                    } else {
                        ((2. * u - u * u).sqrt(), u - 1.)
                    };
                    let dy = rise * s.height_m * (1. - s.shoulder);
                    let dz = run * s.depth_m * (1. - s.tip_fraction) * if crown { -1. } else { 1. };
                    self.along
                        .cross(Vec3::Y * dy + self.across * dz)
                        .normalize()
                });
            }
        }
        Band {
            along: self.along,
            distances: [start, self.distance],
            normals,
            timber_origin: self.timber_origin,
        }
    }
}

impl Band {
    pub(super) fn front(&self, g: &mut Geometry, points: [Vec3; 4]) {
        let [bottom, top] = self.normals;
        let uv = std::array::from_fn(|i| {
            let along = points[i].dot(self.along);
            let across = self.distances[usize::from(i >= 2)];
            self.timber_origin.map_or([along, across], |origin| {
                (origin + Vec2::new(across, -along)).to_array()
            })
        });
        g.vertex_quad(points, [bottom, bottom, top, top], uv);
    }
}
