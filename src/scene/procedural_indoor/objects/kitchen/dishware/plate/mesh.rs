//! Indexed plate sections. The pole is a single vertex; rectangular UV seams
//! are duplicated only on the rolled edge. Topology depends only on side count.
use super::*;
use std::{array, borrow::Cow, sync::OnceLock};

#[derive(Clone, Copy)]
pub(super) enum Patch {
    Underside,
    Edge,
    Rim,
    Well,
}
impl Patch {
    fn stations(self) -> std::ops::RangeInclusive<usize> {
        match self {
            Self::Underside => 0..=7,
            Self::Edge => 7..=11,
            Self::Rim => 11..=12,
            Self::Well => 12..=16,
        }
    }
    fn curved_chart(self) -> bool {
        matches!(self, Self::Edge)
    }
    fn columns(self, station: usize, n: usize) -> usize {
        if matches!(station, 0 | 16) {
            1
        } else {
            n + usize::from(self.curved_chart())
        }
    }
}

#[derive(Clone)]
struct PatchTopology {
    count: usize,
    indices: Vec<u32>,
}
impl PatchTopology {
    fn new(patch: Patch, n: usize) -> Self {
        let mut indices = Vec::new();
        let mut count = 0;
        let mut previous = None;
        for station in patch.stations() {
            let cols = patch.columns(station, n);
            let at = |start: usize, len: usize, col: usize| (start + col % len) as u32;
            if let Some((first, previous_cols)) = previous {
                for j in 0..n {
                    let a = at(first, previous_cols, j);
                    let b = at(count, cols, j);
                    let c = at(first, previous_cols, j + 1);
                    let d = at(count, cols, j + 1);
                    if previous_cols > 1 {
                        indices.extend([a, b, c]);
                    }
                    if cols > 1 {
                        indices.extend([c, b, d]);
                    }
                }
            }
            previous = Some((count, cols));
            count += cols;
        }
        Self { count, indices }
    }
}

type Topology = [PatchTopology; 4];
fn topology(n: usize) -> Cow<'static, Topology> {
    static STOCK: [OnceLock<Topology>; 33] = [const { OnceLock::new() }; 33];
    let build = || {
        [Patch::Underside, Patch::Edge, Patch::Rim, Patch::Well].map(|p| PatchTopology::new(p, n))
    };
    match STOCK.get(n) {
        Some(slot) => Cow::Borrowed(slot.get_or_init(build)),
        None => Cow::Owned(build()),
    }
}

struct Meridian {
    xz: Vec2,
    radial: Vec3,
    gradient: Vec3,
    along: f32,
}
struct Station {
    center: Vec3,
    radius: f32,
    normal: Vec2,
    along: f32,
}
pub(super) struct TurnedPlate {
    meridians: Vec<Meridian>,
    stations: [Station; 17],
    indices: Cow<'static, Topology>,
    normal_y: Vec3,
    axis_y: Vec3,
    unit_circle: bool,
}
impl TurnedPlate {
    pub(super) fn new(p: &DishProgram, form: &PlateForm, tf: Transform) -> Self {
        let profile = form.section(p);
        let curve = profile.map(|(x, y)| Vec2::new(x, y));
        let edges: [Vec2; 16] = array::from_fn(|i| (curve[i + 1] - curve[i]).normalize());
        let (x, y, z) = (
            tf.rotation * Vec3::X,
            tf.rotation * Vec3::Y,
            tf.rotation * Vec3::Z,
        );
        let (right, forward) = (x * tf.scale.x, z * tf.scale.z);
        let (nx, nz) = (x / tf.scale.x, z / tf.scale.z);
        let mut distance = 0.;
        let stations = array::from_fn(|i| {
            let tangent = match i {
                0 => edges[0],
                16 => edges[15],
                i => {
                    let (a, b) = (edges[i - 1], edges[i]);
                    // The flat well and recessed base have true planar normals.
                    if a.y == 0. {
                        a
                    } else if b.y == 0. {
                        b
                    } else {
                        (a + b).normalize()
                    }
                }
            };
            if i > 0 {
                distance += curve[i].distance(curve[i - 1]);
            }
            Station {
                radius: curve[i].x,
                center: tf.translation + y * (tf.scale.y * curve[i].y),
                normal: Vec2::new(tangent.y, -tangent.x),
                along: distance,
            }
        });
        let outline = p.outline();
        let meridians = outline
            .meridians()
            .map(|(v, grad, along)| Meridian {
                xz: v,
                radial: right * v.x + forward * v.y,
                gradient: nx * grad.x + nz * grad.y,
                along: along * p.radius,
            })
            .collect();
        Self {
            meridians,
            stations,
            indices: topology(p.segments as usize),
            normal_y: y / tf.scale.y,
            axis_y: y * tf.scale.y.signum(),
            unit_circle: form.outline_exponent == 2.
                && p.aspect == Vec2::ONE
                && tf.scale == Vec3::ONE,
        }
    }

    pub(super) fn write(&self, g: &mut Geometry, patch: Patch) {
        let topology = &self.indices[patch as usize];
        let start = g.positions.len() as u32;
        g.positions.reserve(topology.count);
        g.normals.reserve(topology.count);
        g.uvs.reserve(topology.count);
        let n = self.meridians.len() - 1;
        for i in patch.stations() {
            let row = &self.stations[i];
            let cols = patch.columns(i, n);
            let meridians = &self.meridians[..cols];
            g.positions.extend(
                meridians
                    .iter()
                    .map(|v| (row.center + v.radial * row.radius).to_array()),
            );
            if row.normal.x == 0. {
                let normal = (self.axis_y * row.normal.y.signum()).to_array();
                g.normals.resize(g.normals.len() + cols, normal);
            } else {
                let axial = self.normal_y * row.normal.y;
                if self.unit_circle {
                    g.normals.extend(
                        meridians
                            .iter()
                            .map(|v| (v.gradient * row.normal.x + axial).to_array()),
                    );
                } else {
                    g.normals.extend(
                        meridians
                            .iter()
                            .map(|v| (v.gradient * row.normal.x + axial).normalize().to_array()),
                    );
                }
            }
            if patch.curved_chart() {
                g.uvs.extend(meridians.iter().map(|v| [v.along, row.along]));
            } else {
                g.uvs
                    .extend(meridians.iter().map(|v| (v.xz * row.radius).to_array()));
            }
        }
        g.indices
            .extend(topology.indices.iter().map(|&idx| idx + start));
    }
}
