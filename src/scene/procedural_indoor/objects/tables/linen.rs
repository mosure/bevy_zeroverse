//! Restaurant linen stock is shared within a venue. Each laid cloth has its own
//! cut, placement and folds. Its top remains the dining support plane.
use super::*;

pub(crate) const INTRODUCED: u32 = 285;
const DRESSED: u32 = 1 << 20;
const NO_POWER_MODULE: u32 = 1 << 19;
const WEAVE_SHIFT: u32 = 8;
const DROP_SHIFT: u32 = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LinenWeave {
    Plain,
    Herringbone,
    AwningStripe,
    BistroCheck,
}
impl LinenWeave {
    pub(crate) const ALL: [Self; 4] = [
        Self::Plain,
        Self::Herringbone,
        Self::AwningStripe,
        Self::BistroCheck,
    ];
    pub(crate) fn slot(self) -> usize {
        880 + self as usize
    }
    fn label(self) -> &'static str {
        [
            "table#finish880#uv_local",
            "table#finish881#uv_local",
            "table#finish882#uv_local",
            "table#finish883#uv_local",
        ][self as usize]
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct DiningLinen {
    pub weave: LinenWeave,
    /// Deepest possible hem, measured from the physical tabletop.
    pub max_drop_m: f32,
}
pub fn dining_linen(o: &IndoorObject) -> Option<DiningLinen> {
    if o.kind != ObjectKind::Table || o.variant & DRESSED == 0 {
        return None;
    }
    Some(DiningLinen {
        weave: LinenWeave::ALL[((o.variant >> WEAVE_SHIFT) & 3) as usize],
        max_drop_m: 0.065 + 0.004 * ((o.variant >> DROP_SHIFT) & 31) as f32,
    })
}
pub(super) fn permits_power_module(o: &IndoorObject) -> bool {
    o.kind != ObjectKind::Table || o.variant & (DRESSED | NO_POWER_MODULE) == 0
}
pub(super) fn frame_variant(o: &IndoorObject) -> u32 {
    if !permits_power_module(o) {
        o.variant & 0xff
    } else {
        o.variant
    }
}

pub(crate) struct DiningService {
    prevalence: f64,
    weave: LinenWeave,
    drop: f32,
    powered: bool,
}
impl DiningService {
    pub(crate) fn sample(room: u64, zone: usize) -> Self {
        let mut r = stream(room, u64::from_le_bytes(*b"linenSvc") ^ zone as u64);
        let prevalence = match r.random_range(0..10) {
            0..=2 => 0.,
            3..=5 => 1.,
            _ => r.random_range(0.18..0.84),
        };
        Self {
            prevalence,
            weave: LinenWeave::ALL[r.random_range(0..4)],
            drop: r.random_range(2.0..29.0),
            powered: r.random_bool(0.11),
        }
    }
    /// Table-service cafés occasionally buy the same stock as restaurants.
    /// Venue selection is independent of the table and menu draws.
    pub(crate) fn for_cafe(room: u64, zone: usize) -> Self {
        let mut choice = stream(room, u64::from_le_bytes(*b"cafeLinn") ^ zone as u64);
        let mut service = Self::sample(room, zone);
        if !choice.random_bool(0.25) {
            service.prevalence = 0.;
        }
        service.powered = choice.random_bool(0.35);
        service
    }
    pub(crate) fn lay(&self, table: &mut IndoorObject) {
        if table.kind != ObjectKind::Table || table.variant >= ORIGINAL_SHARED_ROW_VARIANT {
            return;
        }
        if !self.powered {
            table.variant |= NO_POWER_MODULE;
        }
        if self.prevalence == 0. {
            return;
        }
        let mut r = stream(table.seed, u64::from_le_bytes(*b"laycloth"));
        if !r.random_bool(self.prevalence) {
            return;
        }
        let drop = (self.drop + r.random_range(-3.5..3.5))
            .round()
            .clamp(0., 31.) as u32;
        table.variant |= DRESSED | (self.weave as u32) << WEAVE_SHIFT | drop << DROP_SHIFT;
    }
}

struct Station {
    // Top, shoulder, free hem, turned lip and inner return to the rigid edge.
    rings: [Vec3; 5],
    uv: [Vec2; 5],
}

impl DiningLinen {
    fn cloth(&self, mesh: &mut Geometry, o: &IndoorObject, outline: &[Vec2], core_scale: f32) {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"clothCut"));
        let perimeter = perimeter(outline);
        let folds = (perimeter / r.random_range(0.18..0.36)).round().max(4.);
        let phase = r.random_range(0.0..TAU);
        let bias = r.random_range(0.0..TAU);
        let fullness = r.random_range(0.009..0.029);
        let shoulder = r.random_range(0.004..0.008);
        let hem = r.random_range(0.004..0.009);
        // Bound the displacement even at the far end of an oblong table.
        // The placement oracle keeps tabletop contents at least 12 mm inside.
        let furthest = outline
            .iter()
            .map(|p| p.length_squared())
            .fold(0., f32::max)
            .sqrt();
        let shoulder_scale = 1. - shoulder / furthest;
        let cut = Mat2::from_angle(
            r.random_range(-0.18..0.18) + if r.random_bool(0.5) { FRAC_PI_2 } else { 0. },
        );
        let offset = Vec2::new(r.random(), r.random());
        let mut stations = Vec::with_capacity(outline.len() + (folds * 6.) as usize);
        let mut along = 0.;
        for (i, &p) in outline.iter().enumerate() {
            let q = outline[(i + 1) % outline.len()];
            let length = p.distance(q);
            let count = (length * folds * 6. / perimeter).ceil().max(1.) as usize;
            for j in 0..count {
                let t = j as f32 / count as f32;
                let rim = p.lerp(q, t);
                let theta = TAU * (along + t * length) / perimeter;
                let fold = (folds * theta + phase + 0.68 * (3. * theta + bias).sin()).sin();
                let gather = 0.5 + 0.5 * fold;
                let corner = (rim / (o.size.xz() * 0.5)).length().clamp(1., 1.42) - 1.;
                let depth = self.max_drop_m
                    * (0.72 + 0.13 * (theta + bias).cos() + 0.24 * corner + 0.05 * fold).min(1.);
                let radial = rim.normalize();
                let hem_pos = rim - radial * (fullness * gather * gather);
                let xy = [
                    rim * shoulder_scale,
                    rim,
                    hem_pos,
                    hem_pos - radial * hem,
                    rim * core_scale,
                ];
                let dy = [0., shoulder, depth - hem * 0.25, depth, 0.005];
                let rings = std::array::from_fn(|k| Vec3::new(xy[k].x, o.size.y - dy[k], xy[k].y));
                // Unfold the skirt into the same metric textile chart as the top,
                // including the last/first edge. No stripe resets around corners.
                let uv = std::array::from_fn(|k| {
                    cut * (xy[0] + radial * (dy[k] + if k >= 3 { hem } else { 0. })) + offset
                });
                stations.push(Station { rings, uv });
            }
            along += length;
        }
        let n = stations.len();
        mesh.positions.reserve(5 * n + 1);
        mesh.normals.reserve(5 * n + 1);
        mesh.uvs.reserve(5 * n + 1);
        mesh.indices.reserve(27 * n);
        let base = mesh.positions.len() as u32;
        mesh.positions.push([0., o.size.y, 0.]);
        mesh.normals.push(Vec3::Y.to_array());
        mesh.uvs.push(offset.to_array());
        for (i, s) in stations.iter().enumerate() {
            let prev = &stations[(i + n - 1) % n];
            let next = &stations[(i + 1) % n];
            for k in 0..5 {
                let along = next.rings[k] - prev.rings[k];
                let side = || along.cross(s.rings[2] - s.rings[1]).normalize();
                let normal = match k {
                    0 => Vec3::Y,
                    1 => (side() + 0.32 * Vec3::Y).normalize(),
                    2 => side(),
                    3 => along.cross(s.rings[3] - s.rings[2]).normalize(),
                    _ => along.cross(s.rings[4] - s.rings[3]).normalize(),
                };
                mesh.positions.push(s.rings[k].to_array());
                mesh.normals.push(normal.to_array());
                mesh.uvs.push(s.uv[k].to_array());
            }
            let a = base + 1 + 5 * i as u32;
            let b = base + 1 + 5 * ((i + 1) % n) as u32;
            mesh.indices.extend([base, b, a]);
            for k in 0..4 {
                mesh.indices
                    .extend([a + k, b + k, b + k + 1, a + k, b + k + 1, a + k + 1]);
            }
        }
    }

    pub(super) fn build(
        &self,
        a: &mut Assembly,
        o: &IndoorObject,
        p: &TableProgram,
        outline: &[Vec2],
    ) {
        // Recess the rigid substrate behind the deepest fold. Only its underside
        // and edge can be seen. No second surface sits under the cloth top.
        let radius = outline
            .iter()
            .map(|p| p.length())
            .fold(f32::INFINITY, f32::min);
        let inset = 1. - 0.042 / radius;
        // The return is real, inward-facing geometry. It shares the core edge,
        // so even low cameras can use the ordinary opaque, back-culled pass.
        self.cloth(
            a.part(Surface::Fabric, self.weave.label()),
            o,
            outline,
            inset,
        );
        let edge: Vec<_> = outline.iter().map(|v| *v * inset).collect();
        let thickness = p.top_thickness + 0.007;
        let tf = Transform::from_xyz(0., o.size.y - 0.005 - thickness * 0.5, 0.);
        let g = a.part(Surface::WoodEdge, o.kind.class_name());
        g.slab_cap(&edge, 1., Vec2::ZERO, -thickness * 0.5, &tf, |v| v);
        for (p, q) in edge.iter().zip(edge.iter().cycle().skip(1)) {
            let vertex = |v: &Vec2, y| Vec3::new(v.x, tf.translation.y + y, v.y);
            g.metric_quad(
                [
                    vertex(p, thickness * 0.5),
                    vertex(q, thickness * 0.5),
                    vertex(q, -thickness * 0.5),
                    vertex(p, -thickness * 0.5),
                ],
                Vec3::new(q.y - p.y, 0., p.x - q.x),
            );
        }
    }
}

#[cfg(test)]
mod tests;
