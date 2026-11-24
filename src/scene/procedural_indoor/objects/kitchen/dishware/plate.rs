//! Pressed or thrown plates. Each upper foot rests in the preceding flat well.
//! All clearances, rack projections and rendered faces use this same section.
use super::*;
mod mesh;
use crate::scene::procedural_indoor::materials::variants;
use mesh::{Patch, TurnedPlate};

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PlateForm {
    pub outline_exponent: f32,
    pub foot_height_m: f32,
    pub foot_radius_m: f32,
    pub foot_half_width_m: f32,
    pub edge_radius_m: f32,
    pub rim_band_m: f32,
    pub shoulder_rise: f32,
    pub well_curve: f32,
    /// Local X/Z displacement of each plate above the base, in metres.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stack_offsets_m: Vec<Vec2>,
}

impl PlateForm {
    pub(super) fn sample(p: &mut DishProgram, size: Vec3, seed: u64) -> Self {
        let mut r = stream(seed, 0x504c_4154_4546_4f52);
        let n = p.count as f32;
        let lift = r.random_range(0.42_f32..1.85);
        // Keep enough height for the dished face even in short cupboard stacks.
        p.wall = p.wall.min(size.y / ((n + 1.8) * (1. + lift)));
        let foot_height_m = (p.wall * lift).min(p.height * 0.26).min(0.006);
        let pitch = p.wall + foot_height_m;
        p.height = p.height.min(size.y - (n - 1.) * pitch);
        // Quarter symmetry includes the extrema for ovals and softened squares.
        p.segments = p.segments.next_multiple_of(4);
        let well = p.radius * p.foot;
        let rim_run = p.radius * (1. - p.shoulder);
        let edge_radius_m = (p.wall * r.random_range(0.35..0.94)).min(rim_run * 0.22);
        let mut form = Self {
            outline_exponent: if r.random_bool(0.34) {
                r.random_range(2.45..4.8)
            } else {
                2.
            },
            foot_height_m,
            foot_radius_m: well * r.random_range(0.58..0.78),
            foot_half_width_m: (well * r.random_range(0.045..0.105)).min(p.wall * 1.5),
            edge_radius_m,
            rim_band_m: edge_radius_m + (rim_run - edge_radius_m) * r.random_range(0.15..0.65),
            shoulder_rise: r.random_range(0.32..0.89),
            well_curve: r.random_range(0.22..0.66),
            stack_offsets_m: Vec::new(),
        };
        if p.count > 1 && r.random_bool(0.78) {
            let section = form.section(p);
            let slope = section[4..=7]
                .windows(2)
                .map(|w| (w[1].1 - w[0].1) / (w[1].0 - w[0].0))
                .fold(0., f32::max);
            // At most twice the sampled offset separates neighbours. Away from
            // the foot, shifting the well by that amount must consume less
            // than its underside clearance, including along the oval's minor axis.
            let inscribed = p.aspect.min_element() * (PI / p.segments as f32).cos();
            let margin = (well - form.foot_radius_m - form.foot_half_width_m)
                .min(foot_height_m / slope.max(0.001));
            let limit = (0.39 * inscribed * margin)
                .min((size.x * 0.5 - p.radius).max(0.))
                .min((size.z * 0.5 - p.radius * p.aspect.y).max(0.))
                .min(0.0055);
            let spread = limit * r.random_range(0.32..1.);
            form.stack_offsets_m = (1..p.count)
                .map(|_| {
                    let (s, c) = r.random_range(0.0..TAU).sin_cos();
                    Vec2::new(c, s) * (spread * r.random::<f32>().sqrt())
                })
                .collect();
        }
        form
    }

    /// Lower meridian from the recessed underside to the rim, then the rolled
    /// edge and the upper face back to the centre. The only low point is the foot.
    pub(super) fn section(&self, p: &DishProgram) -> [(f32, f32); 17] {
        let (r, h, w) = (p.radius, p.height, p.wall);
        let f = self.foot_height_m;
        let floor = w + f;
        let well = r * p.foot;
        let shoulder = r * p.shoulder;
        let middle = well + (shoulder - well) * 0.5;
        let dz = (h - floor) * self.shoulder_rise;
        let mid_y = floor + dz * self.well_curve;
        let high = floor + dz;
        let bead = r - self.edge_radius_m;
        let band = r - self.rim_band_m;
        let band_y = high + (h - high) * (band - shoulder) / (bead - shoulder);
        let d = std::f32::consts::FRAC_1_SQRT_2;
        [
            (0., f),
            (self.foot_radius_m - self.foot_half_width_m, f),
            (self.foot_radius_m, 0.),
            (self.foot_radius_m + self.foot_half_width_m, f),
            (well, f),
            (middle, mid_y - w),
            (shoulder, high - w),
            (bead, h - w),
            (bead + self.edge_radius_m * d, h - w * (0.5 + 0.5 * d)),
            (r, h - w * 0.5),
            (bead + self.edge_radius_m * d, h - w * (0.5 - 0.5 * d)),
            (bead, h),
            (band, band_y),
            (shoulder, high),
            (middle, mid_y),
            (well, floor),
            (0., floor),
        ]
    }

    pub(super) fn half_extents(&self, p: &DishProgram, yaw: f32) -> Vec2 {
        let (s, c) = yaw.sin_cos();
        // Support function of the rotated superellipse. Its tessellated chords
        // lie inside. Offsets are in dish coordinates, applied after shaping.
        let axis = |x: f32, z: f32| {
            let a = x.abs() * p.radius * p.aspect.x;
            let b = z.abs() * p.radius * p.aspect.y;
            if self.outline_exponent == 2. {
                a.hypot(b)
            } else {
                let dual = self.outline_exponent / (self.outline_exponent - 1.);
                (a.powf(dual) + b.powf(dual)).powf(1. / dual)
            }
        };
        let extra = self.stack_offsets_m.iter().fold(Vec2::ZERO, |bound, v| {
            bound.max(Vec2::new(c * v.x + s * v.y, -s * v.x + c * v.y).abs())
        });
        Vec2::new(axis(c, s), axis(-s, c)) + extra
    }

    /// Glaze selection belongs to the set finish, so a coordinated place
    /// setting or cupboard stack can carry one fired rim colour throughout.
    pub(crate) fn rim_slot(surface: Surface, finish_seed: u64) -> Option<usize> {
        if surface != Surface::Ceramic {
            return None;
        }
        let mut r = stream(finish_seed, 0x504c_4154_4547_4c5a);
        r.random_bool(0.37).then(|| {
            let slot = variants::object_slot(surface, finish_seed, None);
            // Keep the same ceramic structure, change its glaze tint.
            let structures = variants::structure_count(surface);
            (slot + structures * r.random_range(1..variants::COUNT / structures)) % variants::COUNT
        })
    }

    pub(super) fn build(&self, a: &mut Assembly, tf: Transform, p: &DishProgram, finish_seed: u64) {
        let plate = TurnedPlate::new(p, self, tf);
        let base = variants::object_slot(p.surface, finish_seed, None);
        let accent = Self::rim_slot(p.surface, finish_seed);
        let body = a.part(p.surface, &format!("dishware#finish{base}"));
        let start = (body.positions.len(), body.indices.len());
        plate.write(body, Patch::Underside);
        plate.write(body, Patch::Well);
        if accent.is_none() {
            plate.write(body, Patch::Edge);
            plate.write(body, Patch::Rim);
        }
        self.repeat_stack(body, start, tf, p);
        if let Some(slot) = accent {
            let rim = a.part(p.surface, &format!("dishware#finish{slot}"));
            let start = (rim.positions.len(), rim.indices.len());
            plate.write(rim, Patch::Edge);
            plate.write(rim, Patch::Rim);
            self.repeat_stack(rim, start, tf, p);
        }
    }

    fn repeat_stack(
        &self,
        g: &mut Geometry,
        (vertex, index): (usize, usize),
        tf: Transform,
        p: &DishProgram,
    ) {
        if p.count == 1 {
            return;
        }
        let nv = g.positions.len() - vertex;
        let ni = g.indices.len() - index;
        let more = (p.count - 1) as usize;
        g.positions.reserve(more * nv);
        g.normals.reserve(more * nv);
        g.uvs.reserve(more * nv);
        g.indices.reserve(more * ni);
        for i in 1..p.count as usize {
            let offset = self.stack_offsets_m.get(i - 1).copied().unwrap_or_default();
            let delta = tf.rotation
                * (tf.scale * Vec3::new(offset.x, p.interior_floor() * i as f32, offset.y));
            for k in vertex..vertex + nv {
                g.positions
                    .push((Vec3::from_array(g.positions[k]) + delta).to_array());
            }
            g.normals.extend_from_within(vertex..vertex + nv);
            g.uvs.extend_from_within(vertex..vertex + nv);
            let shift = (i * nv) as u32;
            for k in index..index + ni {
                g.indices.push(g.indices[k] + shift);
            }
        }
    }
}
