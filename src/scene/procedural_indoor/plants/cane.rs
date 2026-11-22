//! Nursery cane crowns. Mature strap leaves arch from wrapping nodes, while
//! shorter leaves near the growing tip retain their upright folded form.
use super::*;

#[derive(Clone, Copy)]
struct Crown {
    root: Vec3,
    tip: Vec3,
    reach: f32,
    half_extent: Vec2,
    habit: CaneHabit,
}

/// Half-width, Bézier control height and twist at mature blade scale.
#[derive(Clone, Copy, serde::Serialize)]
pub(crate) struct CaneHabit {
    pub width_fraction: f32,
    pub arch_fraction: f32,
    pub twist_radians: f32,
}
impl CaneHabit {
    pub(crate) fn sample(seed: u64) -> Self {
        let mut r = stream(seed, 0x4341_4e45_4841_4249);
        Self {
            width_fraction: r.random_range(0.055..0.18),
            arch_fraction: r.random_range(0.50..0.92),
            twist_radians: r.random_range(0.18..0.58),
        }
    }
}

pub(super) fn build(
    a: &mut Assembly,
    o: &IndoorObject,
    g: &Growth,
    soil: Vec3,
    spread: f32,
    soil_radius: f32,
) {
    let mut r = stream(o.seed, 0x4341_4e45_4352_4f57);
    let height = o.size.y - soil.y;
    let habit = CaneHabit::sample(o.seed);
    for cane in 0..g.leader_count {
        let direction = radial(g.angle(cane));
        let foot = soil + direction * soil_radius * r.random_range(0.22..0.55);
        let stature = 0.40 + 0.40 * g.node(cane, g.leader_count - 1);
        let tip =
            foot + Vec3::Y * (height * stature) + direction * spread * r.random_range(0.02..0.19);
        let axis = (tip - foot).normalize();
        let stem_length = tip.distance(foot);
        let radius = (spread * r.random_range(0.043..0.075)).min(stem_length * 0.042);
        let live_length = (spread * r.random_range(0.19..0.37)).min(stem_length * 0.33);
        let root = tip - axis * live_length;
        // Leaf bases stay on this segment of the actual cane. Low annular scars
        // are formed in its bark profile; there are no overlaid ring faces.
        let scars = ((stem_length - live_length) / (radius * 6.0)).clamp(2., 12.) as usize;
        let mut profile = Vec::with_capacity(3 * scars + 5);
        profile.extend([(0., 0.), (radius, 0.)]);
        for i in 0..scars {
            let t = (i as f32 + r.random_range(0.26..0.68)) / scars as f32;
            let y = (stem_length - live_length) * t;
            let bark = radius * (1. - 0.26 * y / stem_length);
            let lip = (radius * 0.15).min((stem_length - live_length) / scars as f32 * 0.12);
            profile.extend([(bark, y - lip), (bark * 1.048, y), (bark, y + lip)]);
        }
        profile.extend([(radius * 0.74, stem_length), (0., stem_length)]);
        a.part(Surface::Bark, "other_prop").lathe(
            &profile,
            10,
            Transform::from_translation(foot).with_rotation(Quat::from_rotation_arc(Vec3::Y, axis)),
        );
        let crown = Crown {
            root,
            tip,
            reach: spread * 1.12,
            half_extent: o.size.xz() * 0.5,
            habit,
        };
        let phase = g.shoot_phase + r.random_range(0.0..TAU);
        let count = g.count(r.random_range(31..=41));
        for k in 0..count {
            let youth = k as f32 / (count - 1) as f32;
            let az = phase + k as f32 * g.phyllotaxis + r.random_range(-0.15..0.15);
            crown.leaf(a, radial(az), youth, o.size.y, soil.y, &mut r);
        }
    }
}

impl Crown {
    fn leaf(
        &self,
        a: &mut Assembly,
        dir: Vec3,
        youth: f32,
        height: f32,
        soil: f32,
        r: &mut impl Rng,
    ) {
        let base = self.root.lerp(self.tip, youth);
        let later = youth * youth;
        let nominal = self.reach * (1. - 0.65 * later) * r.random_range(0.81..1.0);
        let side = dir.cross(Vec3::Y);
        // An off-centre cane has more growing room across the pot than towards
        // its nearest edge. Reserve the actual azimuth, sideways sweep and the
        // full folded blade width, including its change of frame along the arc.
        let radial = dir.xz();
        let offset = base.xz() * radial.signum();
        let margin = side.xz().abs() * 0.13 + Vec2::splat(1.43 * self.habit.width_fraction);
        let outward = (self.half_extent - offset) / (radial.abs() + margin);
        let inward = (self.half_extent + offset) / margin;
        let length = nominal.min(outward.min(inward).min_element() * 0.99);
        let width =
            length * self.habit.width_fraction * (1. - 0.32 * youth) * r.random_range(0.84..1.09);
        let headroom = height - base.y;
        let lift = (length * (self.habit.arch_fraction + 0.30 * later)).min(headroom * 0.85);
        let end_y = (length * (-0.53 + 1.43 * later + r.random_range(-0.10..0.10)))
            .clamp(-(base.y - soil) * 0.71, headroom * 0.79);
        let sweep = length * r.random_range(-0.13..0.13);
        let roll = self.habit.twist_radians * r.random_range(-1.0..1.0);
        let fold = r.random_range(0.21..0.43) + 0.16 * later;
        let mut pos = [Vec3::ZERO; 23];
        let mut uv = [[0.; 2]; 23];
        pos[0] = base;
        uv[0] = [0.5, 0.];
        for i in 1..8 {
            let t = i as f32 / 8.;
            let arc = 2. * t * (1. - t) * lift + t * t * end_y;
            let mid = base
                + dir * (length * t * (0.45 + 0.55 * t))
                + Vec3::Y * arc
                + side * sweep * t * t;
            let tangent = (dir * (length * (0.45 + 1.10 * t))
                + Vec3::Y * (2. * (1. - 2. * t) * lift + 2. * t * end_y)
                + side * (2. * t * sweep))
                .normalize();
            let lateral = tangent.cross(Vec3::Y).normalize();
            let up = lateral.cross(tangent);
            let (sr, cr) = (roll * t).sin_cos();
            let across = lateral * cr + up * sr;
            let camber = up * cr - lateral * sr;
            let envelope = (PI * t).sin().powf(0.62) * (1.12 - 0.42 * t);
            let w = width * envelope;
            for (j, sign) in [-1.0_f32, 0., 1.].into_iter().enumerate() {
                let index = 1 + (i - 1) * 3 + j;
                pos[index] =
                    mid + across * (sign * w) + camber * (sign.abs() * w * fold * (1. - 0.65 * t));
                uv[index] = [0.5 + sign * envelope.min(1.) * 0.5, t];
            }
        }
        pos[22] = base + dir * length + Vec3::Y * end_y + side * sweep;
        uv[22] = [0.5, 1.];
        let surface = if youth > 0.76 && r.random_bool(0.46) {
            Surface::LeafLight
        } else {
            Surface::Leaf
        };
        leaf::append(a, surface, pos, uv);
    }
}
