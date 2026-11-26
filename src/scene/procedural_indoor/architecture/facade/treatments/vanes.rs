use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;
use vertical::{VaneLayout, RAIL_BOTTOM, RAIL_TOP, TRACK_Z};

const RAIL_HALF_DEPTH: f32 = 0.025;
const RAIL_WALL: f32 = 0.0025;
const SLOT_HALF_WIDTH: f32 = 0.005;

impl Builder<'_> {
    /// Extruded channel with a downward-facing slot for the travelling
    /// carriers. The brackets enter the back wall below the top face.
    pub(super) fn vane_track(&mut self, o: &WindowOpening) {
        let tf = self.tf;
        let left = self.lo.x;
        let right = self.hi.x;
        let bottom = o.max.y - RAIL_BOTTOM;
        let (d, h, t, s) = (
            RAIL_HALF_DEPTH,
            RAIL_BOTTOM - RAIL_TOP,
            RAIL_WALL,
            SLOT_HALF_WIDTH,
        );
        let channel = [
            (-d, h),
            (-d, 0.),
            (-s, 0.),
            (-s, t),
            (-d + t, t),
            (-d + t, h - t),
            (d - t, h - t),
            (d - t, t),
            (s, t),
            (s, 0.),
            (d, 0.),
            (d, h),
        ];
        let point = |x, (z, y)| tf.transform_point(Vec3::new(x, bottom + y, TRACK_Z + z));
        let surface = if self.p.identity.is_multiple_of(3) {
            Surface::Plastic
        } else {
            Surface::Metal
        };
        let g = self.a.part(surface, &self.label);
        for (&a, &b) in channel.iter().zip(channel.iter().cycle().skip(1)) {
            let outward = tf.rotation * Vec3::new(0., a.0 - b.0, b.1 - a.1);
            g.metric_quad(
                [
                    point(left, a),
                    point(right, a),
                    point(right, b),
                    point(left, b),
                ],
                outward,
            );
        }
        for (x, normal) in [(left, Vec3::NEG_X), (right, Vec3::X)] {
            for (z0, z1, y0, y1) in [
                (-d, d, h - t, h),
                (-d, -d + t, t, h - t),
                (d - t, d, t, h - t),
                (-d, -s, 0., t),
                (s, d, 0., t),
            ] {
                g.metric_quad(
                    [
                        point(x, (z0, y0)),
                        point(x, (z1, y0)),
                        point(x, (z1, y1)),
                        point(x, (z0, y1)),
                    ],
                    tf.rotation * normal,
                );
            }
        }
        let spans = (self.width() / 1.45).ceil().max(1.) as usize;
        let back = TRACK_Z - d + t * 0.5;
        for i in 0..=spans {
            let x = (left + 0.011).lerp(right - 0.011, i as f32 / spans as f32);
            self.block(
                Surface::Metal,
                Vec3::new(x, bottom + 0.5 * h, 0.002),
                Vec3::new(0.020, 0.032, 0.004),
                0.,
            );
            self.block(
                Surface::Metal,
                Vec3::new(x, bottom + 0.64 * h, (back + 0.002) * 0.5),
                Vec3::new(0.014, 0.0034, back - 0.002),
                0.,
            );
        }
    }

    pub(super) fn vanes(&mut self, stock: VerticalStock, o: &WindowOpening) {
        let l = stock.layout(self.p, o, self.tilt, self.coverage);
        let height = l.top - l.bottom;
        let rotation = Quat::from_rotation_y(l.angle);
        let mut vane = Geometry::default();
        let hem = (stock.hem_m / height).min(0.35);
        // The weight pocket flattens the crown at the foot. Keep the same cut
        // length, yarn direction and mesh topology throughout the traverse.
        cloth::sheet_rows::<false>(
            &mut vane,
            Transform::from_rotation(self.tf.rotation * rotation),
            [3, 2],
            Vec2::new(l.width, height),
            stock.thickness_m,
            |v| if v == 0.5 { hem } else { v },
            |u, v| {
                let x = 2. * u - 1.;
                let crown = stock.crown_m * (1. - x * x) * (0.22 + 0.78 * (v / hem).min(1.));
                Vec3::new(0.5 * x * l.width, v * height, crown)
            },
        );
        let mut clip = Geometry::default();
        let tip = Vec3::Y * (height - 0.001) + rotation * (Vec3::Z * (stock.crown_m * 8. / 9.));
        let bearing = Vec3::Y * (l.rail_y - l.bottom + RAIL_WALL + 0.0015);
        let stem = bearing - tip;
        clip.cuboid(
            Vec3::new(0.006, stem.length(), 0.0018),
            0.,
            Transform::from_translation(self.tf.rotation * (tip + stem * 0.5)).with_rotation(
                self.tf.rotation * Quat::from_rotation_arc(Vec3::Y, stem.normalize()) * rotation,
            ),
        );
        clip.cuboid(
            Vec3::new(0.0075, 0.003, 0.016),
            0.,
            Transform::from_translation(
                self.tf.rotation * (Vec3::Y * (l.rail_y - l.bottom + RAIL_WALL + 0.002)),
            )
            .with_rotation(self.tf.rotation),
        );

        // Slit stock is cut end to end. Wider bolts supply several drops side
        // by side before advancing along the roll. Origin is per installation.
        let mut r = stream(installation_seed(self.p, o), 0x56414e45435554);
        let origin = Vec2::new(r.random_range(0.0..3.), r.random_range(0.0..4.));
        let per_row = if r.random_bool(0.44) {
            1
        } else {
            (r.random_range(1.15..1.85) / (l.width + 0.004))
                .floor()
                .max(1.) as usize
        };
        let allowance = r.random_range(0.018..0.043);
        repeat(
            self.a.part(self.p.fabric, &self.cloth_label),
            &vane,
            l,
            self.tf,
            |i| {
                origin
                    + Vec2::new(
                        (i % per_row) as f32 * (l.width + 0.004),
                        (i / per_row) as f32 * (height + 2. * stock.hem_m + allowance),
                    )
            },
        );
        repeat(
            self.a.part(Surface::Metal, &self.label),
            &clip,
            l,
            self.tf,
            |_| Vec2::ZERO,
        );
    }
}

/// One closed fabric shell and one carrier are tessellated per aperture.
/// The tracks stay batched into their existing material and semantic groups.
fn repeat(
    dest: &mut Geometry,
    template: &Geometry,
    l: VaneLayout,
    tf: Transform,
    uv_origin: impl Fn(usize) -> Vec2,
) {
    dest.positions.reserve(template.positions.len() * l.count);
    dest.normals.reserve(template.normals.len() * l.count);
    dest.uvs.reserve(template.uvs.len() * l.count);
    dest.indices.reserve(template.indices.len() * l.count);
    for i in 0..l.count {
        let base = dest.positions.len() as u32;
        let origin = tf.transform_point(Vec3::new(l.x(i), l.bottom, TRACK_Z));
        let cut = uv_origin(i);
        dest.positions.extend(
            template
                .positions
                .iter()
                .map(|&v| (origin + Vec3::from(v)).to_array()),
        );
        dest.normals.extend_from_slice(&template.normals);
        dest.uvs.extend(
            template
                .uvs
                .iter()
                .map(|&v| (cut + Vec2::from(v)).to_array()),
        );
        dest.indices
            .extend(template.indices.iter().map(|&k| base + k));
    }
}
