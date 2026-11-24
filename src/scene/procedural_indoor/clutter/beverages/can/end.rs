//! Rolled double seam, scored end, attached tongue and riveted stay-on pull tab.
use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CanEnd {
    pub panel_fraction: f32,
    pub recess_fraction: f32,
    pub gauge_m: f32,
    pub port_half_fraction: Vec2,
    pub port_offset_fraction: f32,
    pub hinge_fraction: f32,
    pub fold_radians: f32,
    pub tab_lift_radians: f32,
    pub fill_fraction: Option<f32>,
}
impl CanEnd {
    pub(super) fn sample(seed: u64, neck_radius: f32) -> Self {
        let mut r = stream(seed, u64::from_le_bytes(*b"CanStamp"));
        let panel_fraction = r.random_range(0.78..0.88);
        let recess_fraction = r.random_range(0.085..0.145);
        let gauge_m = r.random_range(0.00020..0.00034);
        // Select a metric score area, then fit both axes of the oval together.
        // Preserve a metal web between the rivet, the score and the rolled rim.
        let area_m2 = r.random_range(0.00018_f32..0.000485);
        let aspect = r.random_range(1.08_f32..1.55);
        let rear_clearance = r.random_range(0.070..0.135);
        let ry = ((area_m2 / (PI * aspect)).sqrt() / (panel_fraction * neck_radius).max(1e-6))
            .min((0.94 - rear_clearance) * 0.5)
            .min(0.67 / aspect);
        Self {
            panel_fraction,
            recess_fraction,
            gauge_m,
            port_half_fraction: Vec2::new(ry * aspect, ry),
            port_offset_fraction: rear_clearance + ry,
            hinge_fraction: r.random_range(0.58..0.83),
            fold_radians: r.random_range(53.0_f32..86.).to_radians(),
            tab_lift_radians: r.random_range(0.0_f32..11.).to_radians(),
            fill_fraction: r.random_bool(0.79).then(|| r.random_range(0.16..0.82)),
        }
    }

    pub(super) fn draw(&self, a: &mut Assembly, p: &CanProgram, neck: f32, h: f32, land_y: f32) {
        let g = a.part(Surface::Metal, package_print::ALUMINUM);
        let t = self.gauge_m;
        let top = h - neck * self.recess_fraction;
        let panel = neck * self.panel_fraction;
        let round = outline(32);
        let tf = Transform::from_rotation(Quat::from_rotation_y(p.lid_yaw));
        // The seam, recessed channel and deck meet along their actual sections.
        round.sweep(
            g,
            &[
                (neck, land_y),
                (neck * 1.018, h - 0.020 * neck),
                (neck * 1.004, h),
                (neck * 0.965, h),
                (neck * 0.948, h - 0.018 * neck),
                (neck * 0.925, top - 0.034 * neck),
                (neck * 0.89, top - 0.041 * neck),
                (panel, top),
            ],
            ProfileJoints::Smooth,
            tf,
        );
        if p.opened {
            round.sweep(
                g,
                &[
                    (panel, top - t),
                    (neck * 0.89, top - t - 0.041 * neck),
                    (neck - t, land_y),
                ],
                ProfileJoints::Smooth,
                tf,
            );
        }

        let offset = Vec2::Y * (self.port_offset_fraction * panel);
        let half = self.port_half_fraction * panel;
        let deck = end_plate::PiercedPlate::new(Vec2::splat(panel), offset, half);
        deck.faces(g, top, top - t, tf);
        let cut: Vec<_> = deck.points.iter().map(|&(_, b)| b).collect();
        let skin_y = top - t * 0.67;
        if p.opened {
            self.fold(g, &cut, offset, half, skin_y, tf);
        } else {
            // The scored channel is continuous metal. The retained plug seals the
            // full aperture and its shallow bevel catches a grazing highlight.
            let inner: Vec<_> = cut.iter().map(|v| offset + (*v - offset) * 0.957).collect();
            for i in 0..cut.len() {
                let j = (i + 1) % cut.len();
                let pts = [
                    (cut[i], skin_y),
                    (cut[j], skin_y),
                    (inner[j], top - 0.07 * t),
                    (inner[i], top - 0.07 * t),
                ]
                .map(|(v, y)| tf.transform_point(Vec3::new(v.x, y, v.y)));
                g.metric_quad(pts, Vec3::Y);
            }
            face(g, &inner, top - 0.07 * t, tf, true);
        }
        let tab_half = Vec2::new(p.tab_width * panel, (0.16 + 0.83 * p.tab_length) * panel);
        let center_z = -0.13 * panel;
        let pivot = 0.035 * panel;
        let gauge = t * 1.9;
        let lift = p.end.tab_lift_radians;
        let seat = top + 0.00009 + (center_z + tab_half.y - pivot) * lift.sin();
        let tab_tf = tf
            * Transform::from_xyz(0., seat, pivot).with_rotation(Quat::from_rotation_x(lift))
            * Transform::from_xyz(0., 0., center_z - pivot);
        let tab = end_plate::PiercedPlate::new(
            tab_half,
            Vec2::NEG_Y * (0.27 * tab_half.y),
            tab_half * Vec2::new(0.60, 0.40),
        );
        tab.embossed_tab(g, tab_half, gauge, tab_tf);
        let head = seat + gauge + 0.23 * t;
        let axis = tf * Transform::from_xyz(0., 0., pivot);
        // The rivet head is only a few millimetres wide.
        static RIVET: OnceLock<RevolvedOutline> = OnceLock::new();
        RIVET
            .get_or_init(|| RevolvedOutline::round(16, 0., 0))
            .sweep(
                g,
                &[
                    (panel * 0.049, top),
                    (panel * 0.049, head - t * 0.46),
                    (panel * 0.093, head - t * 0.46),
                    (panel * 0.089, head),
                    (panel * 0.066, head + t * 0.24),
                    (0., head + t * 0.24),
                ],
                ProfileJoints::Creased,
                axis,
            );
    }

    fn fold(
        &self,
        g: &mut Geometry,
        cut: &[Vec2],
        offset: Vec2,
        half: Vec2,
        y: f32,
        tf: Transform,
    ) {
        // A chord across the rear of the score is an uncut hinge. Both halves
        // use its original endpoints so the retained metal is always attached.
        let target = offset.y - half.y * self.hinge_fraction;
        let endpoint = |positive: bool| {
            cut.iter()
                .enumerate()
                .filter(|(_, v)| (v.x > 0.) == positive)
                .min_by(|(_, a), (_, b)| (a.y - target).abs().total_cmp(&(b.y - target).abs()))
                .unwrap()
                .0
        };
        let (right, left) = (endpoint(true), endpoint(false));
        let arc = |i: usize, j: usize| {
            (0..=(j + cut.len() - i) % cut.len())
                .map(|n| cut[(i + n) % cut.len()])
                .collect::<Vec<_>>()
        };
        let fixed = arc(left, right);
        let moving = arc(right, left);
        let origin = Vec3::new(cut[left].x, y, cut[left].y);
        let axis =
            Vec3::new(cut[right].x - cut[left].x, 0., cut[right].y - cut[left].y).normalize();
        let fold = tf
            * Transform::from_translation(origin)
                .with_rotation(Quat::from_axis_angle(axis, self.fold_radians))
            * Transform::from_translation(-origin);
        face(g, &fixed, y, tf, true);
        face(g, &fixed, y - self.gauge_m, tf, false);
        face(g, &moving, y, fold, true);
        face(g, &moving, y - self.gauge_m, fold, false);
        for pair in moving.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let pts = [(a, y - self.gauge_m), (a, y), (b, y), (b, y - self.gauge_m)]
                .map(|(q, h)| fold.transform_point(Vec3::new(q.x, h, q.y)));
            g.metric_quad(pts, fold.rotation * Vec3::new(b.y - a.y, 0., a.x - b.x));
        }
    }
}

fn face(g: &mut Geometry, polygon: &[Vec2], y: f32, tf: Transform, up: bool) {
    let center = polygon.iter().sum::<Vec2>() / polygon.len() as f32;
    let start = g.positions.len() as u32;
    let normal = (tf.rotation * (if up { Vec3::Y } else { Vec3::NEG_Y })).to_array();
    for &v in std::iter::once(&center).chain(polygon) {
        g.positions
            .push(tf.transform_point(Vec3::new(v.x, y, v.y)).to_array());
        g.normals.push(normal);
        g.uvs.push(v.to_array());
    }
    for i in 0..polygon.len() {
        let a = start + 1 + i as u32;
        let b = start + 1 + ((i + 1) % polygon.len()) as u32;
        g.indices
            .extend(if up { [start, b, a] } else { [start, a, b] });
    }
}
