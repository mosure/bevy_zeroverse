//! Closed book shells share the room's wraparound jacket atlas. Each exterior
//! face is emitted only once, including the visible page edges of leaning books.
use super::*;
use crate::scene::procedural_indoor::materials::{paper::jacket, variants};

#[derive(Clone, Copy)]
pub(super) struct Cell {
    pub left: f32,
    pub width: f32,
    pub floor: f32,
    pub height: f32,
    pub depth: f32,
    pub outer: f32,
    pub side: i8,
}

impl Cell {
    pub(super) fn index(&self, bay: usize, level: usize) -> u64 {
        (bay * 16 + level * 2 + usize::from(self.side > 0)) as u64
    }

    pub fn books(
        &self,
        seed: u64,
        index: u64,
        fill: f32,
        version: u32,
        mut emit: impl FnMut(Book),
    ) {
        if self.width < 0.08 || self.depth < 0.10 || self.height < 0.12 {
            return;
        }
        let reverse = version >= display::INTRODUCED
            && stream(seed, u64::from_le_bytes(*b"LibOrder") + index).random_bool(0.5);
        let mut r = stream(seed, u64::from_le_bytes(*b"LibBooks") + index);
        let width = self.width * (fill * r.random_range(0.53..1.23)).min(0.96);
        let mut x = self.left + r.random_range(0.004..(self.width - width - 0.004).max(0.005));
        let end = x + width;
        let mut edition = r.random();
        let mut height = r.random_range(0.18..0.31);
        let mut depth = r.random_range(0.15..0.24);
        let mut upright_stop: Option<(f32, f32)> = None;
        // Related volumes share print and trim dimensions, interrupted by
        // independent titles and genuine open spaces on each face and tier.
        for i in 0..18 {
            if r.random_bool(0.48) {
                edition = r.random();
                height = r.random_range(0.15..0.34);
                depth = r.random_range(0.14..0.26);
            }
            let h = (height * r.random_range(0.96..1.04_f32)).min(self.height * 0.91);
            let w: f32 = r.random_range(0.017..0.057);
            let d = (depth * r.random_range(0.96..1.04_f32)).min(self.depth * 0.89);
            // A tilted volume rests on a taller upright neighbor. Its centre of
            // mass must load the stop; the bottom bearing edge stays on the deck.
            let mut tilt: f32 = 0.;
            if let Some((edge, height)) = upright_stop {
                if height >= h && r.random_bool(0.34) {
                    tilt = (w / h).atan() + r.random_range(0.025..0.11);
                    x = edge + 0.00025;
                }
            }
            let (sin, cos) = tilt.sin_cos();
            let span = w * cos + h * sin.abs();
            if x + span > end {
                break;
            }
            let side = self.side as f32;
            let setback = r.random_range(0.15..0.85) * (self.depth - d).min(0.061);
            // Mirror only the packing order. Spines still face this aisle and
            // jacket lettering keeps its handedness on both sides of the stack.
            let center = x + span * 0.5;
            let position = Vec3::new(
                if reverse {
                    self.left + (self.width - (center - self.left))
                } else {
                    center
                },
                self.floor + (h * cos + w * sin.abs()) * 0.5,
                side * (self.outer - setback - d * 0.5),
            );
            let rotation = Quat::from_rotation_y(if self.side < 0 { PI } else { 0. })
                * Quat::from_rotation_z(if reverse { -tilt * side } else { tilt * side });
            emit(Book {
                edition,
                size: Vec3::new(w, h, d),
                tf: Transform::from_translation(position).with_rotation(rotation),
            });
            upright_stop = (tilt == 0.).then_some((x + span, h));
            x += span
                + if i > 0 && r.random_bool(0.10) {
                    r.random_range(0.03..0.10)
                } else {
                    r.random_range(0.001..0.007)
                };
        }
    }
}

pub(super) struct Book {
    pub(super) edition: u64,
    pub size: Vec3,
    pub tf: Transform,
}

impl Book {
    pub fn build(&self, a: &mut Assembly, version: u32) {
        let half = self.size * 0.5;
        let w = half.x;
        let d = half.z;
        let shoulder = w * 0.30;
        let crown = w * 0.23;
        let track = [
            (-w, -d),
            (-w, d - crown),
            (-w + shoulder, d),
            (w - shoulder, d),
            (w, d - crown),
            (w, -d),
        ];
        let point = |(x, z), y| self.tf.transform_point(Vec3::new(x, y, z));
        let print = jacket::slot(version, self.edition, false);
        let (substrate, label) = if let Some(slot) = print {
            (
                Surface::PrintedPaper,
                format!("books#finish{slot}#uv_local"),
            )
        } else {
            let substrate =
                [Surface::FabricAlt, Surface::Leather, Surface::Paper][(self.edition % 3) as usize];
            (
                substrate,
                format!("books#finish{}", variants::slot(self.edition)),
            )
        };
        let wrap = a.part(substrate, &label);
        let [left, right] = jacket::SPINE;
        let shoulder_u = (right - left) * 0.13;
        let us = [0., left, left + shoulder_u, right - shoulder_u, right, 1.];
        for (i, edge) in track.windows(2).enumerate() {
            let points = [
                point(edge[0], -half.y),
                point(edge[1], -half.y),
                point(edge[1], half.y),
                point(edge[0], half.y),
            ];
            let n = (points[1] - points[0])
                .cross(points[3] - points[0])
                .normalize();
            if print.is_some() {
                wrap.vertex_quad(
                    points,
                    [n; 4],
                    [[us[i], 1.], [us[i + 1], 1.], [us[i + 1], 0.], [us[i], 0.]],
                );
            } else {
                wrap.metric_quad(points, n);
            }
        }
        let pages = a.part(Surface::Paper, "books");
        pages.metric_quad(
            [
                point(track[5], -half.y),
                point(track[0], -half.y),
                point(track[0], half.y),
                point(track[5], half.y),
            ],
            self.tf.rotation * Vec3::NEG_Z,
        );
        for sign in [-1., 1.] {
            let y = sign * half.y;
            let normal = self.tf.rotation * (Vec3::Y * sign);
            for inds in [[0, 1, 4, 5], [1, 2, 3, 4]] {
                pages.metric_quad(inds.map(|i| point(track[i], y)), normal);
            }
        }
    }
}
