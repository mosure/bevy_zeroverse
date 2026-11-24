//! Molded mains terminations. The occupied volume follows the final straight
//! connector segment; the rest of the lead retains its flexible jacket radius.
use super::*;

pub(crate) const RECESSED_MAINS: u32 = 18;

#[derive(Clone, Copy)]
pub(crate) struct MainsPlug {
    tip: Vec3,
    axis: Vec3,
    length: f32,
    radius: f32,
    recessed: bool,
}

impl MainsPlug {
    fn between(o: &IndoorObject, end: Vec3, next: Vec3) -> Option<Self> {
        o.cable_target_socket?;
        let delta = next - end;
        let straight = delta.length();
        if straight < 0.0001 || !straight.is_finite() {
            return None;
        }
        let size = ((o.seed >> 48) as u16) as f32 / u16::MAX as f32;
        Some(Self {
            tip: end,
            axis: delta / straight,
            length: (straight - 0.0003).min(0.034),
            radius: 0.0158 + size * 0.0020,
            recessed: o.variant == RECESSED_MAINS,
        })
    }

    pub fn from_path(o: &IndoorObject, path: &[Vec3]) -> Option<Self> {
        let [.., prev, end] = path else {
            return None;
        };
        Self::between(o, *end, *prev)
    }
    pub fn world(o: &IndoorObject) -> Option<Self> {
        o.cable_target_socket?;
        let [.., prev, end] = o.cable_path.as_deref()? else {
            return None;
        };
        let tf = o.transform();
        Self::between(o, tf.transform_point(*end), tf.transform_point(*prev))
    }

    pub fn fits(self) -> bool {
        self.length >= if self.recessed { 0.023 } else { 0.016 }
    }
    /// Radial extent is perpendicular to the port normal, so a plug mounted to
    /// a wall does not reserve space behind its mating face.
    pub fn bounds(self) -> (Vec3, Vec3) {
        let d = self.axis;
        let extent = Vec3::new(d.yz().length(), d.xz().length(), d.xy().length()) * self.radius;
        let end = self.tip + self.length * d;
        (self.tip.min(end) - extent, self.tip.max(end) + extent)
    }

    /// Four enclosing edges parallel to the plug axis. Unlike its world AABB,
    /// these retain the plane of contact on an obliquely installed wall plate.
    pub fn outer_edges(self) -> [(Vec3, Vec3); 4] {
        let rotation = Quat::from_rotation_arc(Vec3::Y, self.axis);
        [(-1., -1.), (-1., 1.), (1., 1.), (1., -1.)].map(|(x, z)| {
            let start = self.tip + rotation * Vec3::new(x, 0., z) * self.radius;
            (start, start + self.axis * self.length)
        })
    }

    pub fn build(self, a: &mut Assembly, label: &str, jacket: f32) {
        let l = self.length;
        let r = self.radius;
        let (nose, shoulder) = if self.recessed {
            (0.0114, 0.0112)
        } else {
            (r * 0.94, 0.0020)
        };
        let tf = Transform::from_translation(self.tip)
            .with_rotation(Quat::from_rotation_arc(Vec3::Y, self.axis));
        a.part(Surface::Plastic, label).lathe(
            &[
                (0., 0.00015),
                (nose, 0.00015),
                (nose, shoulder),
                (r, shoulder + 0.0016),
                (r, l * 0.69),
                (r * 0.76, l * 0.82),
                (jacket * 1.5, l * 0.89),
                (jacket * 1.5, l * 0.96),
                (jacket, l),
            ],
            16,
            tf,
        );
    }
}
