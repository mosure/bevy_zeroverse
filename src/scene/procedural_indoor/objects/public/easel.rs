//! Adjustable atelier easels. The forward side of the tilted painting plane is local +Z.
use super::*;
mod construction;
#[cfg(test)]
mod tests;

pub(crate) const VERSION: u32 = 298;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Stand {
    Tripod,
    HFrame,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Canvas {
    Empty,
    Linen,
    Primed,
    Painted,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Easel {
    pub stand: Stand,
    pub canvas: Canvas,
    pub frame_finish: Surface,
    pub lean_radians: f32,
    pub base_width_m: f32,
    pub stile_width_m: f32,
    pub stock_depth_m: f32,
    /// Distance up the inclined mast to the top of the tray.
    pub tray_position_m: f32,
    pub canvas_width_m: f32,
    pub canvas_height_m: f32,
    pub canvas_depth_m: f32,
    pub tray_depth_m: f32,
    pub working_clearance_m: f32,
}

impl Easel {
    pub fn sample(o: &IndoorObject) -> Self {
        let mut form = stream(o.seed, u64::from_le_bytes(*b"EaslForm"));
        let mut content = stream(o.seed, u64::from_le_bytes(*b"EaslWork"));
        let mut stock = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"EaslWood"),
        );
        let stand = if stock.random_bool(0.51) {
            Stand::Tripod
        } else {
            Stand::HFrame
        };
        let frame_finish = if stock.random_bool(if stand == Stand::Tripod { 0.76 } else { 0.88 }) {
            Surface::Wood
        } else {
            Surface::Metal
        };
        let canvas = match content.random_range(0..100) {
            0..=10 => Canvas::Empty,
            11..=23 => Canvas::Linen,
            24..=42 => Canvas::Primed,
            _ => Canvas::Painted,
        };
        let tray_position_m = o.size.y * form.random_range(0.33..0.54);
        let board_wider = canvas != Canvas::Empty && (o.size.x > 0.79 || form.random_bool(0.37));
        let lean_limit = (0.65 * o.size.z / o.size.y).atan();
        let lean_radians = form.random_range(0.047..0.285_f32).min(lean_limit);
        let width = if board_wider {
            o.size.x
        } else {
            o.size.x * content.random_range(0.54..0.96)
        };
        let height = (width * content.random_range(0.58..1.53_f32))
            .clamp(0.24, o.size.y * 0.93 - tray_position_m - 0.025);
        Self {
            stand,
            canvas,
            frame_finish,
            lean_radians,
            base_width_m: if board_wider {
                (o.size.x * form.random_range(0.64..0.94)).min(0.82)
            } else {
                o.size.x
            },
            stile_width_m: form.random_range(0.029..0.056),
            stock_depth_m: form.random_range(0.023..0.042),
            tray_position_m,
            canvas_width_m: width,
            canvas_height_m: height,
            canvas_depth_m: content.random_range(0.019..0.039),
            tray_depth_m: form.random_range(0.073..0.112),
            working_clearance_m: working_clearance(o),
        }
    }

    /// This rigid frame is shared by the jaws, paint surface, stretcher and tray.
    /// The front feet are set back enough for the tray to stay inside the footprint.
    pub(crate) fn rail_frame(&self, o: &IndoorObject) -> Transform {
        let (sin, cos) = self.lean_radians.sin_cos();
        let tray_front =
            (self.canvas_back_z() + self.tray_depth_m) * cos - (self.tray_position_m - 0.020) * sin;
        let toe = (o.size.z * 0.31).min(o.size.z * 0.5 - tray_front);
        Transform::from_xyz(0., 0.024, toe).with_rotation(Quat::from_rotation_x(-self.lean_radians))
    }

    pub(crate) fn canvas_back_z(&self) -> f32 {
        self.stock_depth_m * 1.5
    }
}

pub(crate) fn working_clearance(o: &IndoorObject) -> f32 {
    0.70 + f32::from((o.seed >> 37) as u16) / f32::from(u16::MAX) * 0.31
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    construction::build(a, o, &Easel::sample(o));
}
