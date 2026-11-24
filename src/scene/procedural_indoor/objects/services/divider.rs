//! Unequal acoustic, glazed and slatted panels with dimensioned stable supports.
use super::*;
mod hardware;
mod plan;
mod stock;
#[cfg(test)]
mod tests;

pub use plan::{Arrangement, Plan};

#[derive(Debug, Clone, serde::Serialize)]
pub struct DividerProgram {
    pub weights: Vec<f32>,
    pub plan: Plan,
    /// Thickness of the finished insert inside the retaining channel.
    pub insert_thickness_m: f32,
    pub bottom: f32,
    pub top: f32,
    pub panel_depth: f32,
    pub border: f32,
    pub stand_span: f32,
    pub base: u8,
    pub frame: Surface,
    pub panel: Surface,
    pub slats: bool,
    pub slat_pitch: f32,
    pub slat_fraction: f32,
    pub slat_angle: f32,
    pub lower_fraction: f32,
}
pub fn parameters(o: &IndoorObject) -> DividerProgram {
    let mut r = stream(o.seed, 0x4449564944455250);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x4449_5649_4649_4e49);
    let n = r.random_range(1..=3);
    let weights = (0..n)
        .map(|_| r.random_range(0.60..1.45))
        .collect::<Vec<_>>();
    let plan = Plan::sample(o, &weights);
    let mut insert = stream(o.seed, u64::from_le_bytes(*b"DivSheet"));
    DividerProgram {
        weights,
        plan,
        insert_thickness_m: insert.random_range(0.007..0.012),
        bottom: o.size.y * r.random_range(0.07..0.31),
        top: o.size.y * r.random_range(0.89..1.0),
        panel_depth: r.random_range(0.018_f32..0.048).min(o.size.z * 0.24),
        border: r.random_range(0.016_f32..0.035).min(o.size.x * 0.024),
        stand_span: r.random_range(0.27..0.41),
        base: r.random_range(0..3),
        frame: [Surface::Wood, Surface::Metal, Surface::Chrome][f.random_range(0..3)],
        panel: [
            Surface::Fabric,
            Surface::FabricAlt,
            Surface::Wood,
            Surface::GlassInterior,
        ][f.random_range(0..4)],
        slats: r.random_bool(0.36),
        slat_pitch: r.random_range(0.045..0.115),
        slat_fraction: r.random_range(0.30..0.73),
        slat_angle: r.random_range(-0.30..0.30),
        lower_fraction: r.random_range(0.24..0.61),
    }
}

impl DividerProgram {
    fn frame_depth(&self) -> f32 {
        self.panel_depth + 0.008
    }

    fn slat_field(&self, opening: f32) -> SlatField {
        let count = (opening / self.slat_pitch).round().clamp(2., 24.) as usize;
        let pitch = opening / count as f32;
        let depth = self.panel_depth * 0.55;
        let (sin, cos) = self.slat_angle.sin_cos();
        // Preserve a real aperture even when a narrow louvre is turned edge-on.
        let width = (pitch * self.slat_fraction).min((0.91 * pitch - depth * sin.abs()) / cos);
        SlatField {
            count,
            pitch,
            width,
            depth,
        }
    }
}

struct SlatField {
    count: usize,
    pitch: f32,
    width: f32,
    depth: f32,
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let label = o.kind.class_name();
    let mut cuts = TimberCuts::new(o.seed, u64::from_le_bytes(*b"DivStock"));
    hardware::feet(a, o, &p);
    for i in 0..p.weights.len() {
        let bay = p.plan.bay(i, p.frame_depth());
        build_bay(a, label, &p, &bay, &mut cuts);
        if i > 0 {
            hardware::hinges(a, label, &p, i);
        }
    }
}

fn build_bay(
    a: &mut Assembly,
    label: &str,
    p: &DividerProgram,
    bay: &plan::Bay,
    cuts: &mut TimberCuts,
) {
    let frame = a.part(p.frame, label);
    let opening = bay.width - 2. * p.border;
    let lo = p.bottom + p.border;
    let hi = p.top - p.border;
    let depth = p.frame_depth();
    for side in [-1., 1.] {
        let local = Transform::from_xyz(
            side * (bay.width - p.border) * 0.5,
            (p.bottom + p.top) * 0.5,
            0.,
        );
        stock::member(
            frame,
            Vec3::new(p.border, p.top - p.bottom, depth),
            bay.frame * local,
            cuts.origin(),
        );
    }
    let rail = |g: &mut Geometry, y, h, cuts: &mut TimberCuts| {
        let tf = Transform::from_xyz(0., y, 0.).with_rotation(Quat::from_rotation_z(-FRAC_PI_2));
        stock::member(
            g,
            Vec3::new(h, opening, depth * 0.94),
            bay.frame * tf,
            cuts.origin(),
        );
    };
    rail(frame, p.bottom + p.border * 0.5, p.border, cuts);
    rail(frame, p.top - p.border * 0.5, p.border, cuts);
    let insert_top = if p.slats {
        let mid = lo + (hi - lo) * p.lower_fraction;
        let h = p.border * 0.9;
        rail(frame, mid, h, cuts);
        let slats = p.slat_field(opening);
        let base = mid + h * 0.5;
        for j in 0..slats.count {
            let x = -0.5 * opening + (j as f32 + 0.5) * slats.pitch;
            let tf = Transform::from_xyz(x, (base + hi) * 0.5, 0.)
                .with_rotation(Quat::from_rotation_y(p.slat_angle));
            stock::member(
                frame,
                Vec3::new(slats.width, hi - base, slats.depth),
                bay.frame * tf,
                cuts.origin(),
            );
        }
        mid - h * 0.5
    } else {
        hi
    };
    // The sheet seats inside a continuous channel. The gasket fills just the
    // perimeter, preserving a real aperture in both glass and slatted screens.
    let bead = 0.0035_f32.min(p.border * 0.18);
    let h = insert_top - lo;
    let yc = (lo + insert_top) * 0.5;
    let seal = a.part(Surface::Rubber, label);
    for side in [-1., 1.] {
        let tf = bay.frame * Transform::from_xyz(side * (opening - bead) * 0.5, yc, 0.);
        seal.cuboid(Vec3::new(bead, h, p.insert_thickness_m + 0.003), 0., tf);
        let tf = bay.frame * Transform::from_xyz(0., yc + side * (h - bead) * 0.5, 0.);
        seal.cuboid(
            Vec3::new(opening - 2. * bead, bead, p.insert_thickness_m + 0.003),
            0.,
            tf,
        );
    }
    stock::member(
        a.part(p.panel, label),
        Vec3::new(opening - 2. * bead, h - 2. * bead, p.insert_thickness_m),
        bay.frame * Transform::from_xyz(0., yc, 0.),
        cuts.origin(),
    );
}
