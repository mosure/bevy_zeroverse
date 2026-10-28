//! Recorded wall section and opening returns. All added detail stays inside the
//! rough aperture; panes, wall cutouts and annotation rays use the same geometry.
use super::super::trim::{TrimProfile, TrimSection};
use super::*;
use crate::scene::procedural_indoor::envelope::construction::triangle;

pub fn wall_thickness(seed: u64) -> f32 {
    let mut r = stream(seed, u64::from_le_bytes(*b"WallSect"));
    r.random_range(0.15_f32.ln()..0.72_f32.ln()).exp()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RevealProgram {
    pub wall_thickness_m: f32,
    pub lining_thickness_m: f32,
    pub lining_surface: Surface,
    pub finish_slot: usize,
    /// Profiled inset surround. Its height is the width of the opening border.
    pub surround: Option<TrimSection>,
}
impl RevealProgram {
    pub fn sample(
        seed: u64,
        wall_thickness_m: f32,
        frame_depth: f32,
        frame_width: f32,
    ) -> (Self, f32) {
        let mut r = stream(seed, u64::from_le_bytes(*b"WinRevel"));
        let lining_surface = [
            Surface::Paint,
            Surface::WoodEdge,
            Surface::Concrete,
            Surface::Ceramic,
            Surface::Plastic,
        ][r.random_range(0..5)];
        let finish_slot = r.random_range(0..super::super::super::materials::variants::COUNT);
        let surround = r.random_bool(0.72).then(|| TrimSection {
            height_m: r.random_range(0.008..(frame_width - 0.001).min(0.070)),
            depth_m: r.random_range(0.004..0.028),
            shoulder: r.random_range(0.36..0.85),
            tip_fraction: r.random_range(0.18..0.72),
            profile: [
                TrimProfile::Flat,
                TrimProfile::Chamfer,
                TrimProfile::Stepped,
                TrimProfile::Round,
                TrimProfile::Cove,
            ][r.random_range(0..5)],
            surface: [
                Surface::Paint,
                Surface::WoodEdge,
                Surface::Metal,
                Surface::Plastic,
            ][r.random_range(0..4)],
            finish_slot,
        });
        // Keep the complete frame inside the wall, with real room-side and
        // exterior returns. Setback is independent of overall wall thickness.
        let recess = frame_depth * 0.5
            + 0.012
            + r.random_range(0.05..0.95) * (wall_thickness_m - frame_depth - 0.024);
        (
            Self {
                wall_thickness_m,
                lining_thickness_m: r.random_range(0.004..0.014),
                lining_surface,
                finish_slot,
                surround,
            },
            recess,
        )
    }
    pub fn validate(&self, frame_depth: f32, recess: f32) -> Result<(), String> {
        let count = super::super::super::materials::variants::COUNT;
        if !frame_depth.is_finite()
            || !recess.is_finite()
            || !(0.15..=0.72).contains(&self.wall_thickness_m)
            || !(0.003..=0.015).contains(&self.lining_thickness_m)
            || self.finish_slot >= count
            || !matches!(
                self.lining_surface,
                Surface::Paint
                    | Surface::WoodEdge
                    | Surface::Concrete
                    | Surface::Ceramic
                    | Surface::Plastic
            )
            || recess - frame_depth * 0.5 < 0.011
            || recess + frame_depth * 0.5 > self.wall_thickness_m - 0.011
            || self.surround.as_ref().is_some_and(|s| {
                !(0.006..=0.075).contains(&s.height_m)
                    || !(0.003..=0.03).contains(&s.depth_m)
                    || !(0.3..=0.9).contains(&s.shoulder)
                    || !(0.1..=0.8).contains(&s.tip_fraction)
                    || s.finish_slot >= count
                    || !matches!(
                        s.surface,
                        Surface::Paint | Surface::WoodEdge | Surface::Metal | Surface::Plastic
                    )
            })
        {
            return Err("invalid exterior wall section or window return".into());
        }
        Ok(())
    }
}

pub(super) fn build(a: &mut Assembly, tf: Transform, f: &Facade, o: &WindowOpening) {
    let Some(p) = f.joinery.as_ref().and_then(|j| j.reveal.as_ref()) else {
        return;
    };
    // Stop before the applied frame bead. Running the liner to the structural
    // frame face intersects that bead and duplicates its rear cap at corners.
    let bead = f.joinery.as_ref().map_or(0., |j| {
        if j.face_bead_fraction > 0. {
            j.face_bead_depth_m
        } else {
            0.
        }
    });
    // Rebate the casing/liner joint behind the room face. A casing back on the
    // exact wall plane competes with the end cap of a perpendicular partition.
    let rebate = if p.surround.is_some() { 0.0015 } else { 0. };
    let depth = f.recess - f.frame_depth * 0.5 - bead - 0.0015 - rebate;
    let t = p.lining_thickness_m;
    let c = (o.min + o.max) * 0.5;
    let size = o.max - o.min;
    let label = format!("wall#finish{}", p.finish_slot);
    let g = a.part(p.lining_surface, &label);
    for x in [o.min.x + t * 0.5, o.max.x - t * 0.5] {
        g.cuboid(
            Vec3::new(t, size.y - 0.004, depth),
            0.,
            Transform::from_translation(tf.transform_point(Vec3::new(
                x,
                c.y,
                -rebate - depth * 0.5,
            )))
            .with_rotation(tf.rotation),
        );
    }
    for y in [o.min.y + t * 0.5, o.max.y - t * 0.5] {
        g.cuboid(
            Vec3::new(size.x - 2. * t - 0.004, t, depth),
            0.,
            Transform::from_translation(tf.transform_point(Vec3::new(
                c.x,
                y,
                -rebate - depth * 0.5,
            )))
            .with_rotation(tf.rotation),
        );
    }
    let Some(s) = &p.surround else { return };
    let g = a.part(s.surface, &format!("window#finish{}", s.finish_slot));
    let ring = |u: f32, z: f32| {
        let lo = o.min + Vec2::splat(u * s.height_m);
        let hi = o.max - Vec2::splat(u * s.height_m);
        [lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y)]
            .map(|p| tf.transform_point(Vec3::new(p.x, p.y, z)))
    };
    let mut emit = |a: [Vec3; 4], b: [Vec3; 4], normal: Vec3| {
        for i in 0..4 {
            let j = (i + 1) % 4;
            triangle(g, [a[i], a[j], b[j]], normal);
            triangle(g, [a[i], b[j], b[i]], normal);
        }
    };
    let knots = s.knots();
    for pair in knots.windows(2) {
        emit(
            ring(pair[0].x, pair[0].y * s.depth_m),
            ring(pair[1].x, pair[1].y * s.depth_m),
            tf.rotation * Vec3::Z,
        );
    }
    // Butt the surround against the rebated return, without overlapping edges.
    emit(
        ring(0., -rebate),
        ring(1., -rebate),
        tf.rotation * Vec3::NEG_Z,
    );
    for (u, z, sign) in [
        (0., s.depth_m, 1.),
        (1., knots.last().unwrap().y * s.depth_m, -1.),
    ] {
        let back = ring(u, -rebate);
        let front = ring(u, z);
        for i in 0..4 {
            let j = (i + 1) % 4;
            let normal = (back[j] - back[i]).cross(tf.rotation * Vec3::Z) * sign;
            triangle(g, [back[i], back[j], front[j]], normal);
            triangle(g, [back[i], front[j], front[i]], normal);
        }
    }
}

#[cfg(test)]
mod tests;
