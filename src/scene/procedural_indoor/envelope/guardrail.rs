//! Recorded guard sections shared by mezzanines and raised-floor edges.
//! Continuous dimensions and bay variation retain manufactured regularity
//! within a bay, without fixing every installation to the same grid.
use super::*;
use crate::scene::procedural_indoor::{layout::stream, materials::Surface, objects::Assembly};
use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Infill {
    Glass,
    Pickets,
    Slats,
    Solid,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardSection {
    pub seed: u64,
    pub infill: Infill,
    pub frame: Surface,
    pub handrail: Surface,
    pub panel: Surface,
    pub round_posts: bool,
    pub round_handrail: bool,
    pub post_width: f32,
    pub handrail_width: f32,
    pub handrail_depth: f32,
    pub post_pitch: f32,
    pub bay_jitter: f32,
    pub infill_width: f32,
    pub infill_gap: f32,
    pub panel_thickness: f32,
    pub bottom_gap: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardProgram {
    pub height: f32,
    pub section: GuardSection,
}

impl GuardSection {
    pub fn sample(seed: u64) -> Self {
        let mut r = stream(seed, 0x4755415244534543);
        Self {
            seed,
            infill: [Infill::Glass, Infill::Pickets, Infill::Slats, Infill::Solid]
                [r.random_range(0..4)],
            frame: [Surface::Metal, Surface::Chrome, Surface::Wood][r.random_range(0..3)],
            handrail: [Surface::Wood, Surface::Metal, Surface::Chrome][r.random_range(0..3)],
            panel: [Surface::Metal, Surface::Wood, Surface::Paint][r.random_range(0..3)],
            round_posts: r.random_bool(0.45),
            round_handrail: r.random_bool(0.6),
            post_width: r.random_range(0.035..0.075),
            handrail_width: r.random_range(0.035..0.065),
            handrail_depth: r.random_range(0.025..0.055),
            post_pitch: r.random_range(0.70..1.45),
            bay_jitter: r.random_range(0.0..0.16),
            infill_width: r.random_range(0.014..0.060),
            infill_gap: r.random_range(0.045..0.095),
            panel_thickness: r.random_range(0.008..0.022),
            bottom_gap: r.random_range(0.035..0.085),
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        for (v, lo, hi) in [
            (self.post_width, 0.025, 0.09),
            (self.handrail_width, 0.025, 0.08),
            (self.handrail_depth, 0.02, 0.07),
            (self.post_pitch, 0.6, 1.6),
            (self.bay_jitter, 0., 0.18),
            (self.infill_width, 0.01, 0.08),
            (self.infill_gap, 0.03, 0.10),
            (self.panel_thickness, 0.006, 0.03),
            (self.bottom_gap, 0.02, 0.10),
        ] {
            if !v.is_finite() || !(lo..=hi).contains(&v) {
                return Err("invalid guard section dimensions".into());
            }
        }
        if ![self.frame, self.handrail, self.panel]
            .into_iter()
            .all(|s| {
                matches!(
                    s,
                    Surface::Metal | Surface::Chrome | Surface::Wood | Surface::Paint
                )
            })
        {
            return Err("guard section requires structural finish surfaces".into());
        }
        Ok(())
    }
    pub fn padding(&self) -> f32 {
        (self.post_width * 0.8)
            .max(self.handrail_width.max(self.handrail_depth) * 0.71)
            .max(self.infill_width * 0.71)
            .max(self.panel_thickness * 0.5)
            + 0.004
    }
    pub fn bounds(&self, height: f32, p: Vec3, q: Vec3) -> (Vec3, Vec3) {
        let pad = self.padding();
        (
            p.min(q) - Vec3::new(pad, 0., pad),
            q.max(p) + Vec3::new(pad, height + pad, pad),
        )
    }
}
impl GuardProgram {
    pub fn sample(seed: u64) -> Self {
        let mut r = stream(seed, 0x4755415244484549);
        Self {
            height: r.random_range(1.03..1.22),
            section: GuardSection::sample(seed),
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if !(1.0..=1.25).contains(&self.height) {
            return Err("invalid guard height".into());
        }
        self.section.validate()
    }
}

fn member(
    a: &mut Assembly,
    surface: Surface,
    p: Vec3,
    q: Vec3,
    width: f32,
    depth: f32,
    round: bool,
) {
    let d = q - p;
    if d.length_squared() < 1e-10 {
        return;
    }
    let tangent = d.normalize();
    let side = if tangent.dot(Vec3::Y).abs() > 0.999 {
        Vec3::X
    } else {
        Vec3::Y.cross(tangent).normalize()
    };
    // Keep rectangular rail depth vertical for both X/Z spans. The shortest
    // Y-to-direction rotation alone swaps width/depth between the two axes.
    let tf = Transform::from_translation((p + q) * 0.5).with_rotation(Quat::from_mat3(
        &Mat3::from_cols(side, tangent, side.cross(tangent)),
    ));
    if round {
        let h = d.length() * 0.5;
        a.part(surface, "other_structure").lathe(
            &[(0., -h), (width * 0.5, -h), (width * 0.5, h), (0., h)],
            12,
            tf,
        );
    } else {
        a.part(surface, "other_structure")
            .cuboid(Vec3::new(width, d.length(), depth), 0., tf);
    }
}

pub(super) struct GuardRun {
    pub start: Vec3,
    pub end: Vec3,
    pub height: f32,
    pub stair_rise: f32,
}

pub(super) fn build(
    a: &mut Assembly,
    s: &GuardSection,
    run: GuardRun,
    support: impl Fn(Vec3) -> f32,
    posts: &mut Vec<Vec3>,
) {
    let GuardRun {
        start: p,
        end: q,
        height,
        stair_rise,
    } = run;
    let horizontal = p.xz().distance(q.xz());
    if horizontal < 0.035 {
        return;
    }
    let up = Vec3::Y * height;
    member(
        a,
        s.handrail,
        p + up,
        q + up,
        s.handrail_width,
        s.handrail_depth,
        s.round_handrail,
    );
    let count = (horizontal / s.post_pitch).ceil().max(1.) as u32;
    let mut bays = stream(
        s.seed
            ^ ((p.x.to_bits() as u64) << 32)
            ^ p.z.to_bits() as u64
            ^ (q.x.to_bits() as u64).rotate_left(17)
            ^ (q.z.to_bits() as u64).rotate_left(41),
        0x4755415244424159,
    );
    let positions: Vec<_> = (0..=count)
        .map(|i| {
            let jitter = if i == 0 || i == count {
                0.
            } else {
                bays.random_range(-s.bay_jitter..=s.bay_jitter)
            };
            p.lerp(q, (i as f32 + jitter) / count as f32)
        })
        .collect();
    let at = |i: u32| positions[i as usize];
    for i in 0..=count {
        let c = at(i);
        let base = c.with_y(support(c));
        // Connected bays share their corner/landing post. Duplicate cylinders
        // and foot plates cause coplanar flicker and unnecessary triangles.
        if posts.iter().any(|p| p.distance_squared(c) < 1e-8) {
            continue;
        }
        posts.push(c);
        member(
            a,
            s.frame,
            base,
            c + up,
            s.post_width,
            s.post_width,
            s.round_posts,
        );
        a.box_part(
            s.frame,
            "other_structure",
            base + Vec3::Y * 0.006,
            Vec3::new(s.post_width * 1.6, 0.012, s.post_width * 1.6),
            0.,
        );
    }
    for i in 0..count {
        let start = at(i);
        let end = at(i + 1);
        let length = start.xz().distance(end.xz());
        let inset = (s.post_width * 0.5 + 0.012) / length;
        if inset >= 0.45 {
            continue;
        }
        let p = start.lerp(end, inset);
        let q = start.lerp(end, 1. - inset);
        if matches!(s.infill, Infill::Pickets | Infill::Slats) {
            let length = p.xz().distance(q.xz());
            // Short deck-edge remnants may have no room for a full-width bar.
            // Do not intersect the adjacent posts to force one into the bay.
            if length < s.infill_width + 0.01 {
                continue;
            }
            let bottom = Vec3::Y * (stair_rise + s.bottom_gap);
            member(
                a,
                s.frame,
                start + bottom,
                end + bottom,
                0.025,
                0.025,
                false,
            );
            let bars = (length / (s.infill_width + s.infill_gap)).ceil().max(1.) as u32;
            for j in 0..bars {
                let c = p.lerp(q, (j as f32 + 0.5) / bars as f32);
                let base = c + bottom;
                let tip = c + up - Vec3::Y * s.handrail_depth * 0.5;
                let d = q - p;
                a.part(s.panel, "other_structure").cuboid(
                    Vec3::new(
                        s.infill_width,
                        tip.y - base.y,
                        if s.infill == Infill::Slats {
                            s.panel_thickness
                        } else {
                            s.infill_width
                        },
                    ),
                    0.,
                    Transform::from_translation((base + tip) * 0.5)
                        .with_rotation(Quat::from_rotation_y((-d.z).atan2(d.x))),
                );
            }
        } else {
            let normal = (q - p).cross(Vec3::Y).normalize() * s.panel_thickness;
            // The slope guide lies at most one riser below a tread. Raising
            // infill by that amount prevents panels slicing through stair noses.
            let bottom = Vec3::Y * (stair_rise + s.bottom_gap);
            let top = up - Vec3::Y * (s.handrail_depth * 0.5 + 0.015);
            super::construction::panel(
                a.part(
                    if s.infill == Infill::Glass {
                        Surface::GlassInterior
                    } else {
                        s.panel
                    },
                    "other_structure",
                ),
                &[
                    p + bottom + normal * 0.5,
                    q + bottom + normal * 0.5,
                    q + top + normal * 0.5,
                    p + top + normal * 0.5,
                ],
                normal,
            );
            // Clamp/standoff blocks connect infill edges to the supporting
            // posts. The intentional clearance must not leave floating panels.
            for (post, edge) in [(start, p), (end, q)] {
                for h in [stair_rise + s.bottom_gap + 0.08, height - 0.15] {
                    member(
                        a,
                        s.frame,
                        post + Vec3::Y * h,
                        edge + Vec3::Y * h,
                        0.025,
                        s.panel_thickness + 0.012,
                        false,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rectangular_handrails_preserve_vertical_section_across_axes() {
        for direction in [Vec3::X, Vec3::Z, -Vec3::X, -Vec3::Z] {
            let mut a = Assembly::default();
            member(
                &mut a,
                Surface::Wood,
                Vec3::ZERO,
                direction * 2.,
                0.08,
                0.04,
                false,
            );
            let (lo, hi) = a.bounds();
            assert!((hi.y - lo.y - 0.04).abs() < 1e-6);
            assert!(((hi - lo).xz().min_element() - 0.08).abs() < 1e-6);
        }
    }
}
