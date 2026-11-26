//! Compound fronds grow from elevated palm canes or low fern crowns.
mod leaflet;
#[cfg(test)]
mod tests;
use super::super::geometry::Geometry;
use super::*;

pub(in crate::scene::procedural_indoor) const INTRODUCED: u32 = 273;

#[derive(Clone, Copy, PartialEq, Eq, serde::Serialize)]
enum Kind {
    Palm,
    Fern,
}

#[derive(Clone, Copy, serde::Serialize)]
pub(in crate::scene::procedural_indoor) struct Habit {
    kind: Kind,
    pub(crate) crowns: usize,
    pub(crate) fronds: usize,
    /// Height of the leading crown above the soil as a fraction of living height.
    pub(crate) crown_elevation: f32,
    /// Offset between opposite ranks in units of one leaflet pitch.
    pub(crate) rank_stagger: f32,
    leaflet_half_width_over_length: f32,
    pinna_pairs: usize,
    fold: f32,
}

impl Habit {
    pub(in crate::scene::procedural_indoor) fn sample(
        o: &IndoorObject,
        growth: &Growth,
    ) -> Option<Self> {
        let kind = match o.variant % SPECIES {
            1 => Kind::Palm,
            4 => Kind::Fern,
            _ => return None,
        };
        let mut r = stream(o.seed, u64::from_le_bytes(*b"pinnate1"));
        let palm = kind == Kind::Palm;
        let crowns = r.random_range(1..=3);
        Some(Self {
            kind,
            crowns,
            fronds: growth
                .count(r.random_range(if palm { 10..=15 } else { 12..=17 }))
                .max(4 * crowns)
                .min(17),
            crown_elevation: r.random_range(if palm { 0.23..0.44 } else { 0.035..0.105 }),
            rank_stagger: r.random_range(0.04..0.68),
            leaflet_half_width_over_length: r.random_range(if palm {
                0.105..0.17
            } else {
                0.19..0.30
            }),
            pinna_pairs: r.random_range(if palm { 12..=17 } else { 12..=18 }),
            fold: r.random_range(if palm { 0.32..0.78 } else { 0.09..0.32 }),
        })
    }

    pub(super) fn build(
        &self,
        a: &mut Assembly,
        o: &IndoorObject,
        g: &Growth,
        base: Vec3,
        spread: f32,
        roots: f32,
    ) {
        let live = o.size.y - base.y;
        let space = Envelope {
            soil: base.y,
            top: o.size.y,
            radius: 0.5 * o.size.x.min(o.size.z),
            spread,
        };
        let palm = self.kind == Kind::Palm;
        let mut r = stream(o.seed, u64::from_le_bytes(*b"frondbed"));
        for i in 0..self.crowns {
            let angle =
                g.shoot_phase + TAU * i as f32 / self.crowns as f32 + r.random_range(-0.24..0.24);
            let dir = radial(angle);
            let root = base
                + dir
                    * roots
                    * r.random_range(if self.crowns == 1 {
                        0.02..0.20
                    } else {
                        0.28..0.65
                    });
            let crown_scale = if i == 0 {
                1.0
            } else {
                r.random_range(0.37..0.85)
            };
            let crown = root
                + dir * spread * r.random_range(0.015..0.13) * crown_scale
                + Vec3::Y * live * self.crown_elevation * crown_scale;
            let shaft = (live * if palm { 0.017 } else { 0.013 }).min(roots * 0.14) * crown_scale;
            let green = a.part(Surface::Succulent, "other_prop");
            if palm {
                cane(green, root, crown, shaft, r.random_range(2..=5));
            } else {
                green.tapered_tube(&[root, crown], shaft, shaft * 0.70, 7);
            }
            let count = self.fronds / self.crowns + usize::from(i < self.fronds % self.crowns);
            let phase = r.random_range(0.0..TAU);
            for j in 0..count {
                // Older fronds sit outside the whorl. More recent growth is brighter,
                // more upright, and carries fewer, more closely folded pinnae.
                let age = ((j as f32 + r.random_range(0.10..0.90)) / count as f32).clamp(0., 1.);
                let dir = radial(phase + g.angle(j) + r.random_range(-0.18..0.18));
                let insertion = crown.lerp(root, 0.28 * age * age) + dir * (shaft * 0.43 * age);
                let frond = Frond::sample(insertion, dir, age, space, palm, &mut r);
                let rib = (live * if palm { 0.0043 } else { 0.0031 }).min(spread * 0.016);
                a.part(Surface::Succulent, "other_prop").tapered_tube(
                    &frond.points,
                    rib,
                    rib * 0.22,
                    6,
                );
                let surface = if age < 0.15 + 0.12 * (1. - g.maturity) {
                    Surface::LeafLight
                } else {
                    Surface::Leaf
                };
                let leaves = a.part(surface, "other_prop");
                frond.add_leaves(leaves, self, space, &mut r);
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Envelope {
    soil: f32,
    top: f32,
    radius: f32,
    spread: f32,
}

struct Frond {
    points: [Vec3; 8],
    frames: [Frame; 7],
    age: f32,
}

struct Frame {
    along: Vec3,
    across: Vec3,
    normal: Vec3,
}

impl Frond {
    fn sample(
        crown: Vec3,
        dir: Vec3,
        age: f32,
        space: Envelope,
        palm: bool,
        r: &mut impl Rng,
    ) -> Self {
        let side = Vec3::Y.cross(dir);
        let reach = (space.spread.min(space.radius - crown.xz().length()))
            * (0.39 + 0.61 * age.sqrt())
            * r.random_range(0.85..1.00);
        let headroom = space.top - crown.y;
        let end =
            (headroom * (0.88 - age * if palm { 1.30 } else { 0.98 }) * r.random_range(0.82..1.00))
                .max(space.soil + 0.07 * (space.top - space.soil) - crown.y);
        let arch = headroom * (0.06 + 0.39 * age) * r.random_range(0.74..1.16);
        let sweep = reach * r.random_range(-0.18..0.18);
        let points: [Vec3; 8] = std::array::from_fn(|k| {
            let t = k as f32 / 7.;
            let arc = 4. * t * (1. - t);
            crown + dir * (reach * t) + side * (sweep * arc) + Vec3::Y * (end * t + arch * arc)
        });
        // The blades turn with the stem, including on nearly vertical new growth.
        // Compute its frame once per mesh segment for both ranks of leaflets.
        let roll = r.random_range(-0.48..0.48);
        let frames = std::array::from_fn(|k| {
            let along = (points[k + 1] - points[k]).normalize();
            let across = Quat::from_axis_angle(along, roll * (k as f32 + 0.5) / 7.)
                * Vec3::Y.cross(along).normalize();
            Frame {
                along,
                across,
                normal: along.cross(across),
            }
        });
        Self {
            points,
            frames,
            age,
        }
    }

    fn add_leaves(&self, leaves: &mut Geometry, habit: &Habit, space: Envelope, r: &mut impl Rng) {
        let palm = habit.kind == Kind::Palm;
        let age = self.age;
        let pairs = ((habit.pinna_pairs as f32 * (0.69 + age * 0.31)).round() as usize).max(6);
        let start = r.random_range(if palm { 0.17..0.31 } else { 0.095..0.21 });
        let pitch = (0.97 - start) / pairs as f32;
        let stagger = habit.rank_stagger * if r.random_bool(0.5) { 1. } else { -1. };
        let length_ratio = r.random_range(if palm { 0.58..0.80 } else { 0.37..0.55 });
        let forward = r.random_range(if palm { 0.24..0.75 } else { 0.08..0.32 });
        for j in 0..pairs {
            let node = start + (j as f32 + 0.5 + r.random_range(-0.09..0.09)) * pitch;
            for sign in [-1., 1.] {
                let t = node + sign * stagger * pitch * 0.5;
                let segment = t * self.frames.len() as f32;
                let k = segment as usize;
                let frame = &self.frames[k];
                let root = self.points[k].lerp(self.points[k + 1], segment - k as f32);
                let u = (t - start) / (1. - start);
                let taper = (1. - u * u) * (0.65 + 0.35 * (u * 4.).min(1.));
                // The reserved disk includes the folded blade width. Fit the
                // organ length before emitting it, preserving its proportions.
                let length = (space.spread
                    * length_ratio
                    * taper
                    * (0.75 + 0.25 * age)
                    * r.random_range(0.90..1.08))
                .min((space.radius - root.xz().length()) * 0.71);
                let cant = if palm {
                    0.42 - 0.35 * age - 0.25 * u
                } else {
                    0.06 - 0.36 * age + 0.16 * u
                };
                let mut delta = (frame.across * sign
                    + frame.along * (forward + 0.40 * u)
                    + frame.normal * (cant + r.random_range(-0.12..0.12)))
                .normalize()
                    * length;
                delta.y = delta
                    .y
                    .clamp(-0.50 * (root.y - space.soil), 0.45 * (space.top - root.y));
                let blade = leaflet::Blade {
                    root,
                    delta,
                    normal: frame.normal,
                    arch: (length * if palm { 0.08 + 0.06 * age } else { 0.055 })
                        .min((space.top - root.y) * 0.18),
                    half_width: length * habit.leaflet_half_width_over_length,
                    fold: habit.fold * (1.12 - 0.25 * age),
                    twist: r.random_range(-0.29..0.29),
                    distal_side: sign,
                };
                blade.write(leaves, habit.kind);
            }
        }
        // The rachis terminates in foliage rather than an exposed stick.
        let root = along(&self.points, 0.945);
        leaflet::Blade {
            root,
            delta: self.points[7] - root,
            normal: self.frames[6].normal,
            arch: (self.points[7] - root).length() * 0.035,
            half_width: (self.points[7] - root).length() * habit.leaflet_half_width_over_length,
            fold: habit.fold,
            twist: 0.,
            distal_side: 1.,
        }
        .write(leaves, habit.kind);
    }
}

/// Leaf scars are ridges in one continuous cane surface so they cannot flicker.
fn cane(g: &mut Geometry, root: Vec3, crown: Vec3, radius: f32, nodes: usize) {
    let axis = crown - root;
    let length = axis.length();
    let mut profile = Vec::with_capacity(3 * nodes + 3);
    profile.push((radius, 0.));
    for i in 1..=nodes {
        let t = i as f32 / (nodes + 1) as f32;
        let r = radius * (1. - 0.28 * t);
        let y = length * t;
        let shoulder = (radius * 0.55).min(length * 0.035);
        profile.extend([(r, y - shoulder), (r * 1.16, y), (r, y + shoulder)]);
    }
    profile.extend([(radius * 0.72, length), (0., length)]);
    let tf = Transform::from_translation(root)
        .with_rotation(Quat::from_rotation_arc(Vec3::Y, axis / length));
    g.lathe(&profile, 9, tf);
}
