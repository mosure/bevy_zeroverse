use super::*;

#[derive(serde::Serialize)]
pub(super) struct Edging {
    pub(super) height_m: f32,
    pub(super) width_m: f32,
    pub(super) exponent: f32,
    section: Section,
}

#[derive(serde::Serialize)]
enum Section {
    CastCurb {
        foot_scale: f32,
        bevel_m: f32,
    },
    CappedMasonry {
        foot_scale: f32,
        wall_scale: f32,
        cap_height_m: f32,
        tile_pitch_m: f32,
        joint_m: f32,
        course_phase: f32,
    },
    FoldedMetal {
        toe_scale: f32,
        toe_height_m: f32,
        wall_scale: f32,
        return_height_m: f32,
    },
}

impl Edging {
    pub(super) fn sample(o: &IndoorObject) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"YardEdge"));
        let radius = o.size.xz() * 0.5;
        let (height, width, section): (f32, f32, Section) = match r.random_range(0..3) {
            0 => (
                r.random_range(0.10..0.25),
                r.random_range(0.062..0.15),
                Section::CastCurb {
                    foot_scale: r.random_range(0.90..0.99),
                    bevel_m: r.random_range(0.007..0.019),
                },
            ),
            1 => {
                let wall_scale =
                    1. - (r.random_range(0.018..0.041) / radius.max_element()).min(0.15);
                (
                    r.random_range(0.22..0.56),
                    r.random_range(0.10..0.19),
                    Section::CappedMasonry {
                        foot_scale: wall_scale * r.random_range(0.93..1.0),
                        wall_scale,
                        cap_height_m: r.random_range(0.034..0.074),
                        tile_pitch_m: r.random_range(0.32..0.60),
                        joint_m: r.random_range(0.006..0.014),
                        course_phase: r.random_range(0.0..1.0),
                    },
                )
            }
            _ => {
                let wall_scale =
                    1. - (r.random_range(0.008..0.023) / radius.max_element()).min(0.15);
                (
                    r.random_range(0.24..0.62),
                    r.random_range(0.035..0.071),
                    Section::FoldedMetal {
                        toe_scale: wall_scale * r.random_range(0.86..0.96),
                        wall_scale,
                        toe_height_m: r.random_range(0.028..0.071),
                        return_height_m: r.random_range(0.013..0.033),
                    },
                )
            }
        };
        Self {
            height_m: height.min(o.size.y * 0.38),
            width_m: width.min(radius.min_element() * 0.24),
            exponent: r.random_range(2.0..8.4),
            section,
        }
    }

    pub(super) fn mesh(&self, a: &mut Assembly, radius: Vec2, ground: &Ground) {
        let outside = RevolvedOutline::superellipse(SIDES as u32, self.exponent, radius);
        let h = self.height_m;
        let sweep = |a: &mut Assembly, surface, profile: &[(f32, f32)]| {
            outside.sweep(
                a.part(surface, "plant"),
                profile,
                ProfileJoints::Creased,
                Transform::IDENTITY,
            );
        };
        let (surface, top_scale) = match self.section {
            Section::CastCurb {
                foot_scale,
                bevel_m,
            } => {
                let bevel = bevel_m.min(h * 0.2);
                let top = 1. - bevel.min(self.width_m * 0.4) / radius.max_element();
                sweep(
                    a,
                    Surface::Concrete,
                    &[(foot_scale, 0.), (1., h - bevel), (top, h)],
                );
                (Surface::Concrete, top)
            }
            Section::FoldedMetal {
                toe_scale,
                toe_height_m,
                wall_scale,
                return_height_m,
            } => {
                let toe = toe_height_m.min(h * 0.25);
                let lip = h - return_height_m.min(h * 0.2);
                sweep(
                    a,
                    Surface::Metal,
                    &[
                        (toe_scale, 0.),
                        (toe_scale, toe),
                        (wall_scale, toe),
                        (wall_scale, lip),
                        (1., lip),
                        (1., h),
                    ],
                );
                (Surface::Metal, 1.)
            }
            Section::CappedMasonry {
                foot_scale,
                wall_scale,
                cap_height_m,
                tile_pitch_m,
                joint_m,
                course_phase,
            } => {
                let underside = h - cap_height_m.min(h * 0.32);
                sweep(
                    a,
                    Surface::Concrete,
                    &[(foot_scale, 0.), (wall_scale, underside)],
                );
                // The joints wrap over the tile faces. Mortar and tile share
                // edges, with a single visible surface through each joint.
                let mut distance = 0.;
                let mut next_joint = course_phase * tile_pitch_m;
                for i in 0..SIDES {
                    let outer =
                        [i, i + 1].map(|k| outside.perimeter_point(k as f32 / SIDES as f32));
                    let inner = [ground.boundary[i], ground.boundary[(i + 1) % SIDES]];
                    let length = outer[0].distance(outer[1]);
                    let mut strip = |start: f32, end: f32, surface| {
                        let g = a.part(surface, "plant");
                        let op = [start, end].map(|t| outer[0].lerp(outer[1], t));
                        let ip = [start, end].map(|t| inner[0].lerp(inner[1], t));
                        let from = op.map(|v| at_height(v * wall_scale, underside));
                        let to = op.map(|v| at_height(v, underside));
                        let top = op.map(|v| at_height(v, h));
                        g.metric_quad([from[0], from[1], to[1], to[0]], Vec3::NEG_Y);
                        g.metric_quad([to[0], to[1], top[1], top[0]], at_height(op[0] + op[1], 0.));
                        close_lip(g, op, ip, h, ground.edge_height_m);
                    };
                    if next_joint < distance + length {
                        let t = ((next_joint - distance) / length).clamp(0.10, 0.90);
                        let half = (joint_m / (2. * length)).min(0.075);
                        strip(0., t - half, Surface::Terracotta);
                        strip(t - half, t + half, Surface::Concrete);
                        strip(t + half, 1., Surface::Terracotta);
                        next_joint += tile_pitch_m;
                    } else {
                        strip(0., 1., Surface::Terracotta);
                    }
                    distance += length;
                }
                return;
            }
        };
        let shell = a.part(surface, "plant");
        for i in 0..SIDES {
            let outer =
                [i, i + 1].map(|k| outside.perimeter_point(k as f32 / SIDES as f32) * top_scale);
            let inner = [ground.boundary[i], ground.boundary[(i + 1) % SIDES]];
            close_lip(shell, outer, inner, h, ground.edge_height_m);
        }
    }
}

fn close_lip(g: &mut Geometry, outer: [Vec2; 2], inner: [Vec2; 2], h: f32, soil: f32) {
    let [a, b] = outer.map(|p| at_height(p, h));
    let [c, d] = inner.map(|p| at_height(p, h));
    g.metric_quad([a, b, d, c], Vec3::Y);
    let [e, f] = inner.map(|p| at_height(p, soil));
    g.metric_quad([c, d, f, e], at_height(-inner[0] - inner[1], 0.));
}
