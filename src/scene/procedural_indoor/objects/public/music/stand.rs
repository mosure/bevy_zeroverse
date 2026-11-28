use super::*;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Stand {
    pub folding: bool,
    pub open_frame: bool,
    pub tilt_radians: f32,
    pub desk_height_m: f32,
    pub lip_height_m: f32,
    pub lip_depth_m: f32,
    pub foot_rotation: f32,
    pub desk_finish: Surface,
    pub score_slot: Option<usize>,
    pub spread: bool,
}

impl Stand {
    pub fn sample(o: &IndoorObject) -> Self {
        let mut frame = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"MuscLine"),
        );
        let mut r = stream(o.seed, u64::from_le_bytes(*b"MuscDesk"));
        let folding = frame.random_bool(0.50);
        let open_frame = folding && frame.random_bool(0.82);
        let tilt: f32 = r.random_range(0.15..0.53);
        let height = (o.size.x * r.random_range(0.54..0.72)).min(0.35);
        Self {
            folding,
            open_frame,
            tilt_radians: tilt,
            desk_height_m: height,
            lip_height_m: o.size.y - height * tilt.cos() - 0.006,
            lip_depth_m: r.random_range(0.037..0.063),
            foot_rotation: r.random_range(0.0..std::f32::consts::TAU),
            desk_finish: if !open_frame && frame.random_bool(0.24) {
                Surface::Wood
            } else {
                Surface::Metal
            },
            score_slot: r.random_bool(0.79).then(|| score::slot(o.seed)),
            spread: r.random_bool(0.64),
        }
    }
}

pub(in super::super) fn build(a: &mut Assembly, o: &IndoorObject) {
    const LABEL: &str = "music_stand";
    let s = Stand::sample(o);
    let rotation = Quat::from_rotation_x(-s.tilt_radians);
    let h = s.desk_height_m;
    let w = o.size.x - 0.008;
    // Balance the tilted desk about the mast; the pivot physically touches its back.
    let base = Vec3::new(0., s.lip_height_m, h * 0.43 * s.tilt_radians.sin() + 0.009);
    let at = |x, y, z| base + rotation * Vec3::new(x, y, z);
    let pivot = at(0., h * 0.43, -0.009);
    let joint = Vec3::new(0., pivot.y - 0.03, 0.);
    let metal = a.part(Surface::Metal, LABEL);
    let collar = s.lip_height_m * 0.62;
    metal.rod(Vec3::Y * 0.081, Vec3::Y * collar, 0.014);
    metal.rod(Vec3::Y * collar, joint, 0.009);
    metal.rod(joint, pivot, 0.012);
    metal.rod(
        Vec3::Y * (collar - 0.012),
        Vec3::Y * (collar + 0.014),
        0.018,
    );
    for k in 0..3 {
        let angle = s.foot_rotation + k as f32 * std::f32::consts::TAU / 3.;
        let dir = Vec3::new(angle.cos(), 0., angle.sin());
        let foot = dir * (0.48 * o.size.x.min(o.size.z) - 0.013) + Vec3::Y * 0.012;
        let knee = foot * 0.58 + Vec3::Y * 0.052;
        if s.folding {
            metal.rod(Vec3::Y * 0.25, foot, 0.009);
            metal.rod(Vec3::Y * 0.09, foot * 0.68 + Vec3::Y * 0.077, 0.005);
        } else {
            metal.rod(Vec3::Y * 0.085, knee, 0.012);
            metal.rod(knee, foot, 0.010);
        }
    }
    let pads = a.part(Surface::Rubber, LABEL);
    for k in 0..3 {
        let angle = s.foot_rotation + k as f32 * std::f32::consts::TAU / 3.;
        let foot =
            Vec3::new(angle.cos(), 0., angle.sin()) * (0.48 * o.size.x.min(o.size.z) - 0.013);
        pads.cuboid(
            Vec3::new(0.033, 0.018, 0.021),
            0.004,
            Transform::from_translation(foot + Vec3::Y * 0.009)
                .with_rotation(Quat::from_rotation_y(-angle)),
        );
    }
    let desk = a.part(s.desk_finish, LABEL);
    if s.open_frame {
        for side in [-1., 1.] {
            desk.rod(at(0., 0.016, 0.), at(side * w * 0.48, h, 0.), 0.005);
            desk.rod(
                at(side * w * 0.48, h, 0.),
                at(side * w * 0.48, 0., 0.),
                0.005,
            );
            desk.rod(at(side * w * 0.48, h, 0.), at(0., h * 0.43, 0.), 0.004);
        }
        desk.rod(at(0., 0., 0.), at(0., h - 0.006, 0.), 0.007);
    } else {
        desk.cuboid(
            Vec3::new(w, h, 0.011),
            0.003,
            Transform::from_translation(at(0., h * 0.5, 0.)).with_rotation(rotation),
        );
    }
    desk.cuboid(
        Vec3::new(w, 0.012, s.lip_depth_m),
        0.003,
        Transform::from_translation(at(0., 0., s.lip_depth_m * 0.5 - 0.003))
            .with_rotation(rotation),
    );
    if let Some(slot) = s.score_slot {
        let paper = a.part(
            Surface::PrintedPaper,
            &format!("paper#finish{slot}#uv_local"),
        );
        let page_h = (h - 0.020).min(if s.spread { w * 0.66 } else { h });
        let page_w = (page_h * 0.71).min(w * 0.47);
        let normal = rotation * Vec3::Z;
        let sheet = |g: &mut crate::scene::procedural_indoor::geometry::Geometry,
                     l: f32,
                     r: f32,
                     u: f32| {
            // Two-page atlas, each leaf receives its own part. A 2 mm gutter separates them.
            g.vertex_quad(
                [
                    at(l, 0.009, 0.010),
                    at(r, 0.009, 0.010),
                    at(r, page_h + 0.009, 0.010),
                    at(l, page_h + 0.009, 0.010),
                ],
                [normal; 4],
                [[u, 1.], [u + 0.5, 1.], [u + 0.5, 0.], [u, 0.]],
            );
            if s.open_frame {
                // The back of the leaf remains visible through the wire rack. Use the
                // margin of the same stock so ink reads on the printed side only.
                g.vertex_quad(
                    [
                        at(r, 0.009, 0.0095),
                        at(l, 0.009, 0.0095),
                        at(l, page_h + 0.009, 0.0095),
                        at(r, page_h + 0.009, 0.0095),
                    ],
                    [-normal; 4],
                    [
                        [0.002, 0.996],
                        [0.021, 0.996],
                        [0.021, 0.004],
                        [0.002, 0.004],
                    ],
                );
            }
        };
        if s.spread {
            sheet(paper, -page_w - 0.001, -0.001, 0.);
            sheet(paper, 0.001, page_w + 0.001, 0.5);
        } else {
            sheet(paper, -page_w * 0.5, page_w * 0.5, 0.);
        }
    }
}
