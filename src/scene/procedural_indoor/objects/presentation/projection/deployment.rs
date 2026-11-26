//! Material coordinates follow one continuous sheet. Stock hidden in the cassette
//! retains its vertices so flow and reverse playback track the same fabric.
use super::*;
use crate::scene::procedural_indoor::{
    materials::projection_cloth,
    motion::{self, ClothMotion, MotionKind, RollerFeed},
};

pub(crate) fn selected(o: &IndoorObject, version: u32, fraction: f32) -> bool {
    version >= motion::PROJECTION_MOTION_VERSION
        && fraction > 0.
        && stream(o.seed, u64::from_le_bytes(*b"AVselect")).random_bool(f64::from(fraction))
}

pub(super) fn build_if_selected(
    a: &mut Assembly,
    o: &IndoorObject,
    p: &ScreenProgram,
    i: &Installation,
    version: u32,
) -> bool {
    if !a
        .motion_policy
        .as_ref()
        .is_some_and(|policy| selected(o, version, policy.projection_screen_fraction))
    {
        return false;
    }
    let start = p.sheet_top_m - (p.sheet_bottom_m + 2. * p.bar_radius_m);
    let min = i.sheet_border_m * 0.18;
    let max = i.stock_length_m;
    let mut r = stream(o.seed, u64::from_le_bytes(*b"AVtravel"));
    let extension = (start - min) / (max - min);
    let lower = extension < 0.25 || (extension <= 0.75 && r.random_bool(0.5));
    let stop = if lower { max } else { min };
    let fraction = if r.random_bool(0.3) {
        1.
    } else {
        r.random_range(0.35..1.)
    };

    // The cassette slot is 0.38 housing heights deep. Put the bend inside it.
    // Subdivide only stock that crosses the bend; at every instant the entire
    // transition triangle is hidden above the lip. Bound tessellation for very
    // shallow imported housings by limiting travel in that installation.
    let inset = 0.19 * p.case_height_m;
    let pitch = inset * 0.8;
    let end = start + ((stop - start) * fraction).clamp(-384. * pitch, 384. * pitch);
    let feed_y = p.sheet_top_m + inset;
    let feed = RollerFeed::new(Transform::IDENTITY, feed_y).with_storage(max, inset * 0.75);
    let mut web = stock(p, i, start, end, inset, pitch);
    let mut joint = motion::construction::rigid(
        MotionKind::Covering,
        o.seed,
        Vec3::ZERO,
        Vec3::Y,
        0.,
        Vec3::Y * (start - end),
    );
    joint.cloth_motion = Some(ClothMotion::ProjectionScreen {
        feed,
        start_drop_m: start,
    });
    for (key, part) in &mut web.parts {
        let mut target = part.clone();
        for (a, b) in part.positions.iter_mut().zip(&mut target.positions) {
            let distance = a[1] - feed_y;
            a[1] = feed_y + feed.packed(distance);
            b[1] = feed_y + feed.packed(distance + (start - end));
        }
        joint.morph.insert(key.clone(), target);
    }
    // The weighted bar and finger loop translate together without deforming.
    construction::hem(&mut web, p, i);
    a.articulated(web, joint);
    true
}

fn stock(
    p: &ScreenProgram,
    i: &Installation,
    start: f32,
    end: f32,
    inset: f32,
    pitch: f32,
) -> Assembly {
    let mut web = Assembly::default();
    let mask = format!("{}#roller", projection_cloth::MASK);
    let image = format!("{}#roller", projection_cloth::label(i.coating));
    let x = p.sheet_center_x_m - 0.5 * p.sheet_width_m;
    let right = x + p.sheet_width_m;
    let z = p.sheet_front_m;
    let back = z - 0.0008;
    let bottom = p.sheet_top_m - start;
    let border = i.sheet_border_m;
    let zones = [0., border, border + i.full_image_height_m, i.stock_length_m];
    for (zone, pair) in zones.windows(2).enumerate() {
        let [lo, hi] = [pair[0], pair[1]];
        // Joining very short pieces can collapse triangles in packed f32 coordinates.
        // Extend the refined section to the nearest stock seam in that case.
        let mut enter = (start.min(end) + inset - pitch).clamp(lo, hi);
        let mut leave = (start.max(end) + inset + pitch).clamp(lo, hi);
        if enter - lo < pitch {
            enter = lo;
        }
        if hi - leave < pitch {
            leave = hi;
        }
        let stepped = ((leave - enter) / pitch).ceil() as usize;
        let cuts = (0..stepped)
            .map(|k| enter.lerp(leave, k as f32 / stepped as f32))
            .chain([leave, hi]);
        let mut prev = lo;
        for cut in cuts {
            if cut <= prev {
                continue;
            }
            let y0 = bottom + prev;
            let y1 = bottom + cut;
            prev = cut;
            let left = Vec3::new(x, y0, z);
            let top_right = Vec3::new(right, y1, z);
            if zone == 1 {
                construction::front(
                    web.part(Surface::Paper, &image),
                    left.with_x(x + border),
                    top_right.with_x(right - border),
                );
            }
            let g = web.part(Surface::Ink, &mask);
            if zone == 1 {
                construction::front(g, left, top_right.with_x(x + border));
                construction::front(g, left.with_x(right - border), top_right);
            } else {
                construction::front(g, left, top_right);
            }
            let corners = construction::corners(left, top_right);
            g.metric_quad(corners.map(|v| v.with_z(back)), Vec3::NEG_Z);
            for (a, b, normal) in [
                (corners[0], corners[3], Vec3::NEG_X),
                (corners[1], corners[2], Vec3::X),
            ] {
                g.metric_quad([a, b, b.with_z(back), a.with_z(back)], normal);
            }
        }
    }
    let g = web.part(Surface::Ink, &mask);
    for (s, n) in [(0., Vec3::NEG_Y), (i.stock_length_m, Vec3::Y)] {
        let a = Vec3::new(x, bottom + s, z);
        let b = a.with_x(right);
        g.metric_quad([a, b, b.with_z(back), a.with_z(back)], n);
    }
    web
}
