use super::*;
use crate::scene::procedural_indoor::materials::{input_housing::Housing, point_of_sale};

pub(super) const DISPLAY_DEPTH: f32 = 0.042;

pub(super) fn base(a: &mut Assembly, x: f32, width: f32, depth: f32, top: f32, stock: Surface) {
    let toe = 0.074;
    board(
        a,
        Surface::Rubber,
        COUNTER,
        Vec3::new(x, toe * 0.5, 0.),
        Vec3::new(width - 0.064, toe, depth - 0.070),
    );
    a.box_part(
        stock,
        COUNTER,
        Vec3::new(x, (top + toe) * 0.5, 0.),
        Vec3::new(width - 0.022, top - toe, depth - 0.032),
        0.008,
    );
}

pub(super) fn cabinet(
    a: &mut Assembly,
    o: &IndoorObject,
    d: CheckoutStation,
    cuts: &mut TimberCuts,
) {
    let s = o.size;
    let [(cx, core), (wx, work)] = d.bays(s.x);
    let toe = 0.084;
    let top = d.worktop_height_m - 0.035;
    let t = 0.024;
    board(
        a,
        Surface::Rubber,
        COUNTER,
        Vec3::Y * (toe * 0.5),
        Vec3::new(s.x - 0.13, toe, s.z - 0.11),
    );
    let panel = (d.cabinet, COUNTER);
    a.board(
        cuts,
        panel,
        Vec3::Y * (toe + t * 0.5),
        Vec3::new(s.x - 0.032, t, s.z - 0.036),
        0.,
        GrainAxis::X,
    );
    let bottom = toe + t;
    let height = top - bottom;
    // The cashier-side cubby is genuinely open. The front and end panels meet
    // at their edges, so stock and interior detail cannot flicker through them.
    a.board(
        cuts,
        panel,
        Vec3::new(0., (bottom + top) * 0.5, (s.z - 0.036 - t) * 0.5),
        Vec3::new(s.x - 0.032, height, t),
        0.,
        GrainAxis::X,
    );
    for x in [
        -(s.x - 0.032 - t) * 0.5,
        (s.x - 0.032 - t) * 0.5,
        cx - cx.signum() * core * 0.5,
    ] {
        a.board(
            cuts,
            panel,
            Vec3::new(x, (bottom + top) * 0.5, -0.5 * t),
            Vec3::new(t, height, s.z - 0.036 - t),
            0.,
            GrainAxis::Y,
        );
    }
    let face_z = -s.z * 0.5 + 0.034;
    let rows = if d.form == CheckoutForm::ApparelWrap {
        3
    } else {
        2
    };
    let pitch = height / rows as f32;
    for i in 0..rows {
        let y = bottom + (i as f32 + 0.5) * pitch;
        a.board(
            cuts,
            panel,
            Vec3::new(cx - cx.signum() * 0.008, y, face_z),
            Vec3::new(core - 0.059, pitch - 0.008, 0.020),
            0.002,
            GrainAxis::X,
        );
        board(
            a,
            Surface::Metal,
            COUNTER,
            Vec3::new(cx, y + pitch * 0.27, face_z - 0.016),
            Vec3::new((core * 0.34).min(0.151), 0.012, 0.012),
        );
    }
    let shelf_y = bottom + height * 0.44;
    a.board(
        cuts,
        panel,
        Vec3::new(wx + cx.signum() * 0.008, shelf_y, 0.),
        Vec3::new(work - 0.059, t, s.z - 0.10),
        0.,
        GrainAxis::X,
    );
    if d.wrapping_roll {
        let length = work - 0.134;
        let r = (height * 0.17).min(0.113);
        let center = Vec3::new(wx, shelf_y + t * 0.5 + r + 0.028, -s.z * 0.16);
        let axis = Quat::from_rotation_z(-FRAC_PI_2);
        a.part(Surface::Paper, COUNTER).lathe(
            &[
                (0., -length * 0.5),
                (r, -length * 0.5),
                (r, length * 0.5),
                (0., length * 0.5),
            ],
            16,
            Transform::from_translation(center).with_rotation(axis),
        );
        // Short opposed trunnions span the gap to the cabinet side walls.
        for side in [-1., 1.] {
            let start = center + Vec3::X * (side * length * 0.5);
            a.part(Surface::Metal, COUNTER).rod(
                start,
                center + Vec3::X * (side * (work * 0.5 - 0.010)),
                0.012,
            );
        }
    }
}

pub(super) fn bag_holder(a: &mut Assembly, x: f32, width: f32, height: f32, depth: f32) {
    let rack = a.part(Surface::Chrome, COUNTER);
    let reach = width * 0.31;
    let y = height + (width * 0.49).clamp(0.18, 0.29);
    for side in [-1., 1.] {
        let post = x + side * reach;
        rack.tube(
            &[
                Vec3::new(post, height, -depth * 0.30),
                Vec3::new(post, y - 0.021, -depth * 0.30),
                Vec3::new(post, y, -depth * 0.27),
                Vec3::new(post, y, depth * 0.16),
                Vec3::new(post, y - 0.018, depth * 0.19),
            ],
            0.009,
            8,
        );
    }
}

// A scanner window replaces a section of the load plate. There is no second
// opaque face above or behind an otherwise coplanar sheet.
pub(super) fn scanner(a: &mut Assembly, x: f32, w: f32, h: f32, d: f32) {
    let window_w = (w * 0.54).min(0.235);
    let window_d = (d * 0.25).min(0.170);
    let z = d * 0.045;
    let y = h - 0.0175;
    for side in [-1., 1.] {
        let wing = (w - window_w) * 0.5;
        board(
            a,
            Surface::Metal,
            COUNTER,
            Vec3::new(x + side * (window_w + wing) * 0.5, y, 0.),
            Vec3::new(wing, 0.035, d),
        );
        let edge = side * d * 0.5;
        let lip = z + side * window_d * 0.5;
        board(
            a,
            Surface::Metal,
            COUNTER,
            Vec3::new(x, y, (edge + lip) * 0.5),
            Vec3::new(window_w, 0.035, (edge - lip).abs()),
        );
    }
    // Smoked optical glass is backed by the scanner; its optical return is opaque.
    board(
        a,
        Surface::Ink,
        point_of_sale::stock::SCANNER,
        Vec3::new(x, h - 0.0125 - 0.0015, z),
        Vec3::new(window_w, 0.025, window_d),
    );
}

fn panel(
    a: &mut Assembly,
    tf: Transform,
    size: Vec3,
    bezel: f32,
    housing: (Surface, &str),
    label: &str,
) {
    let half = size * 0.5;
    let rect = |w: f32, h: f32, z: f32| {
        [(-w, -h), (w, -h), (w, h), (-w, h)].map(|(x, y)| tf.transform_point(Vec3::new(x, y, z)))
    };
    let front = rect(half.x, half.y, half.z);
    let back = rect(half.x, half.y, -half.z);
    let display = rect(half.x - bezel, half.y - bezel, half.z);
    let normal = tf.rotation * Vec3::Z;
    let shell = a.part(housing.0, housing.1);
    shell.metric_quad(back, -normal);
    for (i, out) in [-Vec3::Y, Vec3::X, Vec3::Y, -Vec3::X]
        .into_iter()
        .enumerate()
    {
        let j = (i + 1) % 4;
        shell.metric_quad([front[i], front[j], display[j], display[i]], normal);
        shell.metric_quad([back[i], back[j], front[j], front[i]], tf.rotation * out);
    }
    a.part(Surface::Ink, label).vertex_quad(
        display,
        [normal; 4],
        [[0., 1.], [1., 1.], [1., 0.], [0., 0.]],
    );
}

pub(super) fn terminal(a: &mut Assembly, o: &IndoorObject, d: CheckoutStation) {
    let kit = Housing::for_object(o);
    let label = kit.component_case_label(CONTROLS);
    let shell = (kit.surface, label.as_str());
    let s = o.size;
    let [(x, core), _] = d.bays(s.x);
    let top = d.worktop_height_m;
    let tf = d.display(s);
    let mast = Vec3::new(x, top, -s.z * 0.256);
    let bottom = tf.transform_point(Vec3::new(0., -d.screen_height_m * 0.5, -0.003));
    board(
        a,
        Surface::Metal,
        COUNTER,
        mast + Vec3::Y * 0.009,
        Vec3::new(core.min(0.24) * 0.67, 0.018, s.z * 0.16),
    );
    a.part(Surface::Metal, COUNTER)
        .rod(mast + Vec3::Y * 0.018, bottom, 0.026);
    panel(
        a,
        tf,
        Vec3::new(d.screen_width_m, d.screen_height_m, DISPLAY_DEPTH),
        0.012,
        shell,
        &point_of_sale::label(o, d.form),
    );
    let reader = Vec3::new(x + core * 0.19, top, s.z * 0.34);
    let tilt = Quat::from_rotation_x(-0.86);
    let center = reader + Vec3::Y * 0.083;
    let height: f32 = 0.135;
    let thick: f32 = 0.033;
    let cradle = Transform::from_translation(center).with_rotation(tilt);
    board(
        a,
        Surface::Rubber,
        CONTROLS,
        reader + Vec3::Y * 0.006,
        Vec3::new(0.072, 0.012, 0.070),
    );
    let socket = cradle.transform_point(Vec3::new(0., -0.018, -thick * 0.5));
    a.part(Surface::Plastic, CONTROLS)
        .rod(reader + Vec3::Y * 0.012, socket, 0.016);
    // Payment display and physical PIN pad share a single filtered face.
    panel(
        a,
        cradle,
        Vec3::new(0.086, height, thick),
        0.006,
        shell,
        point_of_sale::READER,
    );
}
