//! Sheet enclosure with a real porthole in each fascia. There is no solid
//! cuboid behind the glass. The housing shares its opening with the door seal.
use super::*;
use crate::scene::procedural_indoor::materials::appliance_panel::Glyph;
pub(super) const TOE: f32 = 0.047;
pub(super) const SIDES: u32 = 32;

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &Machine) {
    let s = o.size;
    let class = o.kind.class_name();
    use crate::scene::procedural_indoor::materials::{self, coated_stock, variants};
    let steel = worktop_finish::BrushedHardware::Stainless.label(class);
    let (surface, enamel) = match p.finish {
        CabinetFinish::BrushedSteel => (Surface::Metal, None),
        CabinetFinish::WhiteEnamel => (Surface::Metal, Some(materials::WHITE_ENAMEL)),
        CabinetFinish::AlmondEnamel => (Surface::Metal, Some(materials::ALMOND_ENAMEL)),
        CabinetFinish::GraphiteEnamel => (Surface::Metal, Some(materials::GRAPHITE_ENAMEL)),
        CabinetFinish::TintedCoating => {
            let lot = (o.finish_seed.unwrap_or(o.seed) % variants::COUNT as u64) as usize;
            (Surface::Plastic, coated_stock::slot(Surface::Plastic, lot))
        }
    };
    let paint = enamel.map(|slot| format!("{class}#finish{slot}#uv_local"));
    let shell = (surface, paint.as_deref().unwrap_or(&steel));
    let front = p.front_z_m;
    let back = -s.z * 0.5;
    let w = s.x * 0.5;
    let h = s.y;
    a.box_part(
        Surface::Rubber,
        class,
        Vec3::new(0., TOE * 0.5, (back + front) * 0.5),
        Vec3::new(s.x - 0.042, TOE, front - back - 0.043),
        0.005,
    );
    let g = a.part(shell.0, shell.1);
    let rect = |z| {
        [
            Vec3::new(-w, TOE, z),
            Vec3::new(w, TOE, z),
            Vec3::new(w, h, z),
            Vec3::new(-w, h, z),
        ]
    };
    let f = rect(front);
    let b = rect(back);
    g.metric_quad(b, Vec3::NEG_Z);
    for (j, normal) in [Vec3::NEG_Y, Vec3::X, Vec3::Y, Vec3::NEG_X]
        .into_iter()
        .enumerate()
    {
        let k = (j + 1) % 4;
        g.metric_quad([b[j], f[j], f[k], b[k]], normal);
    }
    let module_h = (h - TOE) / p.pockets.len() as f32;
    for (i, drum) in p.pockets.iter().enumerate() {
        let low = TOE + i as f32 * module_h;
        fascia(
            a.part(shell.0, shell.1),
            w,
            low,
            low + module_h,
            drum,
            front,
        );
        drum::build(a, o, p, drum, &steel);
        console(a, o, p, drum, low, module_h);
    }
}

fn fascia(g: &mut Geometry, w: f32, lo: f32, hi: f32, p: &Pocket, z: f32) {
    let point = |i: u32| {
        let (sin, cos) = (TAU * (i % SIDES) as f32 / SIDES as f32).sin_cos();
        let dir = Vec2::new(cos, -sin);
        let edge = dir / dir.abs().max_element();
        let center = Vec2::new(0., p.center_y_m);
        let outer = center
            + edge
                * Vec2::new(
                    w,
                    if edge.y >= 0. {
                        hi - center.y
                    } else {
                        center.y - lo
                    },
                );
        ((center + dir * p.radius_m).extend(z), outer.extend(z))
    };
    let mut old = point(0);
    for i in 1..=SIDES {
        let new = point(i);
        // The cabinet is a single sheet. Keep the brushed finish continuous
        // around the cutout, including where adjacent sectors meet a corner.
        let face = [old.0, new.0, new.1, old.1];
        g.vertex_quad(face, [Vec3::Z; 4], face.map(|v| v.xy().to_array()));
        old = new;
    }
}

fn console(a: &mut Assembly, o: &IndoorObject, m: &Machine, p: &Pocket, base: f32, h: f32) {
    let c = o.kind.class_name();
    let w = o.size.x;
    let z = m.front_z_m;
    let y = base + h * 0.866;
    let panel_w = w * 0.83;
    let high = (h * 0.128).min(0.164);
    a.box_part(
        Surface::Plastic,
        c,
        Vec3::new(0., y, z + 0.0035),
        Vec3::new(panel_w, high, 0.007),
        0.003,
    );
    let lcd = Vec2::new(w * 0.132, high * 0.54);
    let x = 0.013 * w;
    let face = z + 0.008;
    let digits = [p.remaining_minutes / 10, p.remaining_minutes % 10];
    for (i, digit) in digits.into_iter().enumerate() {
        let l = x + (i as f32 - 1.) * lcd.x * 0.5;
        let r = l + lcd.x * 0.5;
        let lo = y - lcd.y * 0.5;
        let hi = y + lcd.y * 0.5;
        a.part(Surface::Ink, &m.film.label(c)).vertex_quad(
            [
                Vec3::new(l, lo, face),
                Vec3::new(r, lo, face),
                Vec3::new(r, hi, face),
                Vec3::new(l, hi, face),
            ],
            [Vec3::Z; 4],
            Glyph::Number(digit).uvs(),
        );
    }
    let tf =
        Transform::from_xyz(w * 0.27, y, z + 0.014).with_rotation(Quat::from_rotation_x(FRAC_PI_2));
    if m.dial {
        a.part(Surface::Rubber, c).cylinder(high * 0.239, 0.014, tf);
        let angle = (p.remaining_minutes as f32 * 0.15 + 0.35).sin() * 2.2;
        let r = high * 0.14;
        a.box_part(
            Surface::Metal,
            c,
            tf.translation + Vec3::new(r * angle.sin(), r * angle.cos(), 0.008),
            Vec3::new(0.0048, 0.0048, 0.002),
            0.0005,
        );
    } else {
        for i in [-1., 0., 1.] {
            a.box_part(
                Surface::Rubber,
                c,
                Vec3::new(w * 0.255 + i * 0.028, y, z + 0.011),
                Vec3::new(0.018, high * 0.26, 0.008),
                0.002,
            );
        }
    }
    if o.kind == ObjectKind::WashingMachine {
        a.box_part(
            Surface::Metal,
            c,
            Vec3::new(-w * 0.257, y, z + 0.0095),
            Vec3::new(w * 0.213, high * 0.69, 0.005),
            0.0015,
        );
        a.box_part(
            Surface::Rubber,
            c,
            Vec3::new(-w * 0.257, y + high * 0.14, z + 0.0145),
            Vec3::new(w * 0.143, 0.012, 0.005),
            0.0015,
        );
    } else {
        // Flush lint drawer below each pocket, with an accessible pull.
        let y = base + h * 0.083;
        a.box_part(
            Surface::Metal,
            c,
            Vec3::new(0., y, z + 0.002),
            Vec3::new(w * 0.57, h * 0.073, 0.004),
            0.001,
        );
        a.box_part(
            Surface::Rubber,
            c,
            Vec3::new(0., y, z + 0.009),
            Vec3::new(w * 0.24, 0.017, 0.010),
            0.002,
        );
    }
}
