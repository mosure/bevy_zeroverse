use super::*;
use crate::scene::procedural_indoor::{
    geometry::Geometry,
    materials::input_housing::{PIANO_ACCIDENTALS, PIANO_NATURALS},
};

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Keyboard {
    pub keys: usize,
    pub white_keys: usize,
    pub first_midi: usize,
    pub key_pitch_m: f32,
    pub key_depth_m: f32,
    pub x_stand: bool,
    pub cheek_finish: Surface,
}

impl Keyboard {
    pub fn sample(o: &IndoorObject) -> Self {
        let (keys, white_keys, first_midi) = if o.size.x < 0.99 {
            (61, 36, 36)
        } else if o.size.x < 1.19 {
            (73, 43, 24)
        } else {
            (88, 52, 21)
        };
        let mut r = stream(o.seed, u64::from_le_bytes(*b"KeysBody"));
        Self {
            keys,
            white_keys,
            first_midi,
            key_pitch_m: ((o.size.x - 0.053) / white_keys as f32)
                .min(r.random_range(0.0226..0.0237)),
            key_depth_m: r.random_range(0.135..0.164),
            x_stand: r.random_bool(0.60),
            cheek_finish: if r.random_bool(0.34) {
                Surface::Wood
            } else {
                Surface::Plastic
            },
        }
    }
}

pub(in super::super) fn build(a: &mut Assembly, o: &IndoorObject) {
    const LABEL: &str = "musical_instrument";
    let k = Keyboard::sample(o);
    let (w, h, d) = (o.size.x, o.size.y, o.size.z);
    let keybed = h - 0.042;
    let key_width = k.key_pitch_m * k.white_keys as f32;
    let front = d * 0.44;
    let back = -d * 0.42;
    let cabinet_bottom = keybed - 0.078;
    let center = (front + back) * 0.5;
    let depth = front - back;
    a.box_part(
        Surface::Plastic,
        LABEL,
        Vec3::new(0., cabinet_bottom + 0.039, center),
        Vec3::new(w, 0.078, depth),
        0.008,
    );
    let rear_depth = depth - k.key_depth_m - 0.019;
    a.box_part(
        Surface::Metal,
        LABEL,
        Vec3::new(0., (keybed + h - 0.009) * 0.5, back + rear_depth * 0.5),
        Vec3::new(w - 0.013, h - 0.009 - keybed, rear_depth),
        0.005,
    );
    let cheek = (w - key_width) * 0.5;
    for side in [-1., 1.] {
        a.box_part(
            k.cheek_finish,
            LABEL,
            Vec3::new(
                side * (w - cheek) * 0.5,
                keybed + 0.018,
                front - k.key_depth_m * 0.5 - 0.01,
            ),
            Vec3::new(cheek, 0.036, k.key_depth_m + 0.017),
            0.003,
        );
    }
    // Each stock is a single mesh. Keys have a sloped nose and closed sides; their
    // hidden undersides meet the case or keybed. Five quads suffice per key.
    let naturals = a.part(Surface::Plastic, &format!("{LABEL}#finish{PIANO_NATURALS}"));
    let left = -key_width * 0.5;
    let near = front - 0.006;
    let far = near - k.key_depth_m;
    let white_top = keybed + 0.020;
    for i in 0..k.white_keys {
        let x = left + (i as f32 + 0.5) * k.key_pitch_m;
        key(
            naturals,
            Vec3::new(x, keybed, far),
            Vec3::new(k.key_pitch_m - 0.0011, 0.020, k.key_depth_m),
            0.0015,
        );
    }
    let accidentals = a.part(
        Surface::Plastic,
        &format!("{LABEL}#finish{PIANO_ACCIDENTALS}"),
    );
    let mut boundary = 0;
    for midi in k.first_midi..k.first_midi + k.keys {
        if matches!(midi % 12, 1 | 3 | 6 | 8 | 10) {
            key(
                accidentals,
                Vec3::new(
                    left + boundary as f32 * k.key_pitch_m,
                    white_top - 0.0002,
                    far,
                ),
                Vec3::new(k.key_pitch_m * 0.55, 0.015, k.key_depth_m * 0.62),
                0.007,
            );
        } else {
            boundary += 1;
        }
    }
    let mut controls = stream(o.seed, u64::from_le_bytes(*b"KeysCtrl"));
    let panel_y = h - 0.0065;
    let panel_z = back + rear_depth * 0.54;
    let block = if controls.random_bool(0.64) {
        -0.30
    } else {
        0.05
    };
    let buttons = controls.random_range(3..=8);
    // Inset the panel and keys below the maximum specified height, including controls.
    for i in 0..buttons {
        let x = w * block + i as f32 * 0.020;
        a.box_part(
            Surface::Rubber,
            LABEL,
            Vec3::new(x, panel_y, panel_z),
            Vec3::new(0.013, 0.006, 0.016),
            0.002,
        );
    }
    a.part(Surface::Plastic, LABEL).lathe(
        &[(0.012, 0.), (0.012, 0.008), (0.009, 0.011), (0., 0.011)],
        12,
        Transform::from_xyz(w * block - 0.034, h - 0.011, panel_z),
    );
    let frame = a.part(Surface::Metal, LABEL);
    for side in [-1., 1.] {
        if k.x_stand {
            let dx = w * 0.31;
            let z = center + side * depth * 0.22;
            for s in [-1., 1.] {
                frame.rod(
                    Vec3::new(s * dx, 0.028, z + s * 0.012),
                    Vec3::new(-s * dx, cabinet_bottom - 0.009, z + s * 0.012),
                    0.012,
                );
            }
            for height in [0.019, cabinet_bottom - 0.011] {
                frame.rod(
                    Vec3::new(side * dx, height, -d * 0.45),
                    Vec3::new(side * dx, height, d * 0.45),
                    if height < 0.1 { 0.019 } else { 0.012 },
                );
            }
        } else {
            let x = side * w * 0.42;
            frame.cuboid(
                Vec3::new(0.036, cabinet_bottom, depth * 0.76),
                0.006,
                Transform::from_xyz(x, cabinet_bottom * 0.5, center),
            );
        }
    }
    if !k.x_stand {
        frame.cuboid(
            Vec3::new(w * 0.84, 0.09, 0.044),
            0.004,
            Transform::from_xyz(0., 0.045, center),
        );
        let pedals = a.part(Surface::Chrome, LABEL);
        for x in [-0.053, 0., 0.053] {
            pedals.cuboid(
                Vec3::new(0.030, 0.018, 0.095),
                0.005,
                Transform::from_xyz(x, 0.041, center + 0.057)
                    .with_rotation(Quat::from_rotation_x(0.10)),
            );
        }
        a.box_part(
            k.cheek_finish,
            LABEL,
            Vec3::new(0., cabinet_bottom * 0.62, back + 0.08),
            Vec3::new(w * 0.84, 0.105, 0.026),
            0.004,
        );
    }
}

/// A prism with a slight rake towards the pianist. Its local origin is on the back hinge.
fn key(g: &mut Geometry, origin: Vec3, size: Vec3, nose: f32) {
    let p = |x: f32, y: f32, z: f32| origin + Vec3::new(x * size.x, y * size.y, z);
    let l = -0.5;
    let r = 0.5;
    let near = size.z;
    let top = near - nose;
    g.metric_quad(
        [p(l, 1., top), p(r, 1., top), p(r, 1., 0.), p(l, 1., 0.)],
        Vec3::Y,
    );
    g.metric_quad(
        [p(l, 0., near), p(r, 0., near), p(r, 1., top), p(l, 1., top)],
        Vec3::Z,
    );
    g.metric_quad(
        [p(l, 0., 0.), p(l, 0., near), p(l, 1., top), p(l, 1., 0.)],
        Vec3::NEG_X,
    );
    g.metric_quad(
        [p(r, 0., near), p(r, 0., 0.), p(r, 1., 0.), p(r, 1., top)],
        Vec3::X,
    );
    g.metric_quad(
        [p(r, 0., 0.), p(l, 0., 0.), p(l, 1., 0.), p(r, 1., 0.)],
        Vec3::NEG_Z,
    );
}
