use super::*;
use crate::scene::procedural_indoor::materials::worktop_finish;

const SINK: &str = "sink";
const TAP: &str = "faucet";

pub(in super::super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = Fountain::for_object(o);
    let s = o.size;
    let back = -0.5 * s.z;
    let join = p.deck_m - p.basin_depth_m - 0.025;
    let plate = if p.form == Form::BottleFiller {
        0.035
    } else {
        0.014
    };
    let cabinet = if p.cabinet == Surface::Metal {
        worktop_finish::BASIN_LABEL
    } else {
        SINK
    };
    a.box_part(
        p.cabinet,
        cabinet,
        Vec3::new(0., 0.5 * s.y, back + 0.5 * plate),
        Vec3::new(s.x * 0.82, s.y, plate),
        0.006,
    );
    if p.form == Form::Bracket {
        // A wall plate carries two braces; the return bend joins the drain to the wall.
        for sign in [-1., 1.] {
            let x = sign * s.x * 0.257;
            a.part(Surface::Metal, worktop_finish::BASIN_LABEL).rod(
                Vec3::new(x, 0.017, back + plate * 0.83),
                Vec3::new(x, join + 0.012, s.z * 0.10),
                0.012,
            );
        }
        a.part(Surface::Chrome, SINK).tube(
            &[
                Vec3::new(0., join + 0.0095, s.z * 0.012),
                Vec3::new(0., 0.060, s.z * 0.012),
                Vec3::new(0., 0.040, -s.z * 0.022),
                Vec3::new(0., 0.037, -s.z * 0.087),
                Vec3::new(0., 0.049, -s.z * 0.145),
                Vec3::new(0., 0.083, -s.z * 0.172),
                Vec3::new(0., 0.095, -s.z * 0.218),
                Vec3::new(0., 0.095, back + plate * 0.6),
            ],
            0.012,
            12,
        );
    } else {
        // The inward-tapered lower apron leaves the front of the unit open below the bowl.
        let half = Vec2::new(s.x * 0.435, s.z * 0.436);
        let tf = Transform::from_xyz(0., 0., -s.z * 0.054);
        a.part(p.cabinet, cabinet).housing(
            &[
                (0., half * Vec2::new(0.77, 0.67)),
                (0.008, half * Vec2::new(0.82, 0.70)),
                (join * 0.62, half * Vec2::new(0.98, 0.98)),
                (join * 0.70, half),
                (join + 0.005, half),
            ],
            p.roundness,
            tf,
        );
        let contact = s.xz() * 0.28;
        a.part(Surface::Rubber, SINK).housing_loop(
            &[
                (join + 0.004, contact),
                (join + 0.011, contact),
                (join + 0.011, contact * 0.85),
                (join + 0.004, contact * 0.85),
                (join + 0.004, contact),
            ],
            p.roundness,
            Transform::from_translation(Fountain::bowl_center(s)),
        );
        let front = tf.translation.z + half.y;
        a.box_part(
            Surface::Rubber,
            TAP,
            Vec3::new(0., join * 0.84, front),
            Vec3::new(s.x * p.push_width_fraction, join * 0.21, 0.011),
            0.005,
        );
    }
    super::basin::build(a, &p, s);
    let face = p.face(s);
    let h = p.guard_length_m;
    // The open bore continues below the flexible guard. The outlet tips inward over the bowl.
    a.part(Surface::Chrome, worktop_finish::TAP_LABEL).lathe(
        &[(0.017, -0.013), (0.021, 0.005), (0.016, h * 0.51)],
        16,
        face,
    );
    a.part(Surface::Rubber, TAP).lathe(
        &[
            (0.016, h * 0.47),
            (0.021, h * 0.76),
            (0.018, h),
            (0.0065, h),
            (0.005, h * 0.90),
            (0.005, h * 0.67),
            (0., h * 0.67),
        ],
        16,
        face,
    );
    if p.form == Form::Bracket {
        let sign = if p.left_bubbler { -1. } else { 1. };
        a.part(Surface::Chrome, TAP).cylinder(
            0.012,
            0.010,
            Transform::from_xyz(sign * 0.44 * s.x, p.deck_m + 0.002, -0.10 * s.z),
        );
    }
    if p.form == Form::BottleFiller {
        let hood_h = 0.067;
        let hood_front = -0.072 * s.z;
        let hood_depth = hood_front - (back + 0.004);
        a.box_part(
            p.cabinet,
            cabinet,
            Vec3::new(
                0.,
                s.y - hood_h * 0.5 - 0.003,
                hood_front - 0.5 * hood_depth,
            ),
            Vec3::new(s.x * 0.79, hood_h, hood_depth),
            0.014,
        );
        let nozzle = Vec3::new(0., s.y - hood_h - 0.003, -s.z * 0.163);
        a.part(Surface::Chrome, worktop_finish::TAP_LABEL).lathe(
            &[
                (0.013, -0.005),
                (0.013, 0.003),
                (0.009, 0.010),
                (0.006, 0.010),
                (0.006, 0.003),
                (0., 0.003),
            ],
            16,
            Transform::from_translation(nozzle).with_rotation(Quat::from_rotation_x(PI)),
        );
        a.box_part(
            Surface::Ink,
            TAP,
            Vec3::new(0., s.y - hood_h - 0.055, back + plate + 0.001),
            Vec3::new(0.041, 0.030, 0.005),
            0.002,
        );
    }
}
