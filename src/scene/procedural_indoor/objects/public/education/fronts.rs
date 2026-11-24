//! Fitted door leaves. Stock is shared within a manufactured bank, while the
//! working height, hinge and facing are resolved against each actual opening.
use super::*;
use crate::scene::procedural_indoor::materials::{coated_stock, variants};

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub enum Handing {
    Left,
    Right,
    PairedLeft,
    PairedRight,
}
impl Handing {
    /// Positive when the latch is to the right of the hinge.
    pub fn latch_side(self, bay: usize) -> f32 {
        match self {
            Self::Left => 1.,
            Self::Right => -1.,
            Self::PairedLeft | Self::PairedRight => {
                if (bay.is_multiple_of(2)) == matches!(self, Self::PairedLeft) {
                    1.
                } else {
                    -1.
                }
            }
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Fittings {
    pub handing: Handing,
    pub handle_height_m: f32,
    pub tier_handle_fraction: f32,
    pub stile_width_m: f32,
    pub pull_spacing_m: f32,
    pub hinge_radius_m: f32,
    pub coating_lot: usize,
    pub vents: Option<VentStock>,
}

impl Fittings {
    pub(super) fn sample(o: &IndoorObject, front: u8) -> Self {
        let mut r = stream(o.seed, 0x4c4f_434b_444f_4f52);
        let mut color = stream(o.finish_seed.unwrap_or(o.seed), 0x4c4f_434b_434f_4154);
        Self {
            handing: [
                Handing::Left,
                Handing::Right,
                Handing::PairedLeft,
                Handing::PairedRight,
            ][r.random_range(0..4)],
            handle_height_m: r.random_range(0.82..1.17),
            tier_handle_fraction: r.random_range(0.34..0.68),
            stile_width_m: r.random_range(0.029..0.049),
            pull_spacing_m: r.random_range(0.051..0.094),
            hinge_radius_m: r.random_range(0.004..0.006),
            coating_lot: color.random_range(0..variants::COUNT),
            vents: (front == 0).then(|| sheet::VentStock::sample(&mut r)),
        }
    }

    pub(super) fn case_label(&self) -> Option<String> {
        self.vents.as_ref()?;
        Some(format!(
            "locker#finish{}",
            coated_stock::slot(Surface::Metal, self.coating_lot)?
        ))
    }

    pub fn handle_y(&self, lo: f32, hi: f32) -> f32 {
        let h = hi - lo;
        if (lo + 0.18 * h..hi - 0.18 * h).contains(&self.handle_height_m) {
            self.handle_height_m
        } else {
            lo + h * self.tier_handle_fraction
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn door(
        &self,
        a: &mut Assembly,
        o: &IndoorObject,
        p: &LockerProgram,
        bay: usize,
        row: usize,
        (lo, hi): (f32, f32),
        case: &str,
    ) {
        let b = &p.bays[bay];
        let w = b.width - 0.006;
        let h = hi - lo - 0.008;
        let t = p.panel_m;
        let front = -o.size.z * 0.5 + p.body_depth_m;
        let center = Vec3::new(b.left + b.width * 0.5, (lo + hi) * 0.5, front + t * 0.5);
        let stile = self.stile_width_m.min(w * 0.17);
        let mut leaf = Assembly::default();
        if let Some(vents) = self.vents {
            vents.build(leaf.part(p.finish, case), Vec3::new(w, h, t), p.vent_rows);
        } else if p.front == 2 {
            for side in [-1., 1.] {
                leaf.box_part(
                    p.finish,
                    case,
                    Vec3::X * side * ((w - stile) * 0.5),
                    Vec3::new(stile, h, t),
                    0.0015,
                );
                leaf.box_part(
                    p.finish,
                    case,
                    Vec3::Y * side * ((h - stile) * 0.5),
                    Vec3::new(w - 2. * stile, stile, t),
                    0.0015,
                );
            }
            // The glazing edge seats under the rebate on all four sides.
            leaf.box_part(
                Surface::ContainerGlass,
                "locker",
                Vec3::ZERO,
                Vec3::new(w - 2. * stile + 0.003, h - 2. * stile + 0.003, 0.004),
                0.,
            );
        } else {
            leaf.box_part(p.finish, case, Vec3::ZERO, Vec3::new(w, h, t), 0.0025);
        }
        let side = self.handing.latch_side(bay);
        let handle = Vec3::new(
            side * (w - stile) * 0.5,
            self.handle_y(lo, hi) - center.y,
            t * 0.5,
        );
        self.handle(&mut leaf, p, handle);
        // The pin sits in the reveal at the outer face. Its bearing stays fixed
        // on the jamb while the leaf and latch swing away from the shelf.
        let pin = Vec3::new(center.x - side * b.width * 0.5, 0., front + t);
        let radius = self.hinge_radius_m;
        // Neighbouring doors may share a narrow jamb and hinge toward each other.
        // Each bearing flange occupies only its own half of that jamb.
        let flange = (t * 0.44).min(0.009);
        for sign in [-1., 1.] {
            let y = center.y + sign * h * 0.33;
            let hinge = pin.with_y(y);
            a.box_part(
                p.accent,
                "locker",
                hinge + Vec3::new(-side * flange * 0.35, 0., -t * 0.56),
                Vec3::new(flange, 0.021, t * 1.14),
                0.,
            );
            a.part(p.accent, "locker").lathe(
                &[(0., -0.015), (radius, -0.015), (radius, 0.015), (0., 0.015)],
                10,
                Transform::from_translation(hinge),
            );
        }
        let rot = Quat::from_rotation_y(-side * b.angles[row]);
        append_at(
            a,
            leaf,
            Transform::from_translation(pin + rot * (center - pin)).with_rotation(rot),
        );
    }

    fn handle(&self, a: &mut Assembly, p: &LockerProgram, c: Vec3) {
        let label = "locker";
        let metal = a.part(p.accent, label);
        let axle = Transform::from_translation(c).with_rotation(Quat::from_rotation_x(FRAC_PI_2));
        match p.handle {
            0 => {
                // Quarter-turn latch with a flared escutcheon and projecting wing.
                metal.lathe(
                    &[
                        (0.011, 0.),
                        (0.011, 0.002),
                        (0.008, 0.003),
                        (0.008, 0.008),
                        (0., 0.008),
                    ],
                    16,
                    axle,
                );
                a.box_part(
                    p.accent,
                    label,
                    c + Vec3::Z * 0.010,
                    Vec3::new(0.008, 0.033, 0.008),
                    0.002,
                );
            }
            1 => {
                // The pull has finger space behind it and both legs seat on the stile.
                let d = self.pull_spacing_m * 0.5;
                let v = |y, z| c + Vec3::new(0., y, z);
                metal.tube(
                    &[
                        v(-d, -0.001),
                        v(-d, 0.013),
                        v(-d + 0.005, 0.019),
                        v(d - 0.005, 0.019),
                        v(d, 0.013),
                        v(d, -0.001),
                    ],
                    0.0035,
                    8,
                );
            }
            _ => {
                metal.lathe(
                    &[
                        (0.010, 0.),
                        (0.010, 0.003),
                        (0.006, 0.005),
                        (0.006, 0.010),
                        (0.012, 0.015),
                        (0.011, 0.020),
                        (0.007, 0.021),
                        (0., 0.021),
                    ],
                    16,
                    axle,
                );
            }
        }
    }
}
