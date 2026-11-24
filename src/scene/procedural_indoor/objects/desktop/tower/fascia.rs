use super::*;
use super::{
    face::{flat_field, Plane},
    sockets::{Connector, Socket},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Intake {
    Punched,
    Louvered,
    Fluted,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Bank {
    Header,
    Footer,
    Left,
    Right,
}

#[derive(Debug, serde::Serialize)]
pub(crate) struct Fascia {
    pub bank: Bank,
    pub intake: Intake,
    pub front_face: Rect,
    pub rear_face: Rect,
    pub vents: Vec<Rect>,
    pub controls: Vec<Socket>,
    pub corner_m: f32,
    pub nose_inset_m: f32,
    pub rear_bevel_m: f32,
    pub intake_recess_m: f32,
    pub pitch_m: f32,
    pub vertical: bool,
    pub reverse_folds: bool,
    pub side: i8,
    pub side_fraction: Vec2,
    pub side_offset: Vec2,
}
impl Fascia {
    pub(crate) fn sample(o: &IndoorObject, p: &CaseProgram) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"CaseFace"));
        let outer = Rect::new(-o.size.x * 0.5, p.feet_m, o.size.x * 0.5, o.size.y);
        let edge = outer.size().min_element();
        let nose_inset_m = edge * r.random_range(0.008..0.034);
        let rear_bevel_m = edge * r.random_range(0.005..0.016);
        let corner_m = edge * r.random_range(0.012..0.049);
        let face = flat_field(outer.inflate(-nose_inset_m), corner_m);
        let rear_face = flat_field(outer.inflate(-rear_bevel_m), corner_m);
        let vertical_bank = r.random_bool(if o.size.y < o.size.x * 0.67 {
            0.78
        } else {
            0.37
        });
        let bank = match (
            vertical_bank,
            r.random_bool(if vertical_bank { 0.52 } else { 0.70 }),
        ) {
            (true, true) => Bank::Left,
            (true, false) => Bank::Right,
            (false, true) => Bank::Header,
            (false, false) => Bank::Footer,
        };
        let frame = face.inflate(-(edge * r.random_range(0.032..0.076)));
        let mut vent = frame;
        let mut control = frame;
        let gap = edge * r.random_range(0.018..0.040);
        let join = (frame.size() * (1. - p.intake_fraction)).max(Vec2::splat(gap + 0.012));
        match bank {
            Bank::Left => {
                vent.min.x += join.x;
                control.max.x = vent.min.x - gap;
            }
            Bank::Right => {
                vent.max.x -= join.x;
                control.min.x = vent.max.x + gap;
            }
            Bank::Header => {
                vent.max.y -= join.y;
                control.min.y = vent.max.y + gap;
            }
            Bank::Footer => {
                vent.min.y += join.y;
                control.max.y = vent.min.y - gap;
            }
        }
        let intake = match r.random_range(0..100) {
            0..=41 => Intake::Punched,
            42..=78 => Intake::Louvered,
            _ => Intake::Fluted,
        };
        let mut vents = vec![vent];
        if r.random_bool(0.34) && vent.size().max_element() > 0.075 {
            let t = r.random_range(0.29..0.69);
            let mut a = vent;
            let mut b = vent;
            if vent.width() > vent.height() {
                let split = vent.min.x + t * vent.width();
                a.max.x = split - gap * 0.5;
                b.min.x = split + gap * 0.5;
            } else {
                let split = vent.min.y + t * vent.height();
                a.max.y = split - gap * 0.5;
                b.min.y = split + gap * 0.5;
            }
            vents = vec![a, b];
        }
        let controls = control_bank(control, &mut r);
        Self {
            bank,
            intake,
            front_face: face,
            rear_face,
            vents,
            controls,
            corner_m,
            nose_inset_m,
            rear_bevel_m,
            intake_recess_m: r
                .random_range(0.0020_f32..0.0065)
                .min(p.fascia_depth_m * 0.68),
            pitch_m: p.louver_pitch_m,
            vertical: p.vertical_louvers,
            reverse_folds: r.random_bool(0.5),
            side: if r.random_bool(0.71) {
                if r.random_bool(0.53) {
                    1
                } else {
                    -1
                }
            } else {
                0
            },
            side_fraction: Vec2::new(r.random_range(0.26..0.68), r.random_range(0.22..0.61)),
            side_offset: Vec2::new(r.random_range(0.08..0.89), r.random_range(0.12..0.87)),
        }
    }

    pub(super) fn build(
        &self,
        a: &mut Assembly,
        plane: Plane,
        stock: &Housing,
        case: &str,
        trim: &str,
        grille: &str,
    ) {
        let mut cuts = [Rect::default(); 8];
        for (dst, src) in cuts.iter_mut().zip(
            self.vents
                .iter()
                .copied()
                .chain(self.controls.iter().map(|s| s.aperture)),
        ) {
            *dst = src;
        }
        plane.sheet(
            a.part(stock.surface, case),
            self.front_face,
            &cuts[..self.vents.len() + self.controls.len()],
        );
        for vent in &self.vents {
            plane.walls(a.part(Surface::Plastic, trim), *vent, self.intake_recess_m);
            if self.intake == Intake::Punched {
                plane.rect(a.part(Surface::Metal, grille), *vent, self.intake_recess_m);
            } else {
                self.folded(a, plane, *vent, stock, case);
            }
        }
        for socket in &self.controls {
            socket.build(a, plane, stock, case, trim);
        }
    }

    fn folded(&self, a: &mut Assembly, plane: Plane, v: Rect, stock: &Housing, case: &str) {
        let short = if self.vertical { v.width() } else { v.height() };
        let n = (short / self.pitch_m).round().clamp(2., 28.) as usize;
        let profile = if self.intake == Intake::Louvered {
            [(0., 1.), (0.28, 1.), (0.75, 0.14), (0.89, 0.14), (1., 1.)]
        } else {
            [(0., 1.), (0.20, 1.), (0.35, 0.30), (0.77, 0.30), (1., 1.)]
        };
        let point = |across: f32, along: f32, depth: f32| {
            let t = if self.reverse_folds {
                1. - across
            } else {
                across
            };
            let uv = if self.vertical {
                Vec2::new(t, along)
            } else {
                Vec2::new(along, t)
            };
            plane.point(v.min + v.size() * uv, depth * self.intake_recess_m)
        };
        for i in 0..n {
            for edge in profile.windows(2) {
                let [(t0, d0), (t1, d1)] = [edge[0], edge[1]];
                let (x0, x1) = ((i as f32 + t0) / n as f32, (i as f32 + t1) / n as f32);
                let g = if d0 == 1. && d1 == 1. && self.intake == Intake::Louvered {
                    a.part(Surface::Rubber, "pc_tower")
                } else {
                    a.part(stock.surface, case)
                };
                g.metric_quad(
                    [
                        point(x0, 0., d0),
                        point(x1, 0., d1),
                        point(x1, 1., d1),
                        point(x0, 1., d0),
                    ],
                    plane.normal(),
                );
            }
        }
    }
}

fn control_bank(area: Rect, r: &mut impl Rng) -> Vec<Socket> {
    let across = area.width() > area.height();
    let axis = if across { Vec2::X } else { Vec2::Y };
    let room = area.size().dot(axis);
    let spacing = r.random_range(0.003..0.010);
    let mut kinds = vec![
        Connector::Button,
        if r.random_bool(0.65) {
            Connector::UsbA
        } else {
            Connector::UsbC
        },
    ];
    if r.random_bool(0.64) {
        kinds.push(Connector::UsbA);
    }
    if r.random_bool(0.60) {
        kinds.push(Connector::Audio);
    }
    if r.random_bool(0.28) {
        kinds.push(Connector::UsbC);
    }
    let reverse = r.random_bool(0.45);
    let button = Vec2::new(r.random_range(0.007..0.012), r.random_range(0.0065..0.010));
    let mut sockets = Vec::with_capacity(kinds.len());
    let mut used = 0.;
    for k in kinds {
        let size = if k == Connector::Button {
            button
        } else {
            k.size()
        };
        let width = size.dot(axis);
        let start = if sockets.is_empty() {
            0.
        } else {
            used + spacing
        };
        if start + width > room || size.dot(Vec2::ONE - axis) > area.size().dot(Vec2::ONE - axis) {
            continue;
        }
        sockets.push(Socket {
            connector: k,
            aperture: Rect::from_center_size(axis * (start + width * 0.5), size),
        });
        used = start + width;
    }
    let along = (room - used) * r.random_range(0.06..0.94);
    let cross = Vec2::ONE - axis;
    let width = sockets
        .iter()
        .map(|s| s.aperture.size().dot(cross))
        .fold(0., f32::max);
    let offset = area.min
        + axis * along
        + cross * (width * 0.5 + (area.size().dot(cross) - width) * r.random_range(0.12..0.88));
    for s in &mut sockets {
        let c = s.aperture.center().dot(axis);
        let pos = offset + axis * if reverse { used - c } else { c };
        s.aperture = Rect::from_center_size(pos, s.aperture.size());
    }
    sockets
}
