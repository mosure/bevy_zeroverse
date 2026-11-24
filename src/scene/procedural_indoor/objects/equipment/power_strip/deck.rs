//! Topology and fit of portable receptacles. The chassis has an opening for
//! each cartridge. A shared, metric contact profile closes each opening.
use super::*;
use crate::scene::procedural_indoor::objects::electrical::{
    edge, insert, pierced, Contact, OutletFace, Rect,
};

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct SocketDeck {
    pub face: OutletFace,
    pub contact_scale: f32,
    pub insert_width_m: f32,
    pub cut_width_m: f32,
    pub row_pitch_m: f32,
    pub column_pitch_m: f32,
    pub lip_m: f32,
    pub insert: Surface,
    pub well_depth_m: f32,
}

impl StripProgram {
    pub(super) fn fit_deck(&mut self, o: &IndoorObject, face: OutletFace) {
        let s = o.size;
        let h = s.y * self.height_fraction;
        let mut stock = stream(o.seed, u64::from_le_bytes(*b"StripBay"));
        let core = match face {
            OutletFace::Polarized => 0.031,
            OutletFace::RoundWell => 0.0435,
            OutletFace::ThreePin => 0.034,
        };
        // Pin spacing belongs to the plug standard. Change capacity and carrier
        // orientation first. Only unusually small custom stock scales the pins.
        let available_width = (s.z * 0.94).min(s.x * 0.47);
        let k = stock
            .random_range(0.98_f32..1.02)
            .min(available_width / core);
        // Plastic and ceramic carriers have different amounts of material around
        // a common contact pattern. Trim the skirt to fit before shrinking a pin.
        let skirt = 0.001 + 0.010 * ((self.plate_fraction - 0.76) / 0.14);
        let w = ((core + skirt) * k).min(available_width);
        let lip = (0.00055 * k).min(h * 0.055);
        let joint = 0.0012 * k;
        let bevel = (h * self.bevel_fraction)
            .min((s.z - w - joint) * 0.38)
            .min(s.x * 0.04);
        self.bevel_fraction = bevel / h;
        let depth = s.z - 2. * bevel;
        let (sin, cos) = self.socket_yaw.sin_cos();
        let projected = w * (sin.abs() + cos.abs()) + joint;
        let plug_pitch = match face {
            OutletFace::Polarized => 0.039,
            OutletFace::RoundWell => 0.047,
            OutletFace::ThreePin => 0.052,
        } * k;
        let pitch_for = |width: f32| (width + 0.002 * k).max(plug_pitch);
        let end = if self.end_caps { s.x * 0.055 } else { 0. };
        let left = -s.x * 0.5 + bevel + end + joint;
        let right = -left
            - if self.switch {
                (0.020 * k).min(s.x * 0.09)
            } else {
                0.
            };
        let fits_two_rows = |width| depth >= width + pitch_for(width);
        let length = right - left;
        let fits_pair = |width| length >= width + pitch_for(width);
        let orthogonal = w + joint;
        let cut = if projected > depth
            || (!fits_pair(projected) && fits_pair(orthogonal))
            || (self.rows == 2 && !fits_two_rows(projected) && fits_two_rows(orthogonal))
        {
            self.socket_yaw = if self.socket_yaw < 0. { FRAC_PI_2 } else { 0. };
            orthogonal
        } else {
            projected
        };
        let columns = self.socket_count / self.rows;
        let pitch = pitch_for(cut);
        if !fits_two_rows(cut) {
            self.rows = 1;
        }
        // Only the interval between centers consumes a full plug pitch. Leave
        // room for at least a pair when a product can accept two mains plugs.
        let capacity = |length: f32| 1 + ((length - cut).max(0.) / pitch).floor() as usize;
        let preferred = (self.socket_span * s.x).max(cut).min(length);
        let columns = columns
            .min(capacity(preferred).max(2))
            .min(capacity(length));
        let span = if columns == 1 {
            cut
        } else {
            preferred
                .max(cut + (columns - 1) as f32 * pitch)
                .min(length)
        };
        let column_pitch_m = if columns > 1 {
            (span - cut) / (columns - 1) as f32
        } else {
            0.
        };
        self.socket_span = span / s.x;
        let midpoint = (left + right) * 0.5;
        let drift = ((length - span) * 0.5).max(0.);
        self.socket_offset =
            (self.socket_offset * s.x).clamp(midpoint - drift, midpoint + drift) / s.x;
        self.socket_count = self.rows * columns;
        // For two rows the smallest spacing clears both plugs. The largest
        // keeps the cartridge cuts inside the beveled deck.
        let row_pitch_m = if self.rows == 2 {
            pitch + (depth - cut - pitch).max(0.) * stock.random_range(0.12..0.88)
        } else {
            0.
        };
        self.deck = Some(SocketDeck {
            row_pitch_m,
            column_pitch_m,
            face,
            insert: if stream(
                o.finish_seed.unwrap_or(o.seed),
                u64::from_le_bytes(*b"StripMat"),
            )
            .random_bool(0.13)
            {
                Surface::Ceramic
            } else {
                Surface::Plastic
            },
            contact_scale: k,
            insert_width_m: w,
            cut_width_m: cut,
            lip_m: lip,
            well_depth_m: if face == OutletFace::RoundWell {
                stock.random_range(0.006_f32..0.0105).min(h * 0.52)
            } else {
                0.
            },
        });
    }
}

fn square(width: f32) -> Rect {
    Rect::new(Vec2::ZERO, Vec2::splat(width))
}

/// In socket coordinates, connect an oriented square insert to the chassis-aligned
/// cut. The eight sectors split at every corner for all four mounting angles.
fn cartridge(o: &IndoorObject, p: &StripProgram, d: SocketDeck, rot: Mat2) -> Assembly {
    let mut a = Assembly::default();
    let l = o.kind.class_name();
    let carrier = d.insert;
    let outer = square(d.insert_width_m);
    let face = square(d.insert_width_m - 2. * d.lip_m);
    let g = a.part(p.housing, l);
    let corners = [
        Vec2::X,
        Vec2::ONE,
        Vec2::Y,
        Vec2::new(-1., 1.),
        Vec2::NEG_X,
        -Vec2::ONE,
        Vec2::NEG_Y,
        Vec2::new(1., -1.),
    ];
    for (i, &u) in corners.iter().enumerate() {
        let v = corners[(i + 1) % corners.len()];
        let outline =
            |q: Vec2| (q * (d.cut_width_m * 0.5 / (rot * q).abs().max_element())).extend(0.);
        let rim = |q: Vec2| (q * (d.insert_width_m * 0.5)).extend(0.);
        g.metric_quad([outline(u), outline(v), rim(v), rim(u)], Vec3::Z);
    }
    edge(a.part(carrier, l), outer, 0., face, d.lip_m, false);
    let contact = Contact {
        face: d.face,
        contact_scale: d.contact_scale,
        face_z_m: d.lip_m,
        well_depth_m: d.well_depth_m,
        insert: carrier,
        pin_sides: 8,
        well_sides: 16,
    };
    insert(&mut a, &contact, l, face, &[Vec2::ZERO], None);
    a
}

// A molded bevel, with flush wraparound end caps. Adjacent materials meet at
// shared boundaries, including the long side walls.
fn chassis(o: &IndoorObject, p: &StripProgram, d: SocketDeck) -> Assembly {
    let mut a = Assembly::default();
    let l = o.kind.class_name();
    let s = o.size;
    let h = s.y * p.height_fraction;
    let bevel = h * p.bevel_fraction;
    let mid = Rect::new(Vec2::ZERO, s.xz());
    let top = Rect::new(Vec2::ZERO, s.xz() - Vec2::splat(2. * bevel));
    a.part(p.housing, l)
        .metric_quad(top.corners().map(|v| v.extend(0.)), -Vec3::Z);
    let segments: &[(f32, f32, bool)] = if p.end_caps {
        &[(0., 0.055, true), (0.055, 0.945, false), (0.945, 1., true)]
    } else {
        &[(0., 1., false)]
    };
    for (lower, z0, upper, z1) in [
        (top, 0., mid, bevel),
        (mid, bevel, mid, h - bevel),
        (mid, h - bevel, top, h),
    ] {
        let lo = lower.corners();
        let hi = upper.corners();
        for i in 0..4 {
            let j = (i + 1) % 4;
            let ds = lo[j] - lo[i];
            let outside = Vec3::new(ds.y, -ds.x, 0.);
            let bands = if i % 2 == 0 {
                segments
            } else {
                &[(0., 1., p.end_caps)][..]
            };
            for &(start, end, cap) in bands {
                a.part(if cap { p.plate } else { p.housing }, l)
                    .metric_quad(
                        [
                            lo[i].lerp(lo[j], start).extend(z0),
                            lo[i].lerp(lo[j], end).extend(z0),
                            hi[i].lerp(hi[j], end).extend(z1),
                            hi[i].lerp(hi[j], start).extend(z1),
                        ],
                        outside,
                    );
            }
        }
    }
    let holes: Vec<_> = (0..p.socket_count)
        .map(|i| {
            let c = p.socket_center(s, i);
            Rect::new(Vec2::new(c.x, -c.y), Vec2::splat(d.cut_width_m))
        })
        .collect();
    for &(start, end, cap) in segments {
        let tile = Rect {
            lo: Vec2::new(top.lo.x + start * (top.hi.x - top.lo.x), top.lo.y),
            hi: Vec2::new(top.lo.x + end * (top.hi.x - top.lo.x), top.hi.y),
        };
        if cap {
            tile.face(a.part(p.plate, l), h);
        } else {
            pierced(a.part(p.housing, l), tile, h, &holes);
        }
    }
    a
}

/// Orient one template once per product. Stamping only translates positions
/// and rebases indices. Material batches and instance labels are preserved.
fn floor_frame(a: &mut Assembly, rot: Mat2, height: f32) {
    for g in a.parts.values_mut() {
        let turn = |[x, y, z]: [f32; 3]| {
            let q = rot * Vec2::new(x, y);
            [q.x, z, -q.y]
        };
        for v in &mut g.positions {
            *v = turn(*v);
            v[1] += height;
        }
        for v in &mut g.normals {
            *v = turn(*v);
        }
    }
}

pub(super) fn build(out: &mut Assembly, o: &IndoorObject, p: &StripProgram, d: SocketDeck) {
    let mut case = chassis(o, p, d);
    floor_frame(&mut case, Mat2::IDENTITY, 0.);
    for ((s, label), g) in case.parts {
        out.part(s, &label).append(g);
    }
    let rot = Mat2::from_angle(p.socket_yaw);
    let mut module = cartridge(o, p, d, rot);
    floor_frame(&mut module, rot, o.size.y * p.height_fraction);
    for ((s, label), module) in module.parts {
        let g = out.part(s, &label);
        let n = p.socket_count;
        g.positions.reserve(n * module.positions.len());
        g.normals.reserve(n * module.normals.len());
        g.uvs.reserve(n * module.uvs.len());
        g.indices.reserve(n * module.indices.len());
        for i in 0..n {
            let c = p.socket_center(o.size, i);
            let base = g.positions.len() as u32;
            g.positions.extend(
                module
                    .positions
                    .iter()
                    .map(|&[x, y, z]| [x + c.x, y, z + c.y]),
            );
            g.normals.extend_from_slice(&module.normals);
            g.uvs.extend_from_slice(&module.uvs);
            g.indices.extend(module.indices.iter().map(|i| base + i));
        }
    }
}
