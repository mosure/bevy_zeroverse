//! Portable distribution strips. Socket cells, chassis and routed cable ports
//! share one sampled program; electrical layout is coherent within a product.
use super::*;
mod deck;
mod legacy;
pub use deck::SocketDeck;

#[derive(Debug, Clone, serde::Serialize)]
pub struct StripProgram {
    pub height_fraction: f32,
    pub bevel_fraction: f32,
    pub socket_span: f32,
    pub socket_offset: f32,
    pub socket_count: usize,
    pub rows: usize,
    pub socket_yaw: f32,
    pub plate_fraction: f32,
    pub switch: bool,
    pub housing: Surface,
    pub plate: Surface,
    pub end_caps: bool,
    /// Present on products with metric, recessed contacts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deck: Option<SocketDeck>,
}

/// Sample the product before placement, including compact twin-row units.
pub(crate) fn dimensions(r: &mut impl Rng) -> Vec3 {
    if r.random_bool(0.30) {
        Vec3::new(
            r.random_range(0.17..0.30),
            r.random_range(0.029..0.052),
            r.random_range(0.080..0.13),
        )
    } else {
        Vec3::new(
            r.random_range(0.21..0.46),
            r.random_range(0.025..0.046),
            r.random_range(0.048..0.076),
        )
    }
}

pub fn parameters(o: &IndoorObject) -> StripProgram {
    let mut r = stream(o.seed, 0x504f574552535452);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x535452495046494e);
    let rows = if o.size.z >= 0.073 && r.random_bool(0.72) {
        2
    } else {
        1
    };
    let columns = r.random_range(2..=((o.size.x / 0.050).floor() as usize).clamp(2, 8));
    let mut p = StripProgram {
        height_fraction: r.random_range(0.62..0.88),
        bevel_fraction: r.random_range(0.07..0.22),
        socket_span: r.random_range(0.62..0.80),
        socket_offset: r.random_range(-0.035..0.025),
        socket_count: columns * rows,
        rows,
        // Product-level rotation, not implausibly unrelated pin angles.
        socket_yaw: [0., FRAC_PI_2, PI * 0.25, -PI * 0.25][r.random_range(0..4)],
        plate_fraction: r.random_range(0.76..0.90),
        switch: r.random_bool(0.66),
        housing: if f.random_bool(0.30) {
            Surface::Metal
        } else {
            Surface::Plastic
        },
        plate: if f.random_bool(0.30) {
            Surface::Rubber
        } else {
            Surface::Plastic
        },
        end_caps: r.random_bool(0.48),
        deck: None,
    };
    if let Some(face) = super::super::wall::OutletFace::from_variant(o.variant) {
        p.fit_deck(o, face);
    }
    p
}
impl StripProgram {
    pub fn socket_center(&self, s: Vec3, index: usize) -> Vec2 {
        let columns = self.socket_count / self.rows;
        if let Some(deck) = self.deck {
            return Vec2::new(
                s.x * self.socket_offset
                    + ((index % columns) as f32 - (columns - 1) as f32 * 0.5) * deck.column_pitch_m,
                ((index / columns) as f32 - (self.rows - 1) as f32 * 0.5) * deck.row_pitch_m,
            );
        }
        Vec2::new(
            s.x * (self.socket_offset
                + self.socket_span * (((index % columns) as f32 + 0.5) / columns as f32 - 0.5)),
            s.z * (((index / columns) as f32 + 0.5) / self.rows as f32 - 0.5),
        )
    }
    fn plate_width(&self, s: Vec3) -> f32 {
        (s.x * self.socket_span / (self.socket_count / self.rows) as f32)
            .min(s.z / self.rows as f32)
            * self.plate_fraction
    }
    pub fn mains_port(&self, size: Vec3, index: usize) -> Option<Vec3> {
        if index >= self.socket_count {
            return None;
        }
        let center = self.socket_center(size, index);
        Some(Vec3::new(
            center.x,
            size.y * self.height_fraction + self.deck.map_or(0.002, |d| d.lip_m - d.well_depth_m),
            center.y,
        ))
    }

    pub fn output_port(&self, s: Vec3, low_voltage: bool) -> Vec3 {
        if low_voltage {
            Vec3::new(
                s.x * 0.35,
                s.y * self.height_fraction * 0.5,
                s.z * 0.5 + 0.001,
            )
        } else {
            self.mains_port(s, 0).expect("power strip has a receptacle")
        }
    }
    pub fn inlet(&self, s: Vec3) -> Vec3 {
        Vec3::new(-s.x * 0.5 - 0.001, s.y * self.height_fraction * 0.5, 0.)
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let l = o.kind.class_name();
    let h = s.y * p.height_fraction;
    let width = if let Some(deck) = p.deck {
        deck::build(a, o, &p, deck);
        deck.cut_width_m
    } else {
        legacy::build(a, o, &p);
        p.plate_width(s)
    };
    if p.switch {
        let last = p.socket_center(s, p.socket_count / p.rows - 1);
        let left = last.x + width * 0.5 + 0.002;
        let right = s.x * 0.5 - h * p.bevel_fraction;
        if right - left > 0.009 {
            a.box_part(
                Surface::Rubber,
                l,
                Vec3::new((left + right) * 0.5, h + 0.001, 0.),
                Vec3::new((right - left) * 0.7, 0.002, s.z * 0.36),
                0.001,
            );
        }
    }
    let inlet = p.inlet(s) + Vec3::X * 0.0009;
    a.box_part(
        Surface::Ink,
        l,
        inlet,
        Vec3::new(0.0008, h * 0.25, s.z * 0.24),
        0.0003,
    );
    let pd = p.output_port(s, true) - Vec3::Z * 0.0009;
    a.box_part(Surface::Ink, l, pd, Vec3::new(0.008, 0.003, 0.0008), 0.0005);
}

#[cfg(test)]
mod tests;
