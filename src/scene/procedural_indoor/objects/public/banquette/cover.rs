//! Fitted sewn foam. Stock measurements are in metres on the backrest.
use super::*;

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(tag = "cut", rename_all = "snake_case")]
pub enum Stitch {
    Channels {
        vertical: bool,
        pitch_m: f32,
        offset_m: f32,
    },
    Diamonds {
        pitch_m: Vec2,
        offset_m: Vec2,
        button_radius_m: Option<f32>,
    },
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct SewnBack {
    pub stitch: Stitch,
    pub size_m: Vec2,
    pub foam_m: f32,
    pub tuck_m: f32,
    pub edge_roll_m: f32,
    pub corner_radius_m: f32,
    pub lumbar_height: f32,
}

impl SewnBack {
    pub(super) fn fit(o: &IndoorObject, p: &BanquetteProgram, back: &BackFit) -> Option<Self> {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"BoothSew"));
        let sewn = match p.back {
            0 => 0.55,
            1 | 2 => 0.82,
            _ => return None,
        };
        if !r.random_bool(sewn) {
            return None;
        }
        let reveal = r.random_range(0.003..0.012);
        let size = Vec2::new(back.width, back.height) - Vec2::splat(2. * reveal);
        let pitch = |length: f32, desired: f32, max: usize| {
            length / (length / desired).round().clamp(2., max as f32)
        };
        let stitch = if p.back == 0 {
            let repeat = Vec2::new(
                pitch(size.x, r.random_range(0.18..0.33), 14),
                pitch(size.y, r.random_range(0.15..0.29), 6),
            );
            Stitch::Diamonds {
                pitch_m: repeat,
                offset_m: repeat
                    * Vec2::new(r.random_range(-0.45..0.45), r.random_range(-0.40..0.40)),
                button_radius_m: r.random_bool(0.65).then(|| {
                    r.random_range(0.006_f32..0.011)
                        .min(repeat.min_element() * 0.085)
                }),
            }
        } else {
            let vertical = p.back == 1;
            let repeat = pitch(
                if vertical { size.x } else { size.y },
                r.random_range(0.095..0.25),
                if vertical { 20 } else { 7 },
            );
            Stitch::Channels {
                vertical,
                pitch_m: repeat,
                offset_m: r.random_range(-0.38..0.38) * repeat,
            }
        };
        let foam = p.back_thickness_m - FRAME_DEPTH;
        Some(Self {
            stitch,
            size_m: size,
            foam_m: foam,
            tuck_m: foam * r.random_range(0.29..0.58),
            edge_roll_m: r
                .random_range(0.014_f32..0.037)
                .min(size.min_element() * 0.095),
            corner_radius_m: r
                .random_range(0.012_f32..0.042)
                .min(size.min_element() * 0.11),
            lumbar_height: r.random_range(0.24..0.48),
        })
    }

    pub fn construction(&self) -> &'static str {
        match self.stitch {
            Stitch::Channels { vertical: true, .. } => "vertical_flutes",
            Stitch::Channels {
                vertical: false, ..
            } => "horizontal_flutes",
            Stitch::Diamonds {
                button_radius_m: Some(_),
                ..
            } => "button_tufted",
            Stitch::Diamonds { .. } => "blind_tufted",
        }
    }

    // An anchor lies at each intersection of the two diagonal stitch lines.
    // Rows alternate by half a pitch. The actual cropped edges decide which
    // fasteners fit, including their entire head and the rolled margin.
    pub(super) fn buttons(&self, mut add: impl FnMut(Vec2, f32)) {
        let Stitch::Diamonds {
            pitch_m,
            offset_m,
            button_radius_m: Some(radius),
        } = self.stitch
        else {
            return;
        };
        let limit =
            self.size_m * 0.5 - Vec2::splat(self.edge_roll_m.max(self.corner_radius_m) + radius);
        let rows = (limit.y / pitch_m.y).ceil() as i32 + 1;
        let cols = (limit.x / pitch_m.x).ceil() as i32 + 1;
        for row in -rows..=rows {
            let y = offset_m.y + row as f32 * pitch_m.y;
            if y.abs() > limit.y {
                continue;
            }
            for col in -cols..=cols {
                let x = offset_m.x + (col as f32 + 0.5 * row.rem_euclid(2) as f32) * pitch_m.x;
                if x.abs() <= limit.x {
                    add(Vec2::new(x, y), radius);
                }
            }
        }
    }

    // Depth of the real front skin above its backing. Stitch valleys retain
    // foam above the board. The shoulder and lumbar crown vary independently.
    pub(super) fn depth(&self, uv: Vec2) -> f32 {
        let groove = |cycles: f32| {
            let shoulder = 1. - 2. * (cycles - cycles.round()).abs();
            shoulder * shoulder * shoulder
        };
        let tuck = match self.stitch {
            Stitch::Channels {
                vertical,
                pitch_m,
                offset_m,
            } => groove(((if vertical { uv.x } else { uv.y }) - offset_m) / pitch_m),
            Stitch::Diamonds {
                pitch_m, offset_m, ..
            } => {
                let c = (uv - offset_m) / pitch_m;
                let a = groove(c.x + 0.5 * c.y);
                let b = groove(c.x - 0.5 * c.y);
                0.30 * (a + b) + 0.40 * a * b
            }
        };
        let q = uv.abs() - (self.size_m * 0.5 - self.corner_radius_m);
        let edge_distance =
            self.corner_radius_m - q.max(Vec2::ZERO).length() - q.max_element().min(0.);
        let edge = (edge_distance / self.edge_roll_m).clamp(0., 1.);
        // The stitch draw fades into the perimeter hem, leaving a continuous
        // rounded shoulder even where a deep channel crosses a corner.
        let draw = 0.35 + 0.65 * edge * edge * (3. - 2. * edge);
        let lumbar =
            (1. - ((uv.y / self.size_m.y + 0.5 - self.lumbar_height) * 1.8).powi(2)).max(0.);
        (self.foam_m * (0.87 + 0.10 * lumbar) - self.tuck_m * tuck * draw)
            * (1. - 0.57 * (1. - edge).powi(2))
    }
}
