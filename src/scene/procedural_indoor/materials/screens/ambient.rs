//! Clock-only OLED presentation for unattended handsets. Black pixels emit
//! nothing; the underlying panel still reflects the room.
use super::*;

#[derive(Clone, Debug, serde::Serialize)]
pub struct AmbientClock {
    pub stacked: bool,
    center: [f32; 2],
    scale: [i32; 2],
    ink: [u8; 3],
    charge: Option<u8>,
}

pub(super) fn configure(seed: u64, p: &mut ScreenProgram) {
    let mut r = stream(seed, u64::from_le_bytes(*b"IdleOLED"));
    let draw = r.random_range(0..100);
    p.dormant = draw < 44;
    if !(44..62).contains(&draw) {
        return;
    }
    let stacked = r.random_bool(0.42);
    // Cover UVs span the portrait panel. Wider glyphs in texture space undo
    // most of that aspect stretch without a per-handset texture allocation.
    let sx: i32 = if stacked {
        r.random_range(4..=8)
    } else {
        r.random_range(3..=6)
    };
    let sy = ((sx as f32 * r.random_range(0.47..0.62)).round() as i32).max(2);
    let v = r.random_range(180..=246);
    let ink = match r.random_range(0..6) {
        0 => [v, (v as f32 * 0.83) as u8, (v as f32 * 0.58) as u8],
        1 => [(v as f32 * 0.70) as u8, (v as f32 * 0.94) as u8, v],
        _ => [v, v, v],
    };
    p.ambient_clock = Some(AmbientClock {
        stacked,
        center: [r.random_range(0.39..0.61), r.random_range(0.30..0.67)],
        scale: [sx, sy],
        ink,
        charge: r.random_bool(0.47).then(|| r.random_range(9..=100)),
    });
    p.luminance = r.random_range(4.0_f32.ln()..42.0_f32.ln()).exp();
}

impl AmbientClock {
    pub(super) fn pixels(&self, minutes: u32) -> Vec<u8> {
        let mut c = Canvas::new([0; 3]);
        let [sx, sy] = self.scale;
        let width = if self.stacked { 12 } else { 28 } * sx;
        let height = if self.stacked { 20 } else { 9 } * sy;
        let [cx, cy] = self.center.map(|v| (v * N as f32).round() as i32);
        let x = cx - width / 2;
        let y = cy - height / 2;
        let digits = [
            minutes / 600,
            minutes / 60 % 10,
            minutes % 60 / 10,
            minutes % 10,
        ];
        for (i, digit) in digits.into_iter().enumerate() {
            let i = i as i32;
            let (col, row) = if self.stacked {
                ((i % 2) * 7, (i / 2) * 11)
            } else {
                (i * 7 + if i >= 2 { 2 } else { 0 }, 0)
            };
            c.digit_scaled(digit, x + col * sx, y + row * sy, [sx, sy], self.ink);
        }
        if !self.stacked {
            for row in [2, 6] {
                c.rect(x + 14 * sx, y + row * sy, sx, sy, self.ink);
            }
        }
        if let Some(charge) = self.charge {
            let dim = self.ink.map(|v| (v as f32 * 0.63) as u8);
            let by = y + height + 9;
            c.rect(cx - 12, by, 24, 7, dim);
            c.rect(cx - 10, by + 1, 20, 5, [0; 3]);
            c.rect(
                cx - 9,
                by + 2,
                (18 * i32::from(charge) / 100).max(1),
                3,
                dim,
            );
            c.rect(cx + 12, by + 2, 2, 3, dim);
        }
        c.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn handheld_power_states_sample_independently_and_emit_only_on_lit_pixels() {
        let mut counts = [0; 4];
        let mut images = Assets::default();
        for seed in 0..512 {
            let p = parameters_for(seed, DisplayKind::Phone);
            let state = match &p.ambient_clock {
                Some(a) => 1 + usize::from(a.stacked),
                None if p.dormant => 0,
                None => 3,
            };
            counts[state] += 1;
            if let Some(clock) = &p.ambient_clock {
                assert!(!p.dormant && (4.0..42.0).contains(&p.luminance));
                let pixels = clock.pixels(p.minutes);
                let data = pixels.as_chunks::<4>().0;
                assert!(data.iter().all(|p| p[3] == 255));
                let lit = data.iter().filter(|p| p[..3] != [0; 3]).count();
                assert!((80..6600).contains(&lit));
                assert!(data[..256 * 13].iter().all(|p| p[..3] == [0; 3]));
                if counts[state] == 1 {
                    let mut m = StandardMaterial::default();
                    prepare(seed, DisplayKind::Phone).apply(&mut m, &mut images);
                    let texture = images.get(m.emissive_texture.as_ref().unwrap()).unwrap();
                    assert_eq!(&texture.data.as_ref().unwrap()[..pixels.len()], pixels);
                    assert_eq!(texture.texture_descriptor.mip_level_count, 9);
                    assert_ne!(clock.pixels((p.minutes + 1) % 1440), pixels);
                }
            }
        }
        assert!((175..275).contains(&counts[0]));
        assert!(counts[1] > 25 && counts[2] > 20);
        assert!((140..242).contains(&counts[3]));
    }
}
