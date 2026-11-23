//! The physical rows of a staggered typing block, inverted-T cursor cluster
//! and optional navigation/numeric banks. Z increases toward the operator.
use super::*;
use keycaps::legend::*;

#[derive(Clone, Copy)]
pub(super) struct Key {
    pub at: Vec2,
    pub size: Vec2,
    pub legend: u8,
    pub right: bool,
}

pub(super) fn rows(function: bool) -> f32 {
    if function {
        5.88
    } else {
        5.
    }
}

pub(super) fn visit(columns: usize, function: bool, stagger: f32, mut add: impl FnMut(Key)) {
    let main = rows(function) - 5.;
    let nav = matches!(columns, 18 | 22);
    let mut key = |x, z, w, h, legend, right| {
        add(Key {
            at: Vec2::new(x, z),
            size: Vec2::new(w, h),
            legend,
            right,
        })
    };
    if function {
        key(0., 0., 1., 0.68, ESC, false);
        for i in 0..12 {
            let x = 2. + i as f32 + (i / 4) as f32 * 0.42;
            key(x, 0., 1., 0.68, F1 + i, i >= 4);
        }
    }
    for (row, prefix, width, letters, suffix, end_width, split) in [
        (0., b'`', 1., &b"1234567890-="[..], BACK, 2., 5),
        (
            1.,
            TAB,
            1.5 + stagger,
            &b"QWERTYUIOP[]"[..],
            b'\\',
            1.5 - stagger,
            5,
        ),
        (
            2.,
            CAPS,
            1.75 + stagger,
            &b"ASDFGHJKL;'"[..],
            ENTER,
            2.25 - stagger,
            5,
        ),
        (
            3.,
            SHIFT,
            2.25 + stagger,
            &b"ZXCVBNM,./"[..],
            SHIFT,
            2.75 - stagger,
            5,
        ),
    ] {
        let z = main + row;
        key(0., z, width, 1., prefix, false);
        for (i, &letter) in letters.iter().enumerate() {
            key(width + i as f32, z, 1., 1., letter, i >= split);
        }
        key(15. - end_width, z, end_width, 1., suffix, true);
    }
    let z = main + 4.;
    for (i, code) in [CTRL, SUPER, ALT].into_iter().enumerate() {
        key(i as f32 * 1.25, z, 1.25, 1., code, false);
    }
    key(3.75, z, 6.25, 1., b' ', false);
    if nav {
        for (i, code) in [ALT, SUPER, FN, CTRL].into_iter().enumerate() {
            key(10. + i as f32 * 1.25, z, 1.25, 1., code, true);
        }
        for (i, code) in [INS, HOME, PGUP, DEL, END, PGDN].into_iter().enumerate() {
            key(
                15.35 + (i % 3) as f32,
                main + (i / 3) as f32,
                1.,
                1.,
                code,
                true,
            );
        }
        key(16.35, z - 1., 1., 1., UP, true);
        for (i, code) in [LEFT, DOWN, RIGHT].into_iter().enumerate() {
            key(15.35 + i as f32, z, 1., 1., code, true);
        }
    } else {
        key(10., z, 1., 1., ALT, true);
        key(11., z, 1., 1., FN, true);
        key(12., z, 1., 1., LEFT, true);
        key(13., z, 1., 0.5, UP, true);
        key(13., z + 0.5, 1., 0.5, DOWN, true);
        key(14., z, 1., 1., RIGHT, true);
    }
    if matches!(columns, 19 | 22) {
        let x = units(columns) - 4.;
        for (row, letters) in [b"N/*-".as_slice(), b"789", b"456", b"123"]
            .into_iter()
            .enumerate()
        {
            for (col, &letter) in letters.iter().enumerate() {
                key(
                    x + col as f32,
                    main + row as f32,
                    1.,
                    1.,
                    if letter == b'N' { NUM } else { letter },
                    true,
                );
            }
        }
        key(x + 3., main + 1., 1., 2., b'+', true);
        key(x + 3., main + 3., 1., 2., ENTER, true);
        key(x, z, 2., 1., b'0', true);
        key(x + 2., z, 1., 1., b'.', true);
    }
}
