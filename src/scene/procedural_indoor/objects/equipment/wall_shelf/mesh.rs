//! Sweep a paired section across the shelf. The returned lip of a steel shelf
//! is open behind the fascia, so rays from underneath see the actual sheet.
use super::*;
use crate::scene::procedural_indoor::geometry::stock::TimberCuts;

#[derive(Clone, Copy)]
struct Section {
    front: bool,
    // (Z from the named edge, local Y). Each pair spans only solid material.
    skins: [Vec2; 2],
    // (normal Z, normal Y) at a radiused bend; square cuts retain their crease.
    normals: Option<[Vec2; 2]>,
}

impl Section {
    fn new(front: bool, outer: Vec2, inner: Vec2) -> Self {
        Self {
            front,
            skins: [outer, inner],
            normals: None,
        }
    }
    fn at(self, x: f32, rear: f32, front: f32) -> [Vec3; 2] {
        let z = if self.front { front } else { rear };
        self.skins.map(|p| Vec3::new(x, p.y, z + p.x))
    }
}

const QUARTER: [(f32, f32); 5] = [
    (0., 1.),
    (0.38268343, 0.9238795),
    (
        std::f32::consts::FRAC_1_SQRT_2,
        std::f32::consts::FRAC_1_SQRT_2,
    ),
    (0.9238795, 0.38268343),
    (1., 0.),
];

fn section(d: &ShelfDeck, top: f32) -> Vec<Section> {
    let bottom = top - d.depth_m;
    let mut rows = Vec::with_capacity(14);
    if d.stock == ShelfStock::FoldedSteel {
        let r = d.nose_m;
        let g = d.gauge_m;
        rows.push(Section::new(
            false,
            Vec2::new(0., bottom),
            Vec2::new(g, bottom),
        ));
        rows.push(Section::new(
            false,
            Vec2::new(0., top),
            Vec2::new(g, top - g),
        ));
        // Follow the outside of the top fold, vertical fascia, then its
        // lower return. The inner bend has the same centre and a smaller radius.
        for lower in [false, true] {
            for (sin, cos) in QUARTER {
                let n = if lower {
                    Vec2::new(cos, -sin)
                } else {
                    Vec2::new(sin, cos)
                };
                let center = Vec2::new(-r, if lower { bottom + r } else { top - r });
                rows.push(Section {
                    front: true,
                    skins: [center + r * n, center + (r - g) * n],
                    normals: Some([n, -n]),
                });
            }
        }
        let end = -r - d.return_m;
        rows.push(Section::new(
            true,
            Vec2::new(end, bottom),
            Vec2::new(end, bottom + g),
        ));
    } else {
        rows.push(Section::new(
            false,
            Vec2::new(0., top),
            Vec2::new(0., bottom),
        ));
        for (i, (s, c)) in QUARTER.into_iter().enumerate() {
            if d.stock == ShelfStock::BandedPanel && i != 0 && i != QUARTER.len() - 1 {
                continue;
            }
            rows.push(Section {
                front: true,
                skins: [
                    Vec2::new(d.nose_m * (s - 1.), top - d.top_fall_m * (1. - c)),
                    Vec2::new(
                        d.lower_run_m * (s - 1.),
                        bottom + d.bottom_rise_m * (1. - c),
                    ),
                ],
                normals: (d.stock == ShelfStock::RoutedTimber).then_some([
                    Vec2::new(d.top_fall_m * s, d.nose_m * c),
                    Vec2::new(d.bottom_rise_m * s, -d.lower_run_m * c),
                ]),
            });
        }
    }
    rows
}

/// Sweep along increasing X. The two UV coordinates are developed section
/// length and distance along the member; timber fibres and veneer run end to end.
fn sweep(
    skin: &mut Geometry,
    mut trim: Option<&mut Geometry>,
    rows: &[Section],
    rear: f32,
    mut stations: impl Iterator<Item = (f32, f32)>,
    origin: Vec2,
    relieve_sheet_ends: bool,
) {
    let Some(mut left) = stations.next() else {
        return;
    };
    let end_cap = |g: &mut Geometry, (x, front): (f32, f32), out: f32, section: &[Section]| {
        for pair in section.windows(2) {
            let [a, b] = pair[0].at(x, rear, front);
            let [c, d] = pair[1].at(x, rear, front);
            let mut p = [a, b, d, c];
            if (p[1] - p[0]).cross(p[2] - p[0]).x * out < 0. {
                p.reverse();
            }
            g.vertex_quad(
                p,
                [Vec3::X * out; 4],
                p.map(|v| (origin + Vec2::new(v.z, v.y)).to_array()),
            );
        }
    };
    let mut previous = 0;
    for right in stations {
        // A brake-formed fascia has one straight bend line. The shaped ends
        // are cut from the flat sheet and stop at the first tangent of the fold.
        let count = if relieve_sheet_ends && left.1 != right.1 {
            3
        } else {
            rows.len()
        };
        let active = &rows[..count];
        if previous < count {
            end_cap(
                trim.as_deref_mut().unwrap_or(skin),
                left,
                -1.,
                &active[previous.saturating_sub(1)..],
            );
        } else if previous > count {
            end_cap(
                trim.as_deref_mut().unwrap_or(skin),
                left,
                1.,
                &rows[count - 1..previous],
            );
        }
        let slope = (right.1 - left.1) / (right.0 - left.0);
        let mut distances = [[0.; 2]; 2];
        for pair in active.windows(2) {
            let l0 = pair[0].at(left.0, rear, left.1);
            let l1 = pair[1].at(left.0, rear, left.1);
            let r0 = pair[0].at(right.0, rear, right.1);
            let r1 = pair[1].at(right.0, rear, right.1);
            let edge = pair[0].front && pair[1].front;
            let g = if edge {
                trim.as_deref_mut().unwrap_or(skin)
            } else {
                &mut *skin
            };
            for side in 0..2 {
                let mut p = [l0[side], l1[side], r1[side], r0[side]];
                let sign = if side == 0 { 1. } else { -1. };
                let flat = sign * (p[1] - p[0]).cross(p[3] - p[0]).normalize();
                let normal = |s: Section| {
                    s.normals.map_or(flat, |n| {
                        let n = n[side];
                        Vec3::new(-n.x * slope, n.y, n.x).normalize()
                    })
                };
                let mut normals = [
                    normal(pair[0]),
                    normal(pair[1]),
                    normal(pair[1]),
                    normal(pair[0]),
                ];
                let [a, b] = [distances[0][side], distances[1][side]];
                distances[0][side] += p[0].distance(p[1]);
                distances[1][side] += p[3].distance(p[2]);
                let mut uv = [a, distances[0][side], distances[1][side], b];
                if side == 1 {
                    p.reverse();
                    normals.reverse();
                    uv.reverse();
                }
                g.vertex_quad(
                    p,
                    normals,
                    std::array::from_fn(|i| (origin + Vec2::new(uv[i], -p[i].x)).to_array()),
                );
            }
        }
        for (i, sign) in [(0, -1.), (count - 1, 1.)] {
            let [a, b] = rows[i].at(left.0, rear, left.1);
            let [c, d] = rows[i].at(right.0, rear, right.1);
            let section = b - a;
            let g = if i == 0 {
                &mut *skin
            } else {
                trim.as_deref_mut().unwrap_or(skin)
            };
            g.stock_quad(
                [a, b, d, c],
                sign * Vec3::new(0., section.z, -section.y),
                Vec3::X,
                Some(origin),
            );
        }
        previous = count;
        left = right;
    }
    end_cap(trim.unwrap_or(skin), left, 1., &rows[..previous]);
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &ShelfProgram, d: &ShelfDeck) {
    let s = o.size;
    let half = s.xz() * 0.5;
    let label = o.kind.class_name();
    let mut cuts = TimberCuts::new(o.seed, u64::from_le_bytes(*b"ShelfUVs"));
    let mut band = Geometry::default();
    sweep(
        a.part(p.board, label),
        p.edging.map(|_| &mut band),
        &section(d, s.y),
        -half.y,
        d.stations(half),
        cuts.origin(),
        d.stock == ShelfStock::FoldedSteel,
    );
    if let Some(edge) = p.edging {
        a.part(edge, label).append(band);
    }
    if p.bracket_x_m.is_empty() {
        return;
    }
    let g = a.part(p.hardware, label);
    for &x in &p.bracket_x_m {
        bracket(g, p, d, s, x);
    }
}

fn bracket(g: &mut Geometry, p: &ShelfProgram, d: &ShelfDeck, s: Vec3, x: f32) {
    let t = p.strap_thickness_m;
    let w = 0.5 * p.strap_width_m;
    let half = s.xz() * 0.5;
    let rear = -half.y
        + if d.stock == ShelfStock::FoldedSteel {
            d.gauge_m * 0.65
        } else {
            0.
        };
    let under = s.y - d.gauge_m;
    let contact = under + (0.001_f32).min(d.gauge_m * 0.20);
    let front = d.front_at(half, x - w).min(d.front_at(half, x + w));
    let end = (-half.y + s.z * p.reach_fraction).min(front - d.lower_run_m - d.return_m - 0.003);
    // One bent L strap. Its bearing intersects the underside of the deck,
    // clear of the folded hem, without an internal pair of coplanar caps.
    let bend = t * 2.;
    let mut path = [Section::new(false, Vec2::ZERO, Vec2::new(t, 0.)); 7];
    for (row, (sin, cos)) in path[1..6].iter_mut().zip(QUARTER) {
        let n = Vec2::new(-cos, sin);
        let center = Vec2::new(bend, contact - bend);
        *row = Section {
            front: false,
            skins: [center + bend * n, center + (bend - t) * n],
            normals: Some([n, -n]),
        };
    }
    path[6] = Section::new(true, Vec2::new(0., contact), Vec2::new(0., contact - t));
    sweep(
        g,
        None,
        &path,
        rear,
        [(x - w, end), (x + w, end)].into_iter(),
        Vec2::ZERO,
        false,
    );
    let r = p.brace_radius_m;
    if p.style != 1 {
        let from = Vec3::new(x, under * 0.095, rear + t * 0.58);
        let to = Vec3::new(x, contact - t * 0.48, end - r * 1.6);
        if p.style == 0 {
            g.tube(&[from, to], r, 8);
        } else {
            let points: [Vec3; 7] = std::array::from_fn(|i| {
                let u = i as f32 / 6.;
                from.lerp(to, u) + Vec3::Y * (under * p.brace_bend * u * (1. - u))
            });
            g.tube(&points, r, 8);
        }
    }
    let radius = (0.34 * w).min(0.0045);
    for y in [0.25 * under, 0.73 * under] {
        g.lathe(
            &[
                (radius, -0.0003),
                (radius, 0.0003),
                (radius * 0.64, radius * 0.32),
                (0., radius * 0.37),
            ],
            10,
            Transform::from_xyz(x, y, rear + t).with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
        );
    }
}
