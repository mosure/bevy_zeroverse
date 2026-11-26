//! The jug lip is drawn from the shoulder. A tubular spout replaces a patch
//! of both vessel skins, so its bore actually connects to the chamber.
use super::*;
use std::{f32::consts::TAU, sync::LazyLock};

const SIDES: usize = 32;
type Ring = [Vec3; SIDES];
static CIRCLE: LazyLock<[Vec2; SIDES]> = LazyLock::new(|| {
    std::array::from_fn(|i| {
        let (s, c) = (TAU * i as f32 / SIDES as f32).sin_cos();
        Vec2::new(s, c)
    })
});

pub(super) fn build(a: &mut Assembly, label: &str, p: &KettleProgram, hw: &Hardware) {
    let foot = p.radius * p.foot_fraction;
    if p.base > 0. {
        a.part(Surface::Plastic, label).lathe(
            &[
                (0., 0.),
                (foot * 1.07, 0.),
                (foot * 1.07, p.base * 0.72),
                (foot, p.base),
            ],
            SIDES as u32,
            Transform::IDENTITY,
        );
    } else {
        a.part(p.finish, label)
            .lathe(&[(0., 0.), (foot, 0.)], SIDES as u32, Transform::IDENTITY);
    }
    let g = a.part(p.finish, label);
    let chart_width = TAU * p.radius;
    let floor = p.base + hw.wall_m;
    g.lathe(
        &[(foot - hw.wall_m, floor), (0., floor)],
        SIDES as u32,
        Transform::IDENTITY,
    );
    let [low, throat, high] = p.port_levels(*hw);
    let belly = p.height * p.belly_height;
    let shoulder = p.height * 0.78;
    let high_port = hw.outlet == Outlet::TaperedSpout;
    let pipe = hw.outlet != Outlet::FormedLip;
    let lip_cos = hw.lip_half_angle.cos();
    let lip_weights = CIRCLE.map(|v| {
        let t = ((v.y - lip_cos) / (1. - lip_cos)).clamp(0., 1.);
        t * t * (3. - 2. * t)
    });
    // Use only the five profile stations for a drawn lip. Tubular stock has
    // three more rings at the lower side port, shared by the nozzle.
    let (levels, port_start) = if high_port {
        (
            [
                0.,
                belly,
                low,
                throat,
                high,
                shoulder,
                p.height * 0.96,
                p.height,
            ],
            2,
        )
    } else {
        (
            [
                0.,
                low,
                throat,
                high,
                belly,
                shoulder,
                p.height * 0.96,
                p.height,
            ],
            1,
        )
    };
    let ring = |y: f32, inner: bool| -> Ring {
        let radius = p.radius_at(y);
        let amount = if pipe {
            0.
        } else {
            ((y / p.height - 0.64) / 0.36).clamp(0., 1.).powi(2)
        };
        let wall = if inner { hw.wall_m } else { 0. };
        std::array::from_fn(|i| {
            let v = CIRCLE[i] * (radius - wall);
            let draw = amount * lip_weights[i];
            Vec3::new(
                v.x,
                p.base + y + if y == 0. { wall } else { 0. } + draw * p.height * 0.024,
                v.y + draw * (p.spout_reach.min(p.radius * 1.37) - p.radius * p.neck),
            )
        })
    };
    let mut outer = ring(0., false);
    let mut inner = ring(0., true);
    let mut port_outer = [[Vec3::ZERO; SIDES]; 3];
    let mut port_inner = port_outer;
    let mut last_level = 0.;
    let mut previous = None;
    for (i, &level) in levels.iter().enumerate().skip(1) {
        if !pipe && i < 4 {
            continue;
        }
        let next = ring(level, false);
        let lining = ring(level, true);
        let port = pipe && (port_start + 1..=port_start + 2).contains(&i);
        let outside = g.normals.len();
        band(
            g,
            &outer,
            &next,
            false,
            port,
            [last_level, level],
            chart_width,
        );
        let inside = g.normals.len();
        band(
            g,
            &inner,
            &lining,
            true,
            port,
            [last_level, level],
            chart_width,
        );
        if let Some((a, b)) = previous {
            smooth_joint::<SIDES>(g, a, outside);
            smooth_joint::<SIDES>(g, b, inside);
        }
        previous = Some((outside, inside));
        last_level = level;
        if pipe && (port_start..=port_start + 2).contains(&i) {
            port_outer[i - port_start] = next;
            port_inner[i - port_start] = lining;
        }
        outer = next;
        inner = lining;
    }
    band(
        g,
        &outer,
        &inner,
        false,
        false,
        [p.height, p.height + hw.wall_m],
        chart_width,
    );
    if pipe {
        nozzle(
            g,
            p,
            hw,
            throat,
            port_boundary(&port_outer),
            port_boundary(&port_inner),
        );
    }
}

// Viewed from the outlet: right mid, upper right, across the top and back
// around the bottom. Each edge is an exact edge of the missing shell patch.
fn port_boundary(rows: &[Ring; 3]) -> [Vec3; 12] {
    let [lo, mid, hi] = rows;
    [
        mid[2], hi[2], hi[1], hi[0], hi[31], hi[30], mid[30], lo[30], lo[31], lo[0], lo[1], lo[2],
    ]
}

fn nozzle(
    g: &mut Geometry,
    p: &KettleProgram,
    hw: &Hardware,
    throat: f32,
    mut outer: [Vec3; 12],
    mut inner: [Vec3; 12],
) {
    const C: usize = 12;
    static ROUND: LazyLock<[Vec2; C]> = LazyLock::new(|| {
        std::array::from_fn(|i| {
            let (s, c) = (TAU * i as f32 / C as f32).sin_cos();
            Vec2::new(c, s)
        })
    });
    let root = Vec3::new(0., p.base + throat, p.radius_at(throat));
    let tip = Vec3::new(0., p.base + p.height * p.spout_tip, p.spout_reach);
    // A short level socket clears the entire belly before the tube rises.
    let socket = Vec3::new(0., root.y, p.socket_z(*hw));
    let span = tip - socket;
    let mouth = p.spout_radius
        * if hw.outlet == Outlet::Gooseneck {
            0.65
        } else {
            1.
        };
    let chord = span.y.atan2(span.z);
    let turn = if hw.outlet == Outlet::Gooseneck {
        0.74 + hw.bend * 0.20
    } else {
        0.27 + hw.bend * 0.18
    };
    let elevation = chord + (FRAC_PI_2 - chord) * turn;
    let exit = hw.exit_elevation(chord);
    let (sn, cs) = elevation.sin_cos();
    let (se, ce) = exit.sin_cos();
    // Two circular bends and a tangent length between them. Solve both
    // displacements; the remaining forward travel is the straight socket.
    let beak = if hw.outlet == Outlet::Gooseneck {
        span.z * 0.19 / ce
    } else {
        0.
    };
    let bent_span = span - Vec3::new(0., se, ce) * beak;
    let rise = 1. + ce - 2. * cs;
    let advance = 2. * sn - se;
    let bend_r = ((bent_span.z - bent_span.y * cs / sn) / (advance - rise * cs / sn))
        .min(bent_span.y / rise)
        * 0.86;
    let straight = (bent_span.y - bend_r * rise) / sn;
    let lead = bent_span.z - bend_r * advance - straight * cs;
    let lower = socket + Vec3::Z * lead;
    let shoulder = lower + Vec3::new(0., 1. - cs, sn) * bend_r;
    let upper = shoulder + Vec3::new(0., sn, cs) * straight;
    let length = lead + straight + bend_r * (2. * elevation - exit) + beak;
    // Finish the rectangular triangulation boundary as an oval cut directly
    // on the vessel skin. Only the flared round socket projects from the shell.
    let half_y = (outer[3].y - root.y) * 0.78;
    let half_x = outer[0].x.min(outer[11].x) * 0.78;
    let collar = (mouth * 1.6)
        .min(half_y.min(half_x) * 0.94)
        .min(bend_r * 0.72);
    let taper = if hw.outlet == Outlet::TaperedSpout {
        0.70
    } else {
        0.92
    };
    let outlet = mouth.min(collar * taper);
    let chart_width = TAU * collar;
    for (edge, inside) in [(&mut outer, false), (&mut inner, true)] {
        let wall = if inside { hw.wall_m } else { 0. };
        let oval = ROUND.map(|v| {
            let x = v.x * (half_x - wall * 0.5);
            let y = throat + v.y * (half_y - wall * 0.5);
            let radius = p.radius_at(y) - wall;
            Vec3::new(x, p.base + y, (radius * radius - x * x).sqrt())
        });
        let first = g.normals.len();
        band(g, edge, &oval, inside, false, [0., half_y], chart_width);
        // Match the parent meridian across the infill; UVs stay metric.
        for i in first..g.normals.len() {
            let point = Vec3::from(g.positions[i]);
            let y = point.y - p.base;
            let e = p.height * 0.0002;
            let slope = (p.radius_at(y + e) - p.radius_at(y - e)) / (2. * e);
            let r = point.xz().normalize();
            let normal = Vec3::new(r.x, -slope, r.y).normalize();
            g.normals[i] = (if inside { -normal } else { normal }).to_array();
            g.uvs[i] = [r.x.atan2(r.y) * p.radius, y];
        }
        *edge = oval;
    }
    let mut distance = 0.;
    let mut last_center = root;
    let mut station = |center: Vec3, sine: f32, cosine: f32, progress: f32| {
        let up = Vec3::new(0., cosine, -sine);
        // The inside radius of each bend must remain positive.
        let radius = collar + (outlet - collar) * (progress / length);
        let wall = hw.wall_m.min(radius * 0.35);
        let ring = |r| ROUND.map(|v| center + (Vec3::X * v.x + up * v.y) * r);
        let shell = ring(radius);
        let lining = ring(radius - wall);
        let end = distance + center.distance(last_center);
        band(
            g,
            &outer,
            &shell,
            false,
            false,
            [distance, end],
            chart_width,
        );
        band(
            g,
            &inner,
            &lining,
            true,
            false,
            [distance, end],
            chart_width,
        );
        distance = end;
        last_center = center;
        outer = shell;
        inner = lining;
    };
    station(socket, 0., 1., 0.);
    station(lower, 0., 1., lead);
    let segments = if hw.outlet == Outlet::Gooseneck { 5 } else { 4 };
    for j in 1..=segments {
        let t = elevation * j as f32 / segments as f32;
        let (s, c) = t.sin_cos();
        station(
            lower + Vec3::new(0., 1. - c, s) * bend_r,
            s,
            c,
            lead + bend_r * t,
        );
    }
    let past_stem = lead + bend_r * elevation + straight;
    station(upper, sn, cs, past_stem);
    for j in 1..=segments {
        let t = elevation - (elevation - exit) * j as f32 / segments as f32;
        let (s, c) = t.sin_cos();
        station(
            upper + Vec3::new(0., c - cs, sn - s) * bend_r,
            s,
            c,
            past_stem + bend_r * (elevation - t),
        );
    }
    if beak > 0. {
        station(tip, se, ce, length);
    }
    band(
        g,
        &outer,
        &inner,
        false,
        false,
        [distance, distance + hw.wall_m],
        chart_width,
    );
}

/// A fixed cylindrical chart keeps the finish continuous across changing
/// circumferences. Only the longitudinal seam duplicates its UV coordinate.
fn band<const N: usize>(
    g: &mut Geometry,
    a: &[Vec3; N],
    b: &[Vec3; N],
    inward: bool,
    port: bool,
    v_coord: [f32; 2],
    chart_width: f32,
) {
    let first = g.positions.len() as u32;
    let du = chart_width / N as f32;
    for i in 0..=N {
        let k = i % N;
        let prev = (k + N - 1) % N;
        let next = (k + 1) % N;
        let rise = b[k] - a[k];
        for (ring, v_coord) in [a, b].into_iter().zip(v_coord) {
            let mut n = (ring[next] - ring[prev]).cross(rise).normalize();
            if inward {
                n = -n;
            }
            g.positions.push(ring[k].to_array());
            g.normals.push(n.to_array());
            g.uvs.push([i as f32 * du, v_coord]);
        }
        if i < N {
            let v = first + 2 * i as u32;
            if !port || (2..N - 2).contains(&i) {
                g.indices.extend(if inward {
                    [v, v + 3, v + 2, v, v + 1, v + 3]
                } else {
                    [v, v + 2, v + 3, v, v + 3, v + 1]
                });
            }
        }
    }
}

/// Share the shading normal across adjacent meridian sections of rolled stock.
/// The foot, lip edge and lid retain their own hard normals.
fn smooth_joint<const N: usize>(g: &mut Geometry, lower: usize, upper: usize) {
    for i in 0..=N {
        let a = lower + i * 2 + 1;
        let b = upper + i * 2;
        let n = (Vec3::from(g.normals[a]) + Vec3::from(g.normals[b]))
            .normalize()
            .to_array();
        g.normals[a] = n;
        g.normals[b] = n;
    }
}
