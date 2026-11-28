//! Unframed painting surfaces. Each occupied easel owns one image. Still lives
//! use the same vessel meridian as the object on the subject table.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{IndoorObject, ObjectKind},
    objects::{
        public::easel::{Canvas as Work, Easel, VERSION},
        vessels,
    },
};
use std::collections::BTreeMap;

// Ordinary art recipes use a small shared palette. Installed canvases reserve
// the tail of the same sparse finish map so two artists can paint different work.
const FIRST: usize = 3200;

pub(crate) fn label(id: usize) -> String {
    match FIRST.checked_add(id) {
        Some(slot) => format!("picture#uv_local#finish{slot}"),
        None => "picture#uv_local".into(),
    }
}
pub(in super::super) fn supports(surface: Surface, slot: usize) -> bool {
    surface == Surface::Art && slot >= FIRST
}

pub(in super::super) async fn install(
    scene: &IndoorManifest,
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
    cooperative: bool,
) {
    if scene.generator_version < VERSION
        || selection.is_some_and(|s| {
            s.finishes
                .range((Surface::Art, FIRST)..=(Surface::Art, usize::MAX))
                .next()
                .is_none()
        })
    {
        return;
    }
    for o in scene
        .objects
        .iter()
        .filter(|o| o.kind == ObjectKind::ArtistEasel)
    {
        let Some(slot) = FIRST.checked_add(o.id) else {
            continue;
        };
        if selection.is_some_and(|s| !s.finishes.contains(&(Surface::Art, slot))) {
            continue;
        }
        let board = Easel::sample(o);
        if board.canvas != Work::Painted {
            continue;
        }
        let subject = o
            .interaction_target
            .and_then(|id| scene.objects.get(id))
            .filter(|subject| scene.valid_easel_subject(o, subject));
        let seed = stream(
            scene.material_seed() ^ o.seed,
            u64::from_le_bytes(*b"WetPaint"),
        )
        .random();
        let mut finish = ArtFinish::sample(seed);
        finish.canvas = finish.canvas.max(0.62);
        let color = pixels(seed, o, &board, subject);
        let maps = finish.maps(seed, color);
        let mut material = StandardMaterial::default();
        bind(&finish, maps, &mut material, images);
        variants.insert((Surface::Art, slot), materials.add(material));
        if cooperative {
            crate::scene::procedural_indoor::preparation::cooperate().await;
        }
    }
}

fn pixels(seed: u64, o: &IndoorObject, board: &Easel, subject: Option<&IndoorObject>) -> Vec<u8> {
    let p = parameters(seed);
    if let Some(subject) = subject {
        return still_life(seed, o, board, subject).0;
    }
    let mut r = stream(seed, u64::from_le_bytes(*b"PleinAir"));
    if r.random_bool(0.67) {
        return landscape(seed, &p, r.random()).0;
    }
    // Broad marks follow a common gesture but change direction, weight and
    // pigment. The canvas edges crop the composition as a painter would.
    let colors = p.composition.palette;
    let mut c = Canvas::new(colors[0]);
    let field = r.random_range(-1.8..1.8);
    let focus = Vec2::new(p.viewpoint, p.horizon) * N as f32;
    for i in 0..r.random_range(28..74) {
        let a = Vec2::new(r.random_range(-16.0..270.), r.random_range(-16.0..270.));
        let angle = (a.y - focus.y).atan2(a.x - focus.x) + field + r.random_range(-0.42..0.42);
        let direction = Vec2::from_angle(angle);
        let reach = r.random_range(22.0..128.);
        let bend = r.random_range(-0.58..0.58) * reach;
        let width = r.random_range(3..23);
        let pigment = colors[1 + i % 3];
        let mut prev = a;
        for j in 1..=12 {
            let t = j as f32 / 12.;
            let v = a + direction * reach * t + direction.perp() * bend * (t * (1. - t) * 4.);
            c.ink_stroke(
                prev.to_array(),
                v.to_array(),
                ((0.22 + 0.78 * (std::f32::consts::PI * t).sin()) * width as f32)
                    .round()
                    .max(1.) as i32,
                pigment,
            );
            prev = v;
        }
    }
    c.0
}

fn still_life(seed: u64, easel: &IndoorObject, board: &Easel, subject: &IndoorObject) -> Canvas {
    let palettes = [
        [[193, 178, 151], [143, 101, 78], [174, 116, 72]],
        [[166, 172, 158], [112, 103, 95], [99, 131, 137]],
        [[178, 165, 178], [133, 114, 120], [186, 164, 123]],
        [[211, 196, 157], [150, 126, 91], [101, 111, 89]],
        [[162, 163, 173], [107, 99, 98], [179, 138, 113]],
    ];
    let mut r = stream(seed, u64::from_le_bytes(*b"OilStudy"));
    let [wall, table, pigment] = palettes[r.random_range(0..palettes.len())]
        .map(|c: [u8; 3]| Vec3::from_array(c.map(f32::from)));
    let ink = |c: Vec3| {
        c.clamp(Vec3::ZERO, Vec3::splat(255.))
            .to_array()
            .map(|v| v.round() as u8)
    };
    let n = N as f32;
    let vessel = vessels::parameters(subject);
    let knots = vessel.knots();
    let bowl = subject.kind == ObjectKind::Bowl;
    let aspect = board.canvas_height_m / board.canvas_width_m;
    let width_to_height = 0.94 * subject.size.x.min(subject.size.z) * aspect / subject.size.y;
    let height = n * r.random_range(0.42..0.73_f32).min(0.72 / width_to_height);
    let rx = 0.5 * height * width_to_height;
    let cx = n * r.random_range(0.38..0.62);
    let foot = n * r.random_range(0.73..0.86);
    let horizon = foot - height * r.random_range(0.20..0.54);
    let tilt = r.random_range(-0.05..0.05);
    let direction = if r.random_bool(0.5) { 1. } else { -1. };
    let eye = easel.position
        + easel.transform().rotation * Vec3::Z * (easel.size.z * 0.5 + 0.38)
        + Vec3::Y * 1.52;
    let top = subject.position + Vec3::Y * subject.size.y;
    let mouth_tilt = ((eye - top).normalize_or_zero().y / aspect).clamp(0.05, 0.54);
    let mut canvas = Canvas::new(ink(wall));
    let bristles = field::PreparedPeriodic::new([23, 71], seed ^ 0x5041494e54, &mut 12000);
    let brush = bristles.raster::<{ N as usize }>();
    for y in 0..N {
        let level = y as f32 + 0.5;
        let t = ((foot - level) / height).clamp(0., 1.);
        let radius = rx * vessel.radius_fraction(bowl, &knots, t);
        let grain = brush.row(y as usize);
        let row = &mut canvas.0[(y * N * 4) as usize..((y + 1) * N * 4) as usize];
        for (x, px) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let dx = x as f32 + 0.5 - cx;
            let line = horizon + tilt * dx;
            let ground = if level > line { table } else { wall };
            let ambient = 0.96 + 0.10 * (1. - level / n) - direction * dx / n * 0.09;
            let shadow = ((dx - direction * rx * 0.53) / (rx * 1.4)).powi(2)
                + ((level - foot - 1.2) / (rx * 0.14 + 2.)).powi(2);
            let bg = ground * ambient * (1. - 0.26 * (1. - shadow).clamp(0., 1.));
            let u = (dx / radius).clamp(-1., 1.);
            let light = (1. - u * u).sqrt() * 0.63 - direction * u * 0.46;
            let paint = pigment * (0.61 + 0.49 * light.max(-0.37));
            let cover = (radius - dx.abs() + 0.5).clamp(0., 1.)
                * (foot - level + 0.5).clamp(0., 1.)
                * (level - (foot - height) + 0.5).clamp(0., 1.);
            px[..3].copy_from_slice(&ink(bg.lerp(paint, cover) * (0.964 + 0.072 * grain[x])));
        }
    }
    let mouth = rx * vessel.radius_fraction(bowl, &knots, 1.);
    let y = foot - height;
    canvas.ellipse(
        [cx, y],
        [mouth, (mouth * mouth_tilt).max(1.)],
        ink(pigment * 1.08),
    );
    canvas.ellipse(
        [cx, y + 0.4],
        [
            (mouth * (1. - vessel.wall_fraction * 1.9)).max(0.8),
            (mouth * mouth_tilt * 0.78).max(0.65),
        ],
        ink(pigment * 0.53),
    );
    canvas
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::objects::audit;
    #[test]
    fn easel_study_depends_on_its_physical_subject_and_replays() {
        let stand = audit::fixture(ObjectKind::ArtistEasel, 18);
        let board = Easel::sample(&stand);
        let mut vase = audit::fixture(ObjectKind::Vase, 39);
        vase.position = Vec3::new(2., 0.67, 0.);
        let reference = pixels(18, &stand, &board, Some(&vase));
        assert_eq!(reference, pixels(18, &stand, &board, Some(&vase)));
        let mut bowl = vase.clone();
        bowl.kind = ObjectKind::Bowl;
        bowl.size = Vec3::new(0.23, 0.09, 0.23);
        let other = pixels(18, &stand, &board, Some(&bowl));
        let differences = reference
            .as_chunks::<4>()
            .0
            .iter()
            .zip(other.as_chunks::<4>().0)
            .filter(|(a, b)| *a != *b)
            .count();
        assert!((900..45000).contains(&differences));
        assert_ne!(reference, pixels(19, &stand, &board, Some(&vase)));
        assert_ne!(reference, pixels(18, &stand, &board, None));
    }
}
