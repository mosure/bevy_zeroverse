//! A level standing place behind the low edge of the desk, facing the audience.
use super::*;
use crate::scene::procedural_indoor::layout::programs::access;
use geometry::furnishing::lectern;

fn audience(p: &ProgramAssembly) -> Vec2 {
    p.min + (p.max - p.min) * Vec2::new(0.5, 0.68)
}

pub(in crate::scene::procedural_indoor::layout::functional) fn faces_class(
    o: &IndoorObject,
    p: &ProgramAssembly,
) -> bool {
    let forward = o.transform().rotation * Vec3::NEG_Z;
    let toward = audience(p) - lectern::presenter_position(o).xz();
    lectern::installed(o) && forward.xz().dot(toward.normalize_or_zero()) > 0.75
}

// Sample within the support rectangle after accounting for the entire oriented work area.
// The same yaw is used for the placement test, the rendered desk and the presenter.
fn place_on(
    station: &mut IndoorObject,
    region: (Vec2, Vec2),
    uv: Vec2,
    aim: Vec2,
    toe: f32,
) -> bool {
    let (min, max) = region;
    let away = min + (max - min) * uv - aim;
    station.position = Vec3::ZERO;
    station.yaw = away.x.atan2(away.y) + toe;
    let (lo, hi) = access::bounds(station);
    let first = min - lo.xz() + 0.025;
    let last = max - hi.xz() - 0.025;
    if !last.cmpge(first).all() {
        return false;
    }
    let site = first + (last - first) * uv;
    station.position = Vec3::new(site.x, 0., site.y);
    true
}

impl IndoorManifest {
    pub(super) fn install_presenter_station(&mut self, index: usize, size: Vec3) {
        let mut r = stream(self.seed, u64::from_le_bytes(*b"SpeakPos") + index as u64);
        let mut station = self.candidate(ObjectKind::Lectern, Vec3::ZERO, size, 0., &mut r);
        let level = self.clutter.sample(station.seed);
        lectern::install(&mut station, level);
        station.finish_seed = Some(self.seed.wrapping_add(index as u64));
        let p = &self.assembly_diagnostics.programs[index];
        let (min, max, focus) = (p.min, p.max, audience(p));
        let span = max - min;
        let aim = focus
            + Vec2::new(
                span.x * r.random_range(-0.14..0.14),
                span.y * r.random_range(-0.09..0.09),
            );
        // The centre aisle may contain a guard or a column. Check the seated banks.
        let targets = [-0.22, 0.22].map(|fraction| {
            let q = focus + Vec2::X * (span.x * fraction);
            Vec3::new(q.x, self.floor_height(q) + 1.16, q.y)
        });
        let strip = (min, Vec2::new(max.x, min.y + 3.));
        let platform = (p.activity == IndoorLayout::LectureHall).then(|| {
            let stage = super::super::architecture::teaching_platform(min, max);
            (stage.min, stage.max)
        });
        let toe = r.random_range(-0.12..0.12);
        let structure = PlacementStructure::new();
        let board_bounds = self
            .objects
            .iter()
            .rev()
            .find(|o| {
                o.kind == ObjectKind::TeachingBoard
                    && o.position.xz().cmpge(min).all()
                    && o.position.xz().cmple(max).all()
            })
            .map(IndoorObject::bounds);
        // Try a continuously sampled flank first. Stratify the remaining front strip only on failure.
        let flank = if r.random_bool(0.5) { -1. } else { 1. };
        let preferred = Vec2::new(0.5 + flank * 0.37 * r.random::<f32>().sqrt(), r.random());
        let primary = platform.unwrap_or(strip);
        let samples = std::iter::once((primary, preferred)).chain(
            std::iter::once_with(|| variation::recovery_points(6, &mut r))
                .flatten()
                .enumerate()
                .map(|(i, uv)| (if i % 3 == 2 { strip } else { primary }, uv)),
        );
        for (region, uv) in samples {
            if !place_on(&mut station, region, uv, aim, toe) {
                self.assembly_diagnostics.programs[index].rejected_candidates += 1;
                continue;
            }
            station.position.y = self.floor_height(station.position.xz());
            let eye = lectern::presenter_position(&station) + Vec3::Y * 1.55;
            if faces_class(&station, &self.assembly_diagnostics.programs[index])
                && self.placement_clear_with_structure(&station, 0.035, false, &structure)
                && targets.iter().any(|&target| {
                    !board_bounds.is_some_and(|(lo, hi)| segment_hits_box(eye, target, lo, hi))
                        && !structure
                            .get_or_init(|| self.columns())
                            .iter()
                            .any(|&(lo, hi)| segment_hits_box(eye, target, lo, hi))
                })
            {
                self.objects.push(station);
                return;
            }
            self.assembly_diagnostics.programs[index].rejected_candidates += 1;
        }
    }
}
