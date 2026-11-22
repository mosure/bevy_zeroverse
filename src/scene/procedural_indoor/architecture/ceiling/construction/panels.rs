use super::*;

pub(super) fn panels(
    a: &mut Assembly,
    f: &Frame,
    camber: f32,
    taper: f32,
    phase: f32,
) -> Vec<Vec3> {
    let cloud = f.m.system == CeilingSystem::AcousticCloud;
    let wood = f.m.system == CeilingSystem::TimberSlats;
    let material = if wood { Surface::Wood } else { Surface::Fabric };
    let count = if cloud {
        1
    } else {
        (f.width / f.m.pitch_m).round().clamp(2., 28.) as usize
    };
    let span = if cloud {
        f.width - 0.012
    } else {
        f.m.thickness_m.min(f.width / count as f32 * 0.32)
    };
    let depth = if cloud {
        f.m.thickness_m.min(f.height * 0.30)
    } else {
        f.height * 0.52
    };
    let top = f.height * 0.77;
    for i in 0..count {
        let z = if cloud {
            0.
        } else {
            (i as f32 + 0.5) * f.width / count as f32 - f.width * 0.5
        };
        let rings: Vec<_> = (0..=8)
            .map(|j| {
                let t = j as f32 / 8.;
                // Nonuniform lower edges; a constant upper datum seats every baffle
                // in the two carrying rails, even when the silhouette is curved.
                let y = top - depth * 0.5;
                let relief = camber
                    * (0.5 + 0.5 * (phase + i as f32 * 0.4 + t * std::f32::consts::TAU).sin());
                let d = depth * (1. - relief * 0.60);
                let w = span * (taper + (1. - taper) * (std::f32::consts::PI * t).sin());
                (
                    Vec3::new((t - 0.5) * (f.length - 0.018), y + (depth - d) * 0.5, z),
                    Vec2::new(d * 0.5, w * 0.5),
                )
            })
            .collect();
        loft(f.part(a, material), &rings, false);
    }
    let mut anchors = Vec::new();
    for x in [-f.length * 0.32, f.length * 0.32] {
        f.block(
            a,
            Surface::Metal,
            Vec3::new(x, top + 0.007, 0.),
            Vec3::new(0.028, 0.016, f.width * 0.92),
            0.002,
        );
        for z in [-f.width * 0.38, f.width * 0.38] {
            anchors.push(Vec3::new(x, top + 0.014, z));
        }
    }
    anchors
}
