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

pub(super) fn grid(a: &mut Assembly, f: &Frame, aspect: f32, recess: f32) -> Vec<Vec3> {
    let open = f.m.system == CeilingSystem::OpenCell;
    let nx = (f.length / (f.m.pitch_m * aspect)).round().clamp(1., 16.) as usize;
    let nz = (f.width / f.m.pitch_m).round().clamp(1., 12.) as usize;
    let dx = (f.length - 0.028) / nx as f32;
    let dz = (f.width - 0.028) / nz as f32;
    let base = if open { 0.013 } else { recess + 0.009 };
    let rail_h = if open { f.height * 0.76 } else { 0.035 };
    let rail_w = if open { 0.012 } else { 0.023 };
    let material = if open {
        Surface::Metal
    } else {
        Surface::Ceiling
    };
    // Real T extrusions: narrow vertical web and a broad, thin tile ledge.
    // Flat extrusion faces avoid subdividing hidden back edges of every cell.
    let rail = |a: &mut Assembly, p: Vec3, size: Vec3| {
        if open {
            f.block(a, Surface::Metal, p, size, 0.001);
        } else {
            f.block(
                a,
                Surface::Metal,
                p.with_y(base + 0.002),
                size.with_y(0.004),
                0.,
            );
            let web = if size.x > size.z {
                Vec3::new(size.x, rail_h - 0.003, 0.004)
            } else {
                Vec3::new(0.004, rail_h - 0.003, size.z)
            };
            f.block(
                a,
                Surface::Metal,
                p.with_y(base + 0.003 + web.y * 0.5),
                web,
                0.,
            );
        }
    };
    // Continuous main runners; cross tees end at them with no coplanar faces.
    for j in 0..=nz {
        let z = (j as f32 - nz as f32 * 0.5) * dz;
        rail(
            a,
            Vec3::new(0., base + rail_h * 0.5, z),
            Vec3::new(f.length - 0.014, rail_h, rail_w),
        );
    }
    for i in 0..=nx {
        let x = (i as f32 - nx as f32 * 0.5) * dx;
        for j in 0..nz {
            let z = (j as f32 + 0.5 - nz as f32 * 0.5) * dz;
            rail(
                a,
                Vec3::new(x, base + rail_h * 0.5, z),
                Vec3::new(rail_w, rail_h, dz - rail_w - 0.002),
            );
        }
    }
    if !open {
        for i in 0..nx {
            for j in 0..nz {
                let p = Vec3::new(
                    (i as f32 + 0.5 - nx as f32 * 0.5) * dx,
                    base + 0.004 - recess + (recess + 0.016) * 0.5,
                    (j as f32 + 0.5 - nz as f32 * 0.5) * dz,
                );
                // Tegular tiles: real recessed lower face, narrow perimeter reveal,
                // thicker inner tile. The outer lip rests on the T ledge.
                f.block(
                    a,
                    material,
                    p,
                    Vec3::new(dx - rail_w - 0.006, recess + 0.016, dz - rail_w - 0.006),
                    0.002,
                );
                f.block(
                    a,
                    material,
                    p.with_y(base + 0.008),
                    Vec3::new(dx - 0.010, 0.008, dz - 0.010),
                    0.,
                );
            }
        }
    }
    // Vertical fascia closes the raft edge; the slab is visible through open cells.
    let edge_h = (rail_h + 0.016).min(f.height - base - 0.002);
    for z in [-f.width * 0.5 + 0.004, f.width * 0.5 - 0.004] {
        f.block(
            a,
            material,
            Vec3::new(0., base + edge_h * 0.5, z),
            Vec3::new(f.length, edge_h, 0.008),
            0.001,
        );
    }
    for x in [-f.length * 0.5 + 0.004, f.length * 0.5 - 0.004] {
        f.block(
            a,
            material,
            Vec3::new(x, base + edge_h * 0.5, 0.),
            Vec3::new(0.008, edge_h, f.width - 0.018),
            0.001,
        );
    }
    let mut anchors = Vec::new();
    // Attach hangers to actual outside runners.
    for x in [-f.length * 0.32, f.length * 0.32] {
        for z in [-(nz as f32) * dz * 0.5, nz as f32 * dz * 0.5] {
            anchors.push(Vec3::new(x, base + rail_h - 0.002, z));
        }
    }
    anchors
}
