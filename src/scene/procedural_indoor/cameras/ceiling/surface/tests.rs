use super::*;
use crate::scene::procedural_indoor::{
    architecture::ceiling::{
        framing::{CeilingFraming, FrameRole},
        CeilingInstallation,
    },
    layout::{stream, IndoorLayout},
    materials::Surface,
};

fn member(section: FrameSection) -> FrameMember {
    FrameMember {
        top: [
            Vec3::new(-2., 4., -0.5),
            Vec3::new(2., 3.8, -0.5),
            Vec3::new(2., 3.8, 0.5),
            Vec3::new(-2., 4., 0.5),
        ],
        depth: 0.2,
        role: FrameRole::Rafter,
        section,
        material: Surface::Wood,
        finish: 0,
        web_pitch: 0.5,
    }
}

#[test]
fn mounting_faces_match_rendered_structural_undersides() {
    for section in [
        FrameSection::Solid,
        FrameSection::ISection,
        FrameSection::Fluted,
    ] {
        let m = member(section);
        let patch = Patch::underside(&m).unwrap();
        for p in [Vec2::ZERO, Vec2::new(-1.1, 0.15), Vec2::new(1.5, -0.15)] {
            let (point, normal) = patch.at(p).unwrap();
            assert!((point.y - m.attachment_y(point - Vec3::Y).unwrap()).abs() < 1e-6);
            assert!(normal.y < -0.99);
            for delta in [Vec2::X * 0.01, Vec2::Y * 0.01] {
                let (q, _) = patch.at(p + delta).unwrap();
                assert!(normal.dot(q - point).abs() < 1e-6);
            }
        }
        assert!(patch.at(Vec2::new(2.01, 0.)).is_none());
        assert!(patch.at(Vec2::new(0., 0.51)).is_none());
    }
    assert!(Patch::underside(&member(FrameSection::OpenWeb)).is_none());
    assert!(Patch::underside(&member(FrameSection::Fluted))
        .unwrap()
        .at(Vec2::new(0., 0.4))
        .is_none());
}

#[test]
fn moving_mounts_cannot_bridge_even_a_thin_support_gap() {
    let mut s = IndoorManifest::generate_with_humans(6, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    s.room_size = Vec3::new(10., 5., 10.);
    let e = s.envelope.as_mut().unwrap();
    e.footprint = vec![
        Vec2::splat(-5.),
        Vec2::new(5., -5.),
        Vec2::splat(5.),
        Vec2::new(-5., 5.),
    ];
    e.ceiling_drop = Vec2::ZERO;
    e.soffits.clear();
    e.floor_patches.clear();
    e.mezzanine = None;
    let m = member(FrameSection::Solid);
    s.ceiling_installation = Some(CeilingInstallation {
        exposed_slab: true,
        modules: vec![],
        framing: Some(CeilingFraming {
            roof_material: Surface::Wood,
            members: vec![m.clone()],
        }),
    });
    s.camera_settings.ceiling_mount = Some(CeilingCameraSettings::default());
    let prior = super::super::proposal::Prototype::sample(
        &s,
        0,
        s.camera_settings.ceiling_mount.as_ref().unwrap(),
    );
    let mut camera = prior.mounted_at(&s, 0, Vec2::new(-1., 0.), &mut stream(6, 99));
    super::super::trajectory::set_rail(&mut camera, Vec3::new(2., -0.1, 0.));
    assert!(super::super::trajectory::plane_clear(&s, &camera));
    let mut left = m.clone();
    let mut right = m;
    left.top[1] = left.top[0].lerp(left.top[1], 0.503);
    left.top[2] = left.top[3].lerp(left.top[2], 0.503);
    right.top[0] = right.top[0].lerp(right.top[1], 0.5035);
    right.top[3] = right.top[3].lerp(right.top[2], 0.5035);
    s.ceiling_installation
        .as_mut()
        .unwrap()
        .framing
        .as_mut()
        .unwrap()
        .members = vec![left, right];
    assert!(
        !super::super::trajectory::plane_clear(&s, &camera),
        "2 mm unsupported rail gap between checkpoints"
    );
}
