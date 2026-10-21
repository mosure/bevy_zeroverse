//! Live proposal-PDF preview; updating it never rebuilds a slider or a scene.
use super::*;
use crate::camera::lens::LensSampling;
use crate::scene::procedural_indoor::cameras::intrinsics::IntrinsicsSettings;

const BINS: usize = 24;
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LensKind {
    Perspective,
    Fisheye,
}
#[derive(Component)]
struct LensBin(LensKind, usize);
#[derive(Component)]
struct LensSummary(LensKind);

pub(super) fn build(world: &mut World, parent: Entity, kind: LensKind) {
    let plot = world
        .spawn((
            Node {
                width: percent(100),
                height: px(52),
                align_items: AlignItems::End,
                column_gap: px(2),
                flex_shrink: 0.,
                ..default()
            },
            ChildOf(parent),
        ))
        .id();
    for index in 0..BINS {
        world.spawn((
            Node {
                flex_grow: 1.,
                flex_basis: px(0),
                height: px(0),
                ..default()
            },
            BackgroundColor(ACCENT),
            LensBin(kind, index),
            ChildOf(plot),
        ));
    }
    let summary = widgets::text(world, parent, "", 11.5, MUTED);
    world.entity_mut(summary).insert(LensSummary(kind));
}

pub(super) fn synchronize(world: &mut World, state: &EditorState) {
    if state.view.page != Page::Cameras {
        return;
    }
    let perspective = serde_json::from_value::<IntrinsicsSettings>(
        state.draft["indoor_camera"]["intrinsics"].clone(),
    )
    .ok()
    .filter(|lens| lens.validate().is_ok());
    let fisheye = if state.draft["indoor_camera"]["lenses"].is_null() {
        Some(LensSampling {
            probability: 0.,
            ..default()
        })
    } else {
        serde_json::from_value::<LensSampling>(state.draft["indoor_camera"]["lenses"].clone()).ok()
    }
    .filter(|lens| lens.validate().is_ok());
    for kind in [LensKind::Perspective, LensKind::Fisheye] {
        let mut bins = [0_u32; BINS];
        let bounds = match kind {
            LensKind::Perspective => perspective
                .as_ref()
                .map(|p| [p.fov_y_min_degrees, p.fov_y_max_degrees]),
            LensKind::Fisheye => fisheye.as_ref().map(|p| p.fov_degrees),
        };
        let quantile = |q| match kind {
            LensKind::Perspective => perspective.as_ref().unwrap().quantile(q),
            LensKind::Fisheye => fisheye.as_ref().unwrap().fov_quantile(q),
        };
        let text = if let Some([lo, hi]) = bounds {
            let range = (hi - lo).max(0.001);
            for i in 0..1024 {
                let degrees = quantile((i as f32 + 0.5) / 1024.);
                let index = (((degrees - lo) / range * BINS as f32) as usize).min(BINS - 1);
                bins[index] += 1;
            }
            format!(
                "{:.0}° — proposal density — {:.0}°\n10%: {:.0}°   Median: {:.0}°   90%: {:.0}°{}",
                lo,
                hi,
                quantile(0.1),
                quantile(0.5),
                quantile(0.9),
                if kind == LensKind::Fisheye && fisheye.as_ref().unwrap().probability == 0. {
                    "\nFisheye disabled; this distribution is retained for re-enabling."
                } else if kind == LensKind::Perspective
                    && perspective
                        .as_ref()
                        .unwrap()
                        .ultra_wide_probability
                        .is_some()
                {
                    "\nImported two-band policy. Edit preference/spread to use the continuous PDF."
                } else {
                    ""
                }
            )
        } else {
            "Set valid FOV bounds to preview the lens distribution.".into()
        };
        let max = bins.iter().copied().max().unwrap_or(1).max(1) as f32;
        for (bin, mut node) in world.query::<(&LensBin, &mut Node)>().iter_mut(world) {
            if bin.0 == kind {
                node.height = px(50. * bins[bin.1] as f32 / max);
            }
        }
        for (summary, mut value) in world.query::<(&LensSummary, &mut Text)>().iter_mut(world) {
            if summary.0 == kind {
                value.0.clone_from(&text);
            }
        }
    }
}
