//! Live proposal-PDF preview; updating it never rebuilds a slider or a scene.
use super::*;
use crate::camera::lens::LensSampling;
use crate::scene::procedural_indoor::cameras::intrinsics::IntrinsicsSettings;

const BINS: usize = 24;
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DistributionKind {
    Perspective,
    Fisheye,
    Clutter,
}
#[derive(Component)]
struct DistributionBin(DistributionKind, usize);
#[derive(Component)]
struct DistributionSummary(DistributionKind);

pub(super) fn build(world: &mut World, parent: Entity, kind: DistributionKind) {
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
            DistributionBin(kind, index),
            ChildOf(plot),
        ));
    }
    let summary = widgets::text(world, parent, "", 11.5, MUTED);
    world.entity_mut(summary).insert(DistributionSummary(kind));
}

pub(super) fn synchronize(world: &mut World, state: &EditorState) {
    if !matches!(state.view.page, Page::Cameras | Page::Scene) {
        return;
    }
    let perspective = serde_json::from_value::<IntrinsicsSettings>(
        model::optics(&state.draft)["intrinsics"].clone(),
    )
    .ok()
    .filter(|lens| lens.validate().is_ok());
    let fisheye = if model::optics(&state.draft)["lenses"].is_null() {
        Some(LensSampling {
            probability: 0.,
            ..default()
        })
    } else {
        serde_json::from_value::<LensSampling>(model::optics(&state.draft)["lenses"].clone()).ok()
    }
    .filter(|lens| lens.validate().is_ok());
    let clutter = serde_json::from_value::<
        crate::scene::procedural_indoor::items::distribution::ClutterSettings,
    >(state.draft["indoor_clutter"].clone())
    .ok()
    .filter(|p| p.validate().is_ok());
    let kinds: &[DistributionKind] = if state.view.page == Page::Scene {
        &[DistributionKind::Clutter]
    } else {
        &[DistributionKind::Perspective, DistributionKind::Fisheye]
    };
    for &kind in kinds {
        let mut bins = [0_u32; BINS];
        let bounds = match kind {
            DistributionKind::Perspective => perspective
                .as_ref()
                .map(|p| [p.fov_y_min_degrees, p.fov_y_max_degrees]),
            DistributionKind::Fisheye => fisheye.as_ref().map(|p| p.fov_degrees),
            DistributionKind::Clutter => clutter.as_ref().map(|_| [0., 1.]),
        };
        let quantile = |q| match kind {
            DistributionKind::Perspective => perspective.as_ref().unwrap().quantile(q),
            DistributionKind::Fisheye => fisheye.as_ref().unwrap().fov_quantile(q),
            DistributionKind::Clutter => {
                let p = clutter.as_ref().unwrap();
                let probability = p.selection_probability();
                if probability == 0. || q <= 1. - probability {
                    0.
                } else {
                    p.quantile((q - (1. - probability)) / probability)
                }
            }
        };
        let text = if let Some([lo, hi]) = bounds {
            let range = (hi - lo).max(0.001);
            for i in 0..1024 {
                let degrees = quantile((i as f32 + 0.5) / 1024.);
                let index = (((degrees - lo) / range * BINS as f32) as usize).min(BINS - 1);
                bins[index] += 1;
            }
            if kind == DistributionKind::Clutter {
                let p = clutter.as_ref().unwrap();
                let probability = p.selection_probability();
                format!("{:.0}% clear tables · {:.0}% selected\nSelected amount — 10%: {:.2}   Median: {:.2}   90%: {:.2}\nProposal distribution; support and clearance can reduce realized counts.",
                    100. * (1. - probability), 100. * probability, p.quantile(0.1), p.quantile(0.5), p.quantile(0.9))
            } else {
                format!(
                "{:.0}° — proposal density — {:.0}°\n10%: {:.0}°   Median: {:.0}°   90%: {:.0}°{}",
                lo,
                hi,
                quantile(0.1),
                quantile(0.5),
                quantile(0.9),
                if kind == DistributionKind::Fisheye && fisheye.as_ref().unwrap().probability == 0. {
                    "\nFisheye disabled; this distribution is retained for re-enabling."
                } else if kind == DistributionKind::Perspective
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
            }
        } else {
            "Set valid bounds to preview the distribution.".into()
        };
        let max = bins.iter().copied().max().unwrap_or(1).max(1) as f32;
        for (bin, mut node) in world
            .query::<(&DistributionBin, &mut Node)>()
            .iter_mut(world)
        {
            if bin.0 == kind {
                node.height = px(50. * bins[bin.1] as f32 / max);
            }
        }
        for (summary, mut value) in world
            .query::<(&DistributionSummary, &mut Text)>()
            .iter_mut(world)
        {
            if summary.0 == kind {
                value.0.clone_from(&text);
            }
        }
    }
}
