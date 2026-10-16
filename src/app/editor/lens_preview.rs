//! Live proposal-PDF preview; updating it never rebuilds a slider or a scene.
use super::*;
use crate::scene::procedural_indoor::cameras::intrinsics::IntrinsicsSettings;

const BINS: usize = 24;
#[derive(Component)]
struct LensBin(usize);
#[derive(Component)]
struct LensSummary;

pub(super) fn build(world: &mut World, parent: Entity) {
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
            LensBin(index),
            ChildOf(plot),
        ));
    }
    let summary = widgets::text(world, parent, "", 11.5, MUTED);
    world.entity_mut(summary).insert(LensSummary);
}

pub(super) fn synchronize(world: &mut World, state: &EditorState) {
    if state.view.page != Page::Cameras {
        return;
    }
    let lens = serde_json::from_value::<IntrinsicsSettings>(
        state.draft["indoor_camera"]["intrinsics"].clone(),
    )
    .ok()
    .filter(|lens| lens.validate().is_ok());
    let mut bins = [0_u32; BINS];
    let text = if let Some(lens) = lens {
        let range = (lens.fov_y_max_degrees - lens.fov_y_min_degrees).max(0.001);
        for i in 0..1024 {
            let degrees = lens.quantile((i as f32 + 0.5) / 1024.);
            let index =
                (((degrees - lens.fov_y_min_degrees) / range * BINS as f32) as usize).min(BINS - 1);
            bins[index] += 1;
        }
        format!(
            "{:.0}° — proposal density — {:.0}°\n10%: {:.0}°   Median: {:.0}°   90%: {:.0}°{}",
            lens.fov_y_min_degrees,
            lens.fov_y_max_degrees,
            lens.quantile(0.1),
            lens.quantile(0.5),
            lens.quantile(0.9),
            if lens.ultra_wide_probability.is_some() {
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
        node.height = px(50. * bins[bin.0] as f32 / max);
    }
    for mut value in world
        .query_filtered::<&mut Text, With<LensSummary>>()
        .iter_mut(world)
    {
        value.0.clone_from(&text);
    }
}
