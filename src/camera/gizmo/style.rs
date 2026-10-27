//! Presentation colors are separate from the additive co-visibility bit palette.
use bevy::{
    gizmos::config::{GizmoConfig, GizmoLineStyle},
    prelude::*,
};

/// Stable camera-ID palette shared with schematic and predicted camera overlays.
/// The first five retain the live-inference colors; additional IDs do not cycle
/// back onto them. No dependence on camera count, selection or draw order.
pub fn rgb(index: usize) -> [u8; 3] {
    const COLORS: [[u8; 3]; 16] = [
        [114, 211, 208],
        [246, 198, 106],
        [177, 155, 230],
        [123, 190, 132],
        [242, 156, 185],
        [100, 166, 244],
        [239, 139, 97],
        [200, 213, 111],
        [216, 143, 227],
        [149, 187, 219],
        [210, 168, 119],
        [145, 216, 166],
        [226, 113, 119],
        [154, 173, 91],
        [142, 153, 235],
        [95, 184, 229],
    ];
    if let Some(c) = COLORS.get(index) {
        return *c;
    }
    let c = Color::hsl(
        (index as f64 * 137.50776405003785 % 360.) as f32,
        0.62,
        0.64,
    )
    .to_srgba();
    [c.red, c.green, c.blue].map(|v| (v * 255.).round() as u8)
}

pub fn color(index: usize) -> Color {
    let [r, g, b] = rgb(index);
    Color::srgb_u8(r, g, b)
}

/// Existing schematic/live-inference labels carry IDs as `C7`, `Pred C7`,
/// `Predicted <C7>` or `Anchor, predicted lens C7`. Resolve the whole ID token,
/// including sparse/out-of-order predictions; anonymous labels use caller order.
/// New clients should retain the same C-index token in their display labels.
pub fn color_index(label: &str, fallback: usize) -> usize {
    label
        .split(|c: char| !c.is_ascii_alphanumeric())
        .find_map(|token| {
            token
                .strip_prefix('C')
                .filter(|s| !s.is_empty())?
                .parse()
                .ok()
        })
        .unwrap_or(fallback)
}

/// Matching solid/dashed editor-only styles. Capture textures never receive them.
pub fn config(predicted: bool) -> GizmoConfig {
    let mut config = GizmoConfig {
        render_layers: super::super::EDITOR_CAMERA_RENDER_LAYER,
        ..default()
    };
    config.line.width = 1.5;
    config.line.style = if predicted {
        GizmoLineStyle::Dashed {
            gap_scale: 2.,
            line_scale: 3.,
        }
    } else {
        GizmoLineStyle::Solid
    };
    config
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn camera_colors_are_stable_distinct_and_prediction_ids_match() {
        let colors: std::collections::HashSet<_> = (0..128).map(rgb).collect();
        assert_eq!(colors.len(), 128);
        assert_eq!(rgb(0), [114, 211, 208]);
        assert_eq!(rgb(4), [242, 156, 185]);
        for label in [
            "C7",
            "Pred C7 · fisheye · t=7s",
            "Predicted <C7>",
            "Anchor, predicted lens C7",
        ] {
            assert_eq!(color_index(label, 0), 7);
        }
        for label in ["C", "C7x", "Camera7", "prediction"] {
            assert_eq!(color_index(label, 3), 3);
        }
        assert_eq!(config(false).render_layers, config(true).render_layers);
        assert!(!config(false)
            .render_layers
            .intersects(&bevy::camera::visibility::RenderLayers::default()));
        assert_ne!(config(false).line.style, config(true).line.style);
        assert_eq!(config(false).depth_bias, 0.);
    }
}
