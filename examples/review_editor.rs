//! Capture the actual native Feathers editor for visual regression review.
//! Run: cargo run --example review_editor --features human_motion
use bevy::{
    prelude::*,
    render::view::screenshot::{save_to_disk, Screenshot},
};
use bevy_zeroverse::{
    app::{
        editor::{
            model::{EditorState, Page},
            Action, Actions,
        },
        viewer_app, BevyZeroverseConfig,
    },
    scene::{procedural_indoor::layout::IndoorManifest, ZeroverseSceneType},
};
#[derive(Resource, Default)]
struct Review {
    frames: u32,
    stage: usize,
}
fn main() {
    std::fs::create_dir_all("out/ui_revamp/native").unwrap();
    let mut app = viewer_app(
        None,
        Some(BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::ProceduralIndoor,
            indoor_seed: Some(46839170),
            num_cameras: 4,
            width: 512.,
            height: 512.,
            camera_grid: true,
            gizmos: false,
            playback_speed: 0.,
            ..default()
        }),
    );
    for mut window in app
        .world_mut()
        .query::<&mut Window>()
        .iter_mut(app.world_mut())
    {
        window.resolution.set(1360., 900.);
    }
    app.insert_resource(bevy::winit::WinitSettings::continuous());
    app.init_resource::<Review>()
        .add_systems(Update, review)
        .run();
}
fn review(world: &mut World) {
    if !world.contains_resource::<IndoorManifest>() {
        return;
    }
    let mut review = world.remove_resource::<Review>().unwrap();
    review.frames += 1;
    // Exercise real text editing, widget observers and retained layout.
    if review.frames == 90 {
        let entity = labeled(world, "Room seed");
        world
            .resource_mut::<bevy::input_focus::InputFocus>()
            .set(entity, bevy::input_focus::FocusCause::Pressed);
        let mut input = world.get_mut::<bevy::text::EditableText>(entity).unwrap();
        input.queue_edit(bevy::text::TextEdit::SelectAll);
        input.queue_edit(bevy::text::TextEdit::Insert("18446744073709551615".into()));
    }
    if review.frames == 100 {
        assert_eq!(
            world.resource::<EditorState>().draft["indoor_seed"],
            serde_json::json!(u64::MAX)
        );
        assert_eq!(world.resource::<IndoorManifest>().seed, 46839170);
        world
            .resource_mut::<bevy::input_focus::InputFocus>()
            .clear();
        let entity = labeled(world, "Discard");
        world.trigger(bevy::ui_widgets::Activate { entity });
    }
    if review.frames == 110 {
        let entity = labeled(world, "Furnishing density");
        world.trigger(bevy::ui_widgets::ValueChange {
            source: entity,
            value: 0.8f32,
            is_final: false,
        });
    }
    if review.frames == 130 {
        assert_eq!(
            world.resource::<EditorState>().draft["indoor_density"],
            serde_json::json!(0.8f32)
        );
        assert_eq!(world.resource::<BevyZeroverseConfig>().indoor_density, 0.65);
        assert_eq!(world.resource::<IndoorManifest>().seed, 46839170);
        let entity = labeled(world, "Discard");
        world.trigger(bevy::ui_widgets::Activate { entity });
    }
    if review.frames == 150 {
        let entity = labeled(world, "Room seed");
        assert_eq!(
            world
                .get::<bevy::text::EditableText>(entity)
                .unwrap()
                .value()
                .to_string(),
            "46839170"
        );
        assert!(
            world.get::<ComputedNode>(entity).unwrap().size().x > 200.,
            "seed input must have a visible hit target"
        );
        assert_eq!(
            world
                .get::<bevy::text::TextLayoutInfo>(entity)
                .unwrap()
                .glyphs
                .len(),
            8
        );
        assert!(!world.resource::<EditorState>().pending());
    }
    const PAGES: [Page; 10] = [
        Page::Scene,
        Page::Cameras,
        Page::Appearance,
        Page::People,
        Page::View,
        Page::Advanced,
        Page::Scene,
        Page::View,
        Page::View,
        Page::View,
    ];
    if (review.frames == 160 || (review.frames > 160 && (review.frames - 160).is_multiple_of(60)))
        && review.stage < PAGES.len()
    {
        world
            .resource_mut::<Actions>()
            .0
            .push(Action::Page(PAGES[review.stage]));
        if review.stage == 3 {
            world
                .resource_mut::<Actions>()
                .0
                .push(Action::Set("@motion".into(), serde_json::json!(true)));
        }
        if review.stage == 6 {
            world.resource_mut::<Actions>().0.push(Action::Set(
                "/indoor_density".into(),
                serde_json::json!(0.8),
            ));
        }
        if review.stage == 7 || review.stage == 8 {
            world.resource_mut::<Actions>().0.push(Action::Set(
                "/render_mode".into(),
                serde_json::json!(if review.stage == 7 {
                    "Semantic"
                } else {
                    "CoVisibility"
                }),
            ));
        }
    }
    if review.frames == 710 {
        let entity = labeled(world, "Annotation");
        world.trigger(bevy::ui_widgets::Activate { entity });
    }
    if review.frames >= 195
        && (review.frames - 195).is_multiple_of(60)
        && review.stage < PAGES.len()
    {
        let path = format!(
            "out/ui_revamp/native/{:02}_{:?}.png",
            review.stage, PAGES[review.stage]
        );
        world
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
        use bevy::{
            feathers::controls::{ButtonVariant, FeathersSlider},
            ui_widgets::{SliderPrecision, SliderRange, SliderValue},
        };
        let mut widgets = Vec::new();
        for (entity, label) in world.query::<(Entity, &AccessibleLabel)>().iter(world) {
            widgets.push(serde_json::json!({"entity":format!("{entity:?}"),"label":label.0,"slider":world.get::<FeathersSlider>(entity).is_some(),"value":world.get::<SliderValue>(entity).map(|v|v.0),"text":world.get::<bevy::text::EditableText>(entity).map(|t|t.value().to_string()),"glyphs":world.get::<bevy::text::TextLayoutInfo>(entity).map(|t|t.glyphs.len()),"text_size":world.get::<bevy::text::TextLayoutInfo>(entity).map(|t|t.size.to_array()),"size":world.get::<ComputedNode>(entity).map(|n|n.size().to_array()),"font":format!("{:?}",world.get::<TextFont>(entity)),"color":format!("{:?}",world.get::<TextColor>(entity)),"range":format!("{:?}",world.get::<SliderRange>(entity)),"precision":format!("{:?}",world.get::<SliderPrecision>(entity)),"variant":format!("{:?}",world.get::<ButtonVariant>(entity)),"hovered":world.get::<bevy::picking::hover::Hovered>(entity).is_some(),"gradient":world.get::<BackgroundGradient>(entity).is_some()}));
        }
        std::fs::write(
            format!("out/ui_revamp/native/{:02}_widgets.json", review.stage),
            serde_json::to_vec_pretty(&widgets).unwrap(),
        )
        .unwrap();
        review.stage += 1;
    }
    if review.stage >= PAGES.len() && review.frames > 195 + (PAGES.len() as u32) * 60 + 20 {
        let state = world.resource::<EditorState>();
        std::fs::write(
            "out/ui_revamp/native/state.json",
            serde_json::to_vec_pretty(&state.draft).unwrap(),
        )
        .unwrap();
        world.write_message(bevy::app::AppExit::Success);
    }
    world.insert_resource(review);
}

fn labeled(world: &mut World, label: &str) -> Entity {
    world
        .query::<(Entity, &AccessibleLabel)>()
        .iter(world)
        .find(|(_, l)| l.0 == label)
        .unwrap_or_else(|| panic!("missing widget {label}"))
        .0
}
