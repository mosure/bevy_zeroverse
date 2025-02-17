use bevy::{
    prelude::*,
    asset::load_internal_asset,
    pbr::{
        ExtendedMaterial,
        MaterialExtension,
    },
    render::render_resource::*,
};

use crate::render::DisabledPbrMaterial;


pub const SEMANTIC_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(5639572395);

#[derive(Component, Debug, Clone, Default, Reflect, Eq, PartialEq)]
#[reflect(Component)]
pub enum SemanticLabel {
    #[default]
    Wall,
    Floor,
    Cabinet,
    Bed,
    Chair,
    Sofa,
    Table,
    Door,
    Window,
    Bookshelf,
    Picture,
    Counter,
    Blinds,
    Desk,
    Shelves,
    Curtain,
    Dresser,
    Pillow,
    Mirror,
    Floormat,
    Clothes,
    Ceiling,
    Books,
    Refrigerator,
    Television,
    Paper,
    Towel,
    ShowerCurtain,
    Box,
    Whiteboard,
    Person,
    Nightstand,
    Toilet,
    Sink,
    Lamp,
    Bathtub,
    Bag,
    OtherStructure,
    OtherFurniture,
    OtherProp,
}

impl SemanticLabel {
    pub fn color(&self) -> Color {
        // https://github.com/apple/ml-hypersim/blob/main/code/cpp/tools/scene_annotation_tool/semantic_label_descs.csv
        match self {
            SemanticLabel::Wall => Color::srgb_u8(174, 199, 232),
            SemanticLabel::Floor => Color::srgb_u8(152, 223, 138),
            SemanticLabel::Cabinet => Color::srgb_u8(31, 119, 180),
            SemanticLabel::Bed => Color::srgb_u8(255, 187, 120),
            SemanticLabel::Chair => Color::srgb_u8(188, 189, 34),
            SemanticLabel::Sofa => Color::srgb_u8(140, 86, 75),
            SemanticLabel::Table => Color::srgb_u8(255, 152, 150),
            SemanticLabel::Door => Color::srgb_u8(214, 39, 40),
            SemanticLabel::Window => Color::srgb_u8(197, 176, 213),
            SemanticLabel::Bookshelf => Color::srgb_u8(148, 103, 189),
            SemanticLabel::Picture => Color::srgb_u8(196, 156, 148),
            SemanticLabel::Counter => Color::srgb_u8(23, 190, 207),
            SemanticLabel::Blinds => Color::srgb_u8(178, 76, 76),
            SemanticLabel::Desk => Color::srgb_u8(247, 182, 210),
            SemanticLabel::Shelves => Color::srgb_u8(66, 188, 102),
            SemanticLabel::Curtain => Color::srgb_u8(219, 219, 141),
            SemanticLabel::Dresser => Color::srgb_u8(140, 57, 197),
            SemanticLabel::Pillow => Color::srgb_u8(202, 185, 52),
            SemanticLabel::Mirror => Color::srgb_u8(51, 176, 203),
            SemanticLabel::Floormat => Color::srgb_u8(200, 54, 131),
            SemanticLabel::Clothes => Color::srgb_u8(92, 193, 61),
            SemanticLabel::Ceiling => Color::srgb_u8(78, 71, 183),
            SemanticLabel::Books => Color::srgb_u8(172, 114, 82),
            SemanticLabel::Refrigerator => Color::srgb_u8(255, 127, 14),
            SemanticLabel::Television => Color::srgb_u8(91, 163, 138),
            SemanticLabel::Paper => Color::srgb_u8(153, 98, 156),
            SemanticLabel::Towel => Color::srgb_u8(140, 153, 101),
            SemanticLabel::ShowerCurtain => Color::srgb_u8(158, 218, 229),
            SemanticLabel::Box => Color::srgb_u8(100, 125, 154),
            SemanticLabel::Whiteboard => Color::srgb_u8(178, 127, 135),
            SemanticLabel::Person => Color::srgb_u8(120, 185, 128),
            SemanticLabel::Nightstand => Color::srgb_u8(146, 111, 194),
            SemanticLabel::Toilet => Color::srgb_u8(44, 160, 44),
            SemanticLabel::Sink => Color::srgb_u8(112, 128, 144),
            SemanticLabel::Lamp => Color::srgb_u8(96, 207, 209),
            SemanticLabel::Bathtub => Color::srgb_u8(227, 119, 194),
            SemanticLabel::Bag => Color::srgb_u8(213, 92, 176),
            SemanticLabel::OtherStructure => Color::srgb_u8(94, 106, 211),
            SemanticLabel::OtherFurniture => Color::srgb_u8(82, 84, 163),
            SemanticLabel::OtherProp => Color::srgb_u8(100, 85, 144),
        }
    }
}


#[derive(Component, Debug, Clone, Default, Reflect, Eq, PartialEq)]
#[reflect(Component, Default)]
pub struct Semantic;


#[derive(Debug, Default)]
pub struct SemanticPlugin;
impl Plugin for SemanticPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            SEMANTIC_SHADER_HANDLE,
            "semantic.wgsl",
            Shader::from_wgsl
        );

        app.register_type::<Semantic>();
        app.register_type::<SemanticLabel>();

        app.add_plugins(MaterialPlugin::<SemanticMaterial>::default());

        app.add_systems(Update, propagate_semantic_labels);
        app.add_systems(Update, apply_semantic_material);

        // TODO: add system for bounding box render toggle
    }
}


fn propagate_semantic_labels(
    mut commands: Commands,
    semantic_parents: Query<(
        &SemanticLabel,
        &Children,
    )>,
) {
    // TODO: capture frame delay based on hierarchy propagation, determine delay from hierarchy depth or apply full-tree update
    for (label, children) in semantic_parents.iter() {
        for &child in children.iter() {
            commands.entity(child).insert(label.clone());
        }
    }
}


#[allow(clippy::type_complexity)]
fn apply_semantic_material(
    mut commands: Commands,
    semantic: Query<
        (
            Entity,
            &DisabledPbrMaterial,
            &SemanticLabel,
        ),
        (With<Semantic>, Without<MeshMaterial3d<SemanticMaterial>>),
    >,
    mut removed_semantics: RemovedComponents<Semantic>,
    mut materials: ResMut<Assets<SemanticMaterial>>,
) {
    for e in removed_semantics.read() {
        if let Some(mut commands) = commands.get_entity(e) {
            commands.remove::<MeshMaterial3d<SemanticMaterial>>();
        }
    }

    for (e, pbr_material, label) in &semantic {
        let color = label.color().to_linear();
        let semantic_material = materials.add(
            ExtendedMaterial {
                base: StandardMaterial {
                    double_sided: pbr_material.double_sided,
                    cull_mode: pbr_material.cull_mode,
                    ..default()
                },
                extension: SemanticExtension {
                    color,
                },
            },
        );

        commands.entity(e).insert(MeshMaterial3d(semantic_material));
    }
}


pub type SemanticMaterial = ExtendedMaterial<StandardMaterial, SemanticExtension>;


#[derive(Default, AsBindGroup, TypePath, Debug, Clone, Asset)]
pub struct SemanticExtension {
    #[uniform(100)]
    pub color: LinearRgba,
}

impl MaterialExtension for SemanticExtension {
    fn fragment_shader() -> ShaderRef {
        SEMANTIC_SHADER_HANDLE.into()
    }
}
