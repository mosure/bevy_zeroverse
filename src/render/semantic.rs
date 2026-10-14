use bevy::{
    asset::{load_internal_asset, uuid_handle},
    pbr::{ExtendedMaterial, MaterialExtension},
    prelude::*,
    render::render_resource::AsBindGroup,
    shader::ShaderRef,
};

use crate::render::DisabledPbrMaterial;

pub const SEMANTIC_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("e1b272d1-bae5-48ed-8214-2adbafa43cad");

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
    // Extended indoor taxonomy; legacy labels and colors remain stable.
    Plant,
    TrashCan,
    Laptop,
    Monitor,
    Phone,
    Keyboard,
    Mouse,
    Headphones,
    Speaker,
    Microphone,
    Printer,
    Projector,
    ProjectionScreen,
    CoffeeTable,
    SideTable,
    Bench,
    Ottoman,
    MobileCart,
    DrawerUnit,
    CoatRack,
    RoomDivider,
    WallOutlet,
    LightSwitch,
    PowerStrip,
    Cable,
    Clock,
    DrinkContainer,
    Appliance,
    Stationery,
    Vase,
    /// Finite exterior background proxy; its depth is the proxy surface distance.
    Sky,
    FloorOutlet,
    Stove,
    Oven,
    Microwave,
    Dishwasher,
    Faucet,
    Dishware,
    KitchenUtensil,
    Food,
    SoapDispenser,
    CleaningSupply,
    Elevator,
    ControlPanel,
    DisplayCase,
    TrayReturn,
    QueueBarrier,
    Lectern,
    Locker,
    Plinth,
    Sculpture,
    InformationStand,
    ExtractionHood,
    DoorFrame,
    DoorHardware,
    CabinetHardware,
}

impl SemanticLabel {
    /// Versioned, authoritative vocabulary for rendered colors and exports.
    pub const TAXONOMY_VERSION: u32 = 7;
    pub const ALL: &[Self] = &[
        Self::Wall,
        Self::Floor,
        Self::Cabinet,
        Self::Bed,
        Self::Chair,
        Self::Sofa,
        Self::Table,
        Self::Door,
        Self::Window,
        Self::Bookshelf,
        Self::Picture,
        Self::Counter,
        Self::Blinds,
        Self::Desk,
        Self::Shelves,
        Self::Curtain,
        Self::Dresser,
        Self::Pillow,
        Self::Mirror,
        Self::Floormat,
        Self::Clothes,
        Self::Ceiling,
        Self::Books,
        Self::Refrigerator,
        Self::Television,
        Self::Paper,
        Self::Towel,
        Self::ShowerCurtain,
        Self::Box,
        Self::Whiteboard,
        Self::Person,
        Self::Nightstand,
        Self::Toilet,
        Self::Sink,
        Self::Lamp,
        Self::Bathtub,
        Self::Bag,
        Self::OtherStructure,
        Self::OtherFurniture,
        Self::OtherProp,
        Self::Plant,
        Self::TrashCan,
        Self::Laptop,
        Self::Monitor,
        Self::Phone,
        Self::Keyboard,
        Self::Mouse,
        Self::Headphones,
        Self::Speaker,
        Self::Microphone,
        Self::Printer,
        Self::Projector,
        Self::ProjectionScreen,
        Self::CoffeeTable,
        Self::SideTable,
        Self::Bench,
        Self::Ottoman,
        Self::MobileCart,
        Self::DrawerUnit,
        Self::CoatRack,
        Self::RoomDivider,
        Self::WallOutlet,
        Self::LightSwitch,
        Self::PowerStrip,
        Self::Cable,
        Self::Clock,
        Self::DrinkContainer,
        Self::Appliance,
        Self::Stationery,
        Self::Vase,
        Self::Sky,
        Self::FloorOutlet,
        Self::Stove,
        Self::Oven,
        Self::Microwave,
        Self::Dishwasher,
        Self::Faucet,
        Self::Dishware,
        Self::KitchenUtensil,
        Self::Food,
        Self::SoapDispenser,
        Self::CleaningSupply,
        Self::Elevator,
        Self::ControlPanel,
        Self::DisplayCase,
        Self::TrayReturn,
        Self::QueueBarrier,
        Self::Lectern,
        Self::Locker,
        Self::Plinth,
        Self::Sculpture,
        Self::InformationStand,
        Self::ExtractionHood,
        Self::DoorFrame,
        Self::DoorHardware,
        Self::CabinetHardware,
    ];

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
            SemanticLabel::Plant => Color::srgb_u8(166, 109, 53),
            SemanticLabel::TrashCan => Color::srgb_u8(75, 47, 186),
            SemanticLabel::Laptop => Color::srgb_u8(52, 207, 37),
            SemanticLabel::Monitor => Color::srgb_u8(227, 73, 131),
            SemanticLabel::Phone => Color::srgb_u8(41, 125, 166),
            SemanticLabel::Keyboard => Color::srgb_u8(180, 186, 34),
            SemanticLabel::Mouse => Color::srgb_u8(171, 66, 207),
            SemanticLabel::Headphones => Color::srgb_u8(57, 227, 134),
            SemanticLabel::Speaker => Color::srgb_u8(166, 52, 30),
            SemanticLabel::Microphone => Color::srgb_u8(60, 76, 186),
            SemanticLabel::Printer => Color::srgb_u8(117, 207, 52),
            SemanticLabel::Projector => Color::srgb_u8(227, 41, 173),
            SemanticLabel::ProjectionScreen => Color::srgb_u8(53, 166, 165),
            SemanticLabel::CoffeeTable => Color::srgb_u8(186, 145, 47),
            SemanticLabel::SideTable => Color::srgb_u8(107, 37, 207),
            SemanticLabel::Bench => Color::srgb_u8(73, 227, 91),
            SemanticLabel::Ottoman => Color::srgb_u8(166, 41, 63),
            SemanticLabel::MobileCart => Color::srgb_u8(34, 104, 186),
            SemanticLabel::DrawerUnit => Color::srgb_u8(172, 207, 66),
            SemanticLabel::CoatRack => Color::srgb_u8(219, 57, 227),
            SemanticLabel::RoomDivider => Color::srgb_u8(30, 166, 120),
            SemanticLabel::WallOutlet => Color::srgb_u8(186, 106, 60),
            SemanticLabel::LightSwitch => Color::srgb_u8(64, 52, 207),
            SemanticLabel::PowerStrip => Color::srgb_u8(81, 227, 41),
            SemanticLabel::Cable => Color::srgb_u8(166, 53, 110),
            SemanticLabel::Clock => Color::srgb_u8(47, 158, 186),
            SemanticLabel::DrinkContainer => Color::srgb_u8(207, 192, 37),
            SemanticLabel::Appliance => Color::srgb_u8(168, 73, 227),
            SemanticLabel::Stationery => Color::srgb_u8(41, 166, 82),
            SemanticLabel::Vase => Color::srgb_u8(186, 39, 34),
            SemanticLabel::Sky => Color::srgb_u8(75, 145, 245),
            SemanticLabel::FloorOutlet => Color::srgb_u8(98, 158, 171),
            SemanticLabel::Stove => Color::srgb_u8(159, 99, 43),
            SemanticLabel::Oven => Color::srgb_u8(88, 113, 158),
            SemanticLabel::Microwave => Color::srgb_u8(131, 99, 183),
            SemanticLabel::Dishwasher => Color::srgb_u8(74, 146, 180),
            SemanticLabel::Faucet => Color::srgb_u8(183, 196, 208),
            SemanticLabel::Dishware => Color::srgb_u8(217, 169, 93),
            SemanticLabel::KitchenUtensil => Color::srgb_u8(114, 167, 78),
            SemanticLabel::Food => Color::srgb_u8(226, 83, 63),
            SemanticLabel::SoapDispenser => Color::srgb_u8(76, 201, 173),
            SemanticLabel::Elevator => Color::srgb_u8(61, 211, 83),
            SemanticLabel::ControlPanel => Color::srgb_u8(74, 202, 94),
            SemanticLabel::DisplayCase => Color::srgb_u8(87, 193, 105),
            SemanticLabel::TrayReturn => Color::srgb_u8(100, 184, 116),
            SemanticLabel::QueueBarrier => Color::srgb_u8(113, 175, 127),
            SemanticLabel::Lectern => Color::srgb_u8(126, 166, 138),
            SemanticLabel::Locker => Color::srgb_u8(139, 157, 149),
            SemanticLabel::Plinth => Color::srgb_u8(152, 148, 160),
            SemanticLabel::Sculpture => Color::srgb_u8(165, 139, 171),
            SemanticLabel::InformationStand => Color::srgb_u8(178, 130, 182),
            SemanticLabel::ExtractionHood => Color::srgb_u8(191, 121, 193),
            SemanticLabel::DoorFrame => Color::srgb_u8(204, 112, 204),
            SemanticLabel::DoorHardware => Color::srgb_u8(217, 103, 215),
            SemanticLabel::CabinetHardware => Color::srgb_u8(230, 94, 226),
            SemanticLabel::CleaningSupply => Color::srgb_u8(200, 194, 77),
        }
    }

    pub fn from_label(name: &str) -> Option<Self> {
        match name {
            "sky" => Some(SemanticLabel::Sky),
            "floor_outlet" => Some(SemanticLabel::FloorOutlet),
            "stove" => Some(SemanticLabel::Stove),
            "oven" => Some(SemanticLabel::Oven),
            "microwave" => Some(SemanticLabel::Microwave),
            "dishwasher" => Some(SemanticLabel::Dishwasher),
            "faucet" => Some(SemanticLabel::Faucet),
            "dishware" => Some(SemanticLabel::Dishware),
            "kitchen_utensil" => Some(SemanticLabel::KitchenUtensil),
            "food" => Some(SemanticLabel::Food),
            "soap_dispenser" => Some(SemanticLabel::SoapDispenser),
            "elevator" => Some(SemanticLabel::Elevator),
            "control_panel" => Some(SemanticLabel::ControlPanel),
            "display_case" => Some(SemanticLabel::DisplayCase),
            "tray_return" => Some(SemanticLabel::TrayReturn),
            "queue_barrier" => Some(SemanticLabel::QueueBarrier),
            "lectern" => Some(SemanticLabel::Lectern),
            "locker" => Some(SemanticLabel::Locker),
            "plinth" => Some(SemanticLabel::Plinth),
            "sculpture" => Some(SemanticLabel::Sculpture),
            "information_stand" => Some(SemanticLabel::InformationStand),
            "extraction_hood" => Some(SemanticLabel::ExtractionHood),
            "door_frame" => Some(SemanticLabel::DoorFrame),
            "door_hardware" => Some(SemanticLabel::DoorHardware),
            "cabinet_hardware" => Some(SemanticLabel::CabinetHardware),
            "cleaning_supply" => Some(SemanticLabel::CleaningSupply),

            "wall" => Some(SemanticLabel::Wall),
            "floor" => Some(SemanticLabel::Floor),
            "cabinet" => Some(SemanticLabel::Cabinet),
            "bed" => Some(SemanticLabel::Bed),
            "chair" => Some(SemanticLabel::Chair),
            "sofa" => Some(SemanticLabel::Sofa),
            "table" => Some(SemanticLabel::Table),
            "door" => Some(SemanticLabel::Door),
            "window" => Some(SemanticLabel::Window),
            "bookshelf" => Some(SemanticLabel::Bookshelf),
            "picture" => Some(SemanticLabel::Picture),
            "counter" => Some(SemanticLabel::Counter),
            "blinds" => Some(SemanticLabel::Blinds),
            "desk" => Some(SemanticLabel::Desk),
            "shelves" => Some(SemanticLabel::Shelves),
            "curtain" => Some(SemanticLabel::Curtain),
            "dresser" => Some(SemanticLabel::Dresser),
            "pillow" => Some(SemanticLabel::Pillow),
            "mirror" => Some(SemanticLabel::Mirror),
            "floormat" => Some(SemanticLabel::Floormat),
            "clothes" => Some(SemanticLabel::Clothes),
            "ceiling" => Some(SemanticLabel::Ceiling),
            "books" => Some(SemanticLabel::Books),
            "refrigerator" => Some(SemanticLabel::Refrigerator),
            "television" => Some(SemanticLabel::Television),
            "paper" => Some(SemanticLabel::Paper),
            "towel" => Some(SemanticLabel::Towel),
            "shower_curtain" => Some(SemanticLabel::ShowerCurtain),
            "box" => Some(SemanticLabel::Box),
            "whiteboard" => Some(SemanticLabel::Whiteboard),
            "person" => Some(SemanticLabel::Person),
            "nightstand" => Some(SemanticLabel::Nightstand),
            "toilet" => Some(SemanticLabel::Toilet),
            "sink" => Some(SemanticLabel::Sink),
            "lamp" => Some(SemanticLabel::Lamp),
            "bathtub" => Some(SemanticLabel::Bathtub),
            "bag" => Some(SemanticLabel::Bag),
            "other_structure" => Some(SemanticLabel::OtherStructure),
            "other_furniture" => Some(SemanticLabel::OtherFurniture),
            "other_prop" => Some(SemanticLabel::OtherProp),
            "human" => Some(SemanticLabel::Person),
            "plant" => Some(SemanticLabel::Plant),
            "trash_can" => Some(SemanticLabel::TrashCan),
            "laptop" => Some(SemanticLabel::Laptop),
            "monitor" => Some(SemanticLabel::Monitor),
            "phone" => Some(SemanticLabel::Phone),
            "keyboard" => Some(SemanticLabel::Keyboard),
            "mouse" => Some(SemanticLabel::Mouse),
            "headphones" => Some(SemanticLabel::Headphones),
            "speaker" => Some(SemanticLabel::Speaker),
            "microphone" => Some(SemanticLabel::Microphone),
            "printer" => Some(SemanticLabel::Printer),
            "projector" => Some(SemanticLabel::Projector),
            "projection_screen" => Some(SemanticLabel::ProjectionScreen),
            "coffee_table" => Some(SemanticLabel::CoffeeTable),
            "side_table" => Some(SemanticLabel::SideTable),
            "bench" => Some(SemanticLabel::Bench),
            "ottoman" => Some(SemanticLabel::Ottoman),
            "mobile_cart" => Some(SemanticLabel::MobileCart),
            "drawer_unit" => Some(SemanticLabel::DrawerUnit),
            "coat_rack" => Some(SemanticLabel::CoatRack),
            "room_divider" => Some(SemanticLabel::RoomDivider),
            "wall_outlet" => Some(SemanticLabel::WallOutlet),
            "light_switch" => Some(SemanticLabel::LightSwitch),
            "power_strip" => Some(SemanticLabel::PowerStrip),
            "cable" => Some(SemanticLabel::Cable),
            "clock" => Some(SemanticLabel::Clock),
            "drink_container" => Some(SemanticLabel::DrinkContainer),
            "appliance" => Some(SemanticLabel::Appliance),
            "stationery" => Some(SemanticLabel::Stationery),
            "vase" => Some(SemanticLabel::Vase),

            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            SemanticLabel::Sky => "sky",
            SemanticLabel::FloorOutlet => "floor_outlet",
            SemanticLabel::Stove => "stove",
            SemanticLabel::Oven => "oven",
            SemanticLabel::Microwave => "microwave",
            SemanticLabel::Dishwasher => "dishwasher",
            SemanticLabel::Faucet => "faucet",
            SemanticLabel::Dishware => "dishware",
            SemanticLabel::KitchenUtensil => "kitchen_utensil",
            SemanticLabel::Food => "food",
            SemanticLabel::SoapDispenser => "soap_dispenser",
            SemanticLabel::Elevator => "elevator",
            SemanticLabel::ControlPanel => "control_panel",
            SemanticLabel::DisplayCase => "display_case",
            SemanticLabel::TrayReturn => "tray_return",
            SemanticLabel::QueueBarrier => "queue_barrier",
            SemanticLabel::Lectern => "lectern",
            SemanticLabel::Locker => "locker",
            SemanticLabel::Plinth => "plinth",
            SemanticLabel::Sculpture => "sculpture",
            SemanticLabel::InformationStand => "information_stand",
            SemanticLabel::ExtractionHood => "extraction_hood",
            SemanticLabel::DoorFrame => "door_frame",
            SemanticLabel::DoorHardware => "door_hardware",
            SemanticLabel::CabinetHardware => "cabinet_hardware",
            SemanticLabel::CleaningSupply => "cleaning_supply",

            SemanticLabel::Wall => "wall",
            SemanticLabel::Floor => "floor",
            SemanticLabel::Cabinet => "cabinet",
            SemanticLabel::Bed => "bed",
            SemanticLabel::Chair => "chair",
            SemanticLabel::Sofa => "sofa",
            SemanticLabel::Table => "table",
            SemanticLabel::Door => "door",
            SemanticLabel::Window => "window",
            SemanticLabel::Bookshelf => "bookshelf",
            SemanticLabel::Picture => "picture",
            SemanticLabel::Counter => "counter",
            SemanticLabel::Blinds => "blinds",
            SemanticLabel::Desk => "desk",
            SemanticLabel::Shelves => "shelves",
            SemanticLabel::Curtain => "curtain",
            SemanticLabel::Dresser => "dresser",
            SemanticLabel::Pillow => "pillow",
            SemanticLabel::Mirror => "mirror",
            SemanticLabel::Floormat => "floormat",
            SemanticLabel::Clothes => "clothes",
            SemanticLabel::Ceiling => "ceiling",
            SemanticLabel::Books => "books",
            SemanticLabel::Refrigerator => "refrigerator",
            SemanticLabel::Television => "television",
            SemanticLabel::Paper => "paper",
            SemanticLabel::Towel => "towel",
            SemanticLabel::ShowerCurtain => "shower_curtain",
            SemanticLabel::Box => "box",
            SemanticLabel::Whiteboard => "whiteboard",
            SemanticLabel::Person => "person",
            SemanticLabel::Nightstand => "nightstand",
            SemanticLabel::Toilet => "toilet",
            SemanticLabel::Sink => "sink",
            SemanticLabel::Lamp => "lamp",
            SemanticLabel::Bathtub => "bathtub",
            SemanticLabel::Bag => "bag",
            SemanticLabel::OtherStructure => "other_structure",
            SemanticLabel::OtherFurniture => "other_furniture",
            SemanticLabel::OtherProp => "other_prop",
            SemanticLabel::Plant => "plant",
            SemanticLabel::TrashCan => "trash_can",
            SemanticLabel::Laptop => "laptop",
            SemanticLabel::Monitor => "monitor",
            SemanticLabel::Phone => "phone",
            SemanticLabel::Keyboard => "keyboard",
            SemanticLabel::Mouse => "mouse",
            SemanticLabel::Headphones => "headphones",
            SemanticLabel::Speaker => "speaker",
            SemanticLabel::Microphone => "microphone",
            SemanticLabel::Printer => "printer",
            SemanticLabel::Projector => "projector",
            SemanticLabel::ProjectionScreen => "projection_screen",
            SemanticLabel::CoffeeTable => "coffee_table",
            SemanticLabel::SideTable => "side_table",
            SemanticLabel::Bench => "bench",
            SemanticLabel::Ottoman => "ottoman",
            SemanticLabel::MobileCart => "mobile_cart",
            SemanticLabel::DrawerUnit => "drawer_unit",
            SemanticLabel::CoatRack => "coat_rack",
            SemanticLabel::RoomDivider => "room_divider",
            SemanticLabel::WallOutlet => "wall_outlet",
            SemanticLabel::LightSwitch => "light_switch",
            SemanticLabel::PowerStrip => "power_strip",
            SemanticLabel::Cable => "cable",
            SemanticLabel::Clock => "clock",
            SemanticLabel::DrinkContainer => "drink_container",
            SemanticLabel::Appliance => "appliance",
            SemanticLabel::Stationery => "stationery",
            SemanticLabel::Vase => "vase",
        }
    }
}

#[derive(Component, Debug, Clone, Default, Reflect, Eq, PartialEq)]
#[reflect(Component, Default)]
pub struct Semantic;

/// The children already carry authoritative part labels. Keep the parent's
/// object-level class for instance export without overwriting those parts.
#[derive(Component, Debug, Default)]
pub struct PreserveSemanticParts;

#[derive(Debug, Default)]
pub struct SemanticPlugin;
impl Plugin for SemanticPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            SEMANTIC_SHADER_HANDLE,
            "semantic.wesl",
            Shader::from_wesl
        );

        app.register_type::<Semantic>();
        app.register_type::<SemanticLabel>();

        app.add_plugins(MaterialPlugin::<SemanticMaterial>::default());

        app.add_systems(Update, propagate_semantic_labels);
        // Material insertion must precede Bevy's specialization bookkeeping.
        // Otherwise annotation-mode scene regeneration can extract a material
        // without its specialization tick (a renderer panic, notably on Wasm).
        app.add_systems(
            PostUpdate,
            apply_semantic_material
                .before(bevy::pbr::check_entities_needing_specialization::<SemanticMaterial>),
        );

        // TODO: add system for bounding box render toggle
    }
}

#[allow(clippy::type_complexity)]
fn propagate_semantic_labels(
    mut commands: Commands,
    semantic_parents: Query<
        (&SemanticLabel, &Children),
        (
            Or<(Changed<SemanticLabel>, Changed<Children>)>,
            Without<PreserveSemanticParts>,
        ),
    >,
    labels: Query<&SemanticLabel>,
) {
    // TODO: capture frame delay based on hierarchy propagation, determine delay from hierarchy depth or apply full-tree update
    for (label, children) in semantic_parents.iter() {
        for child in children.iter() {
            if labels.get(child) != Ok(label) {
                commands.entity(child).insert(label.clone());
            }
        }
    }
}

#[allow(clippy::type_complexity)]
pub(crate) fn apply_semantic_material(
    mut commands: Commands,
    semantic: Query<
        (Entity, &DisabledPbrMaterial, &SemanticLabel),
        (
            With<Semantic>,
            Or<(
                Without<MeshMaterial3d<SemanticMaterial>>,
                Changed<SemanticLabel>,
            )>,
        ),
    >,
    mut removed_semantics: RemovedComponents<Semantic>,
    mut materials: ResMut<Assets<SemanticMaterial>>,
    mut cache: Local<super::annotation_material::AnnotationMaterialCache<SemanticMaterial>>,
) {
    for e in removed_semantics.read() {
        if let Ok(mut commands) = commands.get_entity(e) {
            commands.remove::<MeshMaterial3d<SemanticMaterial>>();
        }
    }

    for (e, pbr_material, label) in &semantic {
        let base_color = label.color();
        let semantic_material = cache.get(
            &mut materials,
            {
                let mut key = pbr_material.annotation_key();
                key.extend(base_color.to_linear().to_f32_array().map(f32::to_bits));
                key
            },
            || ExtendedMaterial {
                base: StandardMaterial {
                    double_sided: pbr_material.double_sided,
                    cull_mode: pbr_material.cull_mode,
                    base_color,
                    unlit: true,
                    ..default()
                },
                extension: SemanticExtension {},
            },
        );

        commands.entity(e).insert(MeshMaterial3d(semantic_material));
    }
}

pub type SemanticMaterial = ExtendedMaterial<StandardMaterial, SemanticExtension>;

#[derive(Default, AsBindGroup, TypePath, Debug, Clone, Asset)]
pub struct SemanticExtension {}

impl MaterialExtension for SemanticExtension {
    fn enable_shadows() -> bool {
        false
    }
    fn enable_prepass() -> bool {
        false
    }

    fn fragment_shader() -> ShaderRef {
        SEMANTIC_SHADER_HANDLE.into()
    }
}

#[cfg(test)]
mod vocabulary_tests {
    use super::*;
    #[test]
    fn procedural_parts_keep_labels_while_legacy_hierarchies_inherit() {
        let mut app = App::new();
        app.add_systems(Update, propagate_semantic_labels);
        let root = app
            .world_mut()
            .spawn((SemanticLabel::Door, PreserveSemanticParts))
            .id();
        let handle = app
            .world_mut()
            .spawn((SemanticLabel::DoorHardware, ChildOf(root)))
            .id();
        let frame = app
            .world_mut()
            .spawn((SemanticLabel::DoorFrame, ChildOf(root)))
            .id();
        let legacy = app.world_mut().spawn(SemanticLabel::Cabinet).id();
        let mesh = app.world_mut().spawn(ChildOf(legacy)).id();
        app.update();
        assert_eq!(
            app.world().get::<SemanticLabel>(handle),
            Some(&SemanticLabel::DoorHardware)
        );
        assert_eq!(
            app.world().get::<SemanticLabel>(frame),
            Some(&SemanticLabel::DoorFrame)
        );
        assert_eq!(
            app.world().get::<SemanticLabel>(mesh),
            Some(&SemanticLabel::Cabinet)
        );
        app.world_mut()
            .entity_mut(legacy)
            .insert(SemanticLabel::Chair);
        app.update();
        assert_eq!(
            app.world().get::<SemanticLabel>(mesh),
            Some(&SemanticLabel::Chair)
        );
    }
    #[test]
    fn complete_taxonomy_roundtrips_and_has_unique_nonbackground_colors() {
        let mut names = std::collections::BTreeSet::new();
        let mut colors = std::collections::BTreeSet::new();
        for label in SemanticLabel::ALL {
            assert_eq!(
                SemanticLabel::from_label(label.as_str()).as_ref(),
                Some(label)
            );
            assert!(names.insert(label.as_str()));
            let c = label.color().to_srgba();
            let rgb = [c.red, c.green, c.blue].map(|v| (v * 255.).round() as u8);
            assert_ne!(rgb, [0; 3]);
            assert!(colors.insert(rgb), "duplicate palette entry: {label:?}");
        }
        for kind in crate::scene::procedural_indoor::layout::ObjectKind::ALL {
            assert!(
                SemanticLabel::from_label(kind.class_name()).is_some(),
                "{kind:?}"
            );
            assert_ne!(kind.class_name(), "other_prop", "{kind:?}");
        }
    }
}
