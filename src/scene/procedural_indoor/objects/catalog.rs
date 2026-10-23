//! Stable object vocabulary and construction families. Append new kinds at the
//! end: ordinals participate in seeded generation and exported dataset identity.
//! A row owns semantic class, construction family, support and table-outline
//! capabilities together. Geometry algorithms stay in their family modules.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectFamily {
    CeilingFans,
    Signals,
    Public,
    Kitchen,
    Equipment,
    Utilities,
    Tables,
    Services,
    Accessories,
    Hospitality,
    Chairs,
    Seating,
    Storage,
    WallFixtures,
    Desktop,
    Tabletop,
    Plants,
    Bins,
    Displays,
    Laptops,
    WallPanels,
    Literature,
    Clocks,
    FloorLamps,
    Floor,
}

macro_rules! object_catalog {
    ($($kind:ident => ($class:literal, $family:ident, $support:literal, $table:literal),)*) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
        pub enum ObjectKind { $($kind,)* }

        impl ObjectKind {
            pub const COUNT: usize = [$(stringify!($kind)),*].len();
            /// Complete catalog in stable ordinal order.
            pub const ALL: [Self; Self::COUNT] = [$(Self::$kind,)*];
            pub const fn class_name(self) -> &'static str {
                match self { $(Self::$kind => $class,)* }
            }
            pub const fn family(self) -> ObjectFamily {
                match self { $(Self::$kind => ObjectFamily::$family,)* }
            }
            pub const fn is_surface(self) -> bool {
                match self { $(Self::$kind => $support,)* }
            }
            pub const fn has_table_outline(self) -> bool {
                match self { $(Self::$kind => $table,)* }
            }
            pub const fn is_kitchen(self) -> bool {
                matches!(self.family(), ObjectFamily::Kitchen)
            }
            pub const fn is_program_item(self) -> bool {
                matches!(self.family(), ObjectFamily::Public)
            }
        }
    };
}

// Kind => (semantic class, construction family, supports items, table outline).
object_catalog! {
    Table => ("table", Tables, true, true),
    Desk => ("desk", Tables, true, true),
    Chair => ("chair", Chairs, false, false),
    Sofa => ("sofa", Seating, false, false),
    CoffeeTable => ("coffee_table", Tables, true, true),
    Cabinet => ("cabinet", Storage, true, false),
    Bookcase => ("bookshelf", Storage, false, false),
    Plant => ("plant", Plants, false, false),
    TrashCan => ("trash_can", Bins, false, false),
    Whiteboard => ("whiteboard", WallPanels, false, false),
    Display => ("television", Displays, false, false),
    Laptop => ("laptop", Laptops, false, false),
    Monitor => ("monitor", Displays, false, false),
    Notebook => ("paper", Literature, false, false),
    Mug => ("drink_container", Tabletop, false, false),
    Books => ("books", Literature, false, false),
    Rug => ("floormat", Floor, false, false),
    WallArt => ("picture", WallPanels, false, false),
    Clock => ("clock", Clocks, false, false),
    FloorLamp => ("lamp", FloorLamps, false, false),
    Keyboard => ("keyboard", Desktop, false, false),
    Mouse => ("mouse", Desktop, false, false),
    WaterBottle => ("drink_container", Tabletop, false, false),
    PenHolder => ("stationery", Tabletop, false, false),
    Printer => ("printer", Utilities, false, false),
    StorageBox => ("box", Utilities, false, false),
    CoatRack => ("coat_rack", Utilities, false, false),
    Bag => ("bag", Utilities, false, false),
    CoffeeCup => ("drink_container", Tabletop, false, false),
    SodaCan => ("drink_container", Tabletop, false, false),
    Notepad => ("paper", Tabletop, false, false),
    Pencil => ("stationery", Tabletop, false, false),
    Microphone => ("microphone", Tabletop, false, false),
    Phone => ("phone", Tabletop, false, false),
    WallOutlet => ("wall_outlet", WallFixtures, false, false),
    LightSwitch => ("light_switch", WallFixtures, false, false),
    SideTable => ("side_table", Tables, true, true),
    MobileCart => ("mobile_cart", Services, true, false),
    DrawerUnit => ("drawer_unit", Services, true, false),
    Bench => ("bench", Services, false, false),
    Ottoman => ("ottoman", Services, false, false),
    RoomDivider => ("room_divider", Services, false, false),
    DeskLamp => ("lamp", Accessories, false, false),
    CoffeeMaker => ("appliance", Hospitality, false, false),
    Kettle => ("appliance", Hospitality, false, false),
    Speaker => ("speaker", Accessories, false, false),
    Headphones => ("headphones", Accessories, false, false),
    Stapler => ("stationery", Accessories, false, false),
    TapeDispenser => ("stationery", Accessories, false, false),
    DocumentTray => ("stationery", Accessories, false, false),
    DeskOrganizer => ("stationery", Accessories, false, false),
    TissueBox => ("box", Accessories, false, false),
    Vase => ("vase", Hospitality, false, false),
    Bowl => ("vase", Hospitality, false, false),
    DeskFan => ("appliance", Accessories, false, false),
    Projector => ("projector", Equipment, false, false),
    ProjectionScreen => ("projection_screen", Equipment, false, false),
    PowerStrip => ("power_strip", Equipment, false, false),
    Cable => ("cable", Equipment, false, false),
    WallShelf => ("shelves", Equipment, true, false),
    NoticeBoard => ("whiteboard", Equipment, false, false),
    WallSconce => ("lamp", Equipment, false, false),
    CeilingPendant => ("lamp", Equipment, false, false),
    FloorOutlet => ("floor_outlet", Floor, false, false),
    KitchenCounter => ("counter", Kitchen, true, false),
    KitchenSink => ("sink", Kitchen, false, false),
    KitchenCupboard => ("cabinet", Kitchen, false, false),
    Pantry => ("cabinet", Kitchen, false, false),
    Cooktop => ("stove", Kitchen, false, false),
    Range => ("stove", Kitchen, false, false),
    Oven => ("oven", Kitchen, false, false),
    Microwave => ("microwave", Kitchen, false, false),
    Dishwasher => ("dishwasher", Kitchen, false, false),
    Refrigerator => ("refrigerator", Kitchen, false, false),
    DryingRack => ("dishware", Kitchen, false, false),
    SoapDispenser => ("soap_dispenser", Kitchen, false, false),
    Sponge => ("cleaning_supply", Kitchen, false, false),
    KnifeBlock => ("kitchen_utensil", Kitchen, false, false),
    UtensilHolder => ("kitchen_utensil", Kitchen, false, false),
    PaperTowel => ("towel", Kitchen, false, false),
    KitchenTowel => ("towel", Kitchen, false, false),
    Dishware => ("dishware", Kitchen, false, false),
    CuttingBoard => ("kitchen_utensil", Kitchen, true, false),
    FruitBowl => ("food", Kitchen, false, false),
    Bread => ("food", Kitchen, false, false),
    Elevator => ("elevator", Public, false, false),
    ServiceCounter => ("counter", Public, true, false),
    DisplayCase => ("display_case", Public, false, false),
    TrayReturn => ("tray_return", Public, false, false),
    DrinkDispenser => ("appliance", Public, false, false),
    QueueBarrier => ("queue_barrier", Public, false, false),
    Lectern => ("lectern", Public, false, false),
    Locker => ("locker", Public, false, false),
    TeachingBoard => ("whiteboard", Public, false, false),
    ExhibitPlinth => ("plinth", Public, true, false),
    Sculpture => ("sculpture", Public, false, false),
    InformationStand => ("information_stand", Public, false, false),
    ExhibitionPanel => ("room_divider", Public, false, false),
    ExtractionHood => ("extraction_hood", Public, false, false),
    Banquette => ("bench", Public, false, false),
    CeilingFan => ("ceiling_fan", CeilingFans, false, false),
    ExitSign => ("exit_sign", Signals, false, false),
    FireAlarm => ("fire_alarm", Signals, false, false),
    StringLights => ("string_lights", Signals, false, false),
    FireDetector => ("fire_detector", Signals, false, false),
    FirePullStation => ("fire_pull_station", Signals, false, false),
}
