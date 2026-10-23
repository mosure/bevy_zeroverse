//! Stable material binding slots, independent of procedural finish families.
//! Preserve ordinal order: mesh batches, recipes and GPU bindings use these IDs.
macro_rules! surfaces {
    ($($name:ident => $maps:literal,)*) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
        #[repr(usize)]
        pub enum Surface { $($name,)* }
        impl Surface {
            pub const COUNT: usize = [$(stringify!($name)),*].len();
            pub const ALL: [Self; Self::COUNT] = [$(Self::$name,)*];
            /// Uses substrate PBR synthesis, distinct from content/glazing programs.
            pub const fn has_procedural_maps(self) -> bool {
                match self { $(Self::$name => $maps,)* }
            }
        }
    };
}

surfaces! {
    Paint => true,
    Accent => true,
    Wood => true,
    WoodEdge => true,
    Floor => true,
    Ceiling => true,
    Metal => true,
    Chrome => true,
    Plastic => true,
    Fabric => true,
    FabricAlt => true,
    Glass => false,
    Ceramic => true,
    Soil => true,
    Leaf => true,
    LeafLight => true,
    Paper => true,
    Screen => false,
    Ink => false,
    Light => false,
    Concrete => true,
    Art => false,
    Rubber => true,
    LeafVariegated => true,
    Terracotta => true,
    Bark => true,
    GlassInterior => false,
    ContainerGlass => false,
    Liquid => false,
    Drink => false,
    PhoneScreen => false,
    PrintedPaper => false,
    Leather => true,
    Whiteboard => false,
    Television => false,
    Succulent => true,
}
