//! Sales fixtures are a single bounded object. Merchandise retains its own
//! materials and semantic classes while sharing the fixture's rigid transform.
use super::*;
use rand_chacha::ChaCha8Rng;

mod apparel;
pub use apparel::folded::{
    folded_apparel, DisplayTableStand, FoldedCut, FoldedDisplay, FoldedGarment, FoldedLayer,
    FoldedPile,
};
pub(crate) use apparel::folded::{
    wants_opposite_face, FOLDED_APPAREL_VERSION, SINGLE_BROWSING_FACE,
};
const RAIL_MIN_HEIGHT: f32 = 1.22;
pub use apparel::rail::{apparel_rail, ApparelRail, RailForm, DISPLAY_RAIL_VERSION};
pub(crate) mod checkout;
mod cold;
pub(crate) use cold::{is_refrigerated, REFRIGERATED};
pub use cold::{refrigerated_case, CaseStock, RefrigeratedCase};
mod grocery;
pub use grocery::gondola::{gondola_stock, GondolaRow, GondolaStock};
pub(crate) use grocery::print_finishes;
pub use grocery::produce::{
    produce_stand, CrateStyle, ProduceBase, ProduceBin, ProducePiece, ProduceStand,
};
pub(crate) use grocery::produce::{resin_slot as produce_resin, VERSION as PRODUCE_VERSION};

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub(crate) enum Installation {
    Refrigerated(RefrigeratedCase),
    Gondola {
        levels: usize,
        facings: usize,
        stock: f32,
        frame: Surface,
    },
    Produce {
        crates: usize,
        fill: f32,
        frame: Surface,
    },
    Rail {
        hangers: usize,
        length: f32,
        frame: Surface,
    },
    Folded {
        piles: usize,
        layers: usize,
        frame: Surface,
    },
    Register(checkout::CheckoutStation),
    Checkout {
        top_m: f32,
        belt: bool,
        till_side: f32,
        frame: Surface,
    },
}

pub(super) fn dimensions(kind: ObjectKind, r: &mut impl Rng) -> Vec3 {
    let (width, height, depth) = match kind {
        ObjectKind::GroceryDisplay if r.random_bool(0.27) => (1.35..2.28, 0.81..1.07, 0.68..0.98),
        ObjectKind::GroceryDisplay => (1.47..2.68, 1.27..1.94, 0.64..0.93),
        ObjectKind::ApparelDisplay if r.random_bool(0.34) => (1.12..2.08, 0.83..1.16, 0.65..0.94),
        ObjectKind::ApparelDisplay => (1.12..2.12, 1.42..1.91, 0.58..0.82),
        ObjectKind::CheckoutCounter => (1.28..2.55, 1.14..1.38, 0.64..0.87),
        _ => unreachable!("retail fixture"),
    };
    Vec3::new(
        r.random_range(width),
        r.random_range(height),
        r.random_range(depth),
    )
}

pub(crate) fn parameters(o: &IndoorObject) -> Installation {
    if let Some(station) = checkout::checkout_station(o) {
        return Installation::Register(station);
    }
    if let Some(case) = refrigerated_case(o) {
        return Installation::Refrigerated(case);
    }
    let mut r = stream(o.seed, u64::from_le_bytes(*b"ShopFits"));
    let frame = [Surface::Metal, Surface::Wood, Surface::Paint][r.random_range(0..3)];
    match o.kind {
        ObjectKind::GroceryDisplay if o.size.y < 1.15 => Installation::Produce {
            crates: (o.size.x / r.random_range(0.50..0.72))
                .round()
                .clamp(2., 4.) as usize,
            fill: r.random_range(0.50..0.98),
            frame,
        },
        ObjectKind::GroceryDisplay => Installation::Gondola {
            levels: ((o.size.y - 0.10) / r.random_range(0.32..0.48))
                .round()
                .clamp(3., 5.) as usize,
            facings: (o.size.x / r.random_range(0.18..0.29))
                .round()
                .clamp(4., 12.) as usize,
            stock: r.random_range(0.55..0.99),
            frame,
        },
        ObjectKind::ApparelDisplay if o.size.y < RAIL_MIN_HEIGHT => Installation::Folded {
            piles: (o.size.x / r.random_range(0.33..0.48))
                .floor()
                .clamp(2., 6.) as usize,
            layers: r.random_range(2..=5),
            frame,
        },
        ObjectKind::ApparelDisplay => Installation::Rail {
            hangers: (o.size.x / r.random_range(0.20..0.31))
                .round()
                .clamp(4., 8.) as usize,
            length: r.random_range(0.53..1.08),
            frame,
        },
        ObjectKind::CheckoutCounter => Installation::Checkout {
            top_m: r.random_range(0.84..0.98),
            belt: o.size.x > 1.6 && r.random_bool(0.54),
            till_side: if r.random_bool(0.5) { 1. } else { -1. },
            frame,
        },
        _ => unreachable!("retail fixture"),
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    let mut r = stream(o.seed, u64::from_le_bytes(*b"ShopLoad"));
    let s = o.size;
    match parameters(o) {
        Installation::Refrigerated(case) => cold::build(a, o, case),
        Installation::Gondola {
            levels,
            facings,
            stock,
            frame,
        } if version >= grocery::gondola::VERSION => {
            grocery::gondola::build(a, o, levels, facings, stock, frame);
        }
        Installation::Gondola {
            levels,
            facings,
            stock,
            frame,
        } => grocery::legacy_shelves(a, o, levels, facings, stock, frame, &mut r),
        Installation::Produce {
            crates,
            fill,
            frame,
        } if version >= grocery::produce::VERSION => {
            ProduceStand::sample(o, crates, fill).build(a, o, frame);
        }
        Installation::Produce {
            crates,
            fill,
            frame,
        } => grocery::legacy_produce(a, s, crates, fill, frame, &mut r),
        Installation::Rail {
            hangers,
            length,
            frame,
        } if version >= DISPLAY_RAIL_VERSION => {
            ApparelRail::sample(o, hangers, length, frame).build(a, o.size)
        }
        Installation::Rail {
            hangers,
            length,
            frame,
        } => apparel::legacy_rail(a, s, hangers, length, frame, &mut r),
        Installation::Folded {
            piles,
            layers,
            frame,
        } if version >= FOLDED_APPAREL_VERSION => {
            FoldedDisplay::sample(o, piles, layers, frame).build(a, s)
        }
        Installation::Folded {
            piles,
            layers,
            frame,
        } => apparel::legacy_folded(a, s, piles, layers, frame, &mut r),
        Installation::Register(spec) => checkout::build(a, o, spec),
        Installation::Checkout {
            top_m,
            belt,
            till_side,
            frame,
        } => checkout::legacy::build(a, s, top_m, belt, till_side, frame),
    }
}

const FIXTURE: &str = "retail_fixture";

/// Square-cut furniture stock. Repeated shelf boards and hems do not need the
/// tessellation of a rounded casting.
fn board(a: &mut Assembly, surface: Surface, label: &str, center: Vec3, size: Vec3) {
    a.box_part(surface, label, center, size, 0.);
}

fn table_base(a: &mut Assembly, s: Vec3, top: f32, frame: Surface) {
    let t = 0.032;
    board(
        a,
        frame,
        FIXTURE,
        Vec3::Y * (top - t * 0.5),
        Vec3::new(s.x, t, s.z),
    );
    for x in [-0.43, 0.43] {
        for z in [-0.40, 0.40] {
            board(
                a,
                frame,
                FIXTURE,
                Vec3::new(s.x * x, (top - t) * 0.5, s.z * z),
                Vec3::new(0.048, top - t, 0.048),
            );
        }
    }
}

/// Install an upright copy from one product lot. Apparel sizes scale uniformly,
/// so the normals only turn and the metric weave chart follows the cut size.
fn append_facing(
    dest: &mut Geometry,
    g: &Geometry,
    base: Vec3,
    (sin, cos): (f32, f32),
    scale: f32,
) {
    let turn = |[x, y, z]: [f32; 3]| Vec3::new(cos * x + sin * z, y, cos * z - sin * x);
    let offset = dest.positions.len() as u32;
    dest.positions.extend(
        g.positions
            .iter()
            .map(|&p| (base + scale * turn(p)).to_array()),
    );
    dest.normals
        .extend(g.normals.iter().map(|&n| turn(n).to_array()));
    if scale == 1. {
        dest.uvs.extend_from_slice(&g.uvs);
    } else {
        dest.uvs
            .extend(g.uvs.iter().map(|&[u, v]| [u * scale, v * scale]));
    }
    dest.indices.extend(g.indices.iter().map(|i| i + offset));
}
