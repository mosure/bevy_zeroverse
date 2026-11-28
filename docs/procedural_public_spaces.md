# Functional indoor spaces

The indoor generator has explicit programs for hallways, stair halls, courtyards, elevator landings, cafes,
cafeterias, restaurants, fitness rooms, cycling and yoga studios, volleyball gyms, changing rooms, laundromats, grocery and clothing stores, rehearsal rooms, classrooms,
lecture halls, showrooms and galleries, alongside kitchens and kitchenettes.
`PublicSuite` samples compatible adjacent functions. These programs use the
existing `ProceduralIndoor` scene and capture pipeline. Cornell, object,
primitive and semantic-room scenes remain available.

```sh
cargo run --bin bevy_zeroverse -- --scene-type procedural-indoor \
  --indoor-layout lecture-hall --indoor-seed 10000
cargo run --example indoor_validate -- --layout gallery --seed 20000 \
  --audit-seeds 1024 --audit-geometry --renders 64 --cameras 3 \
  --width 512 --height 512 --labels --co-visibility --no-raw \
  --output out/gallery_qualification
```

The native/WebGPU controller's activity selector uses the same enum as CLI,
URL and Rust configuration. URL spelling follows the existing enum parser,
for example `indoor_layout=lecture-hall`. Changes use the existing explicit
Apply/Next controls. `Mixed` draws uniformly from the full list of room programs.

Courtyard metrics include the open roof fraction, slab thickness, column count, soffit and fascia materials. The public-program audit also records rooted beds, container species, paving, seats and direct and diffuse metered daylight.

Elevator landings and hallways can have upholstered banquettes or linked waiting seats facing the through aisle. A linked bank carries two to five bays on a common metal beam. Bent plywood, moulded polymer and upholstered seat/back pairs have different profiles. An occasional bay holds a side table. Door approaches and the main aisle stay clear of seated passengers' knees. A lobby directory can stand next to a lift bank with its face turned toward arriving passengers. Hallway capacity counts each installed bank as one station. The public-program audit separately records each linked bank’s individual seat count, position, orientation and finish. [Vitra Airline beam construction](https://www.vitra.com/en-un/product/airline)

When people are enabled, a passenger can sit on a shell or cushion, rest their hands in their lap, or gesture. The ordinary population draw determines how many people visit. A passenger who cannot fit keeps their standing placement. In the room manifest, `IndoorHuman::chair` identifies the bank and `waiting_seat()` returns its zero-based bay and seat height in metres. On linked banks the index includes occasional side-table bays. On banquettes it follows usable places along the cushions. The public-program audit lists the occupants of each bank.

Bike proportions use commercial indoor cycles as a reference. Treadmill placement uses the clearance specified in the Life Fitness Integrity operator manual. Camera proposals can frame the body of an admitted station alongside broader room and person views. The workout audit records accepted bikes, treadmills, mats and benches per room, aisle width and the instructor station where present. [Keiser M3i dimensions](https://www.keiser.com/shop/cardio-equipment/m3i-indoor-bike) · [Life Fitness Integrity operation manual, page 17](https://fitnesssuperstore.info/pdfs/Life%20Fitness%20Integrity%20Series%20Treadmill%20CLST%20Operation%20Manual.pdf)

The public-program audit records the table count, seats, supported objects, and each dressed table’s object ID, weave and maximum hem drop for restaurants and cafés. Rust clients can recover the same measurements from an archived table with `objects::tables::dining_linen`. The cloth retains its table’s semantic class and bounding box.

Yoga studios and some fitness rooms set out cork or foam blocks and covered bolsters. A class may share a set, individuals may bring their own, and whole rooms or individual mats can be clear. Blocks can lie flat or rest on edge. Bolsters have round, slim or rectangular sections and optional sewn handles. Each prop sits on one flat end panel of its mat, clear of fold hinges and the middle third. The manifest identifies each `YogaBlock` and `YogaBolster` and its supporting mat. Both use the `fitness_equipment` pixel class and have individual 3D boxes. Enabling prop motion allows them to slide, turn or lift and settle back onto their mat within the available clearance. The public-program audit records their construction and placement. Product dimensions draw on commercial yoga equipment. [Block sizes](https://eu.manduka.com/blogs/faq/help-center-yoga-props-manduka-eu) · [Rectangular bolster](https://eu.manduka.com/products/equa-rectangular-bolster)

Select `locker-room` (alias `changing-room`) or `LockerRoom` in a Rust/JSON configuration. Capacity counts installed locker banks. Visible taller compartments can contain a fixed shelf, garment rail or coat hooks. Each locker’s product program exposes the number of individual compartments through `LockerProgram::compartments()`. Each changing bench’s `interaction_target` identifies its locker, and opposing banks in an island reference one another. The layout reserves door approaches and a through aisle. Glazing sits above standing eye level and the entry has an opaque or louvered leaf. A folded towel or personal bag can occupy one end of a seat. Floors sample matte tile, poured terrazzo or resilient granulated stock. Slat and laminate thicknesses and locker proportions draw on commercial changing furniture. [Excelsior bench specifications](https://www.excelsior-cubicles.co.uk/changing-areas/changing-room-benches/) · [Penco locker configurations](https://www.pencoproducts.com/products/lockers/vanguard-metal-storage-lockers/)

In `laundromat`, each machine reserves 1.05 m in front of its loading face, connected to a cross-aisle. Machine dimensions and controls are shared within a bank. Drum loads and displayed times vary independently. The public-program audit records cabinet counts, dryer pockets, finish groups, folding stations and supported towels. Cabinet proportions draw on commercial front-load equipment. [Electrolux WH6-11 washer](https://www.electroluxprofessional.com/commercial-laundry-equipment/commercial-washers/front-load-washer-11-kg-wh6-11-WH6-11/) · [Electrolux TD6-14 dryer](https://www.electroluxprofessional.com/commercial-laundry-equipment/tumble-dryers/tumble-dryer-14-kg-td6-14-TD6-14/)

Laundromats and changing rooms can park rolling laundry carts beside a washer, dryer, folding table, locker or bench. Each cart keeps a standing position in an existing aisle and records the station it serves. A fleet can share its basket dimensions while individual carts carry different loads. Wire baskets have open grids and optional garment rails. Canvas hampers hang from folding frames. Empty carts, loose laundry and folded linen use the same collision and motion policy as other rolling carts. The frame, clothing and towels retain their own semantic labels. The audit’s `fill_fraction` gives the sampled target height relative to the basket depth. Individual bundles settle below it. Product proportions draw on commercial wire carts and canvas X-carts. [R&B Wire laundry cart](https://www.rbwire.com/products/standard) · [Rubbermaid collapsible X-carts](https://www.rubbermaidcommercial.com/material-handling/bulk-trucks/collapsible-x-carts/)

Fresh-produce stands carry removable crates on level or stepped decks. Open frames, panel ends and cabinets vary the view below the table. Wide decks have multiple handling bays. Crate footprints, setbacks, materials, fill and produce lots vary across the display. Each piece sits on the bin floor and clears its neighbors. Use `objects::public::produce_stand` to replay the layout and stock positions from a manifest. [Commercial tiered produce tables](https://cayugadisplays.com/products/produce/tables/island-produce-tables/multi-tier-tables-with-adjustable-shelf)

Grocery stores can include a chilled department. The open side of each cold case faces its reserved customer aisle. Cabinet ends can be insulated or glazed independently, and the rear intake has a recessed grille. Its shelves step back above the return grille and stock dairy tubs, gable-top cartons and bottled drinks in product runs with variable fill. The public-program audit records ambient, produce and refrigerated fixture counts per zone and the construction parameters of each cold case. Rust clients can read the same parameters from a manifest with `objects::public::refrigerated_case`. The case section uses commercial open multidecks as a reference. [Hussmann ID5SU cross-section and merchandising layout](https://www.hussmann.com/ns/Data_Sheets/DS_ID5SU_0539599_EN.pdf)

On ambient grocery shelves, the lower and upper decks can have different heights and depths. Cartons, sealed gusseted paper bags, cans and bottles share printed lots across neighboring facings. Groups can be freshly fronted, partly set back or sold through. Small packages may sit two abreast in a wide bay or run two deep toward the backing. The audit records the occupied product slots on both faces, including empty slots, paired facings and rear copies. Replay them from a manifest with `objects::public::gondola_stock(object, generator_version)`.

In `rehearsal-room`, both seating banks share a walkable center aisle. The music desk, paper and retaining lip tilt together. Full-size keyboards use 52 naturals and 36 raised accidentals. Compact versions have 61 or 73 keys. The dimensions use production furniture and instruments as scale references. [Manhasset music stands](https://manhasset-specialty.com/blog/model-48-49-50-and-54-symphony-music-stand-instructions/) · [Yamaha P-145 specifications](https://europe.yamaha.com/en/musical-instruments/pianos/products/p-series/p-145/specs.html)

Some ensembles also have a percussion section at the rear. One to three pedal timpani share a standing area connected to the center aisle. The semantic image labels drum surfaces as musical instruments, and the capture exports a separate oriented box for each drum. The room audit records the nominal diameters, playing height, bowl finish and pedal orientations. Drum sizes and proportions follow commercial concert instruments. [Yamaha TP-4300R dimensions](https://uk.yamaha.com/en/musical-instruments/percussion/products/timpani/tp-4300r/) · [Adams Universal bowl options](https://www.adams-music.com/en/percussion/timpani/universal)

Classroom activity tables use sheet laminate, a separate edge band and telescoping steel legs. Two tables form a six-sided group with a small installation gap. Each half keeps its own object ID and bounding box. Seat-facing notebooks, pencils and occasional personal items share each student’s writing space. The public-program audit records these relationships, dimensions, stock and seat headings per zone. The proportions draw on [Smith System activity tables](https://smithsystem.com/furniture/planner-trapezoid-activity-table/) and [Virco trapezoid tables](https://virco.com/product/5000-series-table-24-x-48-trapezoid/).

Floor guides beside lift banks identify the landing and the highest served level. Stair-hall guides identify the lower or upper landing from the stair deck elevation. The printed insert fits its stand in portrait or landscape format. A folding stand carries the same guide on both outward faces. These placements follow the use of floor identification at vertical circulation points. [Virginia Tech campus wayfinding plan](https://www.facilities.vt.edu/content/dam/facilities_vt_edu/planning-and-construction/Campus-Wayfinding-Master-Plan.pdf)

## Construction and use

| Program | Construction and placement |
| --- | --- |
| Workshop | Office-based seminars and maker spaces. Standing assembly benches have timber or ruled cutting surfaces. Joinery benches have drilled tops and left- or right-hand face vises. An offcut can lie flat or stand gripped between the jaws. Each piece has its own transform, bounds, semantic class and support record. The layout reserves operator clearance along the working edge. The Workshop audit exports bench construction, contents and contact for each seed. [Sjöbergs Nordic bench dimensions](https://sjobergs.se/app/uploads/2020/04/Sjobergs-produktblad-Nordic-1450.pdf) |
| Studio | Office studios, private painting stations and shared ateliers. Tripod and H-frame easels adjust in height and tilt and can hold raw linen, primed canvas or painted work. The layout reserves room for the artist. A linked still life is a supported vase or bowl with a clear sightline from the working side. Painted still-life studies take their silhouette from that vessel’s profile. Other canvases carry procedural landscapes or abstract brushwork. The `studio` audit records each station and its subject. [Studio easel hardware reference](https://www.mabef.com/en/products/studio-easels/m09) |
| Reception | Waiting rooms may include an arrival-facing check-in counter. Staff work behind a higher visitor ledge, sometimes with a lower signing wing at either end. Walk-up space, staff seating and supported equipment share the accepted floor plan. The two working heights follow dimensions used in commercial modular desks. [Manufacturer drawings](https://mdd.pl/media/product-modular-pdf/VALDE-components.pdf) |
| Library | Reading rooms can have perimeter bookcases, freestanding stacks or parallel browsing aisles. Each side has space to browse. Some shelves feature an aisle-facing cover on a bent-wire or grooved timber stand. Spine-out runs pack from either end, with leaning volumes supported by an upright neighbour. The `library` audit records stack dimensions, finishes, shelf counts, featured titles and seating for each seed. [Shelf display examples](https://www.demco.com/products/library-supplies/shelf-merchandising) |
| Hallway | A straight passage with rooms on one or both sides, fitted doors, glazed or opaque frontages, optional wall guards and waiting seats. Door approaches and the continuous walking aisle are reserved before furnishing |
| Stair hall | A double-height foyer with a lower entrance, guarded staircase, wall-backed upper gallery and optional intermediate landing. Flight width, direction, framing, tread finish and balustrade vary with the seed |
| Courtyard | A paved open-air garden enclosed by four covered walks. The entrance walk passes between columns. Garden islands hold individual specimens or staggered groups in soil, gravel or bark. Container gardens and outdoor benches occupy the remaining bays |
| Elevator landing | Telescoping door banks, recessed jambs/thresholds, call controls, indicators, bounded illuminated car cavities, handrails, waiting benches and information stands |
| Cafe | Modular service counter, glazed display case with real shelves/contents, coffee equipment and beverage dispensers, small tables, banquettes and contextual dishware. Some venues fit their dining tables with coordinated linens |
| Cafeteria | Serving counters, shelf or slide-rail tray returns, beverage taps, optional guarded hot-food wells, larger dining tables, reserved serving queue and through circulation |
| Restaurant | Table-service groups, paired booths, upholstered banquettes and a waiter station with staff access. Restaurants can dress all, some or none of their tables. Tables share the venue’s woven stock, with individual cuts and hem depths. Dishes rest on the textile at tabletop height |
| Fitness room | Free-weight benches, dumbbells and mats, with optional barbell stations, stationary bikes and treadmills. Full cages and half racks have bored uprights, adjustable catches and bare or symmetrically loaded bars. Each treadmill reserves 2 m of clear floor behind the belt and 0.5 m on either side |
| Yoga studio | An instructor mat faces participant mats across a clear aisle. Timber or resilient floors support roll-out and folding pads with pigmented grip or cork facings. Open racks store spare rolls and reserve space to reach them. |
| Laundromat | Banks of front-loading washers and single- or double-pocket dryers share clear loading aisles. Folding tables can carry stacks of towels. |
| Cycling studio | Adjustable bikes from a shared equipment line face an instructor in staggered banks. The front of the class can occupy either end of the room. Each station reserves mounting space and the central aisle stays open |
| Gymnasium | Full-size and community volleyball courts with inlaid lines, hardwood floors and high lighting. Sidelines can have benches or bleachers. Multipurpose halls can also have one or two wall-mounted basketball goals with clear shooting approaches. |
| Grocery store | Open refrigerated multidecks, double-sided dry-goods shelves, partitioned produce islands, a staffed conveyor or basket register, optional self-checkout stations and clear browsing routes connected to crossing aisles |
| Clothing store | Sales rails with shirts, dresses and outerwear, folded-stock tables, a cash wrap, optional self-checkout stations and single or paired fitting rooms facing a shared aisle. Collections share cuts and dye lots |
| Locker room | Dry changing rooms combine fitted storage, occasional back-to-back locker islands and backless benches. Each bench records the bank it serves. Individual compartments, open leaves and towels retain their geometry in every view. |
| Rehearsal room | Chamber seating ranges from shallow ranks to curved sections around a rehearsal leader. Player-facing music stands hold printed parts. An optional accompanist has a stage piano and a fitted seat. |
| Classroom | Board-facing rows, mixed classrooms and small groups. Each trapezoidal activity table can stand alone or join a second table at its long edge. Chairs face their own writing surface. A central aisle and teaching area constrain the installation. |
| Lecture hall | Curved audience rows, linked-frame folding seats, explicit floor tiers, intermediate stair treads, teaching platform and guards at exposed drops |
| Showroom | Product plinths, information stands and vitrines; coordinated content collections and shared finish identity |
| Gallery | Angled exhibition panels with framed artwork, sculptural plinths and vitrines; separate viewing/circulation space and exhibit lighting |
| Public suite | Compatible function pairs with sampled split direction/position, open shared boundaries and one authoritative primary camera region |

In classrooms and lecture halls the low edge of the lectern faces a level standing area 0.72–0.94 m deep. The presenter looks across the sloped desk toward the seats. Placement samples both sides of the teaching area and checks sightlines toward the seating banks against the teaching board and structural obstacles. Standing presenters can use that space with a sampled offset and heading, subject to the usual body, furniture and floor checks. A paper stop holds optional printed sheets or cue cards. The room clutter policy controls their presence. Microphones can be flush-mounted or carried on a bent neck, with the socket on either side. Paper and microphones retain their own pixel labels within the lectern object box. [Safco stand-up lectern](https://www.safcoproducts.com/products/stand-up-lectern-8915) · [Shure gooseneck microphones](https://content-files.shure.com/Pubs/MX405/us_pro_mx405_mx410_specsheet.pdf)

In `gymnasium`, full-size volleyball courts measure 18 × 9 m, with 50 mm lines, a 3 m free zone and a centre net height of 2.24 or 2.43 m. Smaller practice courts vary both dimensions and net height. The manifest records the playing area, clearances, orientation and court centre. The line pigment is selected against the wood finish so markings remain legible on pale maple. Floor paint shares the floor semantic. Padded posts, tape and individually modelled cords share an object box and the `sports_equipment` semantic. Depth and co-visibility can see through the net openings. [FIVB Official Volleyball Rules, sections 1–2](https://www.fivb.com/wp-content/uploads/2025/01/FIVB-Volleyball_Rules2025_2028-EN-v05.pdf)

Practice baskets can use an end wall or a side wall. Paired stations face each other and share a mounting height and backboard specification. Installations require opaque backing, a clear floor approach and space outside the volleyball run-off. Adult rims sit at 3.05 m, and lowered practice hoops sample 2.58–2.89 m. The orange steel ring has a nominal 450–459 mm inside diameter and twelve attachment points for the net. A basket exports one object box and the `sports_equipment` label. The gym's capacity counts its marked volleyball court. [FIBA basketball equipment, section 1](https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2024-official-basketball-rules-and-basketball-equipment.pdf)

Some multipurpose halls mark a shooting station in front of an installed basketball goal. Adult stations put the far edge of the stripe 4.60 m in front of the backboard. A full stripe spans 3.60 m and can carry a semicircle of radius 1.80 m. Lowered practice baskets can have a shorter throw. A narrow bay can take a short stripe, and some hoops have bare timber. The line follows its own board orientation and leaves room for a shooter clear of the net and fixed furniture. The two sports use separate pigments. At a crossing, the volleyball line stays continuous. Timber and paint share mesh edges and the floor label, including at oblique camera angles. The room manifest exposes the goal ID, stripe centre, forward direction, dimensions and semicircle in `program.free_throw_marks`. [FIBA court dimensions, article 2.5](https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2024-v10a.pdf) · [FIBA backboard placement, section 1.1](https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2024-official-basketball-rules-and-basketball-equipment.pdf)

Where the sideline is deep enough, a gymnasium can fit a freestanding bank of two to five tiers. Timber benches or individual moulded seats flank a stepped aisle. Row pitch, rise, bank length and aisle position vary within each installation. Placement leaves at least 0.70 m between the front of the bank and the court run-off and checks the door approach. Room cameras can look diagonally across the playing surface toward the seats as well as along the net. Each bank has one object box and the `bleacher` pixel label. The public-program audit records its orientation, construction and spectator capacity separately from playing courts. The dimensions and seat profiles draw on commercial telescopic gym seating. [Hussey MAXAM dimensional guide](https://www.husseyseating.com/content/MAXAM/MAXAM-Brochure.pdf)



For a hallway, the primary camera and O-voxel region is the passage. Adjacent offices and lounges are visible through glazing and doorways. The topology export identifies which two regions each door connects.

A stair hall exports the ground, upper deck and individual tread tops as metric support regions. The upper gallery can hold facing seats, a freestanding exhibit, a directory or open standing space. Its route reservations connect the ground entrance to the gallery and keep the arrival clear of furniture. It represents a single storey of rise.

Courtyard garden islands have cast curbs, masonry walls with jointed terracotta coping, or folded metal troughs with a recessed foot. Each group has one to three crowns. The bed shares its woodland, tropical or dry-garden collection with the surrounding container plants. Each crown scales to fit the available soil and headroom while retaining its species proportions. Roots, bark chips and gravel follow the height of the mounded substrate.

Use `plant_species_name(bed.variant)` for the principal specimen. The public-program audit’s `garden_beds` entries include the fitted object and its planting details from `objects::audit::program`, with each crown’s root and dimensions in local metres and yaw in radians. The manifest stores the collection alongside the species in the bed variant. A custom bed can also supply a plain species index.

A courtyard archives the bounds of its rectangular sky opening, canopy thickness and finishes. The light meter projects both edges of the opening onto the floor for sun and diffuse sky. Ceiling cameras and luminaires attach to the covered walks. A moving camera’s full mounting rail must stay on the canopy. Sky seen through the opening uses the same finite depth and sky label as the other exterior views.

Fitness rooms fit the dumbbell shelf and its approach first. In a narrow bay the shelf can turn lengthwise or use a shorter commercial size. After fitting a bench, either side of the central aisle can receive a barbell station. Strength-oriented rooms request one more often. Placement reserves 0.72 m past each sleeve and 0.91 m in front of the rack. Each station counts once in workout capacity. The public-program audit records its pose, frame, J-cup and safety heights by pin index, plate stock and paired loads. All components share the station’s instance and `fitness_equipment` pixel class. The camera planner can see through the cage. The bar is racked and static. Bar lengths, sleeve diameters, plate profiles and rack proportions draw on commercial equipment. [Training bar](https://media.eleiko.com/admin/download-product-sheet.aspx?articlecode=3002551&language=en-us&view=true) · [Half rack](https://media.eleiko.com/admin/download-product-sheet.aspx?articlecode=3060249-03&language=en-us) · [Training plates](https://eleiko.com/en-us/equipment/plates/weightlifting/3085440-20-eleiko-iwf-weightlifting-training-plate-20-kg)

Free-weight storage uses open rails or rubber-lined trays for flat-sided dumbbells and individual V saddles for round heads. Two- and three-tier frames fit the admitted footprint. Lower shelves accommodate larger pairs. A set can be full, missing both weights of a pair, or have one weight checked out. Empty saddles stay on the frame. Colour-coded sets carry recessed, mass-marked inserts with a shared mipmapped print sheet. The workout audit records the dimensions, orientation and occupancy of each pair. [Flat rack reference](https://www.lifefitness.com/en-us/catalog/strength-training/storage-racks/life-fitness-flat-dumbbell-rack) · [Saddle rack reference](https://www.lifefitness.com/en-us/catalog/strength-training/storage-racks/life-fitness-saddle-dumbbell-racks)

Retail capacity counts installed merchandise fixtures. Their stock shares the fixture’s bounding box and rigid transform. Pixel semantics distinguish fixture structure (`retail_fixture`), food, drink containers and clothing. Checkout structure and terminals use `counter` and `control_panel`.

Checkout forms follow the merchandise. Grocery lanes have an inset scanner and a separate receiving or conveyor surface. Apparel wraps provide a continuous folding surface, cashier-side storage and an optional suspended wrapping roll. Self-checkouts have a shopper-facing terminal, a lower bagging platform and an independent card reader. One attendant can serve a single or paired bank within 6.2 m of the staffed till. The placement validator checks each customer approach and the cashier's working side. The public-program audit exports these forms, poses, dimensions and finishes. The Rust accessor is `objects::public::checkout_station`. For a mesh audit at the installed size ranges, run `cargo run --example audit_objects -- --category CheckoutCounter --sample-dimensions --output out/checkout_audit`. Checkout screens use static, shop-specific sale or payment pages. [Commercial scan and bagging station reference](https://www.ncrvoyix.com/dam/retail/docs/NCR-Voyix-Hardware-Fabric-Datasheet.pdf)

Tall clothing displays can present a collection side-on along a rail or face garments toward the shopping aisle on straight or sloped arms. On sloped arms each occupied hook rests against a retaining pin. Adjacent copies share a sewn pattern and dye lot, with individual sizes, spacing and swivel clearance. Contrast sleeves carry their own fabric dye through the lining and cuff. The public-program audit records arm and hanger counts, positions and orientations. The object audit includes the garment cuts and exact arm geometry. [Commercial garment-rack hardware](https://www.grand-benedicts.com/apparel-racks/retail-apparel-merchandise-racks-by-grand-benedicts.asp)

Low clothing tables carry folded tees, pullovers, button-down shirts and trousers. Apron frames, panel-end tables with stretchers and recessed plinths support the displays. Deep tables can present stock on opposite faces when a second clear browsing aisle connects to the cross-aisle. Wall-backed tables face the admitted shopping aisle. Piles can run low in any bay. Each has its own dye lot, garment sizes and handling offsets. Exposed pieces show bound neck openings with a cloth backing, pointed collars and buttons, or a waistband and pocket. The public-program audit records the table height, cut, dye, orientation and support height of each folded layer. Rust clients can replay the installation with `objects::public::folded_apparel(object, generator_version)`.

Fitting rooms occupy shop perimeter after the checkout and two merchandise fixtures have clear access. Remaining stock fits around the changing area. A paired unit shares a partition and dye lot. Each bay independently samples curtain draw, pull direction and fold count, a supported bench and a backed dressing mirror. An opaque silvered finish reflects the existing room probe. The entrance faces a clear approach connected to a public cross-aisle. The cubicle is solid for camera and person placement even when the curtain is open. The instance box covers the assembly; pixel labels distinguish the curtain, mirror, bench and retail structure. The public-program audit records each bay’s clear opening in metres.

Dimensions, row pitch/curvature, aisle width, density, group rotation, level
rise, display spacing, construction state, contents and finish identity are
seeded parameters. A room name selects functional requirements; it does not
just bias the old four office activity weights. No external mesh or image
assets are required by the new object builders.

Required anchors are admitted before optional content. A bounded search finds
usable space within the actual polygonal envelope. An unavailable required
function causes validation to fail; unrelated furniture cannot satisfy it.
Physical object bounds, supported contents, mounted equipment and reserved
circulation are checked separately. Doors and folding seats sample static
states within their pre-admitted bounds. Folded auditorium seats are excluded
from seated-person sampling.

Lecture tiers and treads are ordinary authoritative `FloorPatch` geometry;
construction, camera clearance, human placement and annotations consume them.
Exposed drops have guards in both meshes and collision queries. The current
ARDY planner excludes unsupported stair traversal; it does not claim to
synthesize stair-climbing motion. Static people and safe level-floor motion
remain supported without loading a motion model when motion is disabled.

Dining areas flank the public aisle. The seated area on each side starts beyond
its service counter, display or queue, and stops at any cross aisle. Separate
sides can have different spacing and table headings. Broad areas can fit two
runs of tables. Venue density controls how many of the available places
receive a table. Cafeterias also sample communal tables in rows. Dataset
metrics report tabletop-to-aisle clearance in metres and table-centre position
across each bay (zero at the aisle, one at the outer edge) for independent
dining groups alongside the aisle.

At café booths, the usable cushion width and the facing edge of the actual
tabletop determine how many places can be served. Settings fit the tabletop,
stay within 90 cm of their diner and share the normal clutter budget. A booth
may be empty or partly served. `SeatsAt` links a dining banquette to its table,
and `Serves` links each placed dish to its chair or banquette. Perimeter waiting
benches have no table assignment. The `dining_banquette_places` and
`dining_banquette_served_dishes` metrics count places and dishes per paired
banquette. The room's seating-station capacity counts the whole bench once.
Cutlery sets can share a metal finish and a timber or moulded grip.

Restaurants, cafés and some cafeterias set out folded cloth napkins beside
served dishes. A venue stocks a cloth size, weave and usual fold. Tables can
use another fold, and each diner can have a slightly different lay or no napkin.
Book and triangular folds have two fabric leaves. Standing tent folds have an
open space between their feet. Each cut has a turned hem and a pressed crease
in the same woven mesh. Each `TableNapkin` has its own bounds, uses the
`towel` semantic class, and records both its supporting table and the seat it
serves. The public-program audit exports the fold and weave in `dining[].napkins`.

Dining tables can serve water in short rocks glasses, tall highballs, faceted bistro tumblers, stemless glasses and footed goblets. Restaurants favor stemware more often than cafés and cafeterias. A venue can share glass dimensions, shape and tint across tables, while each serving has its own fill level or stays empty. The existing clutter budget and clear tabletop area also apply to glassware. Each `DrinkingGlass` has its own object bounds and uses the `drink_container` semantic class. The public-program audit records its construction and water depth in `dining[].glassware`. The pressed-glass family draws on the proportions of commercial café tumblers. [Duralex Le Picardie](https://eu.duralex.com/en/products/picardie-transparent-cup-table?variant=46598810108210)

Served bowls sample a house soup and individual orders. Stock, tomato, squash and
pea soups vary in pigment and surface roughness. Herbs, beans, diced vegetables
and croutons can float in loose clusters or across the bowl, and some servings
are plain. Object metadata records the recipe, fill elevation and individual
ingredients. Dataset metrics expose `dining_soup_recipe`, `dining_soup_garnish_count`
and `dining_soup_fill_height_m`.

Café display cases stock trays of biscuits, oval rolls, ring bagels and paper-lined muffins. Each batch shares its recipe and nominal portion size. Stock can fill loosely spaced rows, scatter on a board or sell out. Portions rest on the tray and fit below the next shelf. A bagel has an open hole through its mesh, and a muffin's pleated liner carries the paper semantic label. The serving board, food and liner share their cabinet's instance box. Crust and crumb use separate color, metric normal and roughness maps. The public-program audit records the shelves, recipes, portion counts and poses under `bakery_cases`. Bagel diameters include the size of a typical sandwich bagel. [King Arthur Baking sandwich bagels](https://www.kingarthurbaking.com/recipes/ultimate-sandwich-bagels-recipe)

Cafeteria hot lines use 530 × 325 mm Gastronorm openings. Each well accepts a full pan,
two halves, three thirds or a half with two quarters. Independent pans hold soup, roasted
vegetables, potato gratin, a lid or an exposed steel basin. Dry sides favor shallow pans. A gratin can be freshly baked or scooped away along one edge. Rear controls face the staff aisle. Installation
requires a supported counter, an unobstructed guest-side landing and clearance in front of
the beverage taps. The archive distinguishes food, controls and appliance geometry. Dataset
metrics include `hot_food_pan_contents`, `hot_food_pan_depth_m` and `hot_food_glass_guard`.
[Bartscher pan dimensions](https://www.bartscher.com/int/en/product/711150/gn-container-1-1gn-d150-basic-line) ·
[Two-half-pan countertop unit](https://www.bartscher.com/pl/en/product/200232/bain-marie-212-150) ·
[Vollrath guarded serving equipment](https://www.vollrathfoodservice.com/products/countertop-equipment/merchandising-display-solutions/display-cases-cabinets/heated-display-cases/40732)

Café and cafeteria queue barriers follow the edge of the service lane. Hand-set
posts can sit at a slight angle when the full base and the public aisle have
clearance. Retractable webbing runs from a reel to a latch, ropes sag between
metal end sockets, and rigid dividers use round or square pickets or a panel
gripped on both faces by four brackets. Dataset metrics record post spacing,
installed yaw and connection height. Sag applies to flexible spans only.

## Export contract

`IndoorManifest::functional_programs()` exposes required and accepted members,
capacity with an explicit unit, sampled parameters, rejected candidate counts, reserved spaces,
focus regions and service relationships. `assembly_diagnostics.mounts` records
ceiling equipment and its physical host. `room_topology()` includes both real
partition portals and open shared boundaries. IDs refer to the ordinary object
instances, preserving individual OBBs and semantics.

`layout::functional::support_regions()` returns named support surfaces with
metric elevations and exact clipped triangles. Bounds are only an acceleration
structure: the ground surface excludes tier patches, while an overhead deck
can overlap a lower floor in X/Z. Capture metadata exports these relationships
for functional scenes; ordinary scene preparation does not construct this
reporting triangulation. These are architectural surfaces; furniture rugs retain
their exact contours and elevations separately in `IndoorManifest::rug_supports`.
The live `floor_height()` query includes those rug overlays for placement.

The semantic taxonomy includes elevator, control panel, display case, tray return,
queue barrier, lectern, locker, plinth, sculpture, information stand and
extraction hood. Existing semantic IDs and colors stay stable. Embedded
contents retain their own semantic classes and the assembly's object instance.
The elevator car's context pieces carry explicit voxel-exclusion tags that
survive material-finish tagging. Depth and semantic capture still see them,
preventing open doors from exposing a void.

RGB, depth, normal, position, semantic, co-visibility, flow, OBB, pose and
schematic data use the production meshes and existing glass policy. Floor and
camera coordinates remain metric. Voxel exports retain the existing static,
single-timestep restriction. The new programs do not change GPU-to-Burn
transport or introduce per-room model initialization.

## Reproducible qualification

- `audit_public_programs` runs every requested seed, optionally building all
  actual meshes and validating camera/people placement. Its report retains
  failures and capture-engine provenance.
- `audit_public_cameras` sweeps 1–12 cameras and baseline 0, 0.5 and 1 over a
  declared population. It exports overlap-oracle samples, geometric baselines,
  placement diagnostics and metric schematics.
- `indoor_validate` captures production RGB and aligned annotations, writes a
  schematic from recorded capture calibration, and verifies rendered geometry.
- `bevy_zeroverse_publication program-study --input ... --output out/...`
  verifies complete, current-source native cohorts and builds a local gallery
  containing every room and annotation. It cannot write a matched study into
  the published project-page directories.

The report distinguishes requested capacity, admitted content, geometric
validity, rendered visibility and elapsed capture time. Seed diversity and
embedding spread are diagnostics; neither establishes photographic realism
or downstream reconstruction gains. Physical reference comparison, dedicated
hardware-browser qualification and fixed-budget downstream training remain
separate acceptance gates.

## Qualification scope and limits

The ten-profile native gallery contains 64 rooms per selected profile, including
PublicSuite, for 640 rooms and 1,920 calibrated 512×512 views. All aligned
RGB/depth/normal/position/semantic/co-visibility captures passed. Views contain
4–30 semantic classes; maximum geometry reprojection error is 0.00180 pixels.
The Rust report includes all rooms, 108 camera-count schematics, 90 fixed-camera
object examples, source-bound receipts and a generated paper figure/table.

The local campaign uses 1,024 consecutive seeds per profile for the nine named
programs and PublicSuite (10,240 requests). These are stratified, seed-matched
profile populations, not 10,240 IID draws from a proposed default mixture.
Every request passes layout constraints and actual constructed-mesh validation.
A separate 1,280-request boundary sweep uses density 0/1, three cameras and full
human density. Camera qualification covers 16 seeds per named profile, counts
1–12 and baselines 0, 0.5 and 1: 5,184 configurations. All requests, including
any failures, remain in the audit receipts.

Manifest generation P50 ranges from 0.36–1.18 ms across these profiles; P95 is
0.68–2.90 ms. These measurements exclude cameras, people, material synthesis,
rendering and export. Actual geometry construction **plus validation** has
P50 16.6–31.0 ms; it is not a production preparation timing. Native capture
latency and GPU tensor delivery are measured separately on shared hardware.

Laundromat capacity counts washer and dryer cabinets. An installed stacked dryer contributes one cabinet and two independently loaded pockets. To inspect the banks, loading aisles and folding tables across consecutive seeds, run `cargo run --example audit_public_programs -- --profiles laundromat --rooms 1024 --density 0.65 --output out/laundry/audit.json`.

Other capacity units include chair stations (including folded stations), display
stations and elevator cars. Folded stations do not admit seated people. Kitchen
service-run receipts use their existing separate schema; an empty public-program
capacity array is not evidence that the kitchen has no seating.

The fixed-envelope object audit samples 64 constructions for each of the 15
new categories. Six orthographic 24×24 depth maps measure shape variation
without rewarding scale or material changes. Elevators, lockers, queue barriers
and sculptures vary substantially at this resolution. Several simpler fixtures
have only one separated representative; their dimensions and finishes vary,
but broad fixed-envelope shape diversity has not been demonstrated. Thin and
internal details can be under-resolved by this diagnostic.

Two matched linear-radiance Cycles comparisons at 512 samples, with independent
reference repeats, expose a lighting limitation. Mean native/reference luminance
is 1.006 for cafe seed 14000 and 2.976 for gallery seed 26000. Native relative RGB
MAE is 0.341 and 2.130 respectively. The gallery ceiling is particularly bright.
Disabling GI reduces its mean ratio to 1.660 but worsens much of the local
illumination match; GI remains enabled. These are targeted diagnostics, not a
population-wide physical accuracy score or evidence of photographic equivalence.

The WebGPU viewer compiles with optional motion enabled; an M4/Chrome runtime
and responsiveness qualification has not been performed on this host. GPU tensor
parity covers two frames, shifted principal points, both glass annotation policies,
and retained allocations. Motion-planning checks cover safe level-floor routes,
not ARDY stair climbing or a new model-quality evaluation.

Mixed suites currently compose compatible pairs across a sampled open boundary.
Arbitrarily oriented structural partitions, selectable unions of reconstruction
regions, multi-storey motion, richer wet-service routing and default-mixture
promotion remain separate work. Existing irregular exterior envelopes are still
supported. Do not interpret the room catalog, exact structural fingerprint count
or an embedding spread as a claim of ten-million-sample learning utility.

The headless CLI also emitted two elevator-landing datasets with GPU O-voxels
at resolution 64 and schematic sidecars. Their 67,921 and 67,137 voxel coordinates
are sorted, unique and in bounds; voxel AABBs equal the annotated reconstruction
AABBs, and the semantic palettes contain no exterior sky. A 32-room lecture-hall
JIT probe delivered three 512×512 views plus depth/normal/position/semantic tensors
at 0.75 rooms/s with completed GPU consumers, full shadows and diffuse GI. This
shared-device run overlapped rendering and training; it does not establish an
isolated throughput regression bound.

### Café service counters

A café can serve from a compact brewer or a staffed espresso bar. A commercial station fits a one-, two- or three-group machine and a separate grinder on the staff edge. The grinder can sit on either side. A clear section beyond the brewing machine accommodates pickup cups. Counter depth, yaw and the real footprints of both appliances determine whether the pair fits. Product proportions use commercial examples as a reference. [Linea Classic S](https://www.lamarzoccousa.com/commercial-products/espresso-machines/linea-classic-s/) · [E65S grinder](https://www.mahlkoenig.com/products/e65s)

The machine has a raised drip grate, space for cups under the spouts, portafilters, controls and steam wands. The grinder has a transparent hopper with a sampled fill level. Both objects keep independent poses, bounding boxes and the appliance semantic class. The public-program audit records the supporting counter, both objects, brew-group count, cup clearance, hopper fill and pickup-cup count in `dining[].espresso_bars`.

### Drinking water in exercise and transit spaces

Fitness rooms, cycling and yoga studios, gymnasiums, hallways and elevator landings can contain a freestanding bottled-water cooler. The generator sites it after the primary equipment. It checks the cabinet and a standing approach facing the taps against the building, door approaches, exercise stations and circulation routes. A venue can omit the cooler where space is tight.

The cooler can have a bottom-loading cabinet, an open top receiver or an inverted three- or five-gallon bottle. Bottle ribs, shoulder, fill volume, outlets and controls are sampled per unit. The drip tray has an open cup bay. The chassis uses the appliance semantic class and the exposed bottle uses the drink-container class. Both belong to one appliance bounding box. The public-program audit lists installed units in `hydration`. Dataset metrics include capacity, water volume, outlet count and user clearance. Dimensions and the loading arrangements use commercial bottled coolers as references. [Avalon cooler comparison](https://www.avalonh2o.com/pages/compare-water-coolers) · [A14 bottom-loading dispenser](https://www.avalonh2o.com/products/a14-bottom-loading-water-cooler)

### Handbaskets in shops

Grocers and some clothing stores offer handbaskets beside the entrance or near a staffed checkout. The stack sits beyond the aisle edge, with room for a shopper to face it from the clear route. The shop can omit it when no site fits. Adding baskets preserves the merchandise and checkout capacity.

A stack contains one to seven compatible tapered baskets. Slotted molded bodies have open apertures and a solid bottom. Welded wire bodies have bent rods that continue across the base. The carrying handles fold alongside each basket so the next can nest. Some stacks sit on a four-caster dolly with an optional push bar. Basket bodies and handles use the `shopping_basket` semantic class, the dolly uses `retail_fixture`, and the assembly has one bounding box. The public-program audit exports the stock, pose and number of baskets under `basket_stacks`. Product proportions and stacking arrangements draw on commercial examples. [Wanzl shopping baskets](https://www.wanzl.com/en_GB/products/baskets/shopping-baskets/wa-series~p1678) · [INTECH stackable handbaskets](https://www.intech-world.com/en/product_section_injection_molding/en_retail/shopping_basket/shopping_basket.php)

### Sculpture on plinths and inside vitrines

Gallery and showroom exhibits include asymmetric carvings and castings with one or two through-holes. The body bears on a plinth. Standing and horizontal carvings vary in mounting angle, thickness, shoulder shape, aperture positions, taper and bore finish. A view through an opening sees the actual geometry behind it in RGB, depth, normals, semantic labels and co-visibility. The program uses existing stone-like concrete, timber, metal, ceramic and terracotta finishes. The object audit reports the fitted carving in metres. Scene metrics record the installed forms, carving axes and lengths of the openings. A museum example illustrates how carved negative space changes with viewpoint. [The Hepworth Wakefield, Pierced Hemisphere I](https://hepworthwakefield.org/our-art-artists/collections/highlights/pierced_hemisphere_i/)
