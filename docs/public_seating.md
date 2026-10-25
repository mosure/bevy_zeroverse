# Lecture and cafeteria seating

Public programs sample seating arrangements before architecture and furniture construction. Chair-only lecture halls, halls with shared writing desks, cafeteria table rows, and small dining-table clusters all remain possible. These choices are independent of the regular-versus-loose chair placement policy.

`ProgramParameters::table_rows` records the seeded `TableRowSeating` contract:

- Lecture halls select shared desks with probability 0.55. Each bank has multiple seats facing the teaching area. Desk depth varies from 0.50 to 0.64 m.
- Cafeterias select shared table rows with probability 0.60. Each accepted table has at least two chairs on each long side. Table depth varies from 0.72 to 0.94 m. Constrained regions can retain the small-table arrangement if no complete shared row fits.
- Table length follows usable bank width and a sampled coverage fraction. Shared tops use rounded rectangular outlines with mild taper and independently sampled supports, keeping a usable surface across all stations. Freestanding tables retain the wider shape catalogue. Table height, chair dimensions, vacancies, lateral offsets, swivel angles and table orientation vary within coordinated sets. Some sets remain orderly.
- Passages behind rows vary from 0.65 to 0.95 m, in addition to the central circulation aisle. These are procedural clearances, not a building-code compliance claim.

Lecture tier boundaries are computed from the same plan as the desks and seats. Table rows require deeper tiers than chair-only rows. Furniture must fit entirely on one support level; local recovery cannot move a chair across a step or to the other side of its table. Tables and chairs pass the ordinary support, wall, portal, column and overlap checks. Chair-to-table relationships are established before checking the actual chair and table-leg clearance models. Subsequent seated-person placement and chair insertion use these relationships.

A row is accepted as a complete seating group: an unsuccessful group is removed before props, people or annotations can reference it. Accepted passages are reserved against later furniture. Table/desk and chair instance identities, semantic classes, bounding boxes, surface contents and schematic footprints use the existing export paths. Older serialized parameter records without `table_rows` deserialize with the original seating policy.

The `layout::functional::table_rows` tests cover consecutive lecture-hall and cafeteria seeds across four densities, accepted table/seat relationships, support levels, floor and leg clearance, routes, mesh envelopes, serialization and deterministic replay. Local audit records and schematic previews are written under `out/table_rows/`. This is a structural diversity audit; it does not establish a perceptual dataset-size multiplier.
