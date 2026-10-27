//! Cabinet chassis and contents share their physical compartment boundaries.
use super::*;

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let t = p.panel_m;
    let l = o.kind.class_name();
    let rear = -p.body_depth_m * 0.5 + t;
    let front = p.body_depth_m * 0.5;
    let inside_front = front - if p.front == 4 { 0.075 } else { 0.045 };
    let top_bottom = s.y - t;
    a.box_part(
        p.frame,
        l,
        Vec3::new(0., top_bottom * 0.5, rear - t * 0.5),
        Vec3::new(s.x, top_bottom, t),
        0.002,
    );
    // One full top preserves the declared support for items placed on cabinets.
    // Uprights end below it, avoiding overlapping coplanar top faces.
    a.box_part(
        p.shelves,
        l,
        Vec3::new(0., s.y - t * 0.5, 0.),
        Vec3::new(s.x, t, s.z),
        0.002,
    );
    for (index, x) in std::iter::once(-s.x * 0.5 + t * 0.5)
        .chain(p.bays.iter().map(|b| b.right + t * 0.5))
        .enumerate()
    {
        // Keep internal dividers behind both sliding-door tracks.
        let end = if p.front == 4 && index > 0 && index < p.columns {
            inside_front
        } else {
            front
        };
        a.box_part(
            p.frame,
            l,
            Vec3::new(x, top_bottom * 0.5, (rear + end) * 0.5),
            Vec3::new(t, top_bottom, end - rear),
            0.002,
        );
    }
    for bay in &p.bays {
        for &y in &bay.levels[..bay.levels.len() - 1] {
            a.box_part(
                p.shelves,
                l,
                Vec3::new((bay.left + bay.right) * 0.5, y, (rear + inside_front) * 0.5),
                Vec3::new(bay.right - bay.left, t, inside_front - rear),
                0.002,
            );
        }
    }
    if p.front == 3 {
        let (inside, shelves) = content_program(o, &p);
        for (index, bay) in p.bays.iter().enumerate() {
            for level in 0..program::open_levels(&p, bay) {
                for mut child in bookcase::contents::placements(&inside, &shelves, index, level) {
                    child.position.z -= 0.0225;
                    bookcase::contents::build(a, &inside, &child);
                }
            }
        }
    }
    fronts::build(a, o, &p);
}

pub(super) fn content_program(
    o: &IndoorObject,
    p: &ShelfProgram,
) -> (IndoorObject, bookcase::BookcaseProgram) {
    let mut inside = o.clone();
    inside.size.z = p.body_depth_m - 0.045;
    let shelves = bookcase::BookcaseProgram {
        open_frame: false,
        closed_back: true,
        panel_m: p.panel_m,
        fill: p.fill,
        bays: p
            .bays
            .iter()
            .map(|b| bookcase::ShelfBay {
                left: b.left,
                right: b.right,
                levels: b.levels.clone(),
            })
            .collect(),
        frame: p.frame,
        shelves: p.shelves,
    };
    (inside, shelves)
}
