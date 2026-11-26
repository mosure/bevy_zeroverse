use super::*;

/// Horizontal section through a retainer. X starts at the pane edge and
/// increases into the pane. Z is measured across its thickness.
pub(super) struct Section {
    root: f32,
    neck_half: f32,
    web: f32,
    contact: f32,
    inner: f32,
    outer: f32,
    tip: f32,
    bevel: f32,
    lip: f32,
}

impl Section {
    pub(super) fn fit(m: &PanelMount, s: &GuardSection, bite: f32) -> Self {
        let half = s.panel_thickness * 0.5;
        let inner = half + m.gasket_m;
        let outer = inner + m.jaw_wall_m;
        Self {
            // A narrow heel seats inside either a round or a square post;
            // perpendicular mounts at a corner have separate roots.
            root: -m.edge_clearance_m - s.post_width * 0.14,
            neck_half: (outer * 0.78).min(s.post_width * 0.23),
            web: -m.edge_clearance_m * 0.34,
            contact: half - 0.00025,
            inner,
            outer,
            tip: bite,
            bevel: (m.jaw_wall_m * 0.42).min(bite * 0.38),
            lip: bite + 0.0018_f32.min(bite * 0.18),
        }
    }
}

pub(super) struct Edge {
    pub edge: Vec3,
    pub along: Vec3,
    pub normal: Vec3,
    pub rake: f32,
}

impl Edge {
    pub(super) fn extrude(&self, g: &mut Geometry, s: &Section, heights: (f32, f32), lining: bool) {
        let point = |p: Vec2, h| {
            self.edge + self.along * p.x + self.normal * p.y + Vec3::Y * (h + p.x * self.rake)
        };
        let cap_normal = Vec3::Y - self.along * self.rake;
        let cap = |g: &mut Geometry, outline: [Vec2; 4]| {
            for (h, outward) in [(heights.0, -cap_normal), (heights.1, cap_normal)] {
                g.metric_quad(outline.map(|p| point(p, h)), outward);
            }
        };
        let wall = |g: &mut Geometry, p, q, outward: Vec2| {
            g.metric_quad(
                [
                    point(p, heights.0),
                    point(q, heights.0),
                    point(q, heights.1),
                    point(p, heights.1),
                ],
                self.along * outward.x + self.normal * outward.y,
            );
        };
        let pt = Vec2::new;
        if !lining {
            cap(
                g,
                [
                    pt(s.root, -s.neck_half),
                    pt(s.web, -s.outer),
                    pt(s.web, s.outer),
                    pt(s.root, s.neck_half),
                ],
            );
            wall(
                g,
                pt(s.root, -s.neck_half),
                pt(s.root, s.neck_half),
                -Vec2::X,
            );
            wall(g, pt(s.web, -s.contact), pt(s.web, s.contact), Vec2::X);
        }
        for sign in [-1., 1.] {
            let p = |x, z| pt(x, sign * z);
            let outside = Vec2::Y * sign;
            if lining {
                cap(
                    g,
                    [
                        p(s.web, s.contact),
                        p(s.lip, s.contact),
                        p(s.lip, s.inner),
                        p(s.web, s.inner),
                    ],
                );
                wall(g, p(s.tip, s.inner), p(s.lip, s.inner), outside);
                wall(g, p(s.lip, s.contact), p(s.lip, s.inner), Vec2::X);
                wall(g, p(s.web, s.contact), p(s.lip, s.contact), -outside);
            } else {
                let shoulder = s.tip - s.bevel;
                let nose = s.outer - s.bevel;
                cap(
                    g,
                    [
                        p(s.web, s.inner),
                        p(shoulder, s.inner),
                        p(shoulder, s.outer),
                        p(s.web, s.outer),
                    ],
                );
                cap(
                    g,
                    [
                        p(shoulder, s.inner),
                        p(s.tip, s.inner),
                        p(s.tip, nose),
                        p(shoulder, s.outer),
                    ],
                );
                wall(g, p(s.root, s.neck_half), p(s.web, s.outer), outside);
                wall(g, p(s.web, s.outer), p(shoulder, s.outer), outside);
                wall(g, p(shoulder, s.outer), p(s.tip, nose), Vec2::X + outside);
                wall(g, p(s.tip, nose), p(s.tip, s.inner), Vec2::X);
            }
        }
    }
}
